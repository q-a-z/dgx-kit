"""DGX-kit HTTP service: live stats, model control, downloads."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from docker.errors import DockerException
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .api_models import router as models_router
from .api_services import router as services_router
from .collectors.gpu import open_gpu
from .collectors.system import SystemCollector
from .auth import COOKIE, SESSION_SECONDS, Auth
from .downloader import Downloader
from .library import Settings
from .benchmarks import Benchmarks
from .gateway import Gateway, bundled_key
from .images import ImageManager
from .procs import ProcessNamer
from .sampler import Sampler
from .store import ModelStore


@dataclass
class Services:
    root: str
    models_root: str
    state_dir: str
    hf_token: str | None
    sampler: Sampler
    store: ModelStore
    runner: object
    images: ImageManager
    downloader: Downloader
    gateway: Gateway
    bench: Benchmarks | None = None
    auth: Auth | None = None
    settings: Settings | None = None
    external: dict = field(default_factory=dict)  # running model servers DGX-kit didn't start
    actions: deque = field(default_factory=lambda: deque(maxlen=1000))
    updater: object = None  # dgxkit.updater.Updater; tests give a fake
    preparing: dict = field(default_factory=dict)  # model name -> the image it waits for before starting
    firmware_probe: object = None  # tests give a fake; normally fwupd is asked through system_info.probe_firmware
    quick_pending: set = field(default_factory=set)  # models DGX-kit just started, waiting for their first answer
    laya: object = None  # the decision model services (dgxkit/laya.py, lev.py, bekko.py), made on first use; tests give fakes
    lev: object = None
    bekko: object = None
    slots: object = None  # dgxkit.slots.SlotScheduler while GPU time slots are on

    def log(self, action: str, detail: str) -> None:
        self.actions.append({"t": time.time(), "action": action, "detail": detail})


def build_services() -> Services:
    from .control import DockerRunner

    root = os.environ.get("DGXKIT_ROOT", "/")
    state = os.environ.get("DGXKIT_STATE_DIR") or default_state_dir()
    print(f"DGX-kit state folder: {state}", flush=True)
    models = os.environ.get("DGXKIT_MODELS_DIR", os.path.expanduser("~/models"))
    token = os.environ.get("HF_TOKEN") or None
    images = ImageManager(state_dir=state)
    gateway = Gateway(state, port=int(os.environ.get("DGXKIT_GATEWAY_PORT", "4000")),
                      master_key=os.environ.get("LITELLM_MASTER_KEY") or bundled_key(state), image=images.image_for("litellm"),
                      host=os.environ.get("DGXKIT_BIND", "0.0.0.0"), db=True)
    sampler = Sampler(open_gpu(), SystemCollector(root))
    sampler.name_process = ProcessNamer(root)
    return Services(
        root=root, models_root=models, state_dir=state, hf_token=token,
        sampler=sampler,
        store=ModelStore(state), runner=DockerRunner(), images=images,
        downloader=Downloader(models, token), gateway=gateway, bench=Benchmarks(state),
    )


class Login(BaseModel):
    password: str


class NewPassword(BaseModel):
    old: str = ""
    new: str


async def first_run(s: Services) -> None:
    """Pick running models back up, pull missing images if the installer asked, start the gateway."""
    from .api_models import reattach, sync_gateway

    running = await reattach(s)
    if running:
        s.log("reattach", ", ".join(running))

    if os.environ.get("DGXKIT_READONLY") == "1":
        s.log("read-only", "no pulls, no gateway, no container changes")
        return
    if os.environ.get("DGXKIT_PULL_IMAGES") == "yes":
        s.log("pull", "missing images")
        failed = await asyncio.to_thread(s.images.pull_missing)
        if failed:
            s.log("pull failed", ", ".join(failed))
    try:
        await sync_gateway(s)
    except Exception as e:  # Docker down at boot; the next start or stop retries
        s.log("gateway", f"not started: {e}")


async def update_watch(s: Services) -> None:
    """Look on GitHub for a newer version on the schedule in Settings (daily unless changed; never when set to never)."""
    from .api_models import _updater
    from .updater import due
    await asyncio.sleep(90)
    while True:
        try:
            u = _updater(s)
            if due(s.settings.update_check, u.cache.get("checked")):
                await asyncio.to_thread(u.check)
        except Exception as e:
            s.log("update check", f"failed: {e}"[:200])
        await asyncio.sleep(600)


async def firmware_watch(s: Services) -> None:
    """Once a day, note the machine's versions and ask fwupd for updates, so the System tab is current without a click.
    Not in read-only mode (it starts a short-lived container) and never fatal."""
    from . import system_info
    await asyncio.sleep(120)
    while True:
        try:
            if os.environ.get("DGXKIT_READONLY") != "1":
                devices = await asyncio.to_thread(system_info.probe_firmware, s)
                await asyncio.to_thread(system_info.refresh, s, devices)
        except Exception as e:
            s.log("firmware check", f"failed: {e}"[:200])
        await asyncio.sleep(24 * 3600)


def default_state_dir() -> str:
    """The installer's folder when it is there and usable (or we are root and can make it), else one in the user's own
    home, so running from a clone works without root."""
    system = "/var/lib/dgx-kit"
    if os.path.isdir(system) and os.access(system, os.W_OK):
        return system
    if os.geteuid() == 0:
        return system
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "dgx-kit")


def app_version() -> str:
    """The version of the package that is running, from its own metadata (pyproject.toml's version)."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("dgx-kit")
    except PackageNotFoundError:
        return "dev"


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = app.state.services = services or build_services()
        s.auth = s.auth or Auth(s.state_dir)
        paths = [p for p in os.environ.get("DGXKIT_MODEL_PATHS", "").split(":") if p]
        s.settings = s.settings or Settings(s.state_dir, paths or [s.models_root])
        from .api_models import quick_after_up
        s.sampler.on_up = lambda name: asyncio.create_task(quick_after_up(s, name))
        s.downloader.token = s.settings.hf_token or s.downloader.token
        s.sampler.start()
        from .api_models import apply_slots
        await apply_slots(s)
        setup = asyncio.create_task(first_run(s))
        from .api_models import watch_external
        watcher = asyncio.create_task(watch_external(s))
        fw_watch = asyncio.create_task(firmware_watch(s))
        up_watch = asyncio.create_task(update_watch(s))
        yield
        setup.cancel()
        watcher.cancel()
        fw_watch.cancel()
        up_watch.cancel()
        if s.slots:
            await s.slots.stop()  # thaws every engine it froze
        await s.sampler.stop()

    app = FastAPI(title="DGX-kit", lifespan=lifespan)
    app.include_router(models_router)
    app.include_router(services_router)

    @app.exception_handler(ValueError)
    async def bad_value(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(DockerException)
    async def docker_down(request: Request, exc: DockerException):
        return JSONResponse({"detail": "Docker isn't reachable"}, status_code=503)

    @app.exception_handler(FileNotFoundError)
    async def not_found(request: Request, exc: FileNotFoundError):
        return JSONResponse({"detail": "not found"}, status_code=404)

    open_paths = {"/api/login", "/api/me"}
    readonly = os.environ.get("DGXKIT_READONLY") == "1"

    def changes_the_box(method: str, path: str) -> bool:
        """In read-only mode, anything that would touch Docker, images or the models folder."""
        if method not in ("POST", "PUT", "DELETE"):
            return False
        if method == "PUT" and re.fullmatch(r"/api/images/[a-z]+", path):
            return False  # picks which tag DGX-kit uses; nothing on the box changes
        return (path.startswith(("/api/images", "/api/gateway", "/api/downloads", "/api/system/firmware", "/api/system/update", "/api/services"))
                or re.fullmatch(r"/api/models/[^/]+/(start|stop|download)", path) is not None
                or (method == "DELETE" and path.startswith("/api/models/")))

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        auth = request.app.state.services.auth
        if (request.url.path.startswith("/api/") and request.url.path not in open_paths
                and not auth.valid(request.cookies.get(COOKIE))):
            return JSONResponse({"detail": "sign in first"}, status_code=401)
        if readonly and changes_the_box(request.method, request.url.path):
            return JSONResponse({"detail": "read-only mode: DGX-kit won't change containers, images or models here"},
                                status_code=403)
        return await call_next(request)

    @app.get("/api/me")
    async def me(request: Request):
        auth = request.app.state.services.auth
        return {"auth_required": auth.enabled, "signed_in": auth.valid(request.cookies.get(COOKIE)),
                "readonly": readonly, "version": app_version()}

    @app.post("/api/login")
    async def login(body: Login, request: Request):
        s = request.app.state.services
        client = request.client.host if request.client else "?"
        if not s.auth.enabled:
            return {"signed_in": True}
        if s.auth.blocked(client):
            raise HTTPException(429, "too many wrong passwords; wait a few minutes")
        token = await asyncio.to_thread(s.auth.login, body.password, client)
        if not token:
            s.log("sign-in failed", client)
            raise HTTPException(401, "wrong password")
        s.log("sign-in", client)
        resp = JSONResponse({"signed_in": True})
        resp.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True, samesite="strict")
        return resp

    @app.post("/api/logout")
    async def logout():
        resp = JSONResponse({"signed_in": False})
        resp.delete_cookie(COOKIE)
        return resp

    @app.post("/api/password")
    async def change_password(body: NewPassword, request: Request):
        s = request.app.state.services
        try:
            await asyncio.to_thread(s.auth.change, body.old, body.new)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        s.log("password", "changed; other sessions signed out")
        resp = JSONResponse({"changed": True})
        resp.set_cookie(COOKIE, s.auth.new_session(), max_age=SESSION_SECONDS, httponly=True, samesite="strict")
        return resp

    @app.get("/api/snapshot")
    async def snapshot(request: Request):
        return request.app.state.services.sampler.latest

    @app.get("/api/history")
    async def history(request: Request):
        return list(request.app.state.services.sampler.history)

    @app.get("/api/stream")
    async def stream(request: Request):
        s = request.app.state.services.sampler
        q = s.subscribe()

        async def events():
            try:
                while not await request.is_disconnected():
                    try:
                        snap = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"data: {json.dumps(snap)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                s.unsubscribe(q)

        return StreamingResponse(events(), media_type="text/event-stream")

    # The built dashboard page, when present (web/dist in a checkout, DGXKIT_WEB_DIR in the image).
    web = Path(os.environ.get("DGXKIT_WEB_DIR", Path(__file__).resolve().parent.parent / "web" / "dist"))
    if (web / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=web / "assets", check_dir=False), name="assets")

        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
        async def page(path: str, request: Request):
            # Unknown API paths stay 404s; any other GET is the single-page app.
            if path.startswith("api/") or request.method != "GET":
                raise HTTPException(404)
            # A real file in the page folder (the tab icon) is served as itself.
            f = (web / path).resolve()
            if path and f.is_file() and f.is_relative_to(web.resolve()):
                return FileResponse(f)
            return FileResponse(web / "index.html")

    return app


def main():
    import uvicorn

    uvicorn.run(create_app(), host=os.environ.get("DGXKIT_BIND", "0.0.0.0"),
                port=int(os.environ.get("DGXKIT_PORT", "3000")))


if __name__ == "__main__":
    main()
