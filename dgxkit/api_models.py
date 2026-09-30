"""HTTP routes for models, recipes and downloads."""
from __future__ import annotations

import asyncio
import json
import re
import os
import shutil
import time
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .collectors.system import parse_meminfo
from . import quickcheck
from .benchmarks import TESTS as _ALL
from .options import apply_text, to_text
from .paths import expand_home
from .control import check_start, engine_command, model_dir, pick_port, weights_path
from .engines import adapter_for
from .recipes import Recipe, inspect_repo
from .sizing import plan
from .discover import container_models, discover
from .library import recipe_from_folder, scan
from .templates import Templates, apply

router = APIRouter(prefix="/api")


class InspectRequest(BaseModel):
    repo: str
    draft_repo: str | None = None


def svc(request: Request):
    return request.app.state.services


def available_bytes(s) -> int:
    return parse_meminfo((Path(s.root) / "proc/meminfo").read_text())["MemAvailable"]


def total_bytes(s) -> int | None:
    try:
        return parse_meminfo((Path(s.root) / "proc/meminfo").read_text())["MemTotal"]
    except (OSError, KeyError):
        return None


def plan_for(s, r: Recipe, extra_free: int = 0):
    return plan(r.config, r.weights_bytes + r.draft_weights_bytes, available_bytes(s) + max(0, extra_free),
                kv_dtype=r.kv_cache_dtype, max_context=r.max_context, min_context=r.min_context,
                min_concurrency=r.min_concurrency, kv_cache_bytes=r.kv_cache_bytes, total_bytes=total_bytes(s))


def _exists(s, name: str) -> bool:
    try:
        return s.store.exists(name)
    except ValueError:
        return False


def make_adapter(s, engine: str, port: int):
    adapter = adapter_for(engine, f"http://127.0.0.1:{port}")
    if getattr(s, "settings", None):
        adapter.slo = s.settings.slo
    return adapter


def watch(s, name: str, engine: str, port: int, meta: dict) -> None:
    """Feed a running model's live stats into the sampler."""
    adapter = make_adapter(s, engine, port)
    adapter.static = dict(meta)
    s.sampler.add_engine(name, adapter)


async def quick_after_up(s, name: str) -> None:
    """A model DGX-kit just started has answered for the first time: take the ten-second speed check."""
    if name not in s.quick_pending:  # not a start we made: a model that was already serving people
        return
    s.quick_pending.discard(name)
    live = s.sampler.latest["models"]
    if any((m.get("running") or 0) + (m.get("waiting") or 0) > 0 for m in live.values()):
        result = {"ts": time.time(), "skipped": "other models were busy, so the numbers would be wrong"}
    else:
        c = (await asyncio.to_thread(s.runner.status)).get(name)
        if not c:
            return
        try:
            result = await asyncio.to_thread(quickcheck.run, c["port"], name)
        except Exception as e:
            result = {"ts": time.time(), "error": f"{type(e).__name__}: {str(e)[:120]}"}
    quickcheck.save(s.state_dir, name, result)
    s.log("start check", f"{name}: " + (f"decode {result['decode_mean_tps']} t/s, prefill {result['prefill_mean_tps']} t/s"
                                        if "decode_mean_tps" in result else result.get("skipped") or result.get("error", "")))


async def reattach(s) -> list[str]:
    """After a dashboard restart, pick up the models that kept running."""
    names = []
    for name, c in (await asyncio.to_thread(s.runner.status)).items():
        if c["state"] == "running" and c.get("engine"):
            watch(s, name, c["engine"], c["port"], c.get("meta") or {"port": c["port"]})
            names.append(name)
    return names


def published(s) -> dict[str, int]:
    """Running models that are marked to publish, by name, with their engine port."""
    out = {}
    for name, c in s.runner.status().items():
        if c["state"] != "running":
            continue
        try:
            if s.store.get(name).publish:
                out[name] = c["port"]
        except (FileNotFoundError, ValueError):
            continue  # a container whose settings were removed
    return out


def lan_ip() -> str:
    """This box's address as other containers see it (no packet is sent)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            return sk.getsockname()[0]
    except OSError:
        return "127.0.0.1"


async def sync_gateway(s) -> None:
    """Publish running models: through the configured gateway's API if an address is set, else DGX-kit's own LiteLLM."""
    st = getattr(s, "settings", None)
    url = st.gateway_url if st else None
    host = None
    if not url:
        # Someone else's LiteLLM already runs here: publish into it, never start a second one beside it.
        try:
            found = await asyncio.to_thread(s.gateway.status)
        except Exception:
            found = {}
        if found.get("external"):
            if not gateway_key(s):
                s.gateway.problem = s.gateway.remote_problem = "LiteLLM is already running here; add its key in Settings to publish models to it"
                return
            url = f"http://127.0.0.1:{found.get('port') or 4000}/v1"
            host = lan_ip()
    if url:
        from urllib.parse import urlparse
        from .gateway import sync_remote
        pub = published(s)
        s.gateway.published = dict(pub)
        s.gateway.problem = s.gateway.remote_problem = await asyncio.to_thread(sync_remote, url, gateway_key(s), pub, host or urlparse(url).hostname or "127.0.0.1")
        return
    await asyncio.to_thread(s.gateway.sync, published(s))
    if s.gateway.problem and "isn't pulled" in s.gateway.problem:
        asyncio.create_task(_pull_gateway_image(s))


async def _pull_gateway_image(s) -> None:
    """LiteLLM and its Postgres are part of the install: fetch what's missing once, then start them."""
    from .gateway import DB_IMAGE
    if "Postgres" in (s.gateway.problem or ""):
        s.gateway.problem = f"pulling the Postgres image {DB_IMAGE}…"
        s.log("pull", DB_IMAGE)
        repo, _, tag = DB_IMAGE.rpartition(":")
        try:
            await asyncio.to_thread(s.gateway.docker.images.pull, repo, tag=tag or "latest")
        except Exception as e:
            s.gateway.problem = f"couldn't pull {DB_IMAGE}: {str(e)[:160]}"
            return
        await sync_gateway(s)
        return
    try:
        job = s.images.start_pull("litellm")
    except ValueError:  # already pulling
        return
    s.gateway.problem = f"pulling the LiteLLM image {job.image}…"
    s.log("pull", job.image)
    while s.images.busy("litellm"):
        await asyncio.sleep(2)
    if job.state == "failed":
        s.gateway.problem = f"couldn't pull {job.image}: {job.error}"
        return
    await sync_gateway(s)


def recipe_from(body: dict) -> Recipe:
    known = {f.name for f in fields(Recipe)}
    try:
        return Recipe(**{k: v for k, v in body.items() if k in known})
    except TypeError as e:
        raise HTTPException(422, str(e))


@router.post("/recipes/inspect")
async def inspect(req: InspectRequest, request: Request):
    s = svc(request)
    try:
        r = inspect_repo(req.repo, req.draft_repo, token=hf_token(s))
    except Exception as e:
        raise HTTPException(400, f"couldn't read {req.repo}: {e}")
    return {"recipe": r.to_dict(), "plan": asdict(plan_for(s, r)) if r.config else None}


@router.get("/models")
async def list_models(request: Request):
    s = svc(request)
    running = await asyncio.to_thread(s.runner.status)
    live = s.sampler.latest["models"]
    for name in list(s.sampler.engines):
        c = running.get(name)
        if c is not None and c["state"] not in ("running", "restarting"):  # it crashed: stop scraping it
            s.sampler.remove_engine(name)
    out = []
    by_user = stopped_by_user(s)
    for r in s.store.list():
        d = r.to_dict()
        d.pop("config", None)
        d["options"] = to_text(r) if r.engine == "vllm" else None
        d["quick"] = quickcheck.load(s.state_dir, r.name)
        d["container"] = running.get(r.name)
        c = d["container"]
        if c and c["state"] == "exited" and (by_user.get(r.name) == c.get("id")
                                              or await asyncio.to_thread(shut_down_cleanly, s, r.name, c.get("id"))):
            d["container"] = {**c, "stopped_cleanly": True}
        d["live"] = live.get(r.name)
        d["downloaded"] = Path(weights_path(r, s.models_root)).exists()
        out.append(d)
    return out


@router.post("/models", status_code=201)
async def create_model(body: dict, request: Request):
    s = svc(request)
    r = recipe_from(body)
    try:
        exists = s.store.exists(r.name)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if exists:
        raise HTTPException(409, "a model with that name exists")
    s.store.save(r)
    s.log("create", r.name)
    return r.to_dict()


@router.put("/models/{name}")
async def edit_model(name: str, body: dict, request: Request):
    s = svc(request)
    if not _exists(s, name):
        raise HTTPException(404)
    # The page never receives the model's config, so what it sends back mustn't erase it.
    r = recipe_from({"config": s.store.get(name).config, **body, "name": name})
    if body.get("options") is not None and r.engine == "vllm":
        try:
            r = apply_text(r, body["options"])
        except ValueError as e:
            raise HTTPException(422, f"couldn't read the options: {e}")
    s.store.save(r)
    s.log("edit", name)
    await sync_gateway(s)  # publish may have changed
    return {"saved": True, "applies_on_next_start": name in s.runner.status()}


@router.get("/models/{name}/versions")
async def versions(name: str, request: Request):
    return svc(request).store.versions(name)


@router.post("/models/{name}/restore/{version}")
async def restore(name: str, version: str, request: Request):
    s = svc(request)
    try:
        r = s.store.restore(name, version)
    except ValueError as e:
        raise HTTPException(404, str(e))
    s.log("restore", f"{name} {version}")
    return r.to_dict()


@router.get("/models/{name}/plan")
async def preview_plan(name: str, request: Request):
    s = svc(request)
    return asdict(plan_for(s, s.store.get(name)))


PREVIEW_KEYS = {"max_context", "min_context", "min_concurrency", "kv_cache_dtype", "kv_cache_bytes"}


@router.post("/models/{name}/plan")
async def preview_edit(name: str, body: dict, request: Request, own_bytes: int = 0):
    """Plan for unsaved edits. own_bytes is memory the model holds now and would give back on restart."""
    s = svc(request)
    if not _exists(s, name):
        raise HTTPException(404)
    r = s.store.get(name)
    for k in PREVIEW_KEYS & body.keys():
        setattr(r, k, body[k])
    return asdict(plan_for(s, r, own_bytes))


@router.post("/models/{name}/start")
async def start(name: str, request: Request):
    s = svc(request)
    if not _exists(s, name):
        raise HTTPException(404)
    r = s.store.get(name)
    r.image = r.image or s.images.image_for(r.engine)
    p = plan_for(s, r)
    check = check_start(r, p, s.models_root, s.images.is_ready(r.image))
    if not check.ok:
        raise HTTPException(409, {"problems": check.problems})
    port = pick_port(s.runner.ports_in_use())
    cmd = engine_command(r, p, port, s.models_root)
    meta = {"context_tokens": p.context_tokens, "kv_pool_tokens": p.kv_pool_tokens,
            "max_concurrency": p.concurrency, "port": port}
    s.runner.start(r, cmd, port, s.models_root, meta)
    watch(s, name, r.engine, port, meta)
    s.quick_pending.add(name)  # the speed check runs once it answers
    s.log("start", f"{name} on port {port}")
    await sync_gateway(s)
    return {"port": port, "plan": asdict(p), "command": cmd}


TEST_ORDER = _ALL.split(",")
BENCH_TESTS = {"decode", "complex", "hardcore", "conc", "prefill", "stall", "needle", "tools", "sanity"}


@router.get("/bench")
async def list_bench(request: Request, model: str | None = None):
    return await asyncio.to_thread(svc(request).bench.list, model)


@router.post("/models/{name}/bench", status_code=202)
async def run_bench(name: str, request: Request, body: dict | None = None):
    """Run tools/bench.py against a model that is up. Other busy models would spoil the numbers, so the script itself refuses then."""
    s = svc(request)
    if not _exists(s, name):
        raise HTTPException(404)
    c = (await asyncio.to_thread(s.runner.status)).get(name)
    live = s.sampler.latest["models"].get(name) or {}
    if not c or c["state"] != "running" or not live.get("up"):
        raise HTTPException(409, "the model has to be up and answering first")
    body = body or {}
    tests = [t for t in (body.get("tests") or sorted(BENCH_TESTS)) if t in BENCH_TESTS]
    if not tests:
        raise HTTPException(422, "no known tests picked")
    seqs = next((int(m.group(1)) for a in s.store.get(name).extra_args if (m := re.fullmatch(r"--max-num-seqs\s+(\d+)", a.strip()))), 2)
    try:
        run = s.bench.start(name, c["port"], ",".join(t for t in TEST_ORDER if t in tests), int(body.get("conc") or seqs))
    except ValueError as e:
        raise HTTPException(409, str(e))
    s.log("bench", f"{name}: {run['tests']}")
    return run


@router.get("/bench/{run_id}")
async def get_bench(run_id: str, request: Request):
    run = await asyncio.to_thread(svc(request).bench.get, run_id)
    if run is None:
        raise HTTPException(404)
    return run


@router.post("/bench/{run_id}/stop")
async def stop_bench(run_id: str, request: Request):
    s = svc(request)
    if not await asyncio.to_thread(s.bench.stop, run_id):
        raise HTTPException(404, "not running")
    s.log("bench stop", run_id)
    return {"stopped": True}


_CLEAN: dict[str, bool] = {}


def shut_down_cleanly(s, name: str, container_id: str | None) -> bool:
    """Did the engine log "Application shutdown complete"? vLLM prints it when it is stopped properly, whoever
    stopped it. Remembered per container, since an exited container's log never changes."""
    if not container_id:
        return False
    if container_id not in _CLEAN:
        try:
            _CLEAN[container_id] = "Application shutdown complete" in s.runner.logs(name, tail=60)
        except Exception:
            return False
    return _CLEAN[container_id]


def _stopped_file(s) -> Path:
    return Path(s.state_dir) / "user-stopped.json"


def stopped_by_user(s) -> dict[str, str]:
    """{model: id of the container the dashboard stopped}. A stopped container is left behind and vLLM's exit
    code after a stop is 0 or 1, so the exit code can't say whether it crashed; knowing who stopped it can."""
    try:
        return json.loads(_stopped_file(s).read_text())
    except (OSError, ValueError):
        return {}


def remember_stop(s, name: str, container_id: str) -> None:
    d = stopped_by_user(s)
    d[name] = container_id
    _stopped_file(s).parent.mkdir(parents=True, exist_ok=True)
    _stopped_file(s).write_text(json.dumps(d))


class PublishBody(BaseModel):
    publish: bool


@router.post("/models/{name}/publish")
async def set_publish(name: str, body: PublishBody, request: Request):
    """List the model on the gateway while it runs, or stop listing it. It keeps running either way."""
    s = svc(request)
    if not _exists(s, name):
        raise HTTPException(404)
    r = s.store.get(name)
    if r.publish != body.publish:
        r.publish = body.publish
        s.store.save(r)
        s.log("publish" if body.publish else "unpublish", name)
    await sync_gateway(s)  # LiteLLM gets the new list now; no restart of the model
    return {"publish": r.publish}


@router.post("/models/{name}/stop")
async def stop(name: str, request: Request):
    s = svc(request)
    live = s.sampler.latest["models"].get(name) or {}
    busy = (live.get("running") or 0) + (live.get("waiting") or 0)
    if busy and request.query_params.get("force") != "1":
        raise HTTPException(409, {"problems": [f"{int(busy)} requests running or waiting"], "force": "?force=1"})
    before = (await asyncio.to_thread(s.runner.status)).get(name)
    s.runner.stop(name)
    if before and before.get("id"):
        remember_stop(s, name, before["id"])
    s.sampler.remove_engine(name)
    s.log("stop", name)
    await sync_gateway(s)
    return {"stopped": True}


@router.get("/models/{name}/logs")
async def logs(name: str, request: Request, tail: int = 200):
    return {"logs": svc(request).runner.logs(name, tail=min(tail, 5000))}


@router.delete("/models/{name}")
async def delete(name: str, request: Request):
    """Forget a model. Its files on disk are never touched here; that is /library/delete, from Settings only."""
    s = svc(request)
    if not _exists(s, name):
        raise HTTPException(404)
    if name in s.runner.status():
        s.runner.stop(name)
        s.sampler.remove_engine(name)
    s.store.delete(name)
    s.log("delete", name)
    return {"deleted": True}


def _under(parent: Path, child: str | Path) -> bool:
    c = Path(child).resolve()
    return c == parent or parent in c.parents


def _du(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def _disk_delete_check(s, path: str) -> dict:
    """What deleting this folder would remove, or an HTTPException saying why it can't be done."""
    found = scan(s.settings.model_paths, s.settings.kinds)
    items = found["models"] + found["drafts"]
    item = next((i for i in items if i["path"] == path), None)
    if item is None:
        raise HTTPException(404, "that isn't a model folder DGX-kit found in your model folders")
    p = Path(path)
    if p.is_symlink():
        raise HTTPException(409, "that folder is a link; remove the link by hand if you mean to")
    real = p.resolve()
    roots = [Path(x).resolve() for x in [*s.settings.model_paths, s.models_root]]
    if not any(real != r and r in real.parents for r in roots):
        raise HTTPException(409, "that folder is not inside your model folders")
    live = {n for n, c in s.runner.status().items() if c["state"] in ("running", "restarting", "created")}
    for r in s.store.list():
        if r.name in live and any(x and _under(real, x) for x in (weights_path(r, s.models_root), r.draft_path)):
            raise HTTPException(409, f"{r.name} is running from it; stop it first")
    return {"path": path, "name": item["name"], "folder": p.name, "size_bytes": _du(real),
            "used_by": sorted(r.name for r in s.store.list() if any(x and _under(real, x) for x in (weights_path(r, s.models_root), r.draft_path))),
            "also_removes": sorted(i["name"] for i in items if i["path"] != path and _under(real, i["path"]))}


@router.get("/library/delete-preview")
async def disk_delete_preview(path: str, request: Request):
    return await asyncio.to_thread(_disk_delete_check, svc(request), path)


class DiskDelete(BaseModel):
    path: str
    confirm: str  # the folder's own name, typed


@router.post("/library/delete")
async def disk_delete(body: DiskDelete, request: Request):
    """Permanently delete a model folder from disk. The page only offers it in Settings, after four confirmations."""
    s = svc(request)
    info = await asyncio.to_thread(_disk_delete_check, s, body.path)
    if body.confirm != info["folder"]:
        raise HTTPException(422, "the confirmation doesn't match the folder name")
    await asyncio.to_thread(shutil.rmtree, body.path)
    s.log("delete from disk", f"{body.path} ({info['size_bytes']} bytes)")
    return {"deleted": True, "freed_bytes": info["size_bytes"]}


@router.post("/models/{name}/download")
async def download(name: str, request: Request):
    s = svc(request)
    r = s.store.get(name)
    jobs = []
    try:
        patterns = [r.gguf_file, "*.json", "*.txt", "*.model", "*.jinja"] if r.gguf_file else None
        jobs.append(await s.downloader.start(r.repo, r.weights_bytes, patterns, r.revision))
        if r.draft_repo:
            jobs.append(await s.downloader.start(r.draft_repo, r.draft_weights_bytes))
    except RuntimeError as e:
        raise HTTPException(409, {"problems": [str(e)]})
    s.log("download", name)
    return [j.progress() for j in jobs]


@router.get("/downloads")
async def downloads(request: Request):
    return [j.progress() for j in svc(request).downloader.jobs.values()]


@router.post("/downloads/{repo:path}/pause")
async def pause_download(repo: str, request: Request):
    if not svc(request).downloader.pause(repo):
        raise HTTPException(409, "that download isn't running")
    svc(request).log("pause", repo)
    return {"paused": True}


@router.post("/downloads/{repo:path}/resume")
async def resume_download(repo: str, request: Request):
    if not svc(request).downloader.resume(repo):
        raise HTTPException(409, "that download isn't paused")
    svc(request).log("resume", repo)
    return {"resumed": True}


@router.delete("/downloads/{repo:path}")
async def cancel_download(repo: str, request: Request):
    svc(request).downloader.cancel(repo)
    return {"cancelled": True}


@router.get("/images")
async def images(request: Request):
    # Docker calls block; keep them off the event loop so the rest of the page keeps answering.
    return await asyncio.to_thread(svc(request).images.list)


class GatewaySettings(BaseModel):
    url: str | None = None
    key: str | None = None
    clear_key: bool = False


def gateway_url(s, request: Request, port: int | None = None, host: str | None = None) -> str:
    """The configured address, else this box at the port the running LiteLLM actually listens on."""
    st = getattr(s, "settings", None)
    return (st.gateway_url if st else None) or f"http://{host or request.url.hostname or '127.0.0.1'}:{port or s.gateway.port}/v1"


def gateway_key(s) -> str | None:
    st = getattr(s, "settings", None)
    return (st.gateway_key if st else None) or s.gateway.master_key


async def probe_gateway(url: str, key: str | None) -> dict:
    """Ask the gateway itself: is it answering, and which models does it serve."""
    import httpx
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{url}/models", headers=headers)
        if r.status_code in (401, 403):
            return {"reachable": True, "auth": "rejected", "served": []}
        r.raise_for_status()
        return {"reachable": True, "auth": "ok" if key else "none", "served": sorted(m.get("id", "") for m in r.json().get("data", []))}
    except Exception as e:
        return {"reachable": False, "auth": None, "served": [], "error": str(e)[:200]}


@router.get("/gateway")
async def gateway(request: Request):
    s = svc(request)
    st = await asyncio.to_thread(s.gateway.status)
    url = gateway_url(s, request, st.get("port"))
    # Probe over loopback when no address is set: the box may not route to its own LAN address.
    probe = await probe_gateway(gateway_url(s, request, st.get("port"), "127.0.0.1"), gateway_key(s))
    if probe["reachable"]:
        st["state"], st["problem"] = "running", None
    return {**st, **probe, "url": url}


@router.get("/settings/gateway")
async def gateway_settings(request: Request):
    s = svc(request)
    key = gateway_key(s)
    st = await asyncio.to_thread(s.gateway.status)
    return {"url": s.settings.gateway_url, "url_default": f"http://{request.url.hostname or '127.0.0.1'}:{st.get('port') or s.gateway.port}/v1",
            "key_set": bool(key), "key_hint": f"…{key[-4:]}" if key and len(key) > 8 else None}


@router.put("/settings/gateway")
async def set_gateway_settings(body: GatewaySettings, request: Request):
    s = svc(request)
    try:
        s.settings.set_gateway(body.url, body.key, body.clear_key)
    except ValueError as e:
        raise HTTPException(422, str(e))
    s.log("gateway", "address or key changed")
    return await gateway_settings(request)


def hf_token(s) -> str | None:
    """The Settings token, else the one the service was started with (HF_TOKEN)."""
    return (s.settings.hf_token if getattr(s, "settings", None) else None) or s.hf_token


def _hf_status(s) -> dict:
    tok = hf_token(s)
    return {"set": bool(tok), "hint": f"…{tok[-4:]}" if tok and len(tok) > 8 else None,
            "source": "settings" if s.settings.hf_token else "environment" if s.hf_token else None}


class HfToken(BaseModel):
    token: str | None = None
    clear: bool = False


@router.get("/settings/hf")
async def hf_settings(request: Request):
    return _hf_status(svc(request))


@router.put("/settings/hf")
async def set_hf_settings(body: HfToken, request: Request):
    """Save (after Hugging Face accepts it) or remove the token that model lookups and downloads use."""
    s = svc(request)
    out: dict = {}
    if body.clear or not (body.token or "").strip():
        s.settings.set_hf_token(None)
    else:
        token = body.token.strip()
        try:
            from huggingface_hub import HfApi
            who = await asyncio.to_thread(lambda: HfApi().whoami(token=token))
            out = {"checked": True, "account": who.get("name")}
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403):
                raise HTTPException(422, "Hugging Face rejected that token")
            out = {"checked": False}  # offline or Hugging Face unreachable: keep it, say it wasn't verified
        s.settings.set_hf_token(token)
    s.downloader.token = hf_token(s)  # downloads already queued read it when they start
    s.log("hf token", "changed" if s.settings.hf_token else "removed")
    return {**_hf_status(s), **out}


@router.get("/settings/gateway/key")
async def reveal_gateway_key(request: Request):
    """The key itself, for copying into a client. Signed-in users only, like every other route."""
    return {"key": gateway_key(svc(request))}


class ExtraEnvBody(BaseModel):
    text: str
    apply: bool = False


@router.get("/gateway/extra")
async def get_gateway_extra(request: Request):
    """The gateway's own extra settings file (litellm/extra.env in the state folder) and whether it is live."""
    return await asyncio.to_thread(svc(request).gateway.extra_env_state)


@router.put("/gateway/extra")
async def put_gateway_extra(body: ExtraEnvBody, request: Request):
    """Save the file; with apply, make the gateway again now so the settings take effect (it restarts for a moment)."""
    s = svc(request)
    if len(body.text) > 20000:
        raise HTTPException(413, "that is a lot for a list of settings")
    from .gateway import EXTRA_ENV
    f = s.gateway.dir / EXTRA_ENV
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(body.text if body.text.endswith("\n") or not body.text else body.text + "\n")
    tmp.replace(f)
    s.log("litellm settings", "applied" if body.apply else "saved")
    if body.apply:
        await sync_gateway(s)
    return await asyncio.to_thread(s.gateway.extra_env_state)


@router.post("/gateway/sync")
async def gateway_sync(request: Request):
    s = svc(request)
    await sync_gateway(s)
    return await asyncio.to_thread(s.gateway.status)


def _engine(s, engine: str) -> None:
    if engine not in s.images.defaults:
        raise HTTPException(404, f"unknown engine {engine}")


@router.post("/images/{engine}/pull", status_code=202)
async def pull_image(engine: str, request: Request):
    s = svc(request)
    _engine(s, engine)
    job = s.images.start_pull(engine)
    s.log("pull", job.image)
    return job.view()


@router.post("/images/builds/{build}", status_code=202)
async def build_image(build: str, request: Request, make_default: bool = False):
    """Build a local image (llama.cpp for the GB10, or a patched vLLM); make_default switches its engine to it."""
    s = svc(request)
    job = s.images.start_build(build, make_default=make_default)
    s.log("build", job.image)
    return job.view()


@router.get("/images/choices/{engine}")
async def image_choices(engine: str, request: Request):
    """Images one model can be set to run on, for the model editor."""
    s = svc(request)
    _engine(s, engine)
    return {"default": s.images.image_for(engine), "choices": s.images.choices(engine)}


class ImageChoice(BaseModel):
    image: str | None = None


@router.put("/images/{engine}")
async def choose_image(engine: str, body: ImageChoice, request: Request):
    """Point an engine at another tag; null goes back to the pinned default. Applies on next start."""
    s = svc(request)
    _engine(s, engine)
    s.images.set_image(engine, body.image)
    s.log("image", f"{engine} uses {s.images.image_for(engine)}")
    if engine == "litellm":
        s.gateway.image = s.images.image_for(engine)
    return {"image": s.images.image_for(engine)}


@router.post("/images/clean")
async def clean_images(request: Request):
    s = svc(request)
    in_use = {r.image for r in s.store.list() if r.image}
    removed = await asyncio.to_thread(s.images.remove_unused, in_use)
    s.log("clean", ", ".join(removed) or "nothing to remove")
    return {"removed": removed}


@router.get("/log")
async def action_log(request: Request):
    return list(svc(request).actions)


def templates(request: Request) -> Templates:
    return Templates(svc(request).state_dir)


@router.get("/templates")
async def list_templates(request: Request):
    return templates(request).list()


class SaveTemplate(BaseModel):
    name: str
    about: str = ""


@router.post("/models/{name}/save-template", status_code=201)
async def save_template(name: str, body: SaveTemplate, request: Request):
    s = svc(request)
    if not _exists(s, name):
        raise HTTPException(404)
    t = templates(request).save_from(body.name, s.store.get(name), body.about)
    s.log("template", f"saved {body.name} from {name}")
    return t


@router.post("/models/{name}/apply-template/{template}")
async def apply_template(name: str, template: str, request: Request):
    """Apply a template's settings to a model; the old settings stay in its version history."""
    s = svc(request)
    if not _exists(s, name):
        raise HTTPException(404)
    r = apply(s.store.get(name), templates(request).get(template))
    s.store.save(r)
    s.log("template", f"applied {template} to {name}")
    await sync_gateway(s)
    return {"recipe": r.to_dict(), "plan": asdict(plan_for(s, r)) if r.config else None,
            "applies_on_next_start": name in s.runner.status()}


@router.delete("/templates/{template}")
async def delete_template(template: str, request: Request):
    templates(request).delete(template)
    svc(request).log("template", f"deleted {template}")
    return {"deleted": True}


@router.get("/library")
async def library(request: Request):
    """Models and drafts found in the configured folders, whether or not DGX-kit set them up."""
    s = svc(request)
    found = await asyncio.to_thread(scan, s.settings.model_paths, s.settings.kinds)
    used = {r.path: r.name for r in s.store.list() if r.path}
    for item in found["models"] + found["drafts"]:
        item["used_by"] = used.get(item["path"])
    return found


class Paths(BaseModel):
    paths: list[str]


def use_local_image(s, engine: str, image: str | None, notes: list[str]) -> str | None:
    """An imported conf names the image of the machine it came from. When that isn't here, use the GB10 build of
    the same vLLM release, and say so; it may still need building in Settings -> Engine images."""
    from .images import BUILDS, build_for
    if engine != "vllm" or not image or s.images.is_ready(image):
        return image
    key = build_for(image)
    if key is None:
        return image
    tag = BUILDS[key]["tag"]
    notes.append(f"{image} isn't on this box; using {tag}" + ("" if s.images.is_ready(tag) else f" (build it in Settings, Engine images: {BUILDS[key]['about']})"))
    return tag


@router.get("/import/llmctl")
async def import_llmctl_preview(request: Request, folder: str = "~/llm-ctl/models"):
    """What each llmctl .conf in a folder would become. Reads only."""
    from .importer import read_confs
    s = svc(request)
    found = await asyncio.to_thread(scan, s.settings.model_paths, s.settings.kinds)
    items = await asyncio.to_thread(read_confs, folder, found["models"] + found["drafts"])
    for i in items:
        if "recipe" in i:
            i["recipe"]["image"] = use_local_image(s, i["recipe"]["engine"], i["recipe"]["image"], i.setdefault("notes", []))
    for i in items:
        if "recipe" in i:
            i["exists"] = s.store.exists(i["recipe"]["name"])
    return {"folder": expand_home(folder), "items": items}


class ImportConfs(BaseModel):
    files: list[str]


@router.post("/import/llmctl")
async def import_llmctl(body: ImportConfs, request: Request):
    """Save the chosen .conf files as DGX-kit models. Nothing starts, and llmctl's containers are left alone."""
    from .importer import parse_conf, recipe_from_conf, relocate
    s = svc(request)
    done, skipped = [], []
    found = await asyncio.to_thread(scan, s.settings.model_paths, s.settings.kinds)
    library = found["models"] + found["drafts"]
    for f in body.files:
        if not f.endswith(".conf"):
            skipped.append({"file": f, "why": "not a .conf file"})
            continue
        try:
            r, notes = recipe_from_conf(Path(f).stem, parse_conf(Path(f).read_text()))
            relocate(r, library, notes)
            r.image = use_local_image(s, r.engine, r.image, notes)
        except Exception as e:
            skipped.append({"file": f, "why": str(e)})
            continue
        if s.store.exists(r.name):
            c = (await asyncio.to_thread(s.runner.status)).get(r.name)
            if c and c["state"] == "running":
                skipped.append({"file": f, "why": f"{r.name} is running; stop it to import again"})
                continue
            # save() keeps the previous version, so replacing it can be undone from the model's history
        s.store.save(r)
        done.append(r.name)
    if done:
        s.log("import", "from llmctl: " + ", ".join(done))
    return {"imported": done, "skipped": skipped}


@router.post("/import/running/{name}")
async def import_running(name: str, request: Request):
    """Save a model started outside DGX-kit as a DGX-kit model, from its launch command. The container keeps running."""
    from .importer import recipe_from_container
    s = svc(request)
    if name not in s.external:
        raise HTTPException(404)
    r, notes = await asyncio.to_thread(lambda: recipe_from_container(s.runner.docker.containers.get(name)))
    if s.store.exists(r.name):
        raise HTTPException(409, f"a model called {r.name} exists")
    s.store.save(r)
    s.log("import", f"{r.name} from the running container {name}")
    return {"name": r.name, "notes": notes}


@router.get("/library/explain")
async def library_explain(path: str, request: Request):
    """Why one folder is or isn't listed (read-only)."""
    from .library import explain
    roots = [Path(r).resolve() for r in svc(request).settings.model_paths]
    if not any(Path(path).resolve().is_relative_to(r) for r in roots):
        raise HTTPException(422, f"{path} isn't inside a model folder")
    return await asyncio.to_thread(explain, path)


@router.get("/library/paths")
async def get_paths(request: Request):
    """The folders alone, without scanning them, so Settings shows them at once."""
    paths = svc(request).settings.model_paths
    return {"paths": paths, "missing": [p for p in paths if not os.path.isdir(p)]}


@router.put("/library/paths")
async def set_paths(body: Paths, request: Request):
    s = svc(request)
    paths = s.settings.set_model_paths(body.paths)
    s.log("library", "folders: " + ", ".join(paths))
    return {"paths": paths}


class Kind(BaseModel):
    path: str
    kind: str | None = None  # model | draft | None to go back to automatic


@router.put("/library/kind")
async def set_kind(body: Kind, request: Request):
    svc(request).settings.set_kind(body.path, body.kind)
    return {"path": body.path, "kind": body.kind}


class Setup(BaseModel):
    path: str
    draft_path: str | None = None
    name: str | None = None
    gguf_file: str | None = None


@router.post("/library/setup", status_code=201)
async def setup_from_library(body: Setup, request: Request):
    """Make a model entry that runs weights already on disk (plus an optional draft)."""
    s = svc(request)
    roots = [Path(p).resolve() for p in s.settings.model_paths]
    for p in filter(None, (body.path, body.draft_path)):
        if not any(Path(p).resolve().is_relative_to(root) for root in roots):
            raise HTTPException(422, f"{p} isn't inside a model folder")
    r = await asyncio.to_thread(recipe_from_folder, body.path, body.draft_path, body.gguf_file)
    if body.name:
        r.name = body.name
    if s.store.exists(r.name):
        raise HTTPException(409, "a model with that name exists")
    s.store.save(r)
    s.log("create", f"{r.name} from {body.path}")
    return {"recipe": r.to_dict(), "plan": asdict(plan_for(s, r)) if r.config else None}


async def refresh_external(s) -> None:
    """Match the sampler to the model servers DGX-kit didn't start that are running now."""
    try:
        found = await asyncio.to_thread(discover, s.runner.docker)
        if getattr(s.sampler, "name_process", None) is not None:
            s.sampler.name_process.set_containers(await asyncio.to_thread(container_models, s.runner.docker))
    except Exception:  # Docker down; keep what we had
        return
    now = {f["name"]: f for f in found}
    for name in [n for n in s.external if n not in now]:
        s.sampler.remove_engine(name)
    for name, f in now.items():
        if s.external.get(name, {}).get("port") == f["port"]:
            continue
        adapter = make_adapter(s, f["engine"], f["port"])
        adapter.static = {"port": f["port"], "external": True}
        if hasattr(adapter, "read_static"):
            try:
                await adapter.read_static()
                adapter.static.update(port=f["port"], external=True)
            except Exception:
                pass  # still loading; the stats loop shows it as down until it answers
        s.sampler.add_engine(name, adapter)
    s.external = now


async def watch_external(s) -> None:
    while True:
        await refresh_external(s)
        await asyncio.sleep(10)


@router.get("/running")
async def running_elsewhere(request: Request):
    """Model servers on this box that DGX-kit didn't start (view only)."""
    s = svc(request)
    live = s.sampler.latest["models"]
    return [{**f, "live": live.get(name)} for name, f in sorted(s.external.items())]


@router.get("/running/{name}/launch")
async def running_launch(name: str, request: Request):
    """How a model DGX-kit didn't start was launched (read-only)."""
    from .discover import launch
    s = svc(request)
    if name not in s.external:
        raise HTTPException(404)
    return await asyncio.to_thread(lambda: launch(s.runner.docker.containers.get(name)))


@router.get("/running/{name}/logs")
async def running_logs(name: str, request: Request, tail: int = 200):
    """Log tail of a model container DGX-kit didn't start (read-only)."""
    s = svc(request)
    if name not in s.external:
        raise HTTPException(404)
    def read():
        return s.runner.docker.containers.get(name).logs(tail=min(tail, 5000)).decode(errors="replace")
    return {"logs": await asyncio.to_thread(read)}


@router.get("/settings/slo")
async def get_slo(request: Request):
    return svc(request).settings.slo


@router.put("/settings/slo")
async def put_slo(body: dict, request: Request):
    s = svc(request)
    slo = s.settings.set_slo(body)
    for adapter in s.sampler.engines.values():  # applies from the next scrape
        adapter.slo = slo
    return slo


@router.get("/layout")
async def get_layout(request: Request):
    """The dashboard's tile layout, shared by every browser."""
    return {"layout": svc(request).settings.layout}


@router.put("/layout")
async def put_layout(body: dict, request: Request):
    layout = body.get("layout")
    if layout is not None and (not isinstance(layout, dict) or len(str(layout)) > 200_000):
        raise HTTPException(422, "layout must be an object under 200 KB")
    svc(request).settings.set_layout(layout)
    return {"saved": True}
