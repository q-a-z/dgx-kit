"""LiteLLM gateway: publishes every running model behind one OpenAI-compatible URL.

DGX-kit writes LiteLLM's config from the models that are running and marked
to publish, and restarts its own LiteLLM container whenever that list changes.
Clients keep one base URL and one key while models come and go.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from .control import port_free

LABEL = "dgxkit.gateway"
GENERATION = "3"  # bumped when the container's setup changes, so an older one is replaced
DB_LABEL = "dgxkit.gateway-db"
DB_NAME = "dgxkit-gateway-db"
DB_IMAGE = os.environ.get("DGXKIT_IMAGE_POSTGRES", "postgres:16")
DB_PORT = 5433  # loopback only, off 5432 so it never meets a Postgres someone else runs
DB_VOLUME = "dgxkit-gateway-pg"
MEMORY_LIMIT = "4g"  # a leaking proxy gets killed alone instead of taking the models with it
NAME = "dgxkit-gateway"
IMAGE = os.environ.get("DGXKIT_IMAGE_LITELLM", "ghcr.io/berriai/litellm:main-stable")


def litellm_config(published: dict[str, int]) -> dict:
    """LiteLLM config for {model name: engine port}; the key comes from the container's env."""
    return {
        "model_list": [
            {"model_name": name,
             "litellm_params": {"model": f"openai/{name}", "api_base": f"http://127.0.0.1:{port}/v1", "api_key": "none"}}
            for name, port in sorted(published.items())
        ],
        # No cache and no spend logs: both grow without bound in a long-running proxy.
        "litellm_settings": {"drop_params": True, "disable_cache": True, "disable_spend_logs": True},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }


def _secret(state_dir: str, name: str, prefix: str = "") -> str:
    """A generated secret, made once and kept private in the state dir so restarts and clients agree."""
    import secrets
    path = Path(state_dir) / "litellm" / name
    if path.exists() and path.read_text().strip():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = prefix + secrets.token_hex(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(value)
    return value


def bundled_key(state_dir: str) -> str:
    """The master key of DGX-kit's own LiteLLM."""
    return _secret(state_dir, "master.key", "sk-dgxkit-")


class Gateway:
    def __init__(self, state_dir: str, port: int = 4000, master_key: str | None = None,
                 image: str = IMAGE, host: str = "0.0.0.0", client=None, db: bool = False):
        self.dir = Path(state_dir) / "litellm"
        self.port = port
        self.host = host
        self.master_key = master_key
        self.image = image
        self._client = client
        self.db = db  # run a Postgres beside LiteLLM: its web login and stored keys need one
        self.published: dict[str, int] = {}
        self.problem: str | None = None
        self.remote_problem: str | None = None  # from publishing into a LiteLLM someone else runs

    @property
    def docker(self):
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    def _container(self):
        cs = self.docker.containers.list(all=True, filters={"label": LABEL})
        return cs[0] if cs else None

    def _write(self, text: str) -> bool:
        """Write the config; True when it changed."""
        path = self.dir / "config.yaml"
        if path.exists() and path.read_text() == text:
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)
        return True

    def sync(self, published: dict[str, int]) -> None:
        """Bring the gateway in line with the running models. Never raises."""
        self.published = dict(published)
        changed = self._write(yaml.safe_dump(litellm_config(published), sort_keys=False))
        try:
            c = self._container()
            current = getattr(c, "labels", {}).get(LABEL, GENERATION) == GENERATION if c is not None else False
            if c is not None and c.status == "running" and (current or not self.db):
                if changed:
                    c.restart(timeout=10)
                self.problem = None
                return
            if not self.master_key:
                self.problem = "no gateway key set (LITELLM_MASTER_KEY)"
                return
            if not (c is not None and c.status == "running") and not port_free(self.port):  # a stale one of ours holds it until replaced
                self.problem = f"port {self.port} is already used by something else"
                return
            try:
                self.docker.images.get(self.image)
            except Exception:
                self.problem = f"the LiteLLM image {self.image} isn't pulled yet"
                return
            # One worker, so there is no per-worker state for Redis to share; LiteLLM can't tell and nags otherwise.
            env = {"LITELLM_MASTER_KEY": self.master_key, "NUM_WORKERS": "1", "LITELLM_LOG": "ERROR",
                   "LITELLM_DISABLE_NO_REDIS_WARNING": "true"}
            if self.db:
                self.problem = self._ensure_db()
                if self.problem:
                    return
                env["DATABASE_URL"] = f"postgresql://litellm:{_secret(str(self.dir.parent), 'db.password')}@127.0.0.1:{DB_PORT}/litellm"
            if c is not None:
                c.remove(force=True)
            # Mount the folder, not the file: the config is replaced atomically, which
            # changes its inode, and a file bind mount would keep showing the old one.
            self.docker.containers.run(
                self.image, ["--config", "/app/dgxkit/config.yaml",
                             "--host", self.host, "--port", str(self.port)],
                name=NAME, detach=True, labels={LABEL: GENERATION}, network_mode="host",
                environment=env,
                mem_limit=MEMORY_LIMIT,
                volumes={str(self.dir): {"bind": "/app/dgxkit", "mode": "ro"}},
                restart_policy={"Name": "unless-stopped"},
            )
            self.problem = None
        except Exception as e:  # Docker down or refused; the dashboard keeps working
            from docker.errors import DockerException
            self._client = None
            self.problem = ("Docker isn't reachable" if isinstance(e, DockerException)
                            else f"couldn't manage the gateway container: {e}")

    def _ensure_db(self) -> str | None:
        """Make sure the Postgres container runs and answers; a problem to show, or None."""
        import time
        cs = self.docker.containers.list(all=True, filters={"label": DB_LABEL})
        c = cs[0] if cs else None
        if c is None or c.status != "running":
            if not port_free(DB_PORT):
                return f"port {DB_PORT} is already used by something else"
            try:
                self.docker.images.get(DB_IMAGE)
            except Exception:
                return f"the Postgres image {DB_IMAGE} isn't pulled yet"
            if c is not None:
                c.remove(force=True)
            c = self.docker.containers.run(
                DB_IMAGE, ["-c", f"port={DB_PORT}", "-c", "listen_addresses=127.0.0.1"],
                name=DB_NAME, detach=True, labels={DB_LABEL: "1"}, network_mode="host", mem_limit="1g",
                environment={"POSTGRES_USER": "litellm", "POSTGRES_DB": "litellm",
                             "POSTGRES_PASSWORD": _secret(str(self.dir.parent), "db.password")},
                volumes={DB_VOLUME: {"bind": "/var/lib/postgresql/data", "mode": "rw"}},
                restart_policy={"Name": "unless-stopped"},
            )
        for _ in range(30):
            code = c.exec_run(["pg_isready", "-h", "127.0.0.1", "-p", str(DB_PORT), "-U", "litellm"])[0]
            if code == 0:
                return None
            time.sleep(1)
        return "Postgres didn't come up in 30 seconds"

    def _external(self):
        """_external_now(), remembered for 15 seconds, since every page poll asks."""
        import time
        now = time.time()
        cached = self.__dict__.get("_ext_cache")
        if cached and now - cached[0] < 15:
            return cached[1]
        found = self._external_now()
        self._ext_cache = (now, found)
        return found

    def _external_now(self):
        """A LiteLLM someone else runs on this box: (container name, host port) or None."""
        for c in self.docker.containers.list():
            tags = str((c.attrs.get("Config") or {}).get("Image", ""))  # c.image would cost a Docker call per container
            if "litellm" not in tags.lower() and "litellm" not in c.name.lower():
                continue
            # A compose project named litellm also runs its database and cache; skip those.
            if any(x in tags.lower() for x in ("postgres", "redis", "prometheus", "grafana")):
                continue
            port = None
            for binds in (c.attrs.get("NetworkSettings", {}).get("Ports") or {}).values():
                for b in binds or []:
                    port = port or int(b.get("HostPort") or 0) or None
            if port is None and c.attrs.get("HostConfig", {}).get("NetworkMode") == "host":
                port = 4000
            return c.name, port
        return None

    def status(self) -> dict:
        state, external, db = None, None, None
        port = self.port
        try:
            c = self._container()
            state = c.status if c else None
            if self.db:
                cs = self.docker.containers.list(all=True, filters={"label": DB_LABEL})
                db = cs[0].status if cs else "missing"
            if state != "running":
                found = self._external()
                if found:
                    external, port = found[0], found[1] or self.port
                    state = "running"
        except Exception:
            self._client = None
        return {"state": state, "port": port, "image": self.image, "problem": self.remote_problem if external else self.problem,
                "models": sorted(self.published), "external": external,
                "db": db, "key_ready": bool(self.master_key)}


ID_PREFIX = "dgxkit-"


def sync_remote(url: str, key: str | None, published: dict[str, int], engine_host: str, client=None) -> str | None:
    """Publish to a LiteLLM that DGX-kit doesn't run, through its admin API.

    Only entries DGX-kit added (model_info.id starting with dgxkit-) are ever changed or removed,
    so models someone else put on the gateway stay as they are. Returns a problem, or None.
    """
    import httpx
    base = url[:-3] if url.endswith("/v1") else url
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with (client or httpx.Client(timeout=5.0)) as c:
            r = c.get(f"{base}/model/info", headers=headers)
            if r.status_code in (401, 403):
                return "the gateway rejected the key"
            r.raise_for_status()
            ours = {}
            for m in r.json().get("data", []):
                mid = (m.get("model_info") or {}).get("id") or ""
                if mid.startswith(ID_PREFIX):
                    ours[mid] = m
            want = {f"{ID_PREFIX}{name}": (name, port) for name, port in published.items()}
            for mid, m in ours.items():
                name_port = want.get(mid)
                api_base = (m.get("litellm_params") or {}).get("api_base")
                if name_port is None or api_base != f"http://{engine_host}:{name_port[1]}/v1":
                    c.post(f"{base}/model/delete", headers=headers, json={"id": mid}).raise_for_status()
                    ours[mid] = None
            for mid, (name, port) in want.items():
                if ours.get(mid) is None:
                    c.post(f"{base}/model/new", headers=headers, json={
                        "model_name": name,
                        "litellm_params": {"model": f"openai/{name}", "api_base": f"http://{engine_host}:{port}/v1", "api_key": "none"},
                        "model_info": {"id": mid},
                    }).raise_for_status()
        return None
    except Exception as e:
        return f"couldn't publish to {base}: {str(e)[:160]}"
