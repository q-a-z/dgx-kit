"""Model control: build engine command lines and run one container per model.

DGX-kit only ever touches containers carrying its own label, so anything else
on the box (including model servers someone started by hand) is left alone.
"""
from __future__ import annotations

import json
import os
import shlex
import socket
from dataclasses import dataclass
from pathlib import Path

from .recipes import Recipe
from .sizing import Plan

LABEL = "dgxkit.model"
PORT_RANGE = range(8100, 8200)


def model_dir(models_root: str, repo: str) -> str:
    return str(Path(models_root) / repo.replace("/", "--"))


def weights_path(r: Recipe, models_root: str) -> str:
    """Where the model's weights are: a library folder, or DGX-kit's own download folder."""
    return r.path or model_dir(models_root, r.repo)


def gguf_path(r: Recipe, models_root: str) -> Path:
    """The GGUF file of a llama.cpp model. The name in the recipe is meant relative to the model folder, but people also
    write it relative to the models folder (gguf/model.gguf), so try that too, then look for the file by name in the folder."""
    base, name = Path(weights_path(r, models_root)), r.gguf_file or ""
    for c in (base / name, Path(models_root) / name):
        if c.is_file():
            return c
    hit = next(base.rglob(Path(name).name), None) if name and base.is_dir() else None
    return hit or base / name


def draft_path(r: Recipe, models_root: str) -> str | None:
    if r.draft_path:
        return r.draft_path
    return model_dir(models_root, r.draft_repo) if r.draft_repo else None


def mounts(r: Recipe, models_root: str) -> dict:
    """Read-only bind mounts for everything the container reads, at the same paths."""
    dirs = {models_root, weights_path(r, models_root)}
    if draft_path(r, models_root):
        dirs.add(draft_path(r, models_root))
    return {d: {"bind": d, "mode": "ro"} for d in sorted(dirs)}


def jit_caches(engine: str) -> tuple[dict, dict]:
    """Volumes and environment for engines that compile kernels on first use.

    Compiled kernels live on the host, so they're built once, not on every start. The compile is capped at two
    jobs: uncapped, ninja and nvcc run one compiler per core and the container hits its memory limit and is
    killed mid-build (seen on a Nemotron start). Shared with llmctl's caches when it left them in ~/.cache.
    """
    if engine == "llamacpp":
        return {}, {}
    root = Path(os.environ.get("DGXKIT_CACHE_DIR") or Path.home() / ".cache")
    vols = {}
    for host, inside in (("flashinfer", "/root/.cache/flashinfer"), ("vllm-jit", "/root/.cache/vllm")):
        (root / host).mkdir(parents=True, exist_ok=True)
        vols[str(root / host)] = {"bind": inside, "mode": "rw"}
    return vols, {"MAX_JOBS": "2", "NVCC_THREADS": "1"}


def docker_options(r: Recipe, models_root: str) -> dict:
    """Volumes, environment and container limits, including what an imported llmctl .conf asked for."""
    d = r.docker or {}
    vols = mounts(r, models_root)
    cache_vols, cache_env = jit_caches(r.engine)
    vols.update(cache_vols)
    for v in d.get("volumes") or []:
        parts = str(v).split(":")
        if len(parts) >= 2 and parts[0].startswith("/"):
            vols[parts[0]] = {"bind": parts[1], "mode": parts[2] if len(parts) > 2 else "rw"}
    out = {"volumes": vols, "environment": {**cache_env, **(r.env or {})} or None}
    for key in ("mem_limit", "shm_size"):
        if d.get(key):
            out[key] = d[key]
    return out


def extra_tokens(lines: list[str]) -> list[str]:
    """Extra flags are kept as the user wrote them, a flag and its value per line; the engine gets them as separate arguments."""
    out: list[str] = []
    for line in lines:
        if line.lstrip()[:1] in ("{", "["):  # a bare JSON value from an older import stays one argument
            out.append(line)
        else:
            out += shlex.split(line)
    return out


def engine_command(r: Recipe, p: Plan, port: int, models_root: str) -> list[str]:
    path = weights_path(r, models_root)
    draft = draft_path(r, models_root)
    if r.engine == "vllm":
        cmd = ["vllm", "serve", path, "--served-model-name", r.name,
               "--host", "0.0.0.0", "--port", str(port),
               "--max-model-len", str(p.context_tokens),
               "--kv-cache-memory-bytes", str(p.kv_bytes)]
        util = r.gpu_memory_utilization or p.gpu_fraction
        if util:
            cmd += ["--gpu-memory-utilization", str(util)]
        if r.kv_cache_dtype != "auto":
            cmd += ["--kv-cache-dtype", r.kv_cache_dtype]
        if draft:
            spec = {"method": r.draft_method or "draft_model", "model": draft,
                    "num_speculative_tokens": r.num_speculative_tokens, **(r.speculative_extra or {})}
            if spec["method"] == "auto":
                del spec["method"]  # vLLM reads the method from the draft's own config
            cmd += ["--speculative-config", json.dumps(spec)]
    elif r.engine == "sglang":
        cmd = ["python3", "-m", "sglang.launch_server", "--model-path", path,
               "--served-model-name", r.name, "--host", "0.0.0.0", "--port", str(port),
               "--context-length", str(p.context_tokens),
               "--max-total-tokens", str(p.kv_pool_tokens), "--enable-metrics"]
        if r.kv_cache_dtype == "fp8":
            cmd += ["--kv-cache-dtype", "fp8_e4m3"]
        if draft:
            cmd += ["--speculative-algorithm", (r.draft_method or "eagle").upper(),
                    "--speculative-draft-model-path", draft,
                    "--speculative-num-draft-tokens", str(r.num_speculative_tokens)]
    elif r.engine == "llamacpp":
        if not r.gguf_file:
            raise ValueError("pick a GGUF file first")
        slots = max(1, int(p.concurrency))
        # /app/llama-server is where both the upstream server-cuda image and our GB10 build keep it.
        cmd = ["/app/llama-server", "-m", str(gguf_path(r, models_root)), "--alias", r.name,
               "--host", "0.0.0.0", "--port", str(port),
               # llama.cpp splits -c across slots, so ask for context x slots.
               "-c", str(p.context_tokens * slots), "--parallel", str(slots),
               "-ngl", "999", "--metrics", "--slots"]
    else:
        raise ValueError(f"unknown engine {r.engine}")
    return cmd + extra_tokens(r.extra_args)


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def pick_port(taken: set[int], is_free=port_free) -> int:
    for port in PORT_RANGE:
        if port not in taken and is_free(port):
            return port
    raise RuntimeError(f"no free port in {PORT_RANGE.start}-{PORT_RANGE.stop - 1}")


@dataclass
class StartCheck:
    ok: bool
    problems: list[str]


def check_start(r: Recipe, p: Plan, models_root: str, image_ready: bool) -> StartCheck:
    problems = []
    if not image_ready:
        problems.append(f"image for {r.engine} isn't ready yet")
    path = Path(weights_path(r, models_root))
    if not path.exists():
        problems.append("model isn't downloaded")
    elif (path / ".dgxkit-incomplete").exists():
        problems.append("model download hasn't finished")
    d = draft_path(r, models_root)
    if d and not Path(d).exists():
        problems.append("draft isn't downloaded")
    if r.engine == "llamacpp" and r.gguf_file and path.exists() and not gguf_path(r, models_root).is_file():
        problems.append(f"GGUF file not found: {gguf_path(r, models_root)}")
    if not p.fits:
        problems.append(p.reason or "doesn't fit in free memory")
    return StartCheck(not problems, problems)


class DockerRunner:
    """Starts, stops and inspects DGX-kit's own model containers."""

    def __init__(self, client=None):
        self._client = client

    @property
    def docker(self):
        # Connect on first use, so the dashboard still comes up while Docker is down.
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    def available(self) -> bool:
        try:
            self.docker.ping()
            return True
        except Exception:
            self._client = None
            return False

    def _ours(self, name: str | None = None):
        filt = {"label": f"{LABEL}={name}" if name else LABEL}
        return self.docker.containers.list(all=True, filters=filt)

    def ports_in_use(self) -> set[int]:
        return {int(c.labels.get("dgxkit.port", 0)) for c in self._ours() if c.status == "running"}

    def status(self) -> dict[str, dict]:
        try:
            return {c.labels[LABEL]: {"state": c.status, "port": int(c.labels.get("dgxkit.port", 0)), "id": c.id,
                                      "exit_code": ((getattr(c, "attrs", None) or {}).get("State") or {}).get("ExitCode"),
                                      "engine": c.labels.get("dgxkit.engine"),
                                      "meta": json.loads(c.labels.get("dgxkit.meta", "{}"))}
                    for c in self._ours()}
        except Exception:  # Docker down: show nothing running rather than failing the page
            self._client = None
            return {}

    def start(self, r: Recipe, cmd: list[str], port: int, models_root: str, meta: dict | None = None):
        for old in self._ours(r.name):  # a stopped container from last time
            old.remove(force=True)
        from docker.types import DeviceRequest, Ulimit
        # Images ship their own entrypoints (vllm-openai runs `vllm serve`, llama.cpp runs the
        # server); the command's first word replaces it so every engine starts the same way.
        return self.docker.containers.run(
            r.image, cmd[1:], entrypoint=[cmd[0]], name=f"dgxkit-{r.name}", detach=True,
            # meta (planned context and KV pool) lets a restarted dashboard pick running models back up.
            labels={LABEL: r.name, "dgxkit.port": str(port), "dgxkit.engine": r.engine,
                    "dgxkit.meta": json.dumps(meta or {})},
            network_mode="host", ipc_mode="host", ulimits=[Ulimit(name="memlock", soft=-1, hard=-1)],
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            **docker_options(r, models_root),
        )

    def stop(self, name: str, timeout: int = 30):
        for c in self._ours(name):
            if c.status == "running":
                c.stop(timeout=timeout)
            else:  # crashed, crash-looping or never started: clear it so nothing is left holding the name
                c.remove(force=True)

    def logs(self, name: str, tail: int = 200) -> str:
        cs = self._ours(name)
        return cs[0].logs(tail=tail).decode(errors="replace") if cs else ""
