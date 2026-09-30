"""Model servers already running on the box that DGX-kit didn't start.

Looks at every running container's command line (read-only; never changes a
container), recognises vLLM, SGLang and llama.cpp servers, and works out the
port to scrape and the model and draft they serve. Their live stats then show
beside DGX-kit's own models, marked as not managed.
"""
from __future__ import annotations

import json
import shlex

from .control import LABEL

DEFAULT_PORTS = {"vllm": 8000, "sglang": 30000, "llamacpp": 8080}


def _opt(args: list[str], *names: str) -> str | None:
    """Value of --name X or --name=X (underscores and dashes are interchangeable)."""
    want = {n.replace("_", "-") for n in names}
    for i, a in enumerate(args):
        key, eq, val = a.partition("=")
        if key.replace("_", "-") in want:
            if eq:
                return val
            if i + 1 < len(args):
                return args[i + 1]
    return None


def _engine(args: list[str], image: str) -> str | None:
    joined = " ".join(args)
    if "sglang.launch_server" in joined or "sglang serve" in joined:
        return "sglang"
    if any(a.endswith("llama-server") for a in args) or "llama.cpp" in image:
        return "llamacpp"
    if "vllm" in joined or "vllm" in image:
        if "serve" in args or "vllm.entrypoints" in joined:
            return "vllm"
    return None


def _draft(engine: str, args: list[str]) -> str | None:
    if engine == "vllm":
        spec = _opt(args, "--speculative-config", "--speculative_config")
        if spec:
            try:
                return json.loads(spec).get("model")
            except ValueError:
                pass
        # vLLM also takes the dotted form: --speculative_config.model /path
        return _opt(args, "--speculative-config.model", "--speculative_config.model")
    if engine == "sglang":
        return _opt(args, "--speculative-draft-model-path")
    if engine == "llamacpp":
        return _opt(args, "-md", "--model-draft")
    return None


def parse(args: list[str], image: str = "") -> dict | None:
    """Engine, port, model and draft from a server's command line, or None if it isn't one."""
    engine = _engine(args, image)
    if not engine:
        return None
    if engine == "vllm":
        model = _opt(args, "--model")
        if not model and "serve" in args:
            i = args.index("serve")
            model = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("-") else None
    elif engine == "sglang":
        model = _opt(args, "--model-path", "--model")
    else:
        model = _opt(args, "-m", "--model")
    port = _opt(args, "--port")
    return {
        "engine": engine,
        "port": int(port) if port and port.isdigit() else DEFAULT_PORTS[engine],
        "model": model,
        "served_name": _opt(args, "--served-model-name", "--alias"),
        "draft": _draft(engine, args),
    }


def _args(c) -> list[str]:
    attrs = c.attrs
    args = [attrs.get("Path", "")] + list(attrs.get("Args") or [])
    # A shell wrapper (sh -c "vllm serve ...") hides the real arguments in one string.
    if len(args) >= 3 and args[0].endswith("sh") and args[1] == "-c":
        try:
            return shlex.split(args[2])
        except ValueError:
            return args
    return args


def _host_port(c, port: int) -> int | None:
    if (c.attrs.get("HostConfig") or {}).get("NetworkMode") == "host":
        return port
    for binding in (c.ports or {}).get(f"{port}/tcp") or []:
        if binding.get("HostPort", "").isdigit():
            return int(binding["HostPort"])
    return None  # not reachable from the host


def discover(client) -> list[dict]:
    """Running model servers that DGX-kit doesn't manage."""
    found = []
    for c in client.containers.list():  # running only
        if LABEL in (c.labels or {}):
            continue
        image = (c.attrs.get("Config") or {}).get("Image", "")
        info = parse(_args(c), image)
        if not info:
            continue
        host_port = _host_port(c, info["port"])
        if host_port is None:
            continue
        found.append({**info, "name": c.name, "port": host_port, "image": image, "managed": False})
    return sorted(found, key=lambda i: i["name"])


def _context(args: list[str]) -> int | None:
    val = _opt(args, "--max-model-len", "--context-length", "-c", "--ctx-size")
    return int(val) if val and val.isdigit() else None


def container_models(client) -> dict[str, dict]:
    """Every running container's full id -> the model it serves (for naming GPU processes)."""
    out = {}
    for c in client.containers.list():
        labels = c.labels or {}
        if LABEL in labels:
            try:
                meta = json.loads(labels.get("dgxkit.meta", "{}"))
            except ValueError:
                meta = {}
            out[c.id] = {"key": labels[LABEL], "model": labels[LABEL], "container": c.name,
                         "ctx": meta.get("context_tokens"), "managed": True}
            continue
        args = _args(c)
        info = parse(args, (c.attrs.get("Config") or {}).get("Image", ""))
        model = info and (info["served_name"] or (info["model"] or "").rstrip("/").rsplit("/", 1)[-1] or None)
        out[c.id] = {"key": c.name if info else None, "model": model, "container": c.name,
                     "ctx": _context(args), "managed": False}
    return out


SECRET = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS")


def launch(c) -> dict:
    """How a container was started, for reading: image, command, options, env (secrets hidden), mounts, ports."""
    attrs = c.attrs
    cfg, host = attrs.get("Config") or {}, attrs.get("HostConfig") or {}
    args = _args(c)
    opts: list[list[str | None]] = []
    i = 1
    while i < len(args):
        a = args[i]
        if a.startswith("-"):
            flag, _, val = a.partition("=")
            if not val and i + 1 < len(args) and not args[i + 1].startswith("-"):
                val, i = args[i + 1], i + 1
            opts.append([flag, val or None])
        else:
            opts.append([None, a])
        i += 1
    env = []
    for e in cfg.get("Env") or []:
        k, _, v = e.partition("=")
        if k in ("PATH", "HOME", "HOSTNAME", "LANG", "LC_ALL") or k.startswith(("NV_", "NVIDIA_REQUIRE", "CUDA_VERSION", "PYTHON")):
            continue
        env.append([k, "(hidden)" if any(x in k.upper() for x in SECRET) else v])
    ports = sorted({f"{b.get('HostPort')}→{p}" for p, binds in ((attrs.get("NetworkSettings") or {}).get("Ports") or {}).items() for b in binds or []})
    return {
        "image": cfg.get("Image", ""),
        "command": shlex.join(args),
        "options": opts,
        "env": env,
        "mounts": [[m.get("Source", ""), m.get("Destination", "")] for m in attrs.get("Mounts") or []],
        "ports": ports or (["host network"] if host.get("NetworkMode") == "host" else []),
        "restart": (host.get("RestartPolicy") or {}).get("Name") or "no",
        "shm": host.get("ShmSize"),
    }
