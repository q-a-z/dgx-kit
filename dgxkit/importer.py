"""Turn a model someone already runs into a DGX-kit recipe, without touching it.

Memory flags (--gpu-memory-utilization, --kv-cache-memory) are kept as the user set them.

Two sources: a running container (its launch command, read through Docker), and
llmctl-style .conf files (KEY=VALUE lines with ENGINE, MODEL and OPTS). Flags DGX-kit
sets itself (port, host, served name) are dropped; the context length,
KV cache type and speculative decoding map to recipe fields; everything else is kept
as extra arguments, in order.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path

from .library import describe, recipe_name
from .paths import expand_home
from .recipes import Recipe, detect_draft_method, detect_quantization

# Flags DGX-kit decides at start: value-taking ones, then switches.
OWNED = {"--port", "--host", "--served-model-name", "--alias",
         "--mem-fraction-static", "--max-total-tokens", "-ngl", "--n-gpu-layers", "--parallel", "-np", "-m", "--model",
         "--model-path"}
SWITCHES = {"--metrics", "--slots", "--enable-metrics"}
CONTEXT = {"--max-model-len", "--context-length", "-c", "--ctx-size"}


def _engine(args: list[str], hint: str | None = None) -> str:
    text = " ".join(args[:3]).lower() + " " + (hint or "").lower()
    if "sglang" in text:
        return "sglang"
    if "llama" in text:
        return "llamacpp"
    return "vllm"


def _split(args: list[str]) -> tuple[str | None, list[tuple[str, str | None]]]:
    """The model positional (vllm serve <model>) and the options as (flag, value) pairs, in order."""
    model, opts, i = None, [], 0
    if args and args[0].endswith("vllm") and len(args) > 1 and args[1] == "serve":
        i = 2
        if i < len(args) and not args[i].startswith("-"):
            model, i = args[i], i + 1
    elif args and ("python" in args[0] or args[0].endswith(("llama-server", "sh"))):
        i = 1
        while i < len(args) and not args[i].startswith("-"):
            i += 1  # "-m sglang.launch_server" style heads are handled below as options
    while i < len(args):
        a = args[i]
        if a.startswith("-"):
            flag, eq, val = a.partition("=")
            if not eq and i + 1 < len(args) and not args[i + 1].startswith("-"):
                val, i = args[i + 1], i + 1
            opts.append((flag, val if (eq or val) else None))
        elif model is None:
            model = a
        i += 1
    return model, opts


def _line(flag: str, val: str | None) -> str:
    """One extra-flags line: the flag with its value, quoted only where a shell would need it."""
    return flag if val is None else f"{flag} {shlex.quote(val)}"


def _bytes(v: str) -> int | None:
    """8000000000, 8G, 8GiB, 7.5g -> bytes."""
    m = re.fullmatch(r"([\d.]+)\s*([kmgt]?)(i?b)?", v.strip().lower())
    if not m:
        return None
    return int(float(m.group(1)) * 1024 ** "_kmgt".index(m.group(2) or "_")) if m.group(2) else int(float(m.group(1)))


def _host(path: str | None, mounts: list[tuple[str, str]]) -> str | None:
    """A path inside the container, as the same folder on the host."""
    if not path or not path.startswith("/"):
        return None
    for src, dst in sorted(mounts, key=lambda m: -len(m[1])):
        if dst and (path == dst or path.startswith(dst.rstrip("/") + "/")):
            return src.rstrip("/") + path[len(dst.rstrip("/")):]
    return path


def recipe_from_args(name: str, args: list[str], image: str | None = None,
                     mounts: list[tuple[str, str]] | None = None, engine_hint: str | None = None) -> tuple[Recipe, list[str]]:
    """A recipe from a launch command, and notes on anything that needs a look."""
    mounts = mounts or []
    engine = _engine(args, engine_hint or image)
    model, opts = _split(args)
    notes: list[str] = []
    extra: list[str] = []
    r = Recipe(name=recipe_name(name), repo="", engine=engine, image=image)
    for flag, val in opts:
        if flag in ("--model", "--model-path", "-m") and val and not val.startswith("sglang"):
            model = val
        elif flag in CONTEXT and val and val.isdigit():
            r.max_context = int(val)
        elif flag in ("--kv-cache-memory", "--kv-cache-memory-bytes") and val:
            r.kv_cache_bytes = _bytes(val)
        elif flag == "--gpu-memory-utilization" and val:
            try:
                r.gpu_memory_utilization = float(val)
            except ValueError:
                extra.append(_line(flag, val))
        elif flag == "--kv-cache-dtype" and val:
            r.kv_cache_dtype = "fp8" if val.startswith("fp8") else val
        elif flag == "--speculative-config" and val:
            try:
                spec = json.loads(val)
                r.draft_path = _host(spec.get("model"), mounts) or None
                if not r.draft_path and spec.get("model"):
                    r.draft_repo = spec["model"]
                r.draft_method = spec.get("method")
                r.num_speculative_tokens = int(spec.get("num_speculative_tokens") or 3)
            except (ValueError, TypeError):
                notes.append("couldn't read --speculative-config; kept it as an extra argument")
                extra.append(_line(flag, val))
        elif flag.replace("-", "_").startswith("__speculative_config.") and val is not None:
            key = flag.replace("-", "_")[len("__speculative_config."):]
            if key == "model":
                r.draft_path = _host(val, mounts)
                if not r.draft_path:
                    r.draft_repo = val
            elif key == "num_speculative_tokens" and val.isdigit():
                r.num_speculative_tokens = int(val)
            elif key == "method":
                r.draft_method = val
            else:
                r.speculative_extra[key] = json.loads(val) if re.fullmatch(r"-?\d+(\.\d+)?|true|false", val) else val
        elif flag in ("--speculative-draft-model-path",) and val:
            r.draft_path = _host(val, mounts)
        elif flag in ("--speculative-algorithm",) and val:
            r.draft_method = val.lower()
        elif flag in ("--speculative-num-draft-tokens", "--speculative-num-steps") and val and val.isdigit():
            r.num_speculative_tokens = int(val)
        elif flag in OWNED or flag in SWITCHES:
            continue
        elif flag == "-m" and val and val.startswith("sglang"):
            continue
        else:
            extra.append(_line(flag, val))
    r.extra_args = extra
    if engine == "llamacpp" and model and model.endswith(".gguf"):
        r.gguf_file = os.path.basename(model)
        model = os.path.dirname(model)
    host = _host(model, mounts)
    if host:
        r.path = host
        r.repo = f"local/{Path(host).name}"
        _fill_from_disk(r, host, notes)
    elif model:
        r.repo = model
        notes.append(f"{model} isn't a folder on this box; DGX-kit will download it from Hugging Face")
    else:
        notes.append("couldn't find the model in the command")
    if r.draft_path:
        _fill_draft(r, r.draft_path, notes)
    if (r.draft_path or r.draft_repo) and not r.draft_method:
        r.draft_method = "auto"
    return r, notes


def _fill_from_disk(r: Recipe, path: str, notes: list[str]) -> None:
    try:
        item = describe(Path(path), sorted(os.listdir(path)))
    except OSError:
        notes.append(f"can't read {path} from here; sizes will fill in when it's readable")
        return
    if not item:
        notes.append(f"no weights found in {path}")
        return
    r.weights_bytes = item["size_bytes"]
    try:
        r.config = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        r.config = {}
    r.quantization = detect_quantization(r.config, item["gguf_files"] or os.listdir(path))


def _fill_draft(r: Recipe, path: str, notes: list[str]) -> None:
    try:
        files = sorted(os.listdir(path))
        item = describe(Path(path), files)
        cfg = json.loads((Path(path) / "config.json").read_text()) if "config.json" in files else {}
    except (OSError, ValueError):
        notes.append(f"can't read the draft at {path}")
        return
    if item:
        r.draft_weights_bytes = item["size_bytes"]
    if not r.draft_method:
        found = detect_draft_method(cfg, Path(path).name)
        # Leave "draft_model" and friends to the engine: vLLM reads the method from the draft's config.
        r.draft_method = found if found in ("eagle3", "mtp") else "auto"


_TOKEN = re.compile(r"""\{[^\s]*\}|\[[^\s]*\]|"(?:[^"\\]|\\.)*"|'[^']*'|[^\s]+""")


_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def _base_env() -> dict[str, str]:
    """What a shell would already know when it read the file: the person's home, not the service's."""
    home = os.environ.get("DGXKIT_HOME") or os.path.expanduser("~")
    return {"HOME": home, "USER": os.environ.get("USER") or os.path.basename(home)}


def _expand(text: str, env: dict[str, str]) -> str:
    """$VAR / ${VAR} from env, and a leading ~/ as the home, the way a shell reads an unquoted word."""
    if text.startswith("~/"):
        text = env.get("HOME", "~") + text[1:]
    return _VAR.sub(lambda m: env.get(m.group(1), m.group(0)), text)


def _tokens(text: str, env: dict[str, str]) -> list[str]:
    """Shell-like words, but JSON ({...}) stays exactly as written, and $VAR / ${VAR} expand from the file."""
    out = []
    for line in text.splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        for tok in _TOKEN.findall(line):
            if tok[0] in "\"'" and tok[-1] == tok[0] and len(tok) > 1:
                tok = tok[1:-1]
            tok = _expand(tok, env)
            if tok != "\\":
                out.append(tok)
    return out


def parse_conf(text: str) -> dict:
    """An llmctl-style .conf: KEY=VALUE lines and KEY=( ... ) arrays, which may span lines."""
    out: dict = {}
    base = _base_env()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = re.sub(r"^\s*export\s+", "", lines[i]).strip()
        i += 1
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m or line.startswith("#"):
            continue
        key, val = m.group(1).upper(), m.group(2)
        if val.startswith("("):
            body = val[1:]
            while ")" not in _strip_quoted(body) and i < len(lines):
                body += "\n" + lines[i]
                i += 1
            cut = _close(body)
            out[key] = _tokens(body[:cut], {**base, **{k: v for k, v in out.items() if isinstance(v, str)}})
            continue
        try:
            parts = shlex.split(val, comments=True)
        except ValueError:
            parts = [val.strip('"\'')]
        out[key] = _expand(" ".join(parts), {**base, **{k: v for k, v in out.items() if isinstance(v, str)}})
    return out


def _strip_quoted(s: str) -> str:
    return re.sub(r""""[^"]*"|'[^']*'|\{[^\s]*\}""", "", s)


def _close(body: str) -> int:
    """Index of the ) that ends an array, skipping quoted text and JSON."""
    depth, quote = 0, None
    for j, ch in enumerate(body):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ")" and depth == 0:
            return j
    return len(body)


def recipe_from_conf(name: str, conf: dict) -> tuple[Recipe, list[str]]:
    """A recipe from an llmctl .conf. CMD holds the real command; MODEL and OPTS are the user's labels."""
    engine = str(conf.get("ENGINE") or "vllm").lower()
    cmd = conf.get("CMD")
    if isinstance(cmd, list) and cmd:
        args = cmd
    else:
        head = {"sglang": ["python3", "-m", "sglang.launch_server", "--model-path"],
                "llamacpp": ["llama-server", "-m"], "llama.cpp": ["llama-server", "-m"]}.get(engine, ["vllm", "serve"])
        args = head + [str(conf.get("MODEL") or "")] + shlex.split(str(conf.get("OPTS") or ""))
    label = conf.get("LITELLM_MODEL_NAME") or _served(args) or name
    r, notes = recipe_from_args(str(label), args, image=conf.get("IMAGE") or None, engine_hint=engine)
    if isinstance(cmd, list) and cmd:
        r.notes = " · ".join(str(conf[k]) for k in ("MODEL", "OPTS") if conf.get(k))
    env = conf.get("ENV")
    pairs = env if isinstance(env, list) else shlex.split(env or "")
    r.env = dict(p.split("=", 1) for p in pairs if "=" in p)
    for k in ("DOCKER_MEM", "MEM", "MEM_LIMIT"):
        if conf.get(k):
            r.docker["mem_limit"] = str(conf[k])
    for k in ("SHM", "SHM_SIZE", "DOCKER_SHM"):
        if conf.get(k):
            r.docker["shm_size"] = str(conf[k])
    for k in ("EXTRA_DOCKER_ARGS", "DOCKER_ARGS", "DOCKER_OPTS"):
        if conf.get(k):
            left = _docker_args(conf[k] if isinstance(conf[k], list) else shlex.split(str(conf[k])), r)
            if left:
                notes.append(f"{k}: not used: {' '.join(left)}")
    known = {"MODEL", "MODEL_PATH", "ENGINE", "OPTS", "NAME", "IMAGE", "PORT", "CMD", "LITELLM_MODEL_NAME", "ENV",
             "DOCKER_MEM", "MEM", "MEM_LIMIT", "SHM", "SHM_SIZE", "DOCKER_SHM", "EXTRA_DOCKER_ARGS", "DOCKER_ARGS", "DOCKER_OPTS"}
    skipped = sorted(k for k in conf if k not in known)
    if skipped:
        notes.append("not imported: " + ", ".join(skipped) + " (DGX-kit sizes memory and runs containers itself)")
    return r, notes


# docker run flags DGX-kit already sets the same way (GPUs, host network and IPC, detach, cleanup).
_DOCKER_SAME = {"--gpus", "--runtime", "--network", "--net", "--ipc", "-d", "--detach", "--rm", "-it", "-i", "-t", "--init"}


def _docker_args(args: list[str], r: Recipe) -> list[str]:
    """Take env, volumes, shm and memory from extra docker run flags; return what's left unused."""
    left, i = [], 0
    while i < len(args):
        a = args[i]
        flag, eq, val = a.partition("=")
        takes = flag in ("-e", "--env", "-v", "--volume", "--shm-size", "-m", "--memory", "--gpus", "--runtime",
                         "--network", "--net", "--ipc", "--ulimit", "--name", "-p", "--publish")
        if takes and not eq and i + 1 < len(args):
            val, i = args[i + 1], i + 1
        if flag in ("-e", "--env") and "=" in val:
            k, v = val.split("=", 1)
            r.env[k] = v
        elif flag in ("-v", "--volume") and val:
            r.docker.setdefault("volumes", []).append(val)
        elif flag == "--shm-size":
            r.docker["shm_size"] = val
        elif flag in ("-m", "--memory"):
            r.docker["mem_limit"] = val
        elif flag in _DOCKER_SAME or flag in ("--name", "-p", "--publish"):
            pass
        else:
            left += [a] + ([val] if takes and not eq else [])
        i += 1
    return left


def _served(args: list[str]) -> str | None:
    for i, a in enumerate(args):
        if a in ("--served-model-name", "--alias") and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--served-model-name="):
            return a.split("=", 1)[1]
    return None


def relocate(r: Recipe, library: list[dict], notes: list[str] | None = None) -> list[str]:
    """Point the recipe at weights that are on this machine.

    A conf written on another machine names its folders there (/home/other/models/x). When that path
    isn't here, use the model folder of the same name from the library, and read its size and
    architecture from there. `notes` (from building the recipe) are edited in place and returned.
    """
    notes = [] if notes is None else notes
    by_name: dict[str, list[str]] = {}
    for item in library:
        by_name.setdefault(os.path.basename(item["path"].rstrip("/")), []).append(item["path"])
    for attr in ("path", "draft_path"):
        p = getattr(r, attr)
        if not p or os.path.exists(p):
            continue
        name = os.path.basename(p.rstrip("/"))
        found = by_name.get(name, [])
        if len(found) == 1:
            setattr(r, attr, found[0])
            notes[:] = [n for n in notes if p not in n]  # what was said about the old path no longer holds
            notes.append(f"{name} isn't at {p} here; using {found[0]}")
            (_fill_from_disk if attr == "path" else _fill_draft)(r, found[0], notes)
        else:
            notes.append(f"weights not found at {p}" + (f" ({len(found)} folders called {name}; not guessing)" if found else ""))
    return notes


def read_confs(folder: str, library: list[dict] | None = None) -> list[dict]:
    """Every *.conf in a folder (not .bak copies), as the recipe it would become. Nothing is saved."""
    out = []
    for f in sorted(Path(expand_home(folder)).glob("*.conf")):
        try:
            conf = parse_conf(f.read_text())
            r, notes = recipe_from_conf(f.stem, conf)
            if library is not None:
                relocate(r, library, notes)
        except Exception as e:  # one broken file mustn't hide the others
            out.append({"file": str(f), "error": f"{type(e).__name__}: {e}"})
            continue
        out.append({"file": str(f), "recipe": r.to_dict() | {"config": {}}, "notes": notes, "port": conf.get("PORT")})
    return out


def recipe_from_container(c) -> tuple[Recipe, list[str]]:
    """A recipe from a running container's own launch command and mounts. The container is only read."""
    from .discover import _args
    attrs = c.attrs
    args = _args(c)
    mounts = [(m.get("Source", ""), m.get("Destination", "")) for m in attrs.get("Mounts") or []]
    image = (attrs.get("Config") or {}).get("Image")
    name = _served(args) or c.name
    r, notes = recipe_from_args(name, args, image=image, mounts=mounts)
    env = [e for e in (attrs.get("Config") or {}).get("Env") or [] if e.split("=", 1)[0].startswith(("VLLM_", "HF_HUB_", "NCCL_", "TORCH_", "SGLANG_"))]
    r.env = dict(e.split("=", 1) for e in env)
    return r, notes
