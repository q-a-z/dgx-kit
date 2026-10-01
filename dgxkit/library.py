"""Models already on disk: scans the configured folders and sorts what it finds
into models and drafts. Read-only; it never moves or deletes anything.

A model is a folder holding config.json with *.safetensors (or *.bin) weights,
or a folder with *.gguf files. Hugging Face cache folders
(models--org--name/snapshots/<rev>/) are recognised and named org/name.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from .paths import expand_home
from .recipes import Recipe, build_recipe

MAX_DEPTH = 4
DRAFT_ARCH = re.compile(r"eagle|dflash|dspark|medusa|mtp|draft|speculator", re.I)
# Only names that can't belong to a main model: a DFlash-trained main model can be called
# "...-dflash" (ornith-abl-dflash on q's Spark), while EAGLE drafts often keep the base
# architecture (qwen3-coder-eagle reports Qwen3MoeForCausalLM), so the name is what tells.
DRAFT_NAME = re.compile(r"(^|[-_./])(eagle3?|medusa|draft|speculator|dspark)([-_./\\d]|$)", re.I)
DRAFT_MAX_LAYERS = 8  # draft heads are one to a few layers deep (DFlash drafts use up to about 5); main models have 16+


class Settings:
    """User settings kept in <state>/settings.yaml: model folders and draft/model overrides."""

    def __init__(self, state_dir: str, default_paths: list[str]):
        self.file = Path(state_dir) / "settings.yaml"
        self.default_paths = default_paths

    def _load(self) -> dict:
        return (yaml.safe_load(self.file.read_text()) or {}) if self.file.exists() else {}

    def _save(self, data: dict) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(data))
        tmp.replace(self.file)

    @property
    def model_paths(self) -> list[str]:
        return self._load().get("model_paths") or list(self.default_paths)

    def set_model_paths(self, paths: list[str]) -> list[str]:
        clean = []
        for p in paths:
            p = expand_home(p.strip())
            if not p:
                continue
            if not os.path.isabs(p):
                raise ValueError(f"{p}: use a full path")
            if p not in clean:
                clean.append(p)
        data = self._load()
        data["model_paths"] = clean
        self._save(data)
        return self.model_paths

    @property
    def gateway_url(self) -> str | None:
        """The OpenAI-compatible address clients use, e.g. http://spark.local:4000/v1. None means work it out."""
        return self._load().get("gateway_url") or None

    @property
    def gateway_key(self) -> str | None:
        key = self.file.parent / "gateway.key"
        return key.read_text().strip() or None if key.exists() else None

    def set_gateway(self, url: str | None = None, key: str | None = None, clear_key: bool = False) -> None:
        if url is not None:
            url = url.strip().rstrip("/")
            if url and not re.fullmatch(r"https?://[^\s/]+(/\S*)?", url):
                raise ValueError("give a full address like http://spark.local:4000/v1")
            data = self._load()
            data["gateway_url"] = url or None
            self._save(data)
        if key is not None or clear_key:
            path = self.file.parent / "gateway.key"
            path.parent.mkdir(parents=True, exist_ok=True)
            if clear_key or not key.strip():
                path.unlink(missing_ok=True)
            else:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(key.strip())

    @property
    def hf_token(self) -> str | None:
        """The Hugging Face token set in Settings; kept private in the state dir, never returned by the API."""
        f = self.file.parent / "hf.token"
        return f.read_text().strip() or None if f.exists() else None

    def set_hf_token(self, token: str | None) -> None:
        path = self.file.parent / "hf.token"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not token or not token.strip():
            path.unlink(missing_ok=True)
            return
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token.strip())

    @property
    def slo(self) -> dict[str, float]:
        """Latency targets in seconds for SLO goodput (ttft, itl, tpot, e2e)."""
        from .engines.base import DEFAULT_SLO
        return {**DEFAULT_SLO, **(self._load().get("slo") or {})}

    def set_slo(self, slo: dict) -> dict:
        from .engines.base import DEFAULT_SLO
        clean = {}
        for k, v in slo.items():
            if k not in DEFAULT_SLO:
                raise ValueError(f"unknown target {k}")
            if not isinstance(v, (int, float)) or not 0 < v < 3600:
                raise ValueError(f"{k}: give seconds between 0 and 3600")
            clean[k] = float(v)
        data = self._load()
        data["slo"] = clean
        self._save(data)
        return self.slo

    @property
    def layout(self) -> dict | None:
        return self._load().get("layout")

    def set_layout(self, layout: dict | None) -> None:
        data = self._load()
        if layout is None:
            data.pop("layout", None)
        else:
            data["layout"] = layout
        self._save(data)

    @property
    def kinds(self) -> dict[str, str]:
        return self._load().get("kinds") or {}

    def set_kind(self, path: str, kind: str | None) -> None:
        if kind not in (None, "model", "draft"):
            raise ValueError("kind is model or draft")
        data = self._load()
        kinds = data.get("kinds") or {}
        if kind:
            kinds[path] = kind
        else:
            kinds.pop(path, None)
        data["kinds"] = kinds
        self._save(data)


def _size(path: Path, files: list[str]) -> int:
    total = 0
    for f in files:
        try:
            total += (path / f).stat().st_size  # follows HF cache symlinks to the blobs
        except OSError:
            pass
    return total


def _hf_name(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/", 1)
    return None


def describe(path: Path, files: list[str]) -> dict | None:
    weights = [f for f in files if f.endswith((".safetensors", ".bin", ".pt", ".pth"))
               and not f.startswith(("training_args", "optimizer", "scheduler", "rng_state"))]
    ggufs = sorted(f for f in files if f.endswith(".gguf"))
    config = {}
    if "config.json" in files:
        try:
            config = json.loads((path / "config.json").read_text())
        except (OSError, ValueError):
            config = {}
    if not ggufs and not (config and weights):
        return None
    archs = config.get("architectures") or [None]
    arch = archs if isinstance(archs, str) else archs[0]
    if not arch and any(f.startswith("eagle") for f in files):
        arch = "Eagle3"
    quant = (config.get("quantization_config") or {}).get("quant_method")
    name = _hf_name(path) or path.name
    return {
        "name": name,
        "path": str(path),
        "format": "gguf" if ggufs else "safetensors",
        "gguf_files": ggufs,
        "architecture": arch,
        "quantization": quant or ("gguf" if ggufs else None),
        "size_bytes": _size(path, ggufs or weights),
        "kind": "draft" if is_draft(config, arch, name) else "model",
    }


def is_draft(config: dict, arch: str | None, name: str) -> bool:
    if DRAFT_ARCH.search(arch or ""):
        return True
    text = config.get("text_config") or config
    layers = text.get("num_hidden_layers")
    if isinstance(layers, int) and 0 < layers <= DRAFT_MAX_LAYERS:
        return True
    return bool(DRAFT_NAME.search(name))


def explain(path: str) -> dict:
    """What the scanner sees in one folder, for finding out why it is or isn't listed."""
    p = Path(path)
    try:
        files = sorted(os.listdir(p))
    except OSError as e:
        return {"path": path, "error": str(e)}
    try:
        item = describe(p, files)
    except Exception as e:
        return {"path": path, "files": files[:50], "error": f"{type(e).__name__}: {e}"}
    return {"path": path, "files": files[:50], "item": item}


def scan(paths: list[str], kinds: dict[str, str] | None = None) -> dict:
    """Everything that looks like a model under the given folders, split into models and drafts."""
    found, missing, unreadable, incomplete = {}, [], [], []
    for root in paths:
        if not os.path.isdir(root):
            missing.append(root)
            continue
        base_depth = root.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True, onerror=lambda e: unreadable.append(e.filename)):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in ("blobs", "refs", "original"))
            if dirpath.count("/") - base_depth >= MAX_DEPTH:
                dirnames[:] = []
            # Keep walking inside a model: a draft is often kept in a subfolder of the model it serves.
            try:
                item = describe(Path(dirpath), filenames)
            except Exception as e:  # one odd folder mustn't hide the rest
                unreadable.append(f"{dirpath} ({type(e).__name__})")
                continue
            if item:
                found[item["path"]] = item
            elif "config.json" in filenames and not any(f.endswith(".gguf") for f in filenames):
                incomplete.append(dirpath)  # config without weights: still downloading, or weights elsewhere
    for p, kind in (kinds or {}).items():
        if p in found:
            found[p]["kind"] = kind
            found[p]["kind_set_by_user"] = True
    # A draft kept inside its model's folder ("dflash_draft", "eagle3-head") is named after that model too.
    for path, item in found.items():
        parent = found.get(str(Path(path).parent))
        if parent and not item["name"].startswith(parent["name"]):
            item["name"] = f"{parent['name']}/{item['name']}"
    items = sorted(found.values(), key=lambda i: i["name"].lower())
    # An HF snapshot folder's parents hold only refs; don't report those.
    incomplete = [p for p in incomplete if not any(f.startswith(p + "/") or p.startswith(f + "/") for f in found)]
    return {"paths": paths, "missing": missing, "unreadable": sorted(set(unreadable)), "incomplete": sorted(incomplete),
            "models": [i for i in items if i["kind"] == "model"],
            "drafts": [i for i in items if i["kind"] == "draft"]}


def _folder(path: str) -> tuple[dict, list[tuple[str, int | None]]]:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(path)
    siblings = []
    for f in sorted(os.listdir(p)):
        try:
            siblings.append((f, (p / f).stat().st_size))
        except OSError:
            siblings.append((f, None))
    config = {}
    if (p / "config.json").exists():
        config = json.loads((p / "config.json").read_text())
    return config, siblings


def recipe_name(label: str) -> str:
    name = re.sub(r"[^a-z0-9._-]+", "-", label.split("/")[-1].lower()).strip("-._")
    return (name or "model")[:64]


def recipe_from_folder(path: str, draft: str | None = None, gguf_file: str | None = None) -> Recipe:
    """A recipe for weights already on disk; the folders are used in place, read-only."""
    config, siblings = _folder(path)
    label = _hf_name(Path(path)) or f"local/{Path(path).name}"
    draft_config, draft_siblings, draft_label = None, None, None
    if draft:
        draft_config, draft_siblings = _folder(draft)
        draft_label = _hf_name(Path(draft)) or f"local/{Path(draft).name}"
    r = build_recipe(label, config, siblings, draft_repo=draft_label, draft_config=draft_config,
                     draft_siblings=draft_siblings, gguf_file=gguf_file)
    r.name = recipe_name(label)
    r.path, r.draft_path = path, draft
    if not r.config and r.gguf_file:  # a GGUF has no config.json; its header says how big the model is
        from .gguf import read_config
        r.config = read_config(Path(path) / r.gguf_file)
    return r
