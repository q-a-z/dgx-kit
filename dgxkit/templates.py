"""Settings templates: named sets of launch settings a model can take in one click.

A template only holds the knobs that make sense across models (context, KV type,
concurrency, draft length, extra flags, publish), never the repo or engine.
DGX-kit ships a few; users save their own from any model's current settings.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .recipes import Recipe

FIELDS = ("kv_cache_dtype", "max_context", "min_context", "min_concurrency", "fill_memory",
          "num_speculative_tokens", "extra_args", "publish")
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

BUILTIN = {
    "compact": {"about": "8K context for one user: the smallest memory use.",
                "settings": {"kv_cache_dtype": "fp8", "max_context": 8192, "min_concurrency": 1.0, "fill_memory": False}},
    "balanced": {"about": "32K context, two requests at once: takes only the memory that needs.",
                 "settings": {"kv_cache_dtype": "auto", "max_context": None, "min_concurrency": 2.0, "fill_memory": False}},
    "long-context": {"about": "One user, 128K context; FP8 KV cache halves its memory.",
                     "settings": {"kv_cache_dtype": "fp8", "max_context": 131072, "min_concurrency": 1.0, "fill_memory": False}},
    "many-users": {"about": "32K context, sized for eight requests at once; FP8 KV cache.",
                   "settings": {"kv_cache_dtype": "fp8", "max_context": 32768, "min_concurrency": 8.0, "fill_memory": False}},
    "all-memory": {"about": "Dedicated box: the longest context that fits, with every free byte for the KV cache.",
                   "settings": {"kv_cache_dtype": "auto", "max_context": None, "min_concurrency": 2.0, "fill_memory": True}},
}


class Templates:
    def __init__(self, state_dir: str):
        self.dir = Path(state_dir) / "templates"

    def _path(self, name: str) -> Path:
        if not NAME.match(name):
            raise ValueError("template names use lowercase letters, digits, dot, dash and underscore")
        return self.dir / f"{name}.yaml"

    def list(self) -> list[dict]:
        out = [{"name": n, "builtin": True, **t} for n, t in BUILTIN.items()]
        if self.dir.exists():
            for p in sorted(self.dir.glob("*.yaml")):
                t = yaml.safe_load(p.read_text()) or {}
                out.append({"name": p.stem, "builtin": False, "about": t.get("about", ""),
                            "settings": t.get("settings", {})})
        return out

    def get(self, name: str) -> dict:
        for t in self.list():
            if t["name"] == name:
                return t
        raise FileNotFoundError(name)

    def save_from(self, name: str, r: Recipe, about: str = "") -> dict:
        if name in BUILTIN:
            raise ValueError(f"{name} is a built-in template")
        path = self._path(name)
        settings = {k: getattr(r, k) for k in FIELDS}
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"about": about or f"Saved from {r.name}", "settings": settings}))
        return self.get(name)

    def delete(self, name: str) -> None:
        if name in BUILTIN:
            raise ValueError(f"{name} is a built-in template")
        self._path(name).unlink()


def apply(r: Recipe, template: dict) -> Recipe:
    for k, v in template["settings"].items():
        if k in FIELDS:
            setattr(r, k, v)
    return r
