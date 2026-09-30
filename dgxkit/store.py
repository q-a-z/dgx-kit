"""Model settings on disk: one YAML file per model, with every saved version kept."""
from __future__ import annotations

import re
import time
from dataclasses import fields
from pathlib import Path

import yaml

from .recipes import Recipe

NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ModelStore:
    def __init__(self, state_dir: str):
        self.dir = Path(state_dir) / "models"
        self.history = Path(state_dir) / "models-history"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.history.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        if not NAME.match(name):
            raise ValueError("model names use lowercase letters, digits, dot, dash and underscore")
        return self.dir / f"{name}.yaml"

    def list(self) -> list[Recipe]:
        return [self.get(p.stem) for p in sorted(self.dir.glob("*.yaml"))]

    def get(self, name: str) -> Recipe:
        data = yaml.safe_load(self._path(name).read_text()) or {}
        known = {f.name for f in fields(Recipe)}
        return Recipe(**{k: v for k, v in data.items() if k in known})

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def save(self, r: Recipe) -> None:
        path = self._path(r.name)
        if path.exists():  # keep the old version before overwriting
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (self.history / f"{r.name}.{stamp}.yaml").write_text(path.read_text())
        tmp = path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(r.to_dict(), sort_keys=False))
        tmp.replace(path)  # atomic on the same filesystem

    def versions(self, name: str) -> list[str]:
        self._path(name)  # validates the name before it goes into a glob
        return sorted(p.name for p in self.history.glob(f"{name}.*.yaml"))

    def restore(self, name: str, version: str) -> Recipe:
        if version not in self.versions(name):  # also blocks path tricks in `version`
            raise ValueError("no such version")
        data = yaml.safe_load((self.history / version).read_text())
        r = Recipe(**data)
        self.save(r)
        return r

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)
