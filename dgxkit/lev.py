"""Lev, a decision model on Qwen3.5-4B (a small LoRA adapter and a head), as a service: `lev serve` in a container.

It answers the same typed questions as Laya (POST /v1/systemone) from a model about ten times Laya's size, so it is
kept as a spare that is started on purpose rather than left running. Its server has no API key, so by default it
listens on this machine only. The adapter and the Qwen3.5-4B backbone are read from a Hugging Face cache folder in the
models directory (`.lev`, hidden so the backbone isn't listed as a model); nothing is downloaded here.
"""
from __future__ import annotations

import json
from pathlib import Path

from .laya import GIB, DecisionService, hub_snapshot

REPO = "interfaze-ai/lev"
DEFAULT_BASE = "Qwen/Qwen3.5-4B"


class LevService(DecisionService):
    name = "lev"
    title = "Lev"
    container_name = "dgxkit-lev"
    build = "decision-gb10"
    default_port = 8201
    has_key = False
    can_expose = True
    expose_default = False  # no key: this machine only, unless it is asked to listen on the network
    needs_bytes = 11 * GIB  # 8.7 GiB of weights, the CUDA context and working memory
    about = "Decision model on Qwen3.5-4B: the same typed questions as Laya, a second opinion from a bigger model. Needs about 11 GiB, so it is started on purpose."

    def cache(self) -> Path:
        return Path(self.config()["dir"] or Path(self.models_root) / ".lev")

    def base_repo(self) -> str:
        """The backbone the adapter says it was trained on."""
        snap = hub_snapshot(self.cache(), REPO)
        try:
            return json.loads((snap / "lev_release.json").read_text())["base_model"] if snap else DEFAULT_BASE
        except (OSError, ValueError, KeyError):
            return DEFAULT_BASE

    def checkpoints(self):
        cache = self.cache()
        adapter = hub_snapshot(cache, REPO)
        base = hub_snapshot(cache, self.base_repo())
        ok = bool(adapter and (adapter / "lev_release.json").is_file() and (adapter / "adapter_model.safetensors").exists()
                  and base and (base / "config.json").is_file() and any(base.glob("*.safetensors")))
        return (str(cache) if ok else None), (["lev"] if ok else [])

    def choices(self):
        _, have = self.checkpoints()
        return [{"name": "lev", "about": "adapter on Qwen3.5-4B, about 11 GiB", "found": bool(have)}]

    def missing(self):
        return (f"Lev isn't on this box: put {REPO} and {self.base_repo()} in a Hugging Face cache under {self.cache()}/hub "
                f"(huggingface_hub: snapshot_download(repo, cache_dir='{self.cache()}/hub') for each).")

    def env(self, cfg):
        return {"LEV_DEVICE": cfg["device"], "LEV_PORT": str(cfg["port"]), "LEV_HOST": "0.0.0.0" if cfg["expose"] else "127.0.0.1"}

    def command(self, cfg):
        return ["python3", "/opt/lev_local.py"]

    def volumes(self, folder):
        return {folder: {"bind": "/models/lev", "mode": "rw"}}  # the Hub client wants to write lock files beside what it reads

    def loaded(self, health):
        return [REPO]

    def device_in_use(self, health, cfg):
        return cfg["device"]  # it refuses to start rather than fall back to the CPU, so what was asked for is what runs
