"""Bekko System One v0 (400M), a decision model, as a service: its own small HTTP server (images/decision/bekko_serve.py).

Bekko ships only a Python class, so the server in the image gives it the same /v1/systemone as Laya. At 395M parameters
it is about Laya's size and speed. The model is read from a Hugging Face cache folder in the models directory
(`.bekko`, hidden); nothing is downloaded here.
"""
from __future__ import annotations

from pathlib import Path

from .laya import GIB, DecisionService, hub_snapshot

REPO = "hotchpotch/bekko-system-one-v0-400m"


class BekkoService(DecisionService):
    name = "bekko"
    title = "Bekko"
    container_name = "dgxkit-bekko"
    build = "decision-gb10"
    default_port = 8202
    has_key = True
    needs_bytes = 3 * GIB  # 1.5 GiB of weights, the CUDA context and working memory
    about = "Decision model (Bekko System One v0, 400M): the same typed questions as Laya from a different model, a second opinion at about Laya's speed."

    def cache(self) -> Path:
        return Path(self.config()["dir"] or Path(self.models_root) / ".bekko")

    def checkpoints(self):
        cache = self.cache()
        snap = hub_snapshot(cache, REPO)
        ok = bool(snap and (snap / "inference_v0.py").is_file() and (snap / "0_BekkoInference" / "model.safetensors").exists())
        return (str(cache) if ok else None), (["bekko-system-one-v0-400m"] if ok else [])

    def choices(self):
        _, have = self.checkpoints()
        return [{"name": "bekko-system-one-v0-400m", "about": "395M parameters, English", "found": bool(have)}]

    def missing(self):
        return (f"Bekko isn't on this box: put {REPO} in a Hugging Face cache under {self.cache()}/hub "
                f"(huggingface_hub: snapshot_download('{REPO}', cache_dir='{self.cache()}/hub', ignore_patterns=['onnx_browser/*'])).")

    def env(self, cfg):
        return {"BEKKO_DEVICE": cfg["device"], "BEKKO_PORT": str(cfg["port"]), "BEKKO_API_KEY": self.key()}

    def command(self, cfg):
        return ["python3", "/opt/bekko_serve.py"]

    def volumes(self, folder):
        return {folder: {"bind": "/models/bekko", "mode": "rw"}}

    def loaded(self, health):
        return health.get("loaded") or []
