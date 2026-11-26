"""Prepare separate veloGB10 recipes from current Hugging Face metadata."""
from __future__ import annotations

import argparse

import yaml

from .recipes import Recipe, inspect_repo
from .store import ModelStore

PRESETS = {
    "qwen35-0.8b-velo": {
        "repo": "doth4580/Qwen3.5-0.8B-NVFP4-MIXED",
        "title": "Qwen3.5 0.8B · veloGB10",
        "max_context": 8192,
    },
    "qwen35-9b-velo": {
        "repo": "doth4580/Qwen3.5-9B-NVFP4-FULL",
        "title": "Qwen3.5 9B · veloGB10",
        "max_context": 32768,
    },
    "qwen36-35b-velo": {
        "repo": "doth4580/Qwen3.6-35B-A3B-NVFP4-FULL",
        "title": "Qwen3.6 35B A3B · veloGB10",
        "max_context": 32768,
        "draft_method": "mtp",
        "extra_args": ["--prefix-cache on", "--fp8-prefill off"],
    },
    "qwen38-27b-dflash2-velo": {
        "repo": "doth4580/Qwen3.8-27B-NVFP4-FULL",
        "draft_repo": "doth4580/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED",
        "title": "Qwen3.8 27B DFlash2 · veloGB10",
        "max_context": 32768,
        "draft_method": "dflash2",
        "extra_args": ["--prefix-cache on", "--fp8-prefill on", "--df2-block 16", "--df2-round-shard on"],
    },
    "qwen38-flash-next-heretic-velo": {
        "repo": "jeffpeng3/Qwen3.8-Flash-Next-exl3-3.05bpw-heretic",
        "revision": "02b86bf19a3fc61412c8542f6f05400d48d0fb33",
        "title": "Qwen3.8 Flash Next Heretic EXL3 · veloGB10",
        "max_context": 262144,
        "draft_method": "mtp",
        "quantization": "exl3",
        "notes": "EXL3 3.05 bpw; baked ablation; full 256K context; MTP enabled. Single-node veloGB10 v0.8.0; GX10-2 load validation pending.",
    },
}


def prepare(name: str, token: str | None = None) -> Recipe:
    settings = PRESETS[name]
    r = inspect_repo(settings["repo"], draft_repo=settings.get("draft_repo"), token=token,
                     revision=settings.get("revision"))
    for key, value in settings.items():
        setattr(r, key, value)
    r.name = name
    r.engine = "velogb10"
    r.kv_cache_dtype = "q8" if r.quantization == "exl3" else "auto"
    r.min_concurrency = 4.0
    r.publish = False
    r.notes = settings.get("notes") or "Single-node veloGB10 v0.8.0 candidate; hardware validation pending. Separate engine-specific weights; no LoRA."
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=tuple(PRESETS))
    parser.add_argument("--state-dir", help="Save recipes to this DGX-kit state directory; otherwise print YAML")
    args = parser.parse_args()
    names = list(dict.fromkeys(args.names)) or list(PRESETS)[:3]
    store = ModelStore(args.state_dir) if args.state_dir else None
    if store:
        existing = [name for name in names if store.exists(name)]
        if existing:
            parser.error("recipes already exist: " + ", ".join(existing))
    recipes = [prepare(name) for name in names]
    for r in recipes:
        if store:
            store.save(r)
            print(f"Prepared {r.name}; weights are not downloaded and the model is stopped.")
        else:
            print(yaml.safe_dump(r.to_dict(), sort_keys=False, explicit_start=True), end="")


if __name__ == "__main__":
    main()
