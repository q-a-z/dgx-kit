"""Run laya-serve on the checkpoints already on disk (no Hub download), on the GPU when there is one."""
import os
from pathlib import Path

import torch

# The first GPU call in a fresh process has to be a tiny one: when it is the weight copy itself, the
# GB10 reports "CUDA error: out of memory" and Laya quietly falls back to the CPU (~1 s instead of ~35 ms).
# It can fail too: CUDA needs free memory to open its context, and on the GB10 the page cache (a model or
# an image just read) can hold nearly all of it. Then run on the CPU, and say so, instead of not starting.
if os.environ.get("LAYA_DEVICE", "cuda").startswith("cuda"):
    try:
        torch.zeros(1, device="cuda")
    except Exception as e:
        print(f"laya: the GPU is not available ({str(e).splitlines()[0]}); running on the CPU. Free memory, then restart it to get the GPU back.", flush=True)
        os.environ["LAYA_DEVICE"] = "cpu"

import laya.router as router

root = Path(os.environ.get("LAYA_LOCAL_DIR", "/models/laya"))
found = {name: (str(root / sub) if sub else str(root), None) for name, sub in
         (("english", ""), ("multilingual", "multilingual"), ("typed-decisions", "typed-decisions"))
         if (root / sub / "rl_agent_config.json").is_file()}
# LAYA_MODELS (set by DGX-kit) says which of them to run; without it, every one the folder has.
wanted = [n.strip() for n in os.environ.get("LAYA_MODELS", "").split(",") if n.strip()]
if wanted:
    found = {n: v for n, v in found.items() if n in wanted}
router.DEFAULT_MODELS.update(found)
os.environ["LAYA_MODELS"] = ",".join(found)
os.environ.setdefault("LAYA_MAX_LOADED", str(max(1, len(found))))

from fastapi import HTTPException
from laya import serve

# Only the checkpoints picked are loaded, and a missing one can't be fetched (the container is offline), so a request
# must never be routed to one that isn't here: without a `model`, Laya picks English or multilingual by the text's
# language, so when either is missing the first checkpoint that runs answers instead; naming one that isn't running is refused plainly.
running = list(found)
_resolve = serve._resolve_model


def _resolve_running(model):
    chosen = _resolve(model)
    if chosen is not None and chosen not in running:
        raise HTTPException(status_code=422, detail=f"checkpoint {chosen!r} isn't running here (running: {', '.join(running)})")
    if chosen is None and running and not {"english", "multilingual"} <= set(running):
        return running[0]
    return chosen


serve._resolve_model = _resolve_running
serve.main()
