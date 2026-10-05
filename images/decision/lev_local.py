"""Run `lev serve` on the Lev adapter and its Qwen3.5-4B backbone already on disk (no Hub download)."""
import os
import sys

import torch

# Lev is a 4B model: on the CPU a call takes seconds. So when the GPU was asked for and can't be opened
# (on the GB10 CUDA needs truly free memory, and the page cache can hold nearly all of it), stop and say so
# instead of quietly serving at CPU speed.
if os.environ.get("LEV_DEVICE", "cuda").startswith("cuda"):
    try:
        torch.zeros(1, device="cuda")
    except Exception as e:
        print(f"lev: the GPU is not available ({str(e).splitlines()[0]}). Free memory (drop the page cache), then start it again.", flush=True)
        sys.exit(3)

from lev.cli import main

main(["serve", "--checkpoint", "interfaze-ai/lev", "--model-cache", "/models/lev/hub",
      "--host", os.environ.get("LEV_HOST", "127.0.0.1"), "--port", os.environ.get("LEV_PORT", "8201")])
