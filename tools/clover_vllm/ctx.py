"""Per-step description of which batch rows are denoise ("preview") blocks. Set by the model state before the forward pass."""
from dataclasses import dataclass

import torch


@dataclass
class StepCtx:
    rows: torch.Tensor      # flat token rows of all preview requests (n_preview * block, request order)
    req_pos: torch.Tensor   # batch positions of the preview requests
    block: int              # tokens per preview block (the canvas)
    clean_idx: torch.Tensor | None = None   # mamba state slot of each preview request (their clean state)


CTX: StepCtx | None = None


def set_ctx(c: StepCtx | None) -> None:
    global CTX
    CTX = c
