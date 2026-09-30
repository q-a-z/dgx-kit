"""A vLLM model's settings as one block of text, the way an llmctl .conf writes them.

to_text() writes the recipe as engine flags, one flag with its value per line; apply_text() reads
edited text back into a recipe with the importer's own parser, so both directions agree. A line
that is deleted clears its setting; DGX-kit sizes context and KV cache itself when they aren't written.
"""
from __future__ import annotations

import json
import shlex
from dataclasses import replace

from .control import extra_tokens
from .importer import recipe_from_args
from .recipes import Recipe

FIELDS = ("max_context", "kv_cache_bytes", "gpu_memory_utilization", "kv_cache_dtype", "draft_path", "draft_repo",
          "draft_method", "num_speculative_tokens", "speculative_extra", "extra_args")


def to_text(r: Recipe) -> str:
    lines: list[str] = []
    if r.max_context:
        lines.append(f"--max-model-len {r.max_context}")
    if r.kv_cache_bytes:
        lines.append(f"--kv-cache-memory-bytes {r.kv_cache_bytes}")
    if r.gpu_memory_utilization:
        lines.append(f"--gpu-memory-utilization {r.gpu_memory_utilization}")
    if r.kv_cache_dtype != "auto":
        lines.append(f"--kv-cache-dtype {r.kv_cache_dtype}")
    draft = r.draft_path or r.draft_repo
    if draft:
        spec = {"model": draft, **({"method": r.draft_method} if r.draft_method and r.draft_method != "auto" else {}),
                "num_speculative_tokens": r.num_speculative_tokens, **(r.speculative_extra or {})}
        lines.append(f"--speculative-config {shlex.quote(json.dumps(spec))}")
    return "\n".join(lines + list(r.extra_args))


def apply_text(r: Recipe, text: str) -> Recipe:
    """r with the settings the text describes. Raises ValueError if a line can't be split into arguments."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parsed, _ = recipe_from_args(r.name, ["vllm", "serve", *extra_tokens(lines)], image=r.image, engine_hint="vllm")
    out = replace(r, **{f: getattr(parsed, f) for f in FIELDS})
    if not (out.draft_path or out.draft_repo):
        out.draft_weights_bytes = 0
    return out
