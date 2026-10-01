"""Size context and KV cache from the memory actually free.

KV bytes per token = 2 (K and V) x attention layers x KV heads x head size x bytes per element.
Hybrid models (Mamba, linear attention) only pay KV for their full-attention layers.
"""
from __future__ import annotations

from dataclasses import dataclass

GIB = 2**30
KV_DTYPE_BYTES = {"fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "bf16": 2, "fp16": 2, "auto": 2}


def text_config(config: dict) -> dict:
    """Vision-language configs keep the language model under text_config."""
    return config.get("text_config") or config.get("llm_config") or config


def attention_layers(config: dict) -> int:
    c = text_config(config)
    types = c.get("layer_types") or c.get("layers_block_type")  # Qwen3-Next / Qwen3.5, and Nemotron-H's own list
    n = c.get("num_hidden_layers") or c.get("n_layer") or (len(types) if isinstance(types, list) else None)
    if not n:
        raise ValueError("config has no layer count")
    if isinstance(types, list):
        return sum(1 for t in types if t in ("full_attention", "attention"))
    if isinstance(c.get("hybrid_override_pattern"), str):  # Nemotron-H: '*' marks attention
        return c["hybrid_override_pattern"].count("*")
    if c.get("full_attention_interval"):
        return n // int(c["full_attention_interval"])
    return n


def kv_bytes_per_token(config: dict, kv_dtype: str = "auto") -> int:
    c = text_config(config)
    heads = c.get("num_attention_heads")
    kv_heads = c.get("num_key_value_heads") or heads
    head_dim = c.get("head_dim") or (c["hidden_size"] // heads if heads and c.get("hidden_size") else None)
    if not kv_heads or not head_dim:
        raise ValueError("the model's config doesn't say how big its attention is, so the KV cache can't be sized")
    return 2 * attention_layers(config) * kv_heads * head_dim * KV_DTYPE_BYTES.get(kv_dtype, 2)


@dataclass
class Plan:
    context_tokens: int
    kv_pool_tokens: int
    kv_bytes: int
    concurrency: float
    fits: bool
    reason: str = ""
    # Share of total memory to tell vLLM (--gpu-memory-utilization): weights, KV and headroom, never its 0.92
    # default, which fails whenever another model already holds memory on unified-memory boxes.
    gpu_fraction: float | None = None
    # What the model should take in all: weights with headroom, the KV cache, and a little for the engine itself.
    total_bytes: int = 0


def plan(
    config: dict,
    weights_bytes: int,
    available_bytes: int,
    *,
    kv_dtype: str = "auto",
    max_context: int | None = None,
    min_context: int = 4096,
    min_concurrency: float = 2.0,
    reserve_bytes: int = 8 * GIB,
    headroom_fraction: float = 0.10,
    kv_cache_bytes: int | None = None,
    total_bytes: int | None = None,
    cap_pool: bool = False,
) -> Plan:
    """Largest context (halving from max_context) that leaves min_concurrency in the pool.

    available_bytes is unified memory free right now (MemAvailable), so models
    already running are accounted for. reserve_bytes is kept back for the OS;
    headroom_fraction of the weights covers activations and CUDA graphs.

    By default the KV pool takes every byte that is free, which is what a dedicated box wants. With cap_pool it is only
    as big as the chosen context times min_concurrency needs (plus a little), so a model takes the memory it needs, not all of it.
    """
    c = text_config(config)
    model_max = c.get("max_position_embeddings")
    max_context = max_context or model_max or 32768
    if model_max:
        max_context = min(max_context, model_max)  # no point asking for more than the model was trained for
    per_token = kv_bytes_per_token(config, kv_dtype)
    kv_budget = available_bytes - reserve_bytes - int(weights_bytes * (1 + headroom_fraction))
    if kv_budget <= 0:
        return Plan(0, 0, 0, 0.0, False, "weights don't fit in free memory")

    def total(kv: int) -> int:
        return int(weights_bytes * (1 + headroom_fraction) + kv + 2 * GIB)

    def frac(kv: int) -> float | None:
        if not total_bytes:
            return None
        need = weights_bytes * (1 + headroom_fraction) + kv + 2 * GIB
        return round(min(0.92, max(0.05, need / total_bytes)), 3)

    if kv_cache_bytes:  # the user fixed the KV cache size: keep it, only check it fits
        pool = kv_cache_bytes // per_token
        ctx = max_context  # as the user wrote it; the engine checks it against its own pool
        ok = kv_cache_bytes <= kv_budget
        return Plan(ctx, pool, kv_cache_bytes, round(pool / ctx, 2) if ctx else 0.0, ok,
                    "" if ok else "the fixed KV cache doesn't fit in free memory", frac(kv_cache_bytes), total(kv_cache_bytes))
    pool = kv_budget // per_token
    ctx = max_context
    while ctx >= min_context:
        if pool / ctx >= min_concurrency:
            used = min(pool, int(ctx * min_concurrency * 1.05) + 1) if cap_pool else pool
            return Plan(ctx, used, used * per_token, round(used / ctx, 2), True, "", frac(used * per_token), total(used * per_token))
        ctx //= 2
    if pool >= min_context:
        return Plan(min_context, pool, pool * per_token, round(pool / min_context, 2), True,
                    f"concurrency below {min_concurrency} even at the minimum context", frac(pool * per_token), total(pool * per_token))
    return Plan(0, pool, pool * per_token, 0.0, False, "not enough memory for the minimum context")
