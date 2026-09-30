"""vLLM adapter: /metrics for live stats, /v1/models and cache_config_info for sizes."""
from __future__ import annotations

from . import prometheus as P
from .base import PromAdapter


class VllmAdapter(PromAdapter):
    engine = "vllm"
    GAUGES = {
        "running": ("vllm_num_requests_running",),
        "waiting": ("vllm_num_requests_waiting",),
        "kv_used": ("vllm_kv_cache_usage_perc", "vllm_gpu_cache_usage_perc"),
    }
    COUNTERS = {
        "gen": ("vllm_generation_tokens_total",),
        "prompt": ("vllm_prompt_tokens_total",),
        "accepted": ("vllm_spec_decode_num_accepted_tokens_total",),
        "drafted": ("vllm_spec_decode_num_draft_tokens_total",),
        "drafts": ("vllm_spec_decode_num_drafts_total",),
        "done": ("vllm_request_success_total",),
        "hits": ("vllm_prefix_cache_hits_total",),
        "queries": ("vllm_prefix_cache_queries_total",),
        "preempt": ("vllm_num_preemptions_total",),
    }
    HISTOGRAMS = {
        "ttft": ("vllm_time_to_first_token_seconds",),
        "itl": ("vllm_inter_token_latency_seconds", "vllm_time_per_output_token_seconds"),
        "tpot": ("vllm_request_time_per_output_token_seconds",),
        "e2e": ("vllm_e2e_request_latency_seconds",),
        "queue": ("vllm_request_queue_time_seconds",),
        "prefill_time": ("vllm_request_prefill_time_seconds",),
        "decode_time": ("vllm_request_decode_time_seconds",),
        "batch": ("vllm_iteration_tokens_total",),
    }

    async def read_static(self) -> dict:
        """Context length and KV pool, read once after the server comes up."""
        r = await self.client.get(self.base_url + "/v1/models")
        data = r.json().get("data", [])
        ctx = data[0].get("max_model_len") if data else None
        m = P.parse((await self.client.get(self.base_url + "/metrics")).text)
        pool = kv_pool_tokens(m)
        self.static = {
            "model": data[0].get("id") if data else None,
            "context_tokens": ctx,
            "kv_pool_tokens": pool,
            "max_concurrency": round(pool / ctx, 2) if pool and ctx else None,
        }
        return self.static


def kv_pool_tokens(m: dict) -> int | None:
    """KV pool in tokens from vLLM's cache_config_info labels (num_gpu_blocks x block_size)."""
    rows = m.get("vllm_cache_config_info")
    if not rows:
        return None
    labels = rows[0][0]
    try:
        return int(labels["num_gpu_blocks"]) * int(labels["block_size"])
    except (KeyError, ValueError):
        return None
