"""SGLang adapter (server started with --enable-metrics)."""
from __future__ import annotations

from .base import PromAdapter


class SglangAdapter(PromAdapter):
    engine = "sglang"
    GAUGES = {
        "running": ("sglang_num_running_reqs",),
        "waiting": ("sglang_num_queue_reqs",),
        "kv_used": ("sglang_token_usage",),
        "prefix_hit": ("sglang_cache_hit_rate",),
        # SGLang reports mean accepted tokens per step rather than accepted/drafted counters.
        "accept_len": ("sglang_spec_accept_length",),
    }
    COUNTERS = {
        "gen": ("sglang_generation_tokens_total",),
        "prompt": ("sglang_prompt_tokens_total",),
        "done": ("sglang_num_requests_total",),
    }
    HISTOGRAMS = {
        "ttft": ("sglang_time_to_first_token_seconds",),
        "itl": ("sglang_inter_token_latency_seconds", "sglang_time_per_output_token_seconds"),
        "e2e": ("sglang_e2e_request_latency_seconds",),
        "queue": ("sglang_queue_time_seconds",),
    }
