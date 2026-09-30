"""llama.cpp adapter (llama-server started with --metrics). It exports gauges and
token counters only, so there are no latency percentiles for it."""
from __future__ import annotations

from .base import PromAdapter


class LlamaCppAdapter(PromAdapter):
    engine = "llamacpp"
    GAUGES = {
        "running": ("llamacpp_requests_processing",),
        "waiting": ("llamacpp_requests_deferred",),
        "kv_used": ("llamacpp_kv_cache_usage_ratio",),
    }
    COUNTERS = {
        "gen": ("llamacpp_tokens_predicted_total",),
        "prompt": ("llamacpp_prompt_tokens_total",),
    }
    HISTOGRAMS = {}
