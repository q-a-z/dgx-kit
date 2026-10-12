"""Atlas adapter (avarok/atlas-gb10, the `spark serve` binary). Its /metrics names were read from the binary's own
strings, not from a running server, so they are the ones it registers; it exports no KV-cache usage or queue length."""
from __future__ import annotations

from .base import PromAdapter


class AtlasAdapter(PromAdapter):
    engine = "atlas"
    GAUGES = {
        "running": ("atlas_requests_active",),
        "prefix_hit": ("atlas_prefix_cache_hit_rate",),
    }
    COUNTERS = {
        "gen": ("atlas_generation_tokens_total",),
        "prompt": ("atlas_prompt_tokens_total",),
        "done": ("atlas_requests_total",),
    }
    HISTOGRAMS = {
        "ttft": ("atlas_time_to_first_token_seconds",),
    }
