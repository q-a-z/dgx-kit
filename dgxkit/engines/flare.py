"""Flare's native Prometheus metrics adapter."""
from __future__ import annotations

from .base import PromAdapter


class FlareAdapter(PromAdapter):
    engine = "flare"
    GAUGES = {
        "running": ("flare_active_requests",),
        "waiting": ("flare_pending_requests",),
        "kv_used": ("flare_kv_cache_usage_ratio", "flare_kv_cache_usage"),
    }
    COUNTERS = {
        "gen": ("flare_generated_tokens_total",),
        "prompt": ("flare_prefill_tokens_total",),
        "reused": ("flare_cache_reused_tokens_total",),
        "done": ("flare_requests_completed_total",),
    }

    async def read_static(self) -> dict:
        response = await self.client.get(self.base_url + "/v1/models")
        data = response.json().get("data", [])
        self.static = {"model": data[0].get("id") if data else None}
        return self.static
