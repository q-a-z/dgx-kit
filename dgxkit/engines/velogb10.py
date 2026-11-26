"""veloGB10's native serving metrics."""
from __future__ import annotations

import time

from .base import PromAdapter
from . import prometheus as P


class VeloGB10Adapter(PromAdapter):
    engine = "velogb10"
    REQUEST_RATE_HISTOGRAMS = {
        "decode_tps_req": ("velogb10_decode_tokens_per_second",),
        "prefill_tps_req": ("velogb10_prefill_tokens_per_second",),
    }
    GAUGES = {
        "running": ("velogb10_requests_running",),
        "waiting": ("velogb10_requests_waiting",),
    }
    COUNTERS = {
        "gen": ("velogb10_generation_tokens_total",),
        "prompt": ("velogb10_prompt_tokens_total",),
        "done": ("velogb10_requests_total",),
    }
    HISTOGRAMS = {
        "ttft": ("velogb10_time_to_first_token_seconds",),
        "e2e": ("velogb10_request_duration_seconds",),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase_rate_totals: dict[str, tuple[float, float]] = {}
        self._phase_rate_baseline: dict[str, tuple[float, float]] = {}

    def reset_stats(self) -> None:
        super().reset_stats()
        self._phase_rate_baseline = dict(self._phase_rate_totals)

    async def scrape(self) -> dict:
        response = await self.client.get(self.base_url + self.metrics_path)
        response.raise_for_status()
        metrics = P.parse(response.text)
        if P.total(metrics, "velogb10_engine_alive") == 0:
            raise RuntimeError("veloGB10 engine is dead")
        stats = self.compute(metrics, time.monotonic())
        for field, names in self.REQUEST_RATE_HISTOGRAMS.items():
            total = P.sum_count(metrics, *names)
            if not total:
                continue
            current = (total[0], total[1])
            baseline = self._phase_rate_baseline.get(field, (0.0, 0.0))
            if current[0] < baseline[0] or current[1] < baseline[1]:
                baseline = (0.0, 0.0)  # the VeloGB10 process restarted and reset its counters
                self._phase_rate_baseline[field] = baseline
            completed = current[1] - baseline[1]
            if completed > 0:
                stats[field] = round((current[0] - baseline[0]) / completed, 1)
            self._phase_rate_totals[field] = current
        return stats

    async def read_static(self) -> dict:
        response = await self.client.get(self.base_url + "/v1/models")
        response.raise_for_status()
        data = response.json().get("data", [])
        self.static = {
            "model": data[0].get("id") if data else None,
            "context_tokens": data[0].get("max_model_len") if data else None,
        }
        return self.static
