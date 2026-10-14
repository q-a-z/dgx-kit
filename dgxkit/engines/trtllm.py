"""TensorRT-LLM adapter (`trtllm-serve`), read from GET /metrics: the server's own JSON iteration stats, no Prometheus.

Each call returns the iterations since the previous call and clears them, so the sampler must be the only reader. The
list stays empty unless the options file sets `enable_iter_perf_stats: true`; then every number reads as idle. The
stats are folded into running totals and fed to the shared adapter as if they were counters, so the live, window and
lifetime numbers come out the same way as for the other engines. There are no latency histograms in this source."""
from __future__ import annotations

import time
from collections import defaultdict

import httpx

from .base import PromAdapter


class TrtllmAdapter(PromAdapter):
    engine = "trtllm"
    metrics_path = "/metrics"
    GAUGES = {
        "running": ("trtllm_running",),
        "waiting": ("trtllm_waiting",),
        "kv_used": ("trtllm_kv_used",),
        "prefix_hit": ("trtllm_prefix_hit",),
        "accept_len": ("trtllm_accept_len",),
    }
    COUNTERS = {
        "gen": ("trtllm_gen_tokens_total",),
        "prompt": ("trtllm_prompt_tokens_total",),
        "done": ("trtllm_requests_total",),
        "accepted": ("trtllm_spec_accepted_total",),
        "drafted": ("trtllm_spec_drafted_total",),
        "drafts": ("trtllm_spec_drafts_total",),
    }
    HISTOGRAMS: dict = {}

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None, slo: dict | None = None):
        # /metrics can take a few seconds to answer when nothing ran since the last call
        super().__init__(base_url, client or httpx.AsyncClient(timeout=10.0), slo)
        self._tot: dict[str, float] = defaultdict(float)
        self._gauge_now: dict[str, float] = {}

    def fold(self, iterations: list) -> None:
        """Add one /metrics answer (a list of per-iteration stat dicts) to the running totals and gauges."""
        g = self._gauge_now
        for it in iterations:
            b = it.get("inflightBatchingStats") or {}
            s = it.get("specDecodingStats") or {}
            acc, drafted = s.get("numAcceptedTokens") or 0, s.get("numDraftTokens") or 0
            # one token per generating request, plus the draft tokens that were accepted; prompt tokens as scheduled
            self._tot["gen"] += (b.get("numGenRequests") or 0) + acc
            self._tot["prompt"] += b.get("numCtxTokens") or 0
            self._tot["done"] += it.get("numCompletedRequests") or 0
            self._tot["accepted"] += acc
            self._tot["drafted"] += drafted
            self._tot["drafts"] += s.get("numRequestsWithDraftTokens") or 0
            g["running"], g["waiting"] = it.get("numActiveRequests") or 0, it.get("numQueuedRequests") or 0
            kv = it.get("kvCacheStats") or {}
            if kv.get("maxNumBlocks"):
                g["kv_used"] = (kv.get("usedNumBlocks") or 0) / kv["maxNumBlocks"]
                g["prefix_hit"] = kv.get("cacheHitRate") or 0.0
            if drafted:
                g["accept_len"] = s.get("acceptanceLength") or 0.0
        if not iterations:  # nothing ran since the last call: idle
            g["running"] = g["waiting"] = 0

    def _parsed(self) -> dict:
        names = {"gen": "trtllm_gen_tokens_total", "prompt": "trtllm_prompt_tokens_total", "done": "trtllm_requests_total",
                 "accepted": "trtllm_spec_accepted_total", "drafted": "trtllm_spec_drafted_total",
                 "drafts": "trtllm_spec_drafts_total"}
        out = {names[k]: [({}, v)] for k, v in self._tot.items()}
        out.update({f"trtllm_{k}": [({}, v)] for k, v in self._gauge_now.items()})
        return out

    async def scrape(self) -> dict:
        r = await self.client.get(self.base_url + self.metrics_path)
        r.raise_for_status()
        self.fold(r.json())
        return self.compute(self._parsed(), time.monotonic())
