"""TensorRT-LLM adapter (`trtllm-serve`). Names are the `trtllm_*` Prometheus ones NVIDIA documents for 1.1; the server
exports Prometheus on /prometheus/metrics (its /metrics is JSON iteration stats). Not read from a running server here:
an unexpected name shows as a missing number, not an error."""
from __future__ import annotations

from .base import PromAdapter


class TrtllmAdapter(PromAdapter):
    engine = "trtllm"
    metrics_path = "/prometheus/metrics"
    GAUGES = {
        "kv_used": ("trtllm_kv_cache_utilization",),
        "prefix_hit": ("trtllm_kv_cache_hit_rate",),
    }
    COUNTERS = {
        "done": ("trtllm_request_success_total",),
    }
    HISTOGRAMS = {
        "ttft": ("trtllm_time_to_first_token_seconds",),
        "tpot": ("trtllm_time_per_output_token_seconds",),
        "e2e": ("trtllm_e2e_request_latency_seconds",),
        "queue": ("trtllm_request_queue_time_seconds",),
    }
