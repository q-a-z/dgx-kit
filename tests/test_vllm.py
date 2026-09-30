from pathlib import Path

import pytest

from dgxkit.engines import prometheus as P
from dgxkit.engines.vllm import VllmAdapter, kv_pool_tokens

TEXT = (Path(__file__).parent / "fixtures" / "vllm_metrics.txt").read_text()


def test_parse_normalises_names():
    m = P.parse(TEXT)
    assert P.total(m, "vllm_num_requests_running") == 2.0
    assert P.total(m, "missing", "vllm_gpu_cache_usage_perc", "vllm_kv_cache_usage_perc") == 0.25


def test_kv_pool_from_cache_config():
    assert kv_pool_tokens(P.parse(TEXT)) == 320000


def test_percentile_interpolates_inside_bucket():
    b = [(0.1, 2.0), (1.0, 8.0), (float("inf"), 10.0)]
    assert P.percentile(b, 0.5) == pytest.approx(0.1 + 0.9 * (3 / 6))
    assert P.fraction_le(b, 1.0) == pytest.approx(0.8)


def test_rates_and_windowed_percentiles():
    a = VllmAdapter("http://x")
    first = a.compute(P.parse(TEXT), now=0.0)
    assert first["running"] == 2.0 and first["kv_used_pct"] == 25.0
    assert "decode_tps" not in first

    later = (TEXT.replace("generation_tokens_total{engine=\"0\",model_name=\"m\"} 1000.0",
                          "generation_tokens_total{engine=\"0\",model_name=\"m\"} 1100.0")
                 .replace("accepted_tokens_total{engine=\"0\",model_name=\"m\"} 600.0",
                          "accepted_tokens_total{engine=\"0\",model_name=\"m\"} 660.0")
                 .replace("draft_tokens_total{engine=\"0\",model_name=\"m\"} 900.0",
                          "draft_tokens_total{engine=\"0\",model_name=\"m\"} 990.0"))
    second = a.compute(P.parse(later), now=2.0)
    assert second["decode_tps"] == 50.0
    assert second["draft_acceptance"] == pytest.approx(60 / 90, abs=1e-3)
    assert second["ttft_p50_s"] is None  # no new requests in the window
