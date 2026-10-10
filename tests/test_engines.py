import pytest

from dgxkit.engines import LlamaCppAdapter, SglangAdapter, adapter_for
from dgxkit.engines import prometheus as P

SGLANG = """
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="m"} 3.0
sglang:num_queue_reqs{model_name="m"} 1.0
sglang:token_usage{model_name="m"} 0.42
sglang:cache_hit_rate{model_name="m"} 0.5
sglang:spec_accept_length{model_name="m"} 2.75
sglang:generation_tokens_total{model_name="m"} GEN
sglang:prompt_tokens_total{model_name="m"} 5000.0
sglang:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 1.0
sglang:time_to_first_token_seconds_bucket{le="1.0",model_name="m"} TTFT
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} TTFT
"""

LLAMA = """
llamacpp:requests_processing 2
llamacpp:requests_deferred 0
llamacpp:kv_cache_usage_ratio 0.1
llamacpp:tokens_predicted_total GEN
llamacpp:prompt_tokens_total 800
"""


def fill(text, gen, ttft="1.0"):
    return text.replace("GEN", str(gen)).replace("TTFT", ttft)


def test_sglang_gauges_rates_and_percentiles():
    a = SglangAdapter("http://x")
    first = a.compute(P.parse(fill(SGLANG, 1000.0)), now=0.0)
    assert first.items() >= {"running": 3.0, "waiting": 1.0, "kv_used_pct": 42.0,
                             "prefix_hit_rate": 0.5, "draft_accept_len": 2.75, "gen_tokens_total": 1000.0}.items()
    second = a.compute(P.parse(fill(SGLANG, 1300.0, ttft="5.0")), now=2.0)
    assert second["decode_tps"] == 150.0
    assert 0.1 < second["ttft_p50_s"] <= 1.0
    assert second["ttft_goodput"] == pytest.approx(0.444)  # 4 new requests between 0.1 and 1 s, target 0.5 s


def test_reset_stats_counts_from_now_and_survives_a_restart():
    a = SglangAdapter("http://x")
    first = a.compute(P.parse(fill(SGLANG, 1000.0)), now=0.0)
    assert first["gen_tokens_total"] == 1000.0 and first["prompt_tokens_total"] == 5000.0 and "stats_since" not in first
    a.reset_stats()
    again = a.compute(P.parse(fill(SGLANG, 1300.0)), now=2.0)
    assert again["gen_tokens_total"] == 300.0 and again["prompt_tokens_total"] == 0.0 and again["stats_since"]
    assert again["decode_tps"] == 150.0  # the live rate doesn't care
    restarted = a.compute(P.parse(fill(SGLANG, 40.0)), now=4.0)  # the server began counting again
    assert restarted["gen_tokens_total"] == 40.0 and "stats_since" not in restarted


def test_llamacpp_has_rates_but_no_percentiles():
    a = LlamaCppAdapter("http://x")
    a.compute(P.parse(fill(LLAMA, 100)), now=0.0)
    second = a.compute(P.parse(fill(LLAMA, 160)), now=3.0)
    assert second["running"] == 2 and second["kv_used_pct"] == 10.0 and second["decode_tps"] == 20.0
    assert not any(k.endswith("_p50_s") for k in second)


def test_adapter_for_each_engine():
    assert adapter_for("sglang", "http://h:1/").base_url == "http://h:1"
    assert adapter_for("llamacpp", "http://h:1").engine == "llamacpp"


VLLM = """
vllm:num_requests_running{model_name="m"} 1
vllm:generation_tokens_total{model_name="m"} GEN
vllm:prompt_tokens_total{model_name="m"} 1000
vllm:request_success_total{finished_reason="stop",model_name="m"} REQ
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} ACC
vllm:spec_decode_num_draft_tokens_total{model_name="m"} DRAFTED
vllm:spec_decode_num_drafts_total{model_name="m"} DRAFTS
vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="m"} FAST
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} REQ
vllm:time_to_first_token_seconds_sum{model_name="m"} TSUM
vllm:time_to_first_token_seconds_count{model_name="m"} REQ
vllm:request_prefill_time_seconds_sum{model_name="m"} 4.0
vllm:request_prefill_time_seconds_count{model_name="m"} REQ
vllm:request_decode_time_seconds_sum{model_name="m"} 10.0
vllm:request_decode_time_seconds_count{model_name="m"} REQ
"""


def vllm_text(**v):
    t = VLLM
    for k, x in v.items():
        t = t.replace(k.upper(), str(x))
    return t


def test_vllm_window_lifetime_and_slo():
    from dgxkit.engines import VllmAdapter
    a = VllmAdapter("http://x")
    a.compute(P.parse(vllm_text(gen=500, req=4, acc=10, drafted=30, drafts=10, fast=4, tsum=1.0)), now=0.0)
    out = a.compute(P.parse(vllm_text(gen=800, req=8, acc=40, drafted=90, drafts=30, fast=6, tsum=3.0)), now=10.0)
    assert out["decode_tps"] == 30.0 and out["decode_tps_avg"] == 30.0
    assert out["decode_tps_req"] == 80.0 and out["prefill_tps_req"] == 250.0  # lifetime tokens / phase time
    assert out["requests_total"] == 8 and out["spec_accepted_total"] == 40
    assert out["draft_acceptance"] == 0.5 and out["draft_accept_len"] == 1.5
    assert out["ttft_avg_s"] == 0.5          # (3.0 - 1.0) s over 4 new requests
    assert out["slo_ttft"] == 0.5            # 2 of the 4 new requests under 500 ms
    assert out["slo_combined"] == 0.5
