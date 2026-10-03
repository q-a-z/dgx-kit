import json

from dgxkit.benchhistory import changed, extract_metrics, regressions
from dgxkit.benchmarks import Benchmarks


def result(decode=120.0, prefill=5000.0, ttft=7.0, tools=50, correct=9):
    return {
        "decode": {"code_temp0": {"tps": [decode, decode + 2], "acc": [3.0, 3.0]}, "prose": {"tps": [100.0, 102.0]}},
        "complex": {"temp0": {"tps": [110.0, 112.0], "verify": [
            {"correct": correct, "total": 9, "finish": "stop"},
            {"correct": 0, "total": 9, "fail_class": "truncated", "finish": "length"}]}},  # cut off: not a score
        "conc": {"streams": 2, "per_stream": [89.0, 90.0], "aggregate": 176.3},
        "prefill": [{"ptoks": 37000, "ttft": ttft, "prefill_tps": prefill}],
        "stall": [{"gap_before_ms": 24, "gap_during_max_ms": 396}],
        "needle": [{"ok": True}, {"ok": False}],
        "tools": {"valid": tools, "total": 50},
        "sanity": {"words": 800},
    }


def test_the_tracked_numbers_come_out_of_a_result():
    m = extract_metrics(result())
    assert m["decode_code"] == 121.0 and m["decode_prose"] == 101.0 and m["accept_len"] == 3.0
    assert m["conc_total"] == 176.3 and m["prefill_tps"] == 5000.0 and m["ttft_s"] == 7.0 and m["stall_ms"] == 396
    assert m["tools_pct"] == 100.0 and m["needle_pct"] == 50.0
    assert m["complex_pct"] == 100.0  # the answer cut off at the token limit does not count against the model
    assert extract_metrics({"decode": {"error": "boom"}}) == {}  # a test that failed has no numbers


def run(i, decode=120.0, image="vllm:0.29", **kw):
    return {"id": f"r{i}", "started": 1000 + i, "context": {"image": image, "dgxkit": "0.1.8"}, "metrics": extract_metrics(result(decode=decode, **kw))}


def test_a_clear_drop_is_flagged_with_what_changed_and_noise_is_not():
    runs = [run(i) for i in range(4)]
    assert regressions(runs + [run(4, decode=117.0)]) == []  # 3% lower is noise
    bad = regressions(runs + [run(4, decode=100.0, image="vllm:0.30")])
    d = next(r for r in bad if r["metric"] == "decode_code")
    assert d["change"] < -15 and d["since"] == ["image: vllm:0.29 → vllm:0.30"] and not d["points"]
    assert regressions([run(0)]) == []  # one run has nothing to be compared with


def test_latency_up_and_quality_down_are_regressions_too():
    runs = [run(i) for i in range(3)]
    slow = {r["metric"] for r in regressions(runs + [run(3, ttft=9.5)])}
    assert slow == {"ttft_s"}
    worse = regressions(runs + [run(3, tools=40)])
    assert [r["metric"] for r in worse] == ["tools_pct"] and worse[0]["points"] and worse[0]["change"] == -20.0


def test_context_changes_are_described():
    assert changed({"image": "a", "driver": "580"}, {"image": "b", "driver": "580"}) == ["image: a → b"]
    assert changed(None, {"image": "b"}) == [] and changed({"image": "a"}, {"image": "a"}) == []


def test_history_lists_finished_runs_oldest_first_with_their_context(tmp_path):
    b = Benchmarks(str(tmp_path))
    b.dir.mkdir(parents=True)
    for i, (decode, image) in enumerate([(120.0, "vllm:0.29"), (121.0, "vllm:0.29"), (95.0, "vllm:0.30")]):
        rid = f"m-2026100{i + 1}-000000"
        (b.dir / f"{rid}.meta").write_text(json.dumps({"id": rid, "model": "m", "port": 1, "tests": "decode", "conc": 2,
                                                       "started": 100.0 + i, "pid": 1, "context": {"image": image}}))
        (b.dir / f"{rid}.json").write_text(json.dumps(result(decode=decode)))
    (b.dir / "m-running.meta").write_text(json.dumps({"id": "m-running", "model": "m", "port": 1, "tests": "decode", "conc": 2, "started": 999.0, "pid": 1}))
    h = b.history("m")
    assert [r["id"] for r in h["runs"]] == ["m-20261001-000000", "m-20261002-000000", "m-20261003-000000"]
    assert h["regressions"] and h["regressions"][0]["since"] == ["image: vllm:0.29 → vllm:0.30"]
    assert h["metrics"]["decode_code"]["unit"] == "t/s"
    assert b.history("other")["runs"] == []
