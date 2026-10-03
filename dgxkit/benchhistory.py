"""Benchmark history: the numbers worth tracking from each run's result, and whether the latest run got worse.

Each run records what the model ran on (image, DGX-kit version, driver, settings) beside its result, so a drop in speed
can be set against what changed since the runs before it.
"""
from __future__ import annotations

import statistics

# metric -> (label, unit, higher is better)
METRICS = {
    "decode_code": ("Decode, code", "t/s", True),
    "decode_prose": ("Decode, prose", "t/s", True),
    "decode_complex": ("Decode, complex code", "t/s", True),
    "accept_len": ("Draft tokens accepted per step", "", True),
    "conc_total": ("Total speed, concurrent streams", "t/s", True),
    "prefill_tps": ("Prefill", "t/s", True),
    "ttft_s": ("First token after a long prompt", "s", False),
    "stall_ms": ("Worst decode stall while prefilling", "ms", False),
    "tools_pct": ("Valid tool calls", "%", True),
    "needle_pct": ("Needle found", "%", True),
    "complex_pct": ("Complex code, hidden tests passed", "%", True),
}
SPEED_DROP = 0.08       # a speed or acceptance this much below the usual is a regression
LATENCY_RISE = 0.15     # a latency this much above the usual is
QUALITY_DROP = 10.0     # percentage points
WINDOW = 5              # the usual = the median of this many runs before

CONTEXT_LABELS = {"image": "image", "image_id": "image build", "dgxkit": "DGX-kit", "driver": "driver", "engine": "engine",
                  "quantization": "quantization", "kv_cache_dtype": "KV cache type", "max_context": "context",
                  "draft_method": "draft method", "speculative_tokens": "draft tokens"}


def _mean(xs) -> float | None:
    xs = [x for x in (xs or []) if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def _pct(ok: int, total: int) -> float | None:
    return round(100 * ok / total, 1) if total else None


def extract_metrics(res: dict) -> dict[str, float]:
    """The tracked numbers present in one bench.py result (a test that failed or wasn't run has none)."""
    out: dict[str, float | None] = {}
    g = lambda k: res.get(k) if isinstance(res.get(k), (dict, list)) and not (isinstance(res.get(k), dict) and res[k].get("error")) else None
    dec = g("decode") or {}
    out["decode_code"] = _mean((dec.get("code_temp0") or {}).get("tps"))
    out["decode_prose"] = _mean((dec.get("prose") or {}).get("tps"))
    out["accept_len"] = _mean((dec.get("code_temp0") or {}).get("acc"))
    cx = g("complex") or {}
    out["decode_complex"] = _mean((cx.get("temp0") or {}).get("tps"))
    ok = total = 0
    for temp in cx.values():
        for v in (temp.get("verify") or []) if isinstance(temp, dict) else []:
            if v.get("fail_class") != "truncated" and v.get("finish") != "length":  # an answer cut off by the limit isn't a score
                ok, total = ok + v.get("correct", 0), total + v.get("total", 0)
    out["complex_pct"] = _pct(ok, total)
    conc = g("conc") or {}
    out["conc_total"] = conc.get("aggregate")
    pre = g("prefill")
    if isinstance(pre, list):
        out["prefill_tps"] = _mean([p.get("prefill_tps") for p in pre])
        out["ttft_s"] = _mean([p.get("ttft") for p in pre])
    stall = g("stall")
    if isinstance(stall, list) and stall:
        out["stall_ms"] = max(s.get("gap_during_max_ms", 0) for s in stall)
    needle = g("needle")
    if isinstance(needle, list) and needle:
        out["needle_pct"] = _pct(sum(1 for n in needle if n.get("ok")), len(needle))
    tools = g("tools") or {}
    if tools.get("total"):
        out["tools_pct"] = _pct(tools.get("valid", 0), tools["total"])
    return {k: v for k, v in out.items() if v is not None}


def changed(before: dict | None, now: dict | None) -> list[str]:
    """What differs between two runs' contexts, in words: `image: vllm:0.29 → vllm:0.30`."""
    if not before or not now:
        return []
    out = []
    for k, label in CONTEXT_LABELS.items():
        a, b = before.get(k), now.get(k)
        if a is not None and b is not None and a != b:
            out.append(f"{label}: {a} → {b}")
    return out


def regressions(runs: list[dict]) -> list[dict]:
    """Metrics of the newest run that are clearly worse than the median of the runs before it (oldest first in `runs`)."""
    if len(runs) < 2:
        return []
    latest, earlier = runs[-1], runs[-1 - WINDOW:-1]
    found = []
    for key, (label, unit, higher) in METRICS.items():
        now = latest["metrics"].get(key)
        past = [r["metrics"][key] for r in earlier if key in r["metrics"]]
        if now is None or not past:
            continue
        usual = statistics.median(past)
        if usual == 0:
            continue
        if key in ("tools_pct", "needle_pct", "complex_pct"):
            bad = usual - now >= QUALITY_DROP
            change = round(now - usual, 1)
        elif higher:
            bad = now < usual * (1 - SPEED_DROP)
            change = round(100 * (now - usual) / usual, 1)
        else:
            bad = now > usual * (1 + LATENCY_RISE)
            change = round(100 * (now - usual) / usual, 1)
        if bad:
            found.append({"metric": key, "label": label, "unit": unit, "now": now, "usual": round(usual, 2), "change": change,
                          "points": key in ("tools_pct", "needle_pct", "complex_pct"),
                          "since": changed(earlier[-1].get("context"), latest.get("context"))})
    return found
