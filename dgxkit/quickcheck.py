"""A ten-second speed check of a model that has just finished loading: two short decodes and two prefills.

Not a benchmark: tools/bench.py is that, and only runs on request. This gives a number to
see at a glance that a start came up healthy, and it is skipped when other models are busy.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

BUDGET_S = 10.0
DECODE_PROMPTS = ["Write a Python function that merges two sorted lists, with a short docstring.",
                  "Explain in three sentences how a hash table handles collisions."]
POLL_S = 2.0  # how often to ask a loading model whether it is ready
PREFILL_SENTENCES = (800, 2400)  # the filler is ~9 tokens a sentence


def _stream(c: httpx.Client, base: str, model: str, prompt: str, max_tokens: int, deadline: float) -> dict:
    """One streamed chat request: seconds to the first token, tokens after it per second, prompt tokens."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic()
    first = usage = None
    with c.stream("POST", f"{base}/v1/chat/completions", json=body, timeout=max(1.0, deadline - t0)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[6:])
            ch = (d.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning")):
                first = time.monotonic()
            usage = d.get("usage") or usage
            if time.monotonic() > deadline:
                break
    end = time.monotonic()
    n = (usage or {}).get("completion_tokens") or 0
    out = {"ttft_s": (first or end) - t0, "prompt_tokens": (usage or {}).get("prompt_tokens")}
    if first and n > 1 and end > first:
        out["decode_tps"] = round((n - 1) / (end - first), 1)
    return out


def wait_ready(base: str, client: httpx.Client, timeout: float = 300.0) -> bool:
    """llama.cpp answers its metrics while the model is still loading, and refuses completions (503) until it is done:
    wait for /health to say ok, so the check doesn't fail on a model that is only loading."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if client.get(f"{base}/health", timeout=5).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(POLL_S)
    return False


def run(port: int, model: str, budget: float = BUDGET_S, client: httpx.Client | None = None) -> dict:
    base = f"http://127.0.0.1:{port}"
    with (client or httpx.Client()) as c:
        if not wait_ready(base, c):
            return {"ts": time.time(), "skipped": "the model didn't report ready within five minutes"}
        deadline = time.monotonic() + budget  # the ten seconds start once it is ready
        out: dict = {"ts": time.time(), "decode_tps": [], "prefill": []}
        for p in DECODE_PROMPTS:
            if time.monotonic() < deadline:
                r = _stream(c, base, model, p, 96, deadline)
                if "decode_tps" in r:
                    out["decode_tps"].append(r["decode_tps"])
        for n in PREFILL_SENTENCES:
            if time.monotonic() < deadline:
                filler = f"[{uuid.uuid4().hex}] " + "The quick brown fox jumps over the lazy dog. " * n  # unique, so no prefix-cache hit
                r = _stream(c, base, model, filler + "\nSay OK.", 8, deadline)
                if r.get("prompt_tokens") and r["ttft_s"] > 0:
                    out["prefill"].append({"tokens": r["prompt_tokens"], "tps": round(r["prompt_tokens"] / r["ttft_s"])})
    d = out["decode_tps"]
    out["decode_mean_tps"] = round(sum(d) / len(d), 1) if d else None
    out["prefill_mean_tps"] = round(sum(p["tps"] for p in out["prefill"]) / len(out["prefill"])) if out["prefill"] else None
    out["seconds"] = round(budget - max(0.0, deadline - time.monotonic()), 1)
    return out


def save(state_dir: str, name: str, result: dict) -> None:
    d = Path(state_dir) / "quick"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(result))


def load(state_dir: str, name: str) -> dict | None:
    f = Path(state_dir) / "quick" / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None
