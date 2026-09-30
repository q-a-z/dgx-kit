"""Shared engine adapter: scrape Prometheus /metrics, turn counters into live stats.

Each engine names the same things differently (and renames them between
versions), so a subclass only lists its names; every read tries them in order
after prometheus.parse() has turned `x:y` into `x_y`.

Three time frames, as in a good serving dashboard:
- live: rates between the last two scrapes (about a second);
- window: the last WINDOW_S seconds, for averages, percentiles and SLO goodput;
- lifetime: since the server started, for per-request averages and totals.
"""
from __future__ import annotations

import time
from collections import deque

import httpx

from . import prometheus as P

WINDOW_S = 60.0
DEFAULT_SLO = {"ttft": 0.5, "itl": 0.05, "tpot": 0.05, "e2e": 5.0}  # seconds


class PromAdapter:
    engine = ""
    health_path = "/health"
    # stat -> metric names, first present wins
    GAUGES: dict[str, tuple[str, ...]] = {}   # running, waiting, kv_used (0..1), prefix_hit (0..1), accept_len
    COUNTERS: dict[str, tuple[str, ...]] = {}  # gen, prompt, accepted, drafted, drafts, hits, queries, preempt, done
    HISTOGRAMS: dict[str, tuple[str, ...]] = {}  # ttft, itl, tpot, e2e, queue, prefill_time, decode_time, batch

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None, slo: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=2.0)
        self.slo = {**DEFAULT_SLO, **(slo or {})}
        self._prev: dict | None = None
        self._prev_t: float | None = None
        self._window: deque = deque()  # (time, counters)
        self.static: dict = {}

    async def health(self) -> bool:
        try:
            return (await self.client.get(self.base_url + self.health_path)).status_code == 200
        except httpx.HTTPError:
            return False

    async def scrape(self) -> dict:
        text = (await self.client.get(self.base_url + "/metrics")).text
        return self.compute(P.parse(text), time.monotonic())

    def _gauge(self, m: dict, key: str) -> float | None:
        return P.total(m, *self.GAUGES.get(key, ()))

    def _read(self, m: dict) -> dict:
        c = {k: P.total(m, *names) for k, names in self.COUNTERS.items()}
        for k, names in self.HISTOGRAMS.items():
            c[k] = P.buckets(m, *names)
            c[k + ":sc"] = P.sum_count(m, *names)
        return c

    def compute(self, m: dict, now: float) -> dict:
        """Live stats from one parsed scrape; rates and windowed stats need earlier scrapes."""
        cur = self._read(m)
        kv = self._gauge(m, "kv_used")
        out = {
            "running": self._gauge(m, "running"),
            "waiting": self._gauge(m, "waiting"),
            "kv_used_pct": round(kv * 100, 1) if kv is not None else None,
        }
        hit = self._gauge(m, "prefix_hit")
        if hit is not None:
            out["prefix_hit_rate"] = round(hit, 3)
        acc_len = self._gauge(m, "accept_len")
        if acc_len is not None:
            out["draft_accept_len"] = round(acc_len, 2)
        self._lifetime(cur, out)

        # live: since the previous scrape
        prev, dt = self._prev, (now - self._prev_t) if self._prev_t is not None else None
        if prev and dt and dt > 0:
            out["decode_tps"] = _rate(cur, prev, "gen", dt)
            out["prefill_tps"] = _rate(cur, prev, "prompt", dt)
            if "preempt" in cur:
                out["preemptions"] = _delta(cur, prev, "preempt")

        # window: since the oldest scrape still inside WINDOW_S
        self._window.append((now, cur))
        while len(self._window) > 2 and now - self._window[1][0] >= WINDOW_S:
            self._window.popleft()
        t0, old = self._window[0]
        span = now - t0
        if span > 0 and old is not cur:
            self._windowed(cur, old, span, out)

        self._prev, self._prev_t = cur, now
        return out

    def _lifetime(self, cur: dict, out: dict) -> None:
        for key, total_key in (("prompt", "prompt_tokens_total"), ("gen", "gen_tokens_total"),
                               ("done", "requests_total"), ("accepted", "spec_accepted_total"),
                               ("drafted", "spec_drafted_total")):
            if cur.get(key) is not None:
                out[total_key] = cur[key]
        # Per-request speeds: tokens over the time requests spent in each phase.
        pt, dt_ = cur.get("prefill_time:sc"), cur.get("decode_time:sc")
        ptoks, gtoks = cur.get("prompt"), cur.get("gen")
        if pt and pt[0] > 0 and ptoks is not None:
            out["prefill_tps_req"] = round(ptoks / pt[0], 1)
        if dt_ and dt_[0] > 0 and gtoks is not None:
            out["decode_tps_req"] = round(gtoks / dt_[0], 1)

    def _windowed(self, cur: dict, old: dict, span: float, out: dict) -> None:
        out["window_s"] = round(span)
        out["decode_tps_avg"] = _rate(cur, old, "gen", span)
        out["prefill_tps_avg"] = _rate(cur, old, "prompt", span)
        if "accepted" in cur:
            acc, dr = _delta(cur, old, "accepted"), _delta(cur, old, "drafted")
            out["draft_acceptance"] = round(acc / dr, 3) if acc is not None and dr else None
            drafts = _delta(cur, old, "drafts") if "drafts" in cur else None
            if acc is not None and drafts:
                out["draft_accept_len"] = round(acc / drafts, 2)
        if "hits" in cur:
            hits, q = _delta(cur, old, "hits"), _delta(cur, old, "queries")
            out["prefix_hit_rate"] = round(hits / q, 3) if hits is not None and q else out.get("prefix_hit_rate")
            out["prefix_queries"] = q
        slo_parts = []
        for key in self.HISTOGRAMS:
            window = P.diff_buckets(cur[key], old[key])
            for q_name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
                v = P.percentile(window, q)
                out[f"{key}_{q_name}_s"] = round(v, 4) if v is not None else None
            mean = _mean(cur.get(key + ":sc"), old.get(key + ":sc"))
            out[f"{key}_avg_s"] = round(mean, 4) if mean is not None else None
            if key in self.slo:
                ok = P.fraction_le(window, self.slo[key])
                out[f"slo_{key}"] = round(ok, 3) if ok is not None else None
                if ok is not None:
                    slo_parts.append(ok)
        if "ttft" in self.HISTOGRAMS:
            out["ttft_goodput"] = out.get("slo_ttft")
        # A request counts only if it met every target; the strictest share bounds that from above.
        out["slo_combined"] = round(min(slo_parts), 3) if slo_parts else None


def _delta(a: dict, b: dict, k: str):
    x, y = a.get(k), b.get(k)
    return x - y if x is not None and y is not None else None


def _rate(a: dict, b: dict, k: str, dt: float):
    d = _delta(a, b, k)
    return round(d / dt, 1) if d is not None else None


def _mean(now, before):
    """Mean of the observations between two (sum, count) readings; lifetime mean if none arrived."""
    if not now:
        return None
    if before and now[1] > before[1]:
        return (now[0] - before[0]) / (now[1] - before[1])
    return now[0] / now[1] if now[1] else None
