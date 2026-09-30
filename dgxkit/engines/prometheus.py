"""Minimal Prometheus text-format parser and histogram helpers."""
from __future__ import annotations

import math
import re
from collections import defaultdict

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse(text: str) -> dict[str, list[tuple[dict, float]]]:
    """Return {metric name: [(labels, value), ...]}. Comments are skipped.

    Names are normalised so vLLM's `vllm:foo` and `vllm_foo` look the same.
    """
    out: dict[str, list] = defaultdict(list)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name = m.group(1).replace(":", "_")
        labels = dict(_LABEL.findall(m.group(3) or ""))
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        out[name].append((labels, value))
    return dict(out)


def first(metrics: dict, *names: str) -> str | None:
    """First of several alternative metric names present (engines rename across versions)."""
    return next((n for n in names if n in metrics), None)


def total(metrics: dict, *names: str) -> float | None:
    """Sum of a metric across label sets, trying alternative names in order."""
    name = first(metrics, *names)
    if name is None:
        return None
    return sum(v for _, v in metrics[name])


def buckets(metrics: dict, *names: str) -> list[tuple[float, float]]:
    """Cumulative histogram buckets [(upper bound, count)] summed across label sets."""
    for base in names:
        rows = metrics.get(base + "_bucket")
        if rows:
            acc: dict[float, float] = defaultdict(float)
            for labels, v in rows:
                le = labels.get("le", "+Inf")
                acc[math.inf if le == "+Inf" else float(le)] += v
            return sorted(acc.items())
    return []


def diff_buckets(now: list, before: list) -> list[tuple[float, float]]:
    """Buckets for the observations that arrived between two scrapes."""
    prev = dict(before)
    return [(le, c - prev.get(le, 0.0)) for le, c in now]


def percentile(bks: list[tuple[float, float]], q: float) -> float | None:
    """Linear interpolation inside the bucket that holds quantile q (0..1)."""
    if not bks or bks[-1][1] <= 0:
        return None
    target = q * bks[-1][1]
    lo_bound, lo_count = 0.0, 0.0
    for le, count in bks:
        if count >= target:
            if math.isinf(le):
                return lo_bound
            span = count - lo_count
            frac = (target - lo_count) / span if span > 0 else 1.0
            return lo_bound + (le - lo_bound) * frac
        lo_bound, lo_count = le, count
    return None


def fraction_le(bks: list[tuple[float, float]], limit: float) -> float | None:
    """Share of observations at or under `limit`, interpolated like percentile()."""
    if not bks or bks[-1][1] <= 0:
        return None
    lo_bound, lo_count = 0.0, 0.0
    for le, count in bks:
        if le >= limit:
            if math.isinf(le):
                return lo_count / bks[-1][1]
            frac = (limit - lo_bound) / (le - lo_bound) if le > lo_bound else 1.0
            return (lo_count + (count - lo_count) * frac) / bks[-1][1]
        lo_bound, lo_count = le, count
    return 1.0


def sum_count(metrics: dict, *names: str) -> tuple[float, float] | None:
    """(sum, count) of a histogram across label sets, for means."""
    for base in names:
        s, c = metrics.get(base + "_sum"), metrics.get(base + "_count")
        if s and c:
            return sum(v for _, v in s), sum(v for _, v in c)
    return None
