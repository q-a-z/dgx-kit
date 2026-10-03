"""Why did a model's engine stop? Read the end of its log and say so in plain words, with a fix where one is known.

A fix is {"label", "set": {recipe field: value}, "add_args": [flag lines], "remove_args": [flags], "replace_args": [flag lines]};
the page applies it with one click.
"""
from __future__ import annotations

import math
import re

GIB = 2**30


def _round_up(n: float, step: int) -> int:
    return int(math.ceil(n / step) * step)


def diagnose(log: str, recipe) -> dict | None:
    """The most specific known cause found in the log, or None when nothing in it is recognised."""
    log = log[-60000:]

    m = re.search(r"\(([\d.]+) GiB KV cache is needed, which is larger than the available KV cache memory \(([\d.]+) GiB\)"
                  r".*?estimated maximum model length is (\d+)", log, re.S)
    if m:
        need, have, maxlen = float(m[1]), float(m[2]), int(m[3])
        want = _round_up(need * GIB * 1.15, 10**8)  # a margin over the single request vLLM asked for
        ctx = maxlen // 1024 * 1024
        return {"cause": "kv_too_small", "title": "The KV cache is too small for the context",
                "detail": f"vLLM needs {need:.2f} GiB of KV cache to serve one request of the full context, and the model was given {have:.2f} GiB. "
                          f"With that much it can hold a context of about {maxlen:,} tokens. (The draft model and hybrid layers take cache too, which DGX-kit's estimate does not include.)",
                "fixes": [{"label": f"Raise the KV cache to {want / 1e9:.1f} GB", "set": {"kv_cache_bytes": want}},
                          {"label": f"Lower the context to {ctx:,}", "set": {"max_context": ctx}}]}

    m = re.search(r"Free memory on device[^\n]*?\(([\d.]+)/([\d.]+) GiB\)[^\n]*less than desired GPU memory utilization \(([\d.]+), ([\d.]+) GiB\)", log)
    if m:
        free, total, util = float(m[1]), float(m[2]), float(m[3])
        return {"cause": "memory_taken", "title": "Other models hold too much memory to start this one",
                "detail": f"Only {free:.0f} of {total:.0f} GiB were free, and the model asked for {util:.0%} of the memory. Stop a model that is running, or ask for less.",
                "fixes": [{"label": f"Ask for {max(0.05, math.floor((free / total - 0.03) * 100) / 100):.0%} of the memory", "set": {"gpu_memory_utilization": max(0.05, math.floor((free / total - 0.03) * 100) / 100)}}]}

    if re.search(r"illegal memory access", log):
        return {"cause": "illegal_memory_access", "title": "The engine crashed on the GPU (illegal memory access)",
                "detail": "This has been seen with prefix caching on hybrid Mamba models, such as Nemotron and Qwen 3.5 and 3.8: a request that reuses cached text reads garbage state. Starting without prefix caching avoids it, at the cost of re-reading the whole prompt each turn.",
                "fixes": [{"label": "Turn prefix caching off", "add_args": ["--no-enable-prefix-caching"]}]}

    if re.search(r"out of memory|OutOfMemoryError|cudaErrorMemoryAllocation|CUDA error: out of memory", log, re.I):
        ctx = getattr(recipe, "max_context", None)
        fixes = [{"label": f"Lower the context to {max(4096, ctx // 2):,}", "set": {"max_context": max(4096, ctx // 2)}}] if ctx else []
        return {"cause": "out_of_memory", "title": "The GPU ran out of memory",
                "detail": "The model, its cache and what other models hold didn't fit together. Stop another model, lower the context, or let DGX-kit size the cache (clear the fixed KV cache size).",
                "fixes": fixes + ([{"label": "Let DGX-kit size the KV cache", "set": {"kv_cache_bytes": None}}] if getattr(recipe, "kv_cache_bytes", None) else [])}

    m = re.search(r"Model architectures \[([^\]]+)\] (?:are not supported|failed to be inspected)|(?:Unknown|Unsupported) (?:model )?architecture[: ]+([\w.]+)", log)
    if m:
        arch = (m[1] or m[2] or "").strip(" '\"")
        return {"cause": "unsupported_architecture", "title": "This engine image doesn't know the model",
                "detail": f"The engine doesn't support {arch}. It needs a newer or patched image (Settings → Engine images, or this model's own image setting).", "fixes": []}

    m = re.search(r"(?:FileNotFoundError|No such file or directory)[^\n]*?['\"]?(/[\w./@+-]+)['\"]?", log)
    if m:
        return {"cause": "file_missing", "title": "A file the model needs isn't there",
                "detail": f"{m[1]} wasn't found. Check the model folder and, for a GGUF model, the GGUF file name in its settings.", "fixes": []}

    if re.search(r"address already in use|Errno 98", log, re.I):
        return {"cause": "port_taken", "title": "The port is already in use",
                "detail": "Another program holds the port this model was given. Start it again; DGX-kit picks a free one.", "fixes": []}

    m = re.search(r"Value error, (.+?)\s*\[type=value_error", log, re.S)  # vLLM refusing a combination of settings
    if m:
        msg = " ".join(m[1].split())
        fixes = []
        if "expert parallelism is enabled" in msg:
            fixes = [{"label": "Remove --enable-expert-parallel", "remove_args": ["--enable-expert-parallel"]}]
        elif "Stochastic rounding for Mamba cache requires the SSM cache to be float16" in msg:
            fixes = [{"label": "Use a float16 SSM cache", "replace_args": ["--mamba-ssm-cache-dtype float16"]},
                     {"label": "Turn stochastic rounding off", "remove_args": ["--enable-mamba-cache-stochastic-rounding", "--mamba-cache-philox-rounds"]}]
        return {"cause": "bad_settings", "title": "The engine rejected a combination of settings", "detail": msg[:400], "fixes": fixes}

    errs = [l.strip() for l in log.splitlines() if re.search(r"\b(ERROR|Error|Exception|FATAL)\b", l)]
    if errs:
        return {"cause": "unknown", "title": "It stopped with an error", "detail": re.sub(r"^\(\w+ pid=\d+\)\s*|^\w+ \d\d-\d\d [\d:]+ \[[^\]]*\]\s*", "", errs[-1])[:300], "fixes": []}
    return None


def apply_fix(recipe, fix: dict):
    """The recipe with a fix applied (fields set, flag lines added when not already there)."""
    for k, v in (fix.get("set") or {}).items():
        setattr(recipe, k, v)
    flag = lambda line: (line.split() or [""])[0]
    for f in fix.get("remove_args") or []:
        recipe.extra_args = [l for l in recipe.extra_args if flag(l) != f]
    for line in fix.get("replace_args") or []:  # replaces the line of the same flag, or adds it
        out, done = [], False
        for l in recipe.extra_args:
            if flag(l) != flag(line):
                out.append(l)
            elif not done:
                out.append(line)
                done = True
        recipe.extra_args = out if done else [*out, line]
    for line in fix.get("add_args") or []:
        if line not in recipe.extra_args:
            recipe.extra_args = [*recipe.extra_args, line]
    return recipe
