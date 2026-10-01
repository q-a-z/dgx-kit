"""Read the model's shape out of a GGUF file's header, so context and KV cache can be sized for llama.cpp models,
which come without a Hugging Face config.json.

The header holds key/value metadata: <arch>.block_count, <arch>.attention.head_count and so on. Only those are read;
the tokenizer tables that follow are skipped.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


_SHARD = re.compile(r"^(?P<stem>.+)-(?P<n>\d{5})-of-(?P<total>\d{5})\.gguf$")


def shard_of(name: str) -> tuple[str, int, str] | None:
    """(stem, shard number, total) when the name is one piece of a split GGUF (model-00001-of-00002.gguf)."""
    m = _SHARD.match(name)
    return (m["stem"], int(m["n"]), m["total"]) if m else None


def first_shards(names: list[str]) -> list[str]:
    """The model files to offer: of a split GGUF only the first piece, since llama.cpp loads the others from it."""
    return [n for n in names if not (shard_of(Path(n).name) and shard_of(Path(n).name)[1] != 1)]


def split_files(first: Path) -> list[Path]:
    """Every piece of the GGUF that `first` belongs to (just itself when it is not split)."""
    sh = shard_of(first.name)
    if not sh or not first.parent.is_dir():
        return [first]
    stem, _, total = sh
    pieces = sorted(p for p in first.parent.iterdir()
                    if (s := shard_of(p.name)) and s[0] == stem and s[2] == total)
    return pieces or [first]


def total_size(first: Path) -> int:
    """Bytes of all the pieces of a (possibly split) GGUF."""
    return sum(p.stat().st_size for p in split_files(first) if p.exists())


def _read(f, fmt: str):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _string(f) -> str:
    return f.read(_read(f, "<Q")).decode("utf-8", "replace")


def _value(f, kind: int):
    if kind in _SCALARS:
        return _read(f, _SCALARS[kind])
    if kind == _STRING:
        return _string(f)
    if kind == _ARRAY:
        inner, n = _read(f, "<I"), _read(f, "<Q")
        return [_value(f, inner) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {kind}")


def read_metadata(path: str | Path, stop_at: str = "tokenizer.") -> dict:
    """The header's key/values up to the first key starting with stop_at (the model's own numbers come before it)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        _read(f, "<I")  # version
        _read(f, "<Q")  # tensor count
        count = _read(f, "<Q")
        out: dict = {}
        for _ in range(count):
            key = _string(f)
            if key.startswith(stop_at):
                break
            out[key] = _value(f, _read(f, "<I"))
        return out


def read_config(path: str | Path) -> dict:
    """A config.json-like dict with what sizing needs; {} when the file can't be read or says too little."""
    try:
        meta = read_metadata(path)
    except (OSError, ValueError, struct.error):
        return {}
    arch = meta.get("general.architecture")
    if not arch:
        return {}
    g = lambda k: meta.get(f"{arch}.{k}")
    layers, hidden, heads, kv = g("block_count"), g("embedding_length"), g("attention.head_count"), g("attention.head_count_kv")
    if not layers or not hidden:
        return {}
    cfg: dict = {"num_hidden_layers": layers, "hidden_size": hidden, "architectures": [arch]}
    if g("context_length"):
        cfg["max_position_embeddings"] = g("context_length")
    if isinstance(heads, list):  # per-layer counts: take the widest
        heads = max(heads or [0])
    if heads:
        cfg["num_attention_heads"] = heads
    if isinstance(kv, list):  # per-layer KV heads; a 0 is a layer without attention (a Mamba or linear layer)
        cfg["layer_types"] = ["full_attention" if n else "linear" for n in kv]
        kv = max(kv or [0])
    swa = g("attention.sliding_window_pattern")  # True = a sliding-window layer, whose cache stays small whatever the context
    if isinstance(swa, list) and len(swa) == layers and "layer_types" not in cfg:
        cfg["layer_types"] = ["sliding_attention" if x else "full_attention" for x in swa]
    shared = g("attention.shared_kv_layers")  # the last layers reuse earlier layers' KV and store none of their own
    if shared and "layer_types" in cfg:
        cfg["layer_types"][-int(shared):] = ["shared_kv"] * int(shared)
    if kv:
        cfg["num_key_value_heads"] = kv
    if g("attention.key_length"):
        cfg["head_dim"] = g("attention.key_length")
    return cfg
