import struct

from dgxkit.gguf import read_config
from dgxkit.sizing import kv_bytes_per_token


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _s(key) + struct.pack("<I", kind) + payload


def tiny_gguf(path, arch="gemma4"):
    u32 = lambda n: struct.pack("<I", n)
    items = [
        _kv("general.architecture", 8, _s(arch)),
        _kv(f"{arch}.block_count", 4, u32(12)),
        _kv(f"{arch}.embedding_length", 4, u32(2048)),
        _kv(f"{arch}.attention.head_count", 4, u32(8)),
        _kv(f"{arch}.attention.head_count_kv", 4, u32(2)),
        _kv(f"{arch}.attention.key_length", 4, u32(256)),
        _kv(f"{arch}.context_length", 4, u32(65536)),
        # every third layer is full attention, the rest sliding-window
        _kv(f"{arch}.attention.sliding_window_pattern", 9, u32(7) + struct.pack("<Q", 12) + bytes([1, 1, 0] * 4)),
        _kv(f"{arch}.attention.shared_kv_layers", 4, u32(3)),
        _kv("tokenizer.ggml.tokens", 9, u32(8) + struct.pack("<Q", 1) + _s("x")),
    ]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(items)) + b"".join(items))


def test_a_ggufs_header_gives_the_models_shape(tmp_path):
    f = tmp_path / "m.gguf"
    tiny_gguf(f)
    c = read_config(f)
    assert (c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["max_position_embeddings"]) == (12, 2048, 8, 2, 256, 65536)
    # of 12 layers, 4 are full attention; the last 3 share KV, which drops the one among them (layer 11)
    assert kv_bytes_per_token(c, "auto") == 2 * 3 * 2 * 256 * 2
    assert read_config(tmp_path / "missing.gguf") == {}
    (tmp_path / "bad.gguf").write_bytes(b"nope")
    assert read_config(tmp_path / "bad.gguf") == {}
