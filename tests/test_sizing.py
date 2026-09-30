from dgxkit.sizing import GIB, attention_layers, kv_bytes_per_token, plan

LLAMA_8B = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
            "hidden_size": 4096, "max_position_embeddings": 131072}


def test_kv_bytes_dense():
    # 2 x 32 layers x 8 heads x 128 dim x 2 bytes = 128 KiB per token
    assert kv_bytes_per_token(LLAMA_8B) == 131072
    assert kv_bytes_per_token(LLAMA_8B, "fp8") == 65536


def test_hybrid_counts_only_attention_layers():
    nemotron_like = {"num_hidden_layers": 6, "hybrid_override_pattern": "M-M*-*",
                     "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 128}
    assert attention_layers(nemotron_like) == 2
    qwen_next_like = {"num_hidden_layers": 4, "layer_types": ["linear_attention"] * 3 + ["full_attention"]}
    assert attention_layers(qwen_next_like) == 1


def test_vl_config_uses_text_config():
    assert attention_layers({"text_config": LLAMA_8B}) == 32


def test_plan_picks_largest_context_with_two_way_concurrency():
    p = plan(LLAMA_8B, weights_bytes=16 * GIB, available_bytes=100 * GIB)
    # budget = 100 - 8 - 17.6 = 74.4 GiB -> 609,484 tokens at 128 KiB
    assert p.fits and p.context_tokens == 131072
    assert p.concurrency >= 2
    assert p.kv_pool_tokens == int((100 * GIB - 8 * GIB - int(16 * GIB * 1.1)) // 131072)


def test_plan_halves_context_when_memory_is_tight():
    p = plan(LLAMA_8B, weights_bytes=16 * GIB, available_bytes=40 * GIB)
    assert p.fits and p.context_tokens < 131072 and p.concurrency >= 2


def test_plan_reports_when_weights_do_not_fit():
    p = plan(LLAMA_8B, weights_bytes=60 * GIB, available_bytes=50 * GIB)
    assert not p.fits and "weights" in p.reason


def test_fixed_kv_cache_is_kept_and_vllm_gets_a_share_that_fits():
    """ornith from llmctl: 8 GB KV fixed. vLLM's 0.92 default crashed it beside another model."""
    from dgxkit.sizing import GIB, plan
    cfg = {"num_hidden_layers": 40, "num_key_value_heads": 2, "head_dim": 256, "max_position_embeddings": 262144}
    p = plan(cfg, 20 * GIB, 80 * GIB, kv_dtype="fp8", max_context=262144, kv_cache_bytes=8_000_000_000, total_bytes=121 * GIB)
    assert p.fits and p.kv_bytes == 8_000_000_000 and p.context_tokens <= 262144
    assert 0.2 < p.gpu_fraction < 0.35
    q = plan(cfg, 20 * GIB, 80 * GIB, kv_dtype="fp8", total_bytes=121 * GIB)
    assert q.gpu_fraction <= 0.92 and not plan(cfg, 20 * GIB, 30 * GIB, kv_cache_bytes=40 * GIB).fits


def test_nemotron_h_counts_attention_from_layers_block_type():
    from dgxkit.sizing import attention_layers
    cfg = {"layers_block_type": ["linear_attention", "moe", "full_attention", "moe", "full_attention"]}  # no num_hidden_layers
    assert attention_layers(cfg) == 2
