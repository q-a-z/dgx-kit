from dataclasses import dataclass, field

from dgxkit.diagnose import apply_fix, diagnose

KV_LOG = ("(EngineCore pid=158) ERROR 10-03 07:19:42 [core.py:1366] ValueError: To serve at least one request with the model's max seq len (131072), "
          "(5.84 GiB KV cache is needed, which is larger than the available KV cache memory (4.18 GiB). Based on the available memory, "
          "the estimated maximum model length is 86400. Try increasing `gpu_memory_utilization` or decreasing `max_model_len`.\n")


@dataclass
class R:
    max_context: int | None = 131072
    kv_cache_bytes: int | None = 4_500_000_000
    gpu_memory_utilization: float | None = 0.3
    extra_args: list = field(default_factory=lambda: ["--max-num-seqs 2"])


def test_a_kv_cache_that_is_too_small_says_how_much_is_needed_and_offers_both_fixes():
    d = diagnose(KV_LOG, R())
    assert d["cause"] == "kv_too_small" and "5.84 GiB" in d["detail"] and "86,400" in d["detail"]
    raise_kv, lower_ctx = d["fixes"]
    assert raise_kv["set"]["kv_cache_bytes"] >= int(5.84 * 2**30)  # at least what vLLM asked for, with a margin
    assert lower_ctx["set"] == {"max_context": 86016}  # 86400 rounded down to a multiple of 1024


def test_other_known_stops():
    free = diagnose("ValueError: Free memory on device cuda:0 (20.1/121.7 GiB) on startup is less than desired GPU memory utilization (0.3, 36.5 GiB). Decrease it.", R())
    assert free["cause"] == "memory_taken" and free["fixes"][0]["set"]["gpu_memory_utilization"] == 0.13
    assert diagnose("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB", R())["fixes"][0]["set"] == {"max_context": 65536}
    crash = diagnose("CUDA error: an illegal memory access was encountered", R())
    assert crash["cause"] == "illegal_memory_access" and crash["fixes"][0]["add_args"] == ["--no-enable-prefix-caching"]
    assert diagnose("ValueError: Model architectures ['DFlash2DraftModel'] are not supported for now.", R())["cause"] == "unsupported_architecture"
    assert "/home/x/m.gguf" in diagnose("FileNotFoundError: [Errno 2] No such file or directory: '/home/x/m.gguf'", R())["detail"]
    assert diagnose("something odd\nERROR boom happened\n", R())["cause"] == "unknown"
    assert diagnose("all fine, listening on port 8000\n", R()) is None


def test_a_fix_changes_the_recipe_and_does_not_repeat_a_flag():
    r = apply_fix(R(), {"set": {"kv_cache_bytes": 7_000_000_000}, "add_args": ["--no-enable-prefix-caching"]})
    assert r.kv_cache_bytes == 7_000_000_000 and r.extra_args == ["--max-num-seqs 2", "--no-enable-prefix-caching"]
    assert apply_fix(r, {"add_args": ["--no-enable-prefix-caching"]}).extra_args.count("--no-enable-prefix-caching") == 1


EP_LOG = ("(APIServer pid=1) pydantic_core._pydantic_core.ValidationError: 1 validation error for SpeculativeConfig\n"
          "(APIServer pid=1)   Value error, Number of experts in the model must be greater than 0 when expert parallelism is enabled. [type=value_error, input_value=ArgsKwargs((), {})]\n")
SR_LOG = ("(APIServer pid=1) pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig\n"
          "(APIServer pid=1)   Value error, Stochastic rounding for Mamba cache requires the SSM cache to be float16. Please set it explicitly, "
          "by specifying `--mamba-ssm-cache-dtype float16`, or disable stochastic rounding. [type=value_error, input_value=ArgsKwargs((), {})]\n")


def test_settings_the_engine_refuses_are_named_with_a_fix():
    ep = diagnose(EP_LOG, R())
    assert ep["cause"] == "bad_settings" and "expert parallelism" in ep["detail"] and ep["fixes"][0]["remove_args"] == ["--enable-expert-parallel"]
    sr = diagnose(SR_LOG, R())
    assert [f["label"] for f in sr["fixes"]] == ["Use a float16 SSM cache", "Turn stochastic rounding off"]
    r = R()
    r.extra_args = ["--max-num-seqs 2", "--enable-expert-parallel", "--mamba-ssm-cache-dtype bfloat16", "--enable-mamba-cache-stochastic-rounding", "--mamba-cache-philox-rounds 5"]
    assert apply_fix(r, ep["fixes"][0]).extra_args == ["--max-num-seqs 2", "--mamba-ssm-cache-dtype bfloat16", "--enable-mamba-cache-stochastic-rounding", "--mamba-cache-philox-rounds 5"]
    assert apply_fix(r, sr["fixes"][0]).extra_args[1] == "--mamba-ssm-cache-dtype float16"  # replaced in place
    assert apply_fix(r, sr["fixes"][1]).extra_args == ["--max-num-seqs 2", "--mamba-ssm-cache-dtype float16"]
    assert apply_fix(R(), {"replace_args": ["--mamba-ssm-cache-dtype float16"]}).extra_args[-1] == "--mamba-ssm-cache-dtype float16"  # added when absent
