from dgxkit.recipes import build_recipe, detect_draft_method

CFG = {"architectures": ["Qwen3MoeForCausalLM"], "num_hidden_layers": 48,
       "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}


def test_nvfp4_repo_goes_to_vllm_with_fp8_kv():
    r = build_recipe("org/Model-NVFP4", CFG,
                     [("config.json", 1000), ("model-00001.safetensors", 10), ("model-00002.safetensors", 20)])
    assert (r.engine, r.quantization, r.kv_cache_dtype) == ("vllm", "nvfp4", "fp8")
    assert r.weights_bytes == 30 and r.name == "model-nvfp4"


def test_gguf_repo_with_several_quants_waits_for_a_choice():
    r = build_recipe("org/Model-GGUF", {}, [("m-Q4_K_M.gguf", 5), ("m-Q8_0.gguf", 9)])
    assert r.engine == "llamacpp" and r.gguf_file is None and r.weights_bytes == 0
    r = build_recipe("org/Model-GGUF", {}, [("m-Q4_K_M.gguf", 5), ("m-Q8_0.gguf", 9)], gguf_file="m-Q8_0.gguf")
    assert r.weights_bytes == 9


def test_draft_methods():
    assert detect_draft_method({"architectures": ["LlamaForCausalLMEagle3"]}) == "eagle3"
    assert detect_draft_method({"architectures": ["DFlashDraftModel"]}) == "dflash"
    assert detect_draft_method({"architectures": ["LlamaForCausalLM"]}) == "draft_model"
