import json

from dgxkit.library import Settings, recipe_from_folder, scan


def model(root, name, arch, layers=32, files=("model.safetensors",), **cfg):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": [arch], "num_hidden_layers": layers,
                                               "num_attention_heads": 32, "num_key_value_heads": 8,
                                               "hidden_size": 4096, **cfg}))
    for f in files:
        (d / f).write_bytes(b"x" * 1000)
    return d


def test_scan_sorts_models_and_drafts_like_on_q_spark(tmp_path):
    vllm = tmp_path / "models" / "vllm"
    model(vllm, "ornith-abl-dflash", "Qwen3ForCausalLM")                 # main model despite the name
    model(vllm, "ornith-dflash2", "DFlash2DraftModel", layers=5)         # draft by architecture
    model(vllm, "gemma-4-dspark", "DSparkDraftModel", layers=8)
    model(vllm, "qwen3-coder-eagle", "Qwen3MoeForCausalLM", layers=1)    # draft by depth and name
    model(vllm, "nemotron-3.5-abliterated-nvfp4", "NemotronHForCausalLM",
          quantization_config={"quant_method": "modelopt"})
    gguf = tmp_path / "models" / "gguf" / "served"
    gguf.mkdir(parents=True)
    (gguf / "m-Q4_K_M.gguf").write_bytes(b"x" * 500)
    hub = tmp_path / "hub" / "models--org--Tiny" / "snapshots" / "abc"
    model(hub.parent, "abc", "LlamaForCausalLM")
    (tmp_path / "hub" / "models--org--Tiny" / "blobs").mkdir()

    found = scan([str(tmp_path / "models"), str(tmp_path / "hub"), str(tmp_path / "nope")])
    names = [m["name"] for m in found["models"]]
    assert names == ["nemotron-3.5-abliterated-nvfp4", "org/Tiny", "ornith-abl-dflash", "served"]
    assert [d["name"] for d in found["drafts"]] == ["gemma-4-dspark", "ornith-dflash2", "qwen3-coder-eagle"]
    assert found["missing"] == [str(tmp_path / "nope")]
    served = next(m for m in found["models"] if m["name"] == "served")
    assert served["format"] == "gguf" and served["gguf_files"] == ["m-Q4_K_M.gguf"] and served["size_bytes"] == 500


def test_user_can_override_kind_and_paths(tmp_path):
    model(tmp_path / "m", "odd-one", "LlamaForCausalLM")
    s = Settings(str(tmp_path / "state"), [str(tmp_path / "m")])
    s.set_kind(str(tmp_path / "m" / "odd-one"), "draft")
    found = scan(s.model_paths, s.kinds)
    assert found["drafts"][0]["kind_set_by_user"] and not found["models"]
    assert s.set_model_paths([" /a ", "/a", "", "/b"]) == ["/a", "/b"]


def test_recipe_from_folder_uses_weights_in_place(tmp_path):
    main = model(tmp_path, "Laguna-XS", "LagunaForCausalLM", files=("a.safetensors", "b.safetensors"))
    draft = model(tmp_path, "laguna-dflash", "DFlashLagunaForCausalLM", layers=2)
    r = recipe_from_folder(str(main), str(draft))
    assert r.name == "laguna-xs" and r.repo == "local/Laguna-XS"
    assert r.path == str(main) and r.draft_path == str(draft)
    assert r.draft_method == "dflash" and r.weights_bytes == 2000


def test_drafts_on_q_spark_are_found_and_classed(tmp_path):
    """Folder names from q's Spark: DFlash/EAGLE/DSpark drafts, some nested under their model."""
    import json
    from dgxkit.library import scan

    def model(rel, layers, arch="Qwen3MoeForCausalLM", weights="model.safetensors"):
        d = tmp_path / rel
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"architectures": [arch], "num_hidden_layers": layers}))
        (d / weights).write_bytes(b"x")

    model("vllm/ornith-abl-dflash", 40)
    model("vllm/qwen3-coder-dflash", 5)
    model("vllm/qwen3-coder-eagle", 1)
    model("vllm/gemma-4-dspark", 40, weights="pytorch_model.pt")
    model("vllm/laguna-xs-2.1", 32)
    model("vllm/laguna-xs-2.1/laguna-xs-2.1-dflash-nvfp4", 3)
    model("vllm/ornith-abl-dflash/dflash_draft", 2)
    (tmp_path / "vllm/half-done").mkdir()
    (tmp_path / "vllm/half-done/config.json").write_text("{}")
    out = scan([str(tmp_path)])
    names = lambda k: sorted(i["name"] for i in out[k])
    assert names("models") == ["laguna-xs-2.1", "ornith-abl-dflash"]
    assert names("drafts") == ["gemma-4-dspark", "laguna-xs-2.1-dflash-nvfp4", "ornith-abl-dflash/dflash_draft", "qwen3-coder-dflash", "qwen3-coder-eagle"]
    assert out["incomplete"] == [str(tmp_path / "vllm/half-done")]
