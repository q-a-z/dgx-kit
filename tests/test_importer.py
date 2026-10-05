from pathlib import Path

from dgxkit.importer import parse_conf, read_confs, recipe_from_args, recipe_from_conf

FIX = Path(__file__).parent / "fixtures" / "ornith.conf"


def test_llmctl_conf_becomes_a_recipe():
    """A model's conf: the CMD array is the command; MODEL and OPTS are labels."""
    r, notes = recipe_from_conf("ornith-1.5-vllm-dflash2", parse_conf(FIX.read_text()))
    assert (r.name, r.image, r.engine) == ("ornith-1.5", "vllm-spark:0.29-pfxtest", "vllm")
    assert r.path == "/home/user/models/vllm/ornith-abl-dflash" and r.draft_path == "/home/user/models/vllm/ornith-dflash2"
    assert (r.max_context, r.kv_cache_dtype, r.num_speculative_tokens, r.draft_method) == (262144, "fp8", 7, "auto")
    assert r.speculative_extra == {"draft_sample_method": "probabilistic"}
    # Memory flags are kept as fields exactly as set; DGX-kit picks ports and names itself
    assert (r.kv_cache_bytes, r.gpu_memory_utilization) == (8_000_000_000, 0.28)
    for owned in ("--port", "--gpu-memory-utilization", "--kv-cache-memory", "--served-model-name", "--host"):
        assert owned not in r.extra_args
    assert """--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"max_new_tokens":32768}'""" in r.extra_args
    assert r.extra_args[0] == "--moe-backend humming" and "--language-model-only" in r.extra_args  # a flag and its value share a line
    assert "vLLM 0.30 corrupts" in r.notes and r.docker["mem_limit"] == "38g"


def test_speculative_json_and_container_mounts_map_to_host_paths():
    args = ["vllm", "serve", "/models/laguna", "--port", "8095", "--speculative-config",
            '{"method": "dflash", "model": "/models/laguna-dflash", "num_speculative_tokens": 5}']
    r, _ = recipe_from_args("laguna", args, image="vllm-spark:0.30", mounts=[("/home/user/models/vllm", "/models")])
    assert r.path == "/home/user/models/vllm/laguna" and r.draft_path == "/home/user/models/vllm/laguna-dflash"
    assert (r.draft_method, r.num_speculative_tokens, r.extra_args) == ("dflash", 5, [])


def test_read_confs_skips_backups(tmp_path):
    (tmp_path / "a.conf").write_text(FIX.read_text())
    (tmp_path / "a.conf.bak").write_text("broken")
    (tmp_path / "b.conf").write_text("CMD=( vllm serve org/some-model --max-model-len 8192 )\n")
    items = read_confs(str(tmp_path))
    assert [Path(i["file"]).name for i in items] == ["a.conf", "b.conf"]
    assert items[1]["recipe"]["repo"] == "org/some-model" and items[1]["recipe"]["max_context"] == 8192


def test_ornith_conf_starts_with_its_own_memory_settings_and_docker_limits():
    from pathlib import Path
    from dgxkit.control import docker_options, engine_command
    from dgxkit.importer import parse_conf, recipe_from_conf
    from dgxkit.sizing import plan
    text = (Path(__file__).parent / "fixtures" / "ornith.conf").read_text()
    text += '\nEXTRA_DOCKER_ARGS="--gpus all -e VLLM_USE_V1=1 -v /home/user/cache:/root/.cache --shm-size 16g --cap-add SYS_NICE"\n'
    r, notes = recipe_from_conf("ornith", parse_conf(text))
    assert r.docker["mem_limit"] == "38g" and r.docker["shm_size"] == "16g"
    assert r.env["VLLM_USE_V1"] == "1"
    assert any("--cap-add SYS_NICE" in n for n in notes)
    p = plan(weights_bytes=20 * 2**30, config={"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2, "num_hidden_layers": 48}, available_bytes=80 * 2**30, max_context=r.max_context,
             kv_cache_bytes=r.kv_cache_bytes, total_bytes=120 * 2**30)
    cmd = engine_command(r, p, 8100, "/models")
    val = lambda f: cmd[cmd.index(f) + 1]
    assert val("--gpu-memory-utilization") == "0.28"
    assert val("--kv-cache-memory-bytes") == "8000000000"
    assert val("--max-model-len") == "262144"
    for f in ("--moe-backend", "--max-num-seqs", "--mamba-ssm-cache-dtype", "--chat-template", "--language-model-only",
              "--override-generation-config", "--tool-call-parser", "--reasoning-parser", "--enable-auto-tool-choice"):
        assert f in cmd, f
    o = docker_options(r, "/models")
    assert o["mem_limit"] == "38g" and o["volumes"]["/home/user/cache"]["bind"] == "/root/.cache"


def test_extra_flag_lines_become_separate_engine_arguments():
    from dgxkit.control import extra_tokens
    lines = ["--max-num-seqs 2", "--language-model-only", """--override-generation-config '{"temperature":1,"top_p":0.95}'""",
             '{"bare":"json from an older import"}', "2"]
    assert extra_tokens(lines) == ["--max-num-seqs", "2", "--language-model-only", "--override-generation-config",
                                   '{"temperature":1,"top_p":0.95}', '{"bare":"json from an older import"}', "2"]


def test_options_text_round_trips_and_deleting_a_line_clears_it():
    from dgxkit.options import apply_text, to_text
    from dgxkit.recipes import Recipe
    r = Recipe(name="n", repo="x", engine="vllm", max_context=393216, kv_cache_bytes=4500000000, gpu_memory_utilization=0.3,
               kv_cache_dtype="fp8", draft_path="/m/draft", draft_method="auto", num_speculative_tokens=3,
               extra_args=["--moe-backend marlin", """--override-generation-config '{"temperature":1}'"""])
    text = to_text(r)
    assert text.splitlines()[0] == "--max-model-len 393216" and "--moe-backend marlin" in text.splitlines()
    back = apply_text(r, text)
    assert (back.max_context, back.kv_cache_bytes, back.gpu_memory_utilization, back.kv_cache_dtype, back.draft_path,
            back.num_speculative_tokens, back.extra_args) == (393216, 4500000000, 0.3, "fp8", "/m/draft", 3, r.extra_args)
    cleared = apply_text(r, "\n".join(l for l in text.splitlines() if "kv-cache-memory" not in l and "speculative" not in l))
    assert cleared.kv_cache_bytes is None and cleared.draft_path is None


def test_home_in_a_conf_is_the_persons_home(monkeypatch):
    from dgxkit.importer import parse_conf
    monkeypatch.setenv("DGXKIT_HOME", "/home/alice")
    conf = parse_conf('MODELS=$HOME/models\nCMD=(\n  vllm serve ${MODELS}/vllm/x --tokenizer ~/tok --port $PORT\n)\nPORT=8081\n')
    assert conf["MODELS"] == "/home/alice/models"
    assert conf["CMD"][:3] == ["vllm", "serve", "/home/alice/models/vllm/x"] and conf["CMD"][3:5] == ["--tokenizer", "/home/alice/tok"]


def test_weights_named_for_another_machine_are_found_here_by_folder_name(tmp_path):
    from dgxkit.importer import relocate
    from dgxkit.recipes import Recipe
    here = tmp_path / "models" / "vllm" / "nemotron-abliterated"
    here.mkdir(parents=True)
    r = Recipe(name="n", repo="local/n", path="/home/other/models/vllm/nemotron-abliterated", draft_path="/home/other/models/vllm/gone")
    notes = relocate(r, [{"path": str(here)}])
    assert r.path == str(here) and r.draft_path == "/home/other/models/vllm/gone"  # found one, left the other
    assert any("using" in n for n in notes) and any("not found" in n for n in notes)
    dup = Recipe(name="d", repo="local/d", path="/x/same")
    relocate(dup, [{"path": "/a/same"}, {"path": "/b/same"}])
    assert dup.path == "/x/same"  # two folders with that name: don't guess


def test_relocated_weights_are_read_from_their_new_place(tmp_path):
    import json
    from dgxkit.importer import relocate
    from dgxkit.recipes import Recipe
    here = tmp_path / "m" / "qwen"
    here.mkdir(parents=True)
    (here / "config.json").write_text(json.dumps({"architectures": ["Qwen3MoeForCausalLM"], "num_hidden_layers": 4}))
    (here / "model.safetensors").write_bytes(b"x" * 500)
    r = Recipe(name="q", repo="local/q", path="/home/other/models/qwen")
    notes = ["can't read /home/other/models/qwen from here; sizes will fill in when it's readable", "an unrelated note"]
    relocate(r, [{"path": str(here)}], notes)
    assert r.weights_bytes == 500 and r.config["num_hidden_layers"] == 4  # sizing has what it needs
    assert not any("can't read" in n for n in notes) and "an unrelated note" in notes
