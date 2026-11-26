import asyncio

import httpx
import pytest

from dgxkit.control import docker_options, engine_command
from dgxkit.discover import parse
from dgxkit.engines import adapter_for
from dgxkit.engines import prometheus as P
from dgxkit.images import BUILDS, DEFAULTS, ImageManager
from dgxkit.recipes import Recipe
from dgxkit.sizing import Plan
from dgxkit.store import ModelStore
from dgxkit.velogb10 import PRESETS, main, prepare

PLAN = Plan(32768, 34406, 2**30, 1.05, True)


def test_launch_preserves_paths_alias_and_lane_sizing():
    r = Recipe("test-velo", "org/model", engine="velogb10", path="/models/with spaces",
               draft_method="mtp", extra_args=["--prefix-cache on"])
    cmd = engine_command(r, PLAN, 8111, "/models")
    for flag, value in {"--model-dir": r.path, "--model-name": r.name, "--port": "8111",
                        "--max-seq-len": "32768", "--max-batch": "1", "--kv-cache": "bf16",
                        "--mtp": "auto", "--prefix-cache": "on"}.items():
        assert cmd[cmd.index(flag) + 1] == value
    assert cmd[:2] == ["/opt/velogb10/gb10_inference", "--server"]
    assert "--kv-cache-memory-bytes" not in cmd
    opts = docker_options(r, "/models")
    assert opts["environment"] is None
    assert all(v["mode"] == "ro" for v in opts["volumes"].values())


@pytest.mark.parametrize("method", ["dflash", "dflash2"])
def test_external_draft_uses_velo_flags(method):
    r = Recipe("test", "org/model", engine="velogb10", draft_repo="org/draft",
               draft_method=method, kv_cache_dtype="fp8")
    cmd = engine_command(r, PLAN, 8100, "/models")
    assert cmd[cmd.index("--draft-dir") + 1] == "/models/org--draft"
    assert cmd[cmd.index("--spec-source") + 1] == method
    assert cmd[cmd.index("--kv-cache") + 1] == "k8v8"


@pytest.mark.parametrize("settings", [
    {"kv_cache_bytes": 1024}, {"fill_memory": True}, {"kv_cache_dtype": "q8"},
    {"draft_method": "eagle3", "draft_repo": "org/draft"}, {"draft_method": "dflash2"},
    {"cluster": {"workers": [{"host": "peer"}], "head_ip": "192.0.2.1"}},
])
def test_unsupported_settings_refused(settings):
    with pytest.raises(ValueError):
        engine_command(Recipe("test", "org/model", engine="velogb10", **settings), PLAN, 8100, "/models")


def test_velo_fabric_uses_the_deployed_head_and_node_commands():
    from dgxkit.control import DockerRunner, RDMA_DEVICES
    from tests.test_cluster import Box

    config = {"head_ip": "192.0.2.1", "workers": [{"host": "q@192.0.2.2"}], "node_port": 29542,
              "rdma_dev": "roce0"}
    r = Recipe("test", "org/model", engine="velogb10", image="img:1", cluster=config, quantization="exl3")
    cmd = engine_command(r, PLAN, 8100, "/models")
    assert cmd[0].endswith("/run_tp_server.sh")
    assert cmd[cmd.index("--node") + 1] == "192.0.2.2:29542"
    assert cmd[cmd.index("--tp") + 1] == "2"
    assert docker_options(r, "/models")["devices"] == RDMA_DEVICES
    local, worker = Box(), Box()
    runner = DockerRunner(local)
    runner.remote = lambda _: worker
    runner.start(r, cmd, 8100, "/models")
    assert worker.ran == [("dgxkit-test-w1", ["--node", "--host", "192.0.2.2", "--port", "29542",
                                             "--rdma-dev", "roce0"],
                           {"dgxkit.worker": "test", "dgxkit.engine": "velogb10-node"})]
    config["discovery_wait"] = 5
    cmd = engine_command(r, PLAN, 8100, "/models")
    assert cmd[0].endswith("/gb10_inference")
    assert cmd[cmd.index("--discover-wait") + 1] == "5"


def test_loaded_velogb10_request_rates_are_reported_from_histograms():
    async def check():
        body = ("velogb10_decode_tokens_per_second_sum 110\n"
                "velogb10_decode_tokens_per_second_count 2\n"
                "velogb10_prefill_tokens_per_second_sum 3000\n"
                "velogb10_prefill_tokens_per_second_count 2\n")
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = adapter_for("velogb10", "http://velo")
            await adapter.client.aclose()
            adapter.client = client
            result = await adapter.scrape()
            assert result["decode_tps_req"] == 55 and result["prefill_tps_req"] == 1500

    asyncio.run(check())


def test_metrics_count_finished_requests_and_do_not_double_count_cached_prompt():
    adapter = adapter_for("velogb10", "http://localhost:8100")
    def metrics(gen):
        return P.parse(f'''velogb10_requests_running 1
velogb10_requests_waiting 2
velogb10_prompt_tokens_total 1000
velogb10_prompt_tokens_cached_total 400
velogb10_generation_tokens_total {gen}
velogb10_requests_total{{finish="stop"}} 3
velogb10_requests_total{{finish="tool_calls"}} 2
velogb10_time_to_first_token_seconds_bucket{{le="0.5"}} 4
velogb10_time_to_first_token_seconds_bucket{{le="+Inf"}} 5
velogb10_time_to_first_token_seconds_sum 2
velogb10_time_to_first_token_seconds_count 5
''')
    adapter.compute(metrics(100), 0)
    stats = adapter.compute(metrics(160), 2)
    assert stats["decode_tps"] == 30
    assert stats["requests_total"] == 5
    assert stats["prompt_tokens_total"] == 1000
    assert stats["running"] == 1 and stats["waiting"] == 2
    assert stats["kv_used_pct"] is None
    assert stats["ttft_avg_s"] == 0.4


def test_health_and_model_metadata_use_native_endpoints():
    async def check():
        def respond(request):
            if request.url.path == "/health":
                return httpx.Response(503)
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"data": [{"id": "velo", "max_model_len": 8192}]})
        adapter = adapter_for("velogb10", "http://velo")
        await adapter.client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter.client = client
            assert not await adapter.health()
            assert await adapter.read_static() == {"model": "velo", "context_tokens": 8192}
    asyncio.run(check())


@pytest.mark.parametrize("status,body", [(503, ""), (200, "velogb10_engine_alive 0")])
def test_sampler_refuses_failed_or_dead_engine_metrics(status, body):
    async def check():
        adapter = adapter_for("velogb10", "http://velo")
        await adapter.client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(status, text=body)
        )) as client:
            adapter.client = client
            with pytest.raises((httpx.HTTPStatusError, RuntimeError)):
                await adapter.scrape()
    asyncio.run(check())


def test_exl3_uses_its_distinct_kv_format():
    r = Recipe("exl", "org/pack", engine="velogb10", quantization="exl3")
    cmd = engine_command(r, PLAN, 8100, "/models")
    assert cmd[cmd.index("--kv-cache") + 1] == "q8"
    r.kv_cache_dtype = "fp16"
    cmd = engine_command(r, PLAN, 8100, "/models")
    assert cmd[cmd.index("--kv-cache") + 1] == "f16"
    r.kv_cache_dtype = "fp8"
    cmd = engine_command(r, PLAN, 8100, "/models")
    assert cmd[cmd.index("--kv-cache") + 1] == "fp8"


def test_discovery_recognizes_servers_but_not_peer_supervisors():
    args = ["/opt/velogb10/gb10_inference", "--server", "--model-dir=/models/pack",
            "--port", "8112", "--model-name", "velo", "--draft-dir", "/models/draft"]
    assert parse(args) == {"engine": "velogb10", "port": 8112, "model": "/models/pack",
                           "served_name": "velo", "draft": "/models/draft"}
    assert parse([args[0], "--node", "--port", "29500"]) is None


def test_image_is_a_pinned_local_build():
    manager = ImageManager(catalog={"velogb10": DEFAULTS["velogb10"]})
    assert manager.build_for("dgx-kit/velogb10:0.8.0") == "velogb10-0.8.0"
    assert BUILDS["velogb10-0.8.0"]["args"] == {"VERSION": "v0.8.0"}


def test_presets_hydrate_metadata_without_downloading_or_overwriting(monkeypatch, tmp_path):
    def inspect(repo, draft_repo=None, token=None, revision=None):
        return Recipe("original", repo, config={"num_hidden_layers": 24}, weights_bytes=123,
                      revision=revision, draft_repo=draft_repo, draft_weights_bytes=10 if draft_repo else 0)
    monkeypatch.setattr("dgxkit.velogb10.inspect_repo", inspect)
    for name in PRESETS:
        r = prepare(name)
        assert r.name == name and r.engine == "velogb10" and not r.publish
        assert r.config == {"num_hidden_layers": 24} and r.weights_bytes == 123
        engine_command(r, PLAN, 8100, "/models")
    name = next(iter(PRESETS))
    monkeypatch.setattr("sys.argv", ["prepare", name, "--state-dir", str(tmp_path)])
    main()
    assert ModelStore(str(tmp_path)).get(name).engine == "velogb10"
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
