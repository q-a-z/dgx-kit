from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dgxkit.app import Services, create_app
from dgxkit.collectors.gpu import FakeGpu
from dgxkit.collectors.system import SystemCollector
from dgxkit.downloader import Downloader
from dgxkit.sampler import Sampler
from dgxkit.store import ModelStore

ROOT = Path(__file__).parent / "fixtures" / "root"
CFG = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
       "hidden_size": 4096, "max_position_embeddings": 131072}


class FakeRunner:
    def __init__(self):
        self.running, self.started = {}, []

    def status(self):
        return {n: {"state": "running", "port": p, "id": f"id-{n}", "engine": "vllm", "meta": {}} for n, p in self.running.items()}

    def ports_in_use(self):
        return set(self.running.values())

    def start(self, r, cmd, port, models_root, meta=None):
        self.running[r.name] = port
        self.started.append(cmd)

    def stop(self, name):
        self.running.pop(name, None)

    def logs(self, name, tail=200):
        return "log line"


class FakeGateway:
    port, master_key, problem = 4000, None, None

    def __init__(self):
        self.synced = []

    def sync(self, published):
        self.synced.append(dict(published))

    def status(self):
        return {"models": sorted(self.synced[-1]) if self.synced else []}


class FakeImages:
    def __init__(self, ready=True):
        self.ready = ready

    def image_for(self, engine):
        return f"dgx-kit/{engine}:test"

    def is_ready(self, image):
        return self.ready

    defaults = {"vllm": "dgx-kit/vllm:test"}

    def list(self):
        return []

    def start_pull(self, engine):
        from dgxkit.images import Job
        self.ready = True
        return Job(engine, "pull", self.image_for(engine), state="done")


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    if "page_dir" not in request.fixturenames:  # API tests run without a built page
        monkeypatch.setenv("DGXKIT_WEB_DIR", str(tmp_path / "no-page"))
    models = tmp_path / "models"
    models.mkdir()
    s = Services(root=str(ROOT), models_root=str(models), state_dir=str(tmp_path / "state"), hf_token=None,
                 sampler=Sampler(FakeGpu(), SystemCollector(str(ROOT))), store=ModelStore(str(tmp_path / "state")),
                 runner=FakeRunner(), images=FakeImages(), downloader=Downloader(str(models)),
                 gateway=FakeGateway())
    with TestClient(create_app(s)) as client:
        yield client, s, models


def model_body(**over):
    return {"name": "llama", "repo": "org/Llama-8B", "config": CFG, "weights_bytes": 16 * 2**30, **over}


def test_create_edit_and_version_history(env):
    client, s, _ = env
    assert client.post("/api/models", json=model_body()).status_code == 201
    assert client.post("/api/models", json=model_body()).status_code == 409
    r = client.put("/api/models/llama", json=model_body(max_context=32768))
    assert r.json() == {"saved": True, "applies_on_next_start": False}
    versions = client.get("/api/models/llama/versions").json()
    assert len(versions) == 1
    assert client.post(f"/api/models/llama/restore/{versions[0]}").json()["max_context"] is None
    assert client.post("/api/models/llama/restore/..%2F..%2Fetc").status_code == 404


def test_start_refuses_until_downloaded_then_runs(env):
    client, s, models = env
    client.post("/api/models", json=model_body())
    r = client.post("/api/models/llama/start")
    assert r.status_code == 409 and "model isn't downloaded" in r.json()["detail"]["problems"]
    (models / "org--Llama-8B").mkdir()
    r = client.post("/api/models/llama/start").json()
    # Fixture MemAvailable is ~28.6 GiB: after 8 GiB reserve and 17.6 GiB weights, context shrinks.
    assert r["port"] == 8100 and r["plan"]["fits"]
    assert r["command"][:3] == ["vllm", "serve", str(models / "org--Llama-8B")]
    assert "llama" in s.sampler.engines
    assert s.gateway.synced[-1] == {"llama": 8100}
    assert client.put("/api/models/llama", json=model_body(publish=False)).status_code == 200
    assert s.gateway.synced[-1] == {}
    assert client.post("/api/models/llama/stop").json() == {"stopped": True}
    assert "llama" not in s.sampler.engines
    assert [a["action"] for a in client.get("/api/log").json()] == ["create", "start", "edit", "stop"]


def test_names_are_validated(env):
    client, _, _ = env
    assert client.post("/api/models", json=model_body(name="../evil")).status_code == 422
    assert client.post("/api/models/..%2Fevil/start").status_code in (404, 422)


def test_unknown_and_bad_names_are_4xx_not_500(env):
    client, _, _ = env
    assert client.get("/api/models/nope/plan").status_code == 404
    assert client.delete("/api/models/nope").status_code == 404
    assert client.get("/api/models/BAD NAME/versions").status_code == 422


def test_image_pull(env):
    client, s, _ = env
    s.images.ready = False
    r = client.post("/api/images/vllm/pull")
    assert r.status_code == 202 and r.json()["image"] == "dgx-kit/vllm:test"
    assert s.images.ready
    assert client.post("/api/images/nope/pull").status_code == 404


@pytest.fixture
def page_dir(tmp_path, monkeypatch):
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<title>DGX-kit</title>")
    (web / "assets" / "app.js").write_text("ok")
    monkeypatch.setenv("DGXKIT_WEB_DIR", str(web))


def test_page_is_served_without_hiding_api_404s(page_dir, env):
    client, _, _ = env
    assert "DGX-kit" in client.get("/").text
    assert "DGX-kit" in client.get("/models").text
    assert client.get("/assets/app.js").text == "ok"
    assert client.get("/api/nope").status_code == 404
    assert client.post("/nope").status_code == 404


def test_running_models_are_picked_up_after_a_restart(env):
    client, s, _ = env
    from dgxkit.api_models import reattach
    s.runner.running["kept"] = 8105
    assert client.portal.call(reattach, s) == ["kept"]
    assert s.sampler.engines["kept"].static == {"port": 8105}
    assert s.sampler.engines["kept"].base_url == "http://127.0.0.1:8105"


def test_templates_apply_save_and_delete(env):
    client, s, _ = env
    client.post("/api/models", json=model_body())
    names = [t["name"] for t in client.get("/api/templates").json()]
    assert names[:3] == ["balanced", "long-context", "many-users"]
    r = client.post("/api/models/llama/apply-template/many-users").json()
    assert r["recipe"]["max_context"] == 32768 and r["recipe"]["min_concurrency"] == 8.0
    assert r["recipe"]["repo"] == "org/Llama-8B"  # templates never touch the model itself
    assert len(client.get("/api/models/llama/versions").json()) == 1  # old settings kept
    assert client.post("/api/models/llama/save-template", json={"name": "mine"}).status_code == 201
    assert client.get("/api/templates").json()[-1]["settings"]["max_context"] == 32768
    assert client.post("/api/models/llama/save-template", json={"name": "balanced"}).status_code == 422
    assert client.delete("/api/templates/balanced").status_code == 422
    assert client.delete("/api/templates/mine").status_code == 200
    assert client.post("/api/models/llama/apply-template/mine").status_code == 404


def test_readonly_mode_blocks_everything_that_changes_the_box(tmp_path, monkeypatch):
    monkeypatch.setenv("DGXKIT_READONLY", "1")
    monkeypatch.setenv("DGXKIT_WEB_DIR", str(tmp_path / "no-page"))
    models = tmp_path / "models"
    models.mkdir()
    s = Services(root=str(ROOT), models_root=str(models), state_dir=str(tmp_path / "state"), hf_token=None,
                 sampler=Sampler(FakeGpu(), SystemCollector(str(ROOT))), store=ModelStore(str(tmp_path / "state")),
                 runner=FakeRunner(), images=FakeImages(), downloader=Downloader(str(models)),
                 gateway=FakeGateway())
    with TestClient(create_app(s)) as client:
        assert client.get("/api/me").json()["readonly"] is True
        assert client.post("/api/models", json=model_body()).status_code == 201  # settings only
        (models / "org--Llama-8B").mkdir()
        for method, path in [("post", "/api/models/llama/start"), ("post", "/api/models/llama/stop"),
                             ("post", "/api/models/llama/download"), ("delete", "/api/models/llama"),
                             ("post", "/api/images/vllm/pull"), ("post", "/api/gateway/sync"),
                             ("post", "/api/images/clean"), ("post", "/api/system/firmware/check")]:
            assert getattr(client, method)(path).status_code == 403, path
        assert s.runner.started == [] and s.gateway.synced == []
        assert client.get("/api/models").status_code == 200
        assert client.get("/api/system").status_code == 200


def test_library_lists_and_sets_up_models_on_disk(env, tmp_path):
    import json
    client, s, models = env
    d = models / "vllm" / "my-model"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({**CFG, "architectures": ["LlamaForCausalLM"]}))
    (d / "model.safetensors").write_bytes(b"x")
    lib = client.get("/api/library").json()
    assert [m["name"] for m in lib["models"]] == ["my-model"] and lib["drafts"] == []
    r = client.post("/api/library/setup", json={"path": str(d)})
    assert r.status_code == 201 and r.json()["recipe"]["path"] == str(d)
    row = client.get("/api/models").json()[0]
    assert row["downloaded"] is True
    assert client.get("/api/library").json()["models"][0]["used_by"] == "my-model"
    outside = client.post("/api/library/setup", json={"path": str(tmp_path)})
    assert outside.status_code == 422
    assert client.put("/api/library/paths", json={"paths": ["relative/path"]}).status_code == 422


def test_models_dgxkit_did_not_start_show_as_view_only(env):
    from dgxkit.api_models import refresh_external
    from tests.test_discover import C, Client
    client, s, _ = env
    s.runner.docker = Client([C("llm-laguna", ["vllm", "serve", "/m", "--port", "8095"])])
    client.portal.call(refresh_external, s)
    running = client.get("/api/running").json()
    assert [(r["name"], r["port"], r["managed"]) for r in running] == [("llm-laguna", 8095, False)]
    assert s.sampler.engines["llm-laguna"].base_url == "http://127.0.0.1:8095"
    s.runner.docker = Client([])
    client.portal.call(refresh_external, s)
    assert client.get("/api/running").json() == [] and "llm-laguna" not in s.sampler.engines


def test_slo_targets_and_layout_are_saved(env):
    client, s, _ = env
    assert client.get("/api/settings/slo").json()["ttft"] == 0.5
    assert client.put("/api/settings/slo", json={"ttft": 1.0}).json()["ttft"] == 1.0
    assert client.put("/api/settings/slo", json={"ttft": -1}).status_code == 422
    assert client.put("/api/settings/slo", json={"nope": 1}).status_code == 422
    assert client.get("/api/layout").json() == {"layout": None}
    client.put("/api/layout", json={"layout": {"hardware": [{"id": "gpu-util", "w": 1}]}})
    assert client.get("/api/layout").json()["layout"]["hardware"][0]["id"] == "gpu-util"
    client.put("/api/layout", json={"layout": None})
    assert client.get("/api/layout").json() == {"layout": None}


def test_plan_preview_for_unsaved_edits(env):
    client, _, _ = env
    client.post("/api/models", json=model_body())
    saved = client.get("/api/models/llama/plan").json()
    half = client.post("/api/models/llama/plan", json={"max_context": saved["context_tokens"] // 2, "name": "ignored"}).json()
    assert half["context_tokens"] == saved["context_tokens"] // 2
    assert half["concurrency"] > saved["concurrency"]
    more = client.post("/api/models/llama/plan?own_bytes=8589934592", json={}).json()
    assert more["kv_bytes"] > saved["kv_bytes"]
    assert client.get("/api/models/llama/plan").json() == saved
    assert client.post("/api/models/nope/plan", json={}).status_code == 404


def test_gateway_settings_keep_the_key_out_of_reads(env):
    client, s, _ = env
    r = client.put("/api/settings/gateway", json={"url": "http://spark.local:4000/v1/", "key": "sk-secret-1234"}).json()
    assert r["url"] == "http://spark.local:4000/v1" and r["key_set"] and r["key_hint"] == "…1234"
    assert "sk-secret" not in str(r)
    assert client.get("/api/settings/gateway/key").json() == {"key": "sk-secret-1234"}
    assert client.put("/api/settings/gateway", json={"url": "not a url"}).status_code == 422
    assert client.put("/api/settings/gateway", json={"clear_key": True}).json()["key_set"] is (s.gateway.master_key is not None)


def test_an_existing_litellm_is_never_doubled(env, monkeypatch):
    """With q's LiteLLM already on the box, publishing goes into it (or waits for its key), never a second container."""
    import asyncio
    from dgxkit import api_models
    import time
    client, s = env[0], env[1]
    for _ in range(200):  # let the startup publish finish first
        if s.gateway.synced:
            break
        time.sleep(0.01)
    s.gateway.status = lambda: {"external": "litellm-litellm-1", "port": 4000, "models": []}
    own = []
    s.gateway.sync = own.append  # starting DGX-kit's own LiteLLM
    asyncio.run(api_models.sync_gateway(s))
    assert own == [] and "key" in s.gateway.remote_problem
    sent = []
    monkeypatch.setattr("dgxkit.gateway.sync_remote", lambda url, key, pub, host, client=None: sent.append((url, key)))
    s.gateway.master_key = "sk-q"
    asyncio.run(api_models.sync_gateway(s))
    assert own == [] and sent == [("http://127.0.0.1:4000/v1", "sk-q")]


def test_saving_from_the_page_keeps_the_stored_config_and_reads_the_options_text(env):
    client, s, _ = env
    assert client.post("/api/models", json=model_body(config={"num_hidden_layers": 12})).status_code == 201
    row = next(m for m in client.get("/api/models").json() if m["name"] == "llama")
    assert "config" not in row  # the page never gets it
    body = {k: v for k, v in row.items() if k not in ("container", "live", "downloaded")}
    body["options"] = "--max-model-len 8192\n--max-num-seqs 2"
    assert client.put("/api/models/llama", json=body).status_code == 200
    saved = s.store.get("llama")
    assert saved.config == {"num_hidden_layers": 12}
    assert (saved.max_context, saved.extra_args) == (8192, ["--max-num-seqs 2"])


def test_a_model_is_only_down_after_it_has_answered_once():
    import asyncio
    from dgxkit.sampler import Sampler

    class Adapter:
        static, calls = {}, 0

        async def scrape(self):
            self.calls += 1
            if self.calls == 2:
                return {"decode_tps": 1}
            raise ConnectionError

    async def run():
        sm = Sampler(FakeGpu(), SystemCollector(str(ROOT)), interval=0.01)
        task = asyncio.create_task(sm._engine_loop("m", Adapter()))
        await asyncio.sleep(0.005)
        before = dict(sm.latest["models"]["m"])  # first scrape failed, nothing has answered yet
        await asyncio.sleep(0.05)
        task.cancel()
        return before, sm.latest["models"]["m"]

    before, after = asyncio.run(run())
    assert before["up"] is False and not before.get("was_up")
    assert after["up"] is False and after["was_up"] is True  # answered once, then went away


def _model_folder(root, rel, weights=b"x" * 1000):
    import json as _json
    d = root / rel
    d.mkdir(parents=True)
    (d / "config.json").write_text(_json.dumps({"architectures": ["Qwen3MoeForCausalLM"], "num_hidden_layers": 4}))
    (d / "model.safetensors").write_bytes(weights)
    return d


def test_removing_a_model_never_touches_its_files(env):
    client, s, models = env
    d = _model_folder(models, "org--Llama-8B")
    assert client.post("/api/models", json=model_body()).status_code == 201
    assert client.delete("/api/models/llama").status_code == 200
    assert d.exists()  # the weights stay; only the recipe is gone


def test_deleting_from_disk_needs_the_typed_name_and_stays_inside_the_model_folders(env, tmp_path):
    client, s, models = env
    client.put("/api/library/paths", json={"paths": [str(models)]})
    d = _model_folder(models, "vllm/keep-me")
    outside = _model_folder(tmp_path, "elsewhere/other")
    pre = client.get("/api/library/delete-preview", params={"path": str(d)}).json()
    assert pre["folder"] == "keep-me" and pre["size_bytes"] > 1000
    assert client.post("/api/library/delete", json={"path": str(d), "confirm": "nope"}).status_code == 422
    assert d.exists()
    assert client.get("/api/library/delete-preview", params={"path": str(outside)}).status_code == 404  # not in a model folder
    assert client.get("/api/library/delete-preview", params={"path": str(models)}).status_code == 404  # a folder root is no model
    r = client.post("/api/library/delete", json={"path": str(d), "confirm": "keep-me"})
    assert r.status_code == 200 and not d.exists() and outside.exists()


def test_a_running_model_blocks_deleting_its_weights(env):
    client, s, models = env
    client.put("/api/library/paths", json={"paths": [str(models)]})
    d = _model_folder(models, "org--Llama-8B")
    client.post("/api/models", json=model_body())
    s.runner.running = {"llama": 8100}
    r = client.get("/api/library/delete-preview", params={"path": str(d)})
    assert r.status_code == 409 and "llama" in r.json()["detail"]


def test_the_start_check_only_runs_for_a_start_we_made_and_skips_when_others_are_busy(env):
    import asyncio
    from dgxkit import quickcheck
    from dgxkit.api_models import quick_after_up
    client, s, _ = env
    asyncio.run(quick_after_up(s, "was-already-running"))  # not pending: nothing is probed or saved
    assert quickcheck.load(s.state_dir, "was-already-running") is None
    s.quick_pending.add("llama")
    s.sampler.latest["models"]["other"] = {"up": True, "running": 3}
    asyncio.run(quick_after_up(s, "llama"))
    assert "busy" in quickcheck.load(s.state_dir, "llama")["skipped"] and "llama" not in s.quick_pending


def test_hf_token_is_checked_saved_privately_and_never_returned(env, monkeypatch):
    import json
    import huggingface_hub
    client, s, _ = env
    assert client.get("/api/settings/hf").json() == {"set": False, "hint": None, "source": None}
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", lambda self, token=None: {"name": "q"})
    r = client.put("/api/settings/hf", json={"token": "hf_abcdefghijklmnop"}).json()
    assert r["set"] and r["account"] == "q" and r["hint"] == "…mnop" and "hf_abcdefghijklmnop" not in json.dumps(r)
    f = s.settings.file.parent / "hf.token"
    assert f.read_text() == "hf_abcdefghijklmnop" and oct(f.stat().st_mode & 0o777) == "0o600"
    assert s.downloader.token == "hf_abcdefghijklmnop"  # what downloads will use

    class Rejected(Exception):
        response = type("R", (), {"status_code": 401})()
    def reject(self, token=None):
        raise Rejected()
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", reject)
    assert client.put("/api/settings/hf", json={"token": "hf_bad"}).status_code == 422
    assert f.read_text() == "hf_abcdefghijklmnop"  # a rejected token doesn't replace the good one
    assert client.put("/api/settings/hf", json={"clear": True}).json()["set"] is False and not f.exists()


def test_a_model_stopped_from_the_dashboard_is_not_shown_as_crashed(env):
    client, s, _ = env
    client.post("/api/models", json=model_body())
    s.runner.running["llama"] = 8100
    assert client.post("/api/models/llama/stop").status_code == 200

    def left_behind(cid, code=1):  # Docker leaves the stopped container; vLLM exits 0 or 1 when stopped
        return lambda: {"llama": {"state": "exited", "port": 8100, "id": cid, "exit_code": code, "engine": "vllm", "meta": {}}}

    row = lambda: next(m for m in client.get("/api/models").json() if m["name"] == "llama")["container"]
    s.runner.status = left_behind("id-llama")
    assert row()["stopped_cleanly"] is True  # the container the dashboard stopped
    s.runner.status = left_behind("id-another-container")
    assert "stopped_cleanly" not in row()  # a different one that exited, with nothing saying it shut down properly, is a crash
    s.runner.logs = lambda name, tail=200: "INFO: Waiting for application shutdown.\nINFO: Application shutdown complete.\n"
    s.runner.status = left_behind("id-stopped-elsewhere")  # stopped from a shell: the engine's own log says it went down properly
    assert row()["stopped_cleanly"] is True


def test_the_running_version_is_reported_before_sign_in(env):
    client, s, _ = env
    v = client.get("/api/me").json()["version"]
    assert isinstance(v, str) and v  # the package's own version, or "dev" when it isn't installed


def test_publish_and_unpublish_update_the_gateway_without_touching_the_model(env):
    client, s, _ = env
    client.post("/api/models", json=model_body())
    s.runner.running["llama"] = 8100
    assert client.post("/api/models/llama/publish", json={"publish": False}).json() == {"publish": False}
    assert s.store.get("llama").publish is False and s.gateway.synced[-1] == {}  # off the gateway now
    assert s.runner.running == {"llama": 8100}  # still running
    assert client.post("/api/models/llama/publish", json={"publish": True}).json() == {"publish": True}
    assert s.gateway.synced[-1] == {"llama": 8100}  # and back on it
    assert client.post("/api/models/nope/publish", json={"publish": True}).status_code == 404
    assert client.post("/api/models/llama/publish", json={}).status_code == 422


def test_system_tab_tracks_versions_and_firmware_updates(env):
    client, s, _ = env
    first = client.get("/api/system").json()
    labels = {i["label"]: i["value"] for i in first["items"]}
    assert labels["NVIDIA driver"] and labels["DGX-kit"] and first["changes"] == []  # the first reading is only a baseline
    devices = [{"id": "ec", "name": "Embedded Controller", "version": "0x02000006", "vendor": "Asus", "summary": None,
                "plugin": "uefi_capsule", "updatable": True, "updates": [{"version": "0x03000001", "summary": "new EC", "remote": "lvfs"}]},
               {"id": "kek", "name": "KEK CA", "version": "2023", "vendor": None, "summary": None, "plugin": "uefi_kek",
                "updatable": False, "updates": []}]
    s.firmware_probe = lambda: devices
    r = client.post("/api/system/firmware/check").json()
    assert r["firmware"]["updates"] == 1 and r["firmware"]["checked"] and r["firmware"]["error"] is None
    devices[0]["version"] = "0x03000001"; devices[0]["updates"] = []  # it was flashed
    r = client.post("/api/system/firmware/check").json()
    assert r["firmware"]["updates"] == 0
    assert [(c["label"], c["from"], c["to"]) for c in r["changes"] if c["key"] == "fw:ec"] == [("Embedded Controller", "0x02000006", "0x03000001")]
    s.firmware_probe = lambda: (_ for _ in ()).throw(RuntimeError("fwupd isn't reachable"))
    r = client.post("/api/system/firmware/check").json()
    assert "fwupd" in r["firmware"]["error"] and r["firmware"]["devices"]  # the last good result is kept

