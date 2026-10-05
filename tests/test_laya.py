import time

import pytest

from dgxkit import laya as laya_mod
from dgxkit.laya import BUILD, LayaService, Problems, find_checkpoints


class FakeContainer:
    def __init__(self, status="running", started="2026-10-05T10:00:00.000000000Z"):
        self.status = status
        self.attrs = {"State": {"StartedAt": started}}
        self.removed = self.stopped = False

    def stop(self, timeout=None):
        self.stopped = True

    def remove(self, force=False):
        self.removed = True

    def logs(self, tail=200):
        return b"started"


class FakeDocker:
    def __init__(self, existing=None, label=None):
        from dgxkit.images import BUILDS
        self.existing, self.runs = existing, []
        self.label = label or BUILDS[BUILD]["args"]["REV"]  # what the image on the box says it was built from
        outer = self

        class Images:
            def get(self, tag):
                return type("Image", (), {"labels": {"dgxkit.build": outer.label}})()

        self.images = Images()

        class Containers:
            def list(self, all=False, filters=None):
                return [outer.existing] if outer.existing is not None else []

            def run(self, image, **kw):
                outer.runs.append((image, kw))
                outer.existing = FakeContainer()
                return outer.existing

        self.containers = Containers()


class FakeJob:
    def __init__(self, state="running", error=None):
        self.state, self.error = state, error

    def view(self):
        return {"state": self.state, "error": self.error}


class FakeImages:
    def __init__(self, ready=True):
        self.ready, self.jobs, self.built = ready, {}, 0

    def is_ready(self, image):
        return self.ready

    def start_build(self, build):
        assert build == BUILD
        self.built += 1
        self.jobs[build] = FakeJob()
        return self.jobs[build]


def bundle(root, names=("", "multilingual", "typed-decisions")):
    for sub in names:
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "rl_agent_config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"x")
        (d / "encoder").mkdir(exist_ok=True)
    return root


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setattr(laya_mod, "port_free", lambda p: True)
    models = tmp_path / "models"
    bundle(models / "vllm" / "laya")
    docker, images = FakeDocker(), FakeImages()
    s = LayaService(str(tmp_path / "state"), str(models), images, docker=docker, health=lambda port: None)
    return s, docker, images, models


def test_the_bundle_with_most_checkpoints_is_found_and_inner_folders_are_not_bundles(tmp_path):
    bundle(tmp_path / "a" / "laya-english-only", names=("",))
    bundle(tmp_path / "b" / "laya")
    folder, have = find_checkpoints(str(tmp_path))
    assert folder == str(tmp_path / "b" / "laya") and have == ["english", "multilingual", "typed-decisions"]
    assert find_checkpoints(str(tmp_path / "nothing")) == (None, [])


def test_status_follows_the_container_and_what_the_server_says(svc):
    s, docker, _, _ = svc
    assert s.status()["state"] == "stopped"
    docker.existing = FakeContainer("running")
    assert s.status()["state"] == "starting"  # up, but /health doesn't answer yet
    s._health = lambda port: {"device": "cuda", "loaded": ["english", "typed-decisions"]}
    st = s.status()
    assert st["state"] == "running" and st["device_in_use"] == "cuda" and st["loaded"] == ["english", "typed-decisions"]
    assert st["started"] == pytest.approx(1791194400.0)  # 2026-10-05 10:00 UTC
    docker.existing = FakeContainer("exited")
    assert s.status()["state"] == "exited"


def test_start_runs_the_server_on_the_checkpoints_with_a_key_and_the_gpu(svc):
    s, docker, _, models = svc
    assert s.start() == {"started": True}
    image, kw = docker.runs[0]
    assert image == "dgx-kit/laya:gb10" and kw["name"] == "dgxkit-laya" and kw["network_mode"] == "host"
    assert kw["volumes"] == {str(models / "vllm" / "laya"): {"bind": "/models/laya", "mode": "ro"}}
    assert kw["environment"]["LAYA_DEVICE"] == "cuda" and kw["environment"]["LAYA_PORT"] == "8200"
    assert kw["environment"]["LAYA_API_KEY"] == s.key() and len(s.key()) >= 24
    assert kw["device_requests"] and kw["ulimits"][0]["Name"] == "memlock"
    assert s.start() == {"started": True} and len(docker.runs) == 1  # already running: nothing to do


def test_cpu_mode_asks_for_no_gpu(svc):
    s, docker, _, _ = svc
    s.set_config(device="cpu", port=8222)
    s.start()
    kw = docker.runs[0][1]
    assert "device_requests" not in kw and kw["environment"]["LAYA_DEVICE"] == "cpu" and kw["environment"]["LAYA_PORT"] == "8222"


def test_start_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(laya_mod, "port_free", lambda p: False)
    (tmp_path / "models").mkdir()
    s = LayaService(str(tmp_path / "state"), str(tmp_path / "models"), FakeImages(), docker=FakeDocker(), health=lambda p: None)
    with pytest.raises(Problems) as e:
        s.start()
    assert any("No Laya checkpoint" in p for p in e.value.problems) and any("Port 8200" in p for p in e.value.problems)


def test_a_missing_image_is_built_first_and_the_server_starts_when_it_is_done(svc):
    s, docker, images, _ = svc
    images.ready = False
    out = s.start()
    assert "preparing" in out and images.built == 1 and not docker.runs
    assert s.status()["state"] == "preparing"
    images.ready = True
    images.jobs[BUILD].state = "done"
    s._waiter.join(5)
    assert len(docker.runs) == 1 and s.error is None


def test_a_failed_build_is_reported_not_swallowed(svc):
    s, docker, images, _ = svc
    images.ready = False
    s.start()
    images.jobs[BUILD].state, images.jobs[BUILD].error = "failed", "no space left"
    s._waiter.join(5)
    assert "no space left" in s.status()["error"] and not docker.runs


def test_stop_removes_the_container_and_restart_replaces_a_crashed_one(svc):
    s, docker, _, _ = svc
    s.start()
    running = docker.existing
    assert s.stop() and running.stopped and running.removed
    docker.existing = crashed = FakeContainer("exited")
    s.start()
    assert crashed.removed and len(docker.runs) == 2
    docker.existing = None
    assert s.stop() is False


def test_settings_are_checked_and_the_key_is_private(svc):
    s, _, _, _ = svc
    with pytest.raises(ValueError):
        s.set_config(device="tpu")
    with pytest.raises(ValueError):
        s.set_config(port=80)
    assert s.set_config(device="cpu", port=8300) == {"device": "cpu", "port": 8300, "dir": None, "checkpoints": None}
    k = s.key()
    assert oct(s.key_file.stat().st_mode)[-3:] == "600" and s.key() == k


def test_the_sample_request_reports_time_and_the_answer(svc, monkeypatch):
    s, _, _, _ = svc
    import io
    import json

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    body = {"routing": {"model": "english"}, "answers": {"jailbreak": {"noul": 0.98}, "topic": {"choice": "security_testing"}}}
    seen = {}

    def fake_open(req, timeout=None):
        seen["url"], seen["auth"] = req.full_url, req.headers.get("Authorization")
        return Resp(json.dumps(body).encode())

    monkeypatch.setattr(laya_mod.urllib.request, "urlopen", fake_open)
    r = s.test()
    assert r["jailbreak"] == 0.98 and r["topic"] == "security_testing" and r["model"] == "english" and r["ms"] >= 0
    assert seen["url"] == "http://127.0.0.1:8200/v1/systemone" and seen["auth"] == f"Bearer {s.key()}"


def test_which_checkpoints_run_is_chosen_from_the_ones_found(svc):
    s, docker, _, models = svc
    assert s.selected() == ["english", "multilingual", "typed-decisions"]  # nothing picked: every one the folder has
    assert s.status()["selected"] == s.selected()
    with pytest.raises(ValueError):
        s.set_config(checkpoints=[])
    with pytest.raises(ValueError):
        s.set_config(checkpoints=["french"])
    s.set_config(checkpoints=["typed-decisions", "english"])
    assert s.selected() == ["english", "typed-decisions"]  # in the order Laya lists them
    s.start()
    env = docker.runs[0][1]["environment"]
    assert env["LAYA_MODELS"] == "english,typed-decisions" and env["LAYA_MAX_LOADED"] == "2"


def test_a_pick_the_folder_does_not_have_is_a_problem_not_an_empty_server(tmp_path, monkeypatch):
    monkeypatch.setattr(laya_mod, "port_free", lambda p: True)
    bundle(tmp_path / "models" / "laya", names=("",))  # English only
    s = LayaService(str(tmp_path / "state"), str(tmp_path / "models"), FakeImages(), docker=FakeDocker(), health=lambda p: None)
    s.set_config(checkpoints=["multilingual"])
    assert s.selected() == []
    with pytest.raises(Problems) as e:
        s.start()
    assert any("None of the checkpoints you picked" in p for p in e.value.problems)


def test_an_image_built_from_older_files_is_rebuilt_before_it_runs(svc):
    s, docker, images, _ = svc
    docker.label = "old"
    assert s.status()["image_ready"] is False
    out = s.start()
    assert "preparing" in out and images.built == 1 and not docker.runs
    docker.label = s.docker.label  # (the build would replace it)
    from dgxkit.images import BUILDS
    docker.label = BUILDS[BUILD]["args"]["REV"]
    images.jobs[BUILD].state = "done"
    s._waiter.join(5)
    assert len(docker.runs) == 1
