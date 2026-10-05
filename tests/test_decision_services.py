import json

import pytest

from dgxkit import laya as laya_mod
from dgxkit.bekko import BekkoService
from dgxkit.laya import GIB, Problems, hub_snapshot
from dgxkit.lev import LevService

from tests.test_laya import FakeDocker, FakeImages


def hub(cache, repo, files, rev="abc123"):
    snap = cache / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    for f, content in files.items():
        (snap / f).parent.mkdir(parents=True, exist_ok=True)
        (snap / f).write_text(content)
    return snap


def lev_cache(models, with_base=True):
    cache = models / ".lev"
    hub(cache, "interfaze-ai/lev", {"lev_release.json": json.dumps({"base_model": "Qwen/Qwen3.5-4B"}), "adapter_model.safetensors": "x"})
    if with_base:
        hub(cache, "Qwen/Qwen3.5-4B", {"config.json": "{}", "model-00001.safetensors": "x"})
    return cache


def bekko_cache(models):
    cache = models / ".bekko"
    hub(cache, "hotchpotch/bekko-system-one-v0-400m", {"inference_v0.py": "x", "0_BekkoInference/model.safetensors": "x"})
    return cache


def make(cls, tmp_path, monkeypatch, free=64 * GIB, **kw):
    monkeypatch.setattr(laya_mod, "port_free", lambda p: True)
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    docker = FakeDocker()
    return cls(str(tmp_path / "state"), str(models), FakeImages(), docker=docker, health=lambda p: None,
               meminfo=lambda: {"MemFree": free, "MemAvailable": free}, **kw), docker, models


def test_lev_needs_both_the_adapter_and_its_backbone(tmp_path, monkeypatch):
    s, _, models = make(LevService, tmp_path, monkeypatch)
    assert s.checkpoints() == (None, []) and "isn't on this box" in s.problems()[0]
    lev_cache(models, with_base=False)
    assert s.checkpoints()[1] == [] and "Qwen/Qwen3.5-4B" in s.missing()  # the backbone it names is the one asked for
    lev_cache(models)
    assert s.checkpoints() == (str(models / ".lev"), ["lev"]) and s.selected() == ["lev"]
    assert s.status()["choices"] == [{"name": "lev", "about": "adapter on Qwen3.5-4B, about 11 GiB", "found": True}]


def test_lev_starts_on_this_machine_only_and_refuses_the_cpu_fallback(tmp_path, monkeypatch):
    s, docker, models = make(LevService, tmp_path, monkeypatch)
    lev_cache(models)
    s.start()
    image, kw = docker.runs[0]
    assert image == "dgx-kit/decision:gb10" and kw["name"] == "dgxkit-lev"
    assert kw["entrypoint"] == ["python3"] and kw["command"] == ["/opt/lev_local.py"]
    assert kw["environment"] == {"LEV_DEVICE": "cuda", "LEV_PORT": "8201", "LEV_HOST": "127.0.0.1"}  # no key, so not on the network
    assert kw["volumes"] == {str(models / ".lev"): {"bind": "/models/lev", "mode": "rw"}} and kw["device_requests"]
    st = s.status()
    assert st["has_key"] is False and st["can_expose"] is True and st["expose"] is False
    s.set_config(expose=True)
    assert s.env(s.config())["LEV_HOST"] == "0.0.0.0"
    with pytest.raises(ValueError):
        make(BekkoService, tmp_path, monkeypatch)[0].set_config(expose=True)  # only Lev has the setting


def test_a_start_that_wants_more_free_memory_than_there_is_is_held_back(tmp_path, monkeypatch):
    s, docker, models = make(LevService, tmp_path, monkeypatch, free=7 * GIB)
    lev_cache(models)
    with pytest.raises(Problems) as e:
        s.start()
    assert "needs about 11 GiB free" in e.value.problems[0] and "drop_caches" in e.value.problems[0] and not docker.runs
    s._meminfo = lambda: {"MemFree": 12 * GIB, "MemAvailable": 40 * GIB}
    s.start()
    assert len(docker.runs) == 1
    # on the CPU what could be made free counts, not what is empty right now
    s.stop()
    s.set_config(device="cpu")
    s._meminfo = lambda: {"MemFree": 1 * GIB, "MemAvailable": 40 * GIB}
    assert s.problems() == []


def test_bekko_has_a_key_a_port_and_its_own_server_command(tmp_path, monkeypatch):
    s, docker, models = make(BekkoService, tmp_path, monkeypatch)
    assert s.status()["state"] == "stopped" and "isn't on this box" in s.status()["problems"][0]
    bekko_cache(models)
    s.start()
    image, kw = docker.runs[0]
    assert image == "dgx-kit/decision:gb10" and kw["name"] == "dgxkit-bekko" and kw["command"] == ["/opt/bekko_serve.py"]
    assert kw["environment"]["BEKKO_API_KEY"] == s.key() and kw["environment"]["BEKKO_PORT"] == "8202"
    assert kw["volumes"] == {str(models / ".bekko"): {"bind": "/models/bekko", "mode": "rw"}}
    assert s.status()["has_key"] is True and s.auth_headers() == {"Authorization": f"Bearer {s.key()}"}


def test_the_three_services_do_not_share_state_files_or_containers(tmp_path, monkeypatch):
    lev, _, _ = make(LevService, tmp_path, monkeypatch)
    bekko, _, _ = make(BekkoService, tmp_path, monkeypatch)
    assert lev.file != bekko.file and lev.container_name != bekko.container_name
    lev.set_config(port=8300)
    assert bekko.config()["port"] == 8202 and lev.config()["port"] == 8300


def test_hub_snapshot_is_the_newest_revision_or_none(tmp_path):
    assert hub_snapshot(tmp_path, "a/b") is None
    hub(tmp_path, "a/b", {"f": "1"}, rev="aaa")
    hub(tmp_path, "a/b", {"f": "2"}, rev="bbb")
    assert hub_snapshot(tmp_path, "a/b").name == "bbb"
