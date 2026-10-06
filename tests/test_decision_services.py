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
    models.mkdir(parents=True, exist_ok=True)
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
    assert s.status()["state"] == "missing" and "isn't on this box" in s.status()["problems"][0]
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


# ---- the Download button: fetch what is missing, then build the image, without starting anything

import asyncio
from types import SimpleNamespace


class FakeDownloader:
    """Puts the files of each repo in place, as a finished download would."""
    token = None

    def __init__(self, write, fail=None, cancel=False):
        self.jobs, self.started, self.write, self.fail, self.cancel = {}, [], write, fail, cancel

    async def start(self, repo, total, ignore_patterns=None, cache_dir=None, local_dir=None, **kw):
        self.started.append({"repo": repo, "total": total, "ignore": ignore_patterns, "cache_dir": cache_dir, "local_dir": local_dir})
        job = SimpleNamespace(repo=repo, state="done", error=None, progress=lambda: {"repo": repo, "state": "running", "bytes": 1, "total_bytes": total})
        if self.fail:
            job.state, job.error = "failed", self.fail
        elif self.cancel:
            job.state = "cancelled"
        else:
            self.write(repo)
        self.jobs[repo] = job
        return job


class BuildingImages(FakeImages):
    """An image that gets built when asked: the job is done at once and the image is there afterwards."""

    def __init__(self):
        super().__init__(ready=False)

    def start_build(self, build):
        self.built += 1
        self.jobs[build] = job = laya_job("done")
        self.ready = True
        return job


def laya_job(state):
    from tests.test_laya import FakeJob
    return FakeJob(state)


def lev_writer(models):
    def write(repo):
        if repo == "interfaze-ai/lev":
            lev_cache(models, with_base=False)
        else:
            hub(models / ".lev", "Qwen/Qwen3.5-4B", {"config.json": "{}", "model-00001.safetensors": "x"})
    return write


def test_download_fetches_lev_then_its_backbone_then_builds_the_image(tmp_path, monkeypatch):
    s, docker, models = make(LevService, tmp_path, monkeypatch)
    s.images, s.downloader = BuildingImages(), FakeDownloader(lev_writer(models))
    s._sizer = lambda repo, ignore: 123
    assert s.status()["state"] == "missing"
    asyncio.run(s._setup())
    assert [x["repo"] for x in s.downloader.started] == ["interfaze-ai/lev", "Qwen/Qwen3.5-4B"]  # the backbone is the one the adapter names
    assert all(x["cache_dir"] == str(models / ".lev") and x["total"] == 123 for x in s.downloader.started)
    assert s.setup["state"] == "done" and s.images.built == 1 and not docker.runs  # nothing was started
    st = s.status()
    assert st["state"] == "stopped" and st["image_ready"] and st["problems"] == [] and s.ready()


def test_bekko_skips_the_browser_model_and_laya_goes_to_its_own_folder(tmp_path, monkeypatch):
    b, _, models = make(BekkoService, tmp_path, monkeypatch)
    b.images, b.downloader = BuildingImages(), FakeDownloader(lambda repo: bekko_cache(models))
    b._sizer = lambda repo, ignore: 1
    asyncio.run(b._setup())
    assert b.downloader.started[0]["ignore"] == ["onnx_browser/*"] and b.setup["state"] == "done"
    from dgxkit.laya import LayaService
    ly, _, models2 = make(LayaService, tmp_path / "x", monkeypatch)
    assert ly.next_step() == {"repo": "convaiinnovations/laya", "local_dir": str(models2 / "laya")}


def test_a_failed_or_cancelled_download_is_reported_and_can_be_tried_again(tmp_path, monkeypatch):
    s, _, models = make(BekkoService, tmp_path, monkeypatch)
    s.images, s.downloader = BuildingImages(), FakeDownloader(lambda r: None, fail="no space left")
    s._sizer = lambda repo, ignore: 1
    asyncio.run(s._setup())
    st = s.status()
    assert st["state"] == "failed" and "no space left" in st["error"] and s.images.built == 0
    s.downloader = FakeDownloader(lambda r: None, cancel=True)
    asyncio.run(s._setup())
    assert s.setup["state"] == "cancelled" and s.status()["state"] == "missing"  # a cancel isn't an error


def test_download_refuses_what_is_already_done_or_running(tmp_path, monkeypatch):
    s, _, models = make(BekkoService, tmp_path, monkeypatch)
    bekko_cache(models)
    s.downloader = FakeDownloader(lambda r: None)
    with pytest.raises(Problems) as e:
        asyncio.run(_start(s))
    assert "already downloaded" in e.value.problems[0]
    s2, _, _ = make(BekkoService, tmp_path / "other", monkeypatch)
    with pytest.raises(Problems) as e:
        s2.start_download()  # no downloader here
    assert "aren't available" in e.value.problems[0]
    s3, _, _ = make(BekkoService, tmp_path / "third", monkeypatch)
    s3.downloader, s3.setup = FakeDownloader(lambda r: None), {"state": "running", "step": 1, "repo": "x", "error": None}
    with pytest.raises(Problems) as e:
        asyncio.run(_start(s3))
    assert e.value.problems == ["Already downloading."]


async def _start(s):
    return s.start_download()


def test_a_laya_folder_still_downloading_is_not_taken_for_a_checkpoint(tmp_path):
    from dgxkit.downloader import MARKER
    from dgxkit.laya import find_checkpoints
    from tests.test_laya import bundle
    folder = bundle(tmp_path / "laya")
    assert find_checkpoints(str(tmp_path))[0] == str(folder)
    (folder / MARKER).touch()  # the downloader's mark that it isn't finished
    assert find_checkpoints(str(tmp_path)) == (None, [])
    (folder / MARKER).unlink()
    (folder / "model.safetensors").unlink()  # a config without its weights isn't one either
    assert "english" not in find_checkpoints(str(tmp_path))[1]


def test_the_downloader_can_fill_a_hugging_face_cache_and_skip_files(tmp_path, monkeypatch):
    from dgxkit.downloader import MARKER, Downloader

    async def go():
        d = Downloader(str(tmp_path / "models"))
        ran = []

        async def fake_run(job):
            ran.append((job.cache_dir, job.ignore_patterns))
            job.state = "done"
        d._run = fake_run
        job = await d.start("org/name", 10, ignore_patterns=["onnx_browser/*"], cache_dir=str(tmp_path / "models" / ".x"))
        await asyncio.sleep(0)
        return d, job, ran

    (tmp_path / "models").mkdir()
    d, job, ran = asyncio.run(go())
    assert job.dest == tmp_path / "models" / ".x" / "hub" / "models--org--name" and (job.dest / MARKER).exists()
    assert ran == [(str(tmp_path / "models" / ".x"), ["onnx_browser/*"])]
