import json

import pytest

from dgxkit.control import LABEL, DockerRunner, check_start, engine_command, pick_port
from dgxkit.recipes import Recipe
from dgxkit.sizing import Plan

P = Plan(context_tokens=65536, kv_pool_tokens=200000, kv_bytes=10 * 2**30, concurrency=3.05, fits=True)


def test_vllm_command_with_draft():
    r = Recipe(name="m", repo="org/M", draft_repo="org/M-eagle3", draft_method="eagle3", kv_cache_dtype="fp8")
    cmd = engine_command(r, P, 8100, "/models")
    assert cmd[:3] == ["vllm", "serve", "/models/org--M"]
    assert cmd[cmd.index("--max-model-len") + 1] == "65536"
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec == {"method": "eagle3", "model": "/models/org--M-eagle3", "num_speculative_tokens": 3}


def test_llamacpp_needs_a_gguf_choice_and_scales_context_by_slots():
    r = Recipe(name="g", repo="org/G", engine="llamacpp")
    with pytest.raises(ValueError):
        engine_command(r, P, 8100, "/models")
    r.gguf_file = "g-Q8_0.gguf"
    cmd = engine_command(r, P, 8100, "/models")
    assert cmd[cmd.index("-c") + 1] == str(65536 * 3)
    assert cmd[cmd.index("--parallel") + 1] == "3"


def test_pick_port_skips_taken_and_bound():
    assert pick_port({8100}, is_free=lambda p: p != 8101) == 8102


def test_check_start_lists_every_problem(tmp_path):
    r = Recipe(name="m", repo="org/M", draft_repo="org/D")
    bad = Plan(0, 0, 0, 0.0, False, "weights don't fit in free memory")
    c = check_start(r, bad, str(tmp_path), image_ready=False)
    assert not c.ok and len(c.problems) == 4
    (tmp_path / "org--M").mkdir()
    (tmp_path / "org--D").mkdir()
    assert check_start(r, P, str(tmp_path), image_ready=True).ok


class FakeContainer:
    def __init__(self, labels, status="running", exit_code=None, started="2026-10-03T20:06:32.648744384Z"):
        self.labels, self.status, self.id = labels, status, "abc123"
        self.attrs = {"State": {"ExitCode": exit_code, "StartedAt": started}}
        self.stopped = self.removed = False

    def stop(self, timeout=None):
        self.stopped = True

    def remove(self, force=False):
        self.removed = True


class FakeDocker:
    def __init__(self, containers):
        self.all = containers
        self.containers = self

    def run(self, image, command, **kw):
        self.ran = (image, command, kw)

    def list(self, all=False, filters=None):
        want = filters["label"]
        key, _, val = want.partition("=")
        return [c for c in self.all if key in c.labels and (not val or c.labels[key] == val)]


def test_runner_only_sees_its_own_containers():
    mine = FakeContainer({LABEL: "m", "dgxkit.port": "8100"})
    someone_elses = FakeContainer({"com.example": "vllm"})
    runner = DockerRunner(FakeDocker([mine, someone_elses]))
    assert runner.status() == {"m": {"state": "running", "port": 8100, "id": "abc123", "exit_code": None, "engine": None, "started": 1791057992.648744, "meta": {}}}
    assert runner.ports_in_use() == {8100}


def test_stop_clears_a_crashed_container_and_stops_a_running_one():
    crashed = FakeContainer({LABEL: "a"}, status="exited", exit_code=1)
    live = FakeContainer({LABEL: "b"})
    runner = DockerRunner(FakeDocker([crashed, live]))
    assert runner.status()["a"]["exit_code"] == 1
    runner.stop("a")
    runner.stop("b")
    assert crashed.removed and not crashed.stopped
    assert live.stopped and not live.removed


def test_start_replaces_the_image_entrypoint_with_the_command():
    d = FakeDocker([])
    r = Recipe(name="m", repo="org/M", image="vllm/vllm-openai:v1")
    DockerRunner(d).start(r, ["vllm", "serve", "/models/org--M"], 8100, "/models")
    image, command, kw = d.ran
    assert kw["entrypoint"] == ["vllm"] and command == ["serve", "/models/org--M"]
    assert kw["labels"][LABEL] == "m"


def test_engines_that_compile_get_persistent_caches_and_a_capped_compile(tmp_path, monkeypatch):
    from dgxkit.control import docker_options
    from dgxkit.recipes import Recipe
    monkeypatch.setenv("DGXKIT_CACHE_DIR", str(tmp_path))
    r = Recipe(name="n", repo="x", engine="vllm", env={"VLLM_MARLIN_USE_ATOMIC_ADD": "1", "MAX_JOBS": "4"})
    o = docker_options(r, "/m")
    assert o["volumes"][str(tmp_path / "flashinfer")] == {"bind": "/root/.cache/flashinfer", "mode": "rw"}
    assert o["volumes"][str(tmp_path / "vllm-jit")]["bind"] == "/root/.cache/vllm"
    assert o["environment"] == {"MAX_JOBS": "4", "NVCC_THREADS": "1", "VLLM_MARLIN_USE_ATOMIC_ADD": "1"}  # the recipe wins
    assert docker_options(Recipe(name="l", repo="x", engine="llamacpp"), "/m")["environment"] is None


def test_status_says_when_a_container_started():
    from dgxkit.control import started_at
    c = FakeContainer({LABEL: "a", "dgxkit.port": "8100"})
    assert DockerRunner(FakeDocker([c])).status()["a"]["started"] == 1791057992.648744  # 2026-10-03 20:06:32.65 UTC
    assert started_at(FakeContainer({}, started="0001-01-01T00:00:00Z")) is None  # never started
    assert started_at(FakeContainer({}, started="")) is None
