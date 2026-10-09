"""GPU time slots: the freezer picks a backend, the scheduler rotates busy models and never leaves one frozen."""
import asyncio
from types import SimpleNamespace

from dgxkit.slots import Freezer, SlotScheduler, has_work


class FakeContainer:
    def __init__(self, log, cid):
        self.log, self.id = log, cid

    def pause(self):
        self.log.append((self.id, True))

    def unpause(self):
        self.log.append((self.id, False))


class FakeRunner:
    def __init__(self, running):
        self.running = running  # name -> container id
        self.paused = []
        self.docker = SimpleNamespace(containers=SimpleNamespace(get=lambda cid: FakeContainer(self.paused, cid)))

    def status(self):
        return {n: {"state": "running", "id": cid, "engine": "vllm"} for n, cid in self.running.items()}


def test_freezer_writes_cgroup_file_when_it_can_else_asks_docker(tmp_path):
    runner = FakeRunner({"a": "aaa"})
    cg = tmp_path / "sys/fs/cgroup/system.slice/docker-aaa.scope"
    cg.mkdir(parents=True)
    (cg / "cgroup.freeze").write_text("0")
    fz = Freezer(runner, root=str(tmp_path))
    fz.set("aaa", True)
    assert (cg / "cgroup.freeze").read_text() == "1" and fz.backend == "cgroup" and runner.paused == []
    fz.set("aaa", True)  # already frozen: nothing happens
    fz.set("bbb", True)  # no cgroup file for this one: Docker pauses it
    assert runner.paused == [("bbb", True)] and fz.backend == "docker pause"
    fz.thaw_all()
    assert (cg / "cgroup.freeze").read_text() == "0" and runner.paused[-1] == ("bbb", False) and fz.frozen == set()


def test_has_work_needs_an_answering_engine_with_requests():
    assert not has_work(None) and not has_work({"up": False, "running": 3}) and not has_work({"up": True, "running": 0, "waiting": 0})
    assert has_work({"up": True, "running": 0, "waiting": 1})


def test_scheduler_rotates_busy_models_and_thaws_on_stop():
    asyncio.run(_rotation())


async def _rotation():
    runner = FakeRunner({"a": "aaa", "b": "bbb", "c": "ccc"})
    sampler = SimpleNamespace(latest={"models": {}}, slots={})
    s = SimpleNamespace(runner=runner, sampler=sampler, root="/nonexistent")
    live = sampler.latest["models"]
    live["a"] = {"up": True, "running": 2}
    live["b"] = {"up": True, "running": 0, "waiting": 0}
    live["c"] = {"up": True, "running": 1}
    sch = SlotScheduler(s, quantum=0.1, tick=0.01)
    sch.start()
    await asyncio.sleep(0.03)
    assert sch.owner == "a" and sampler.slots == {"a": "running", "c": "waiting"}  # b idle: left alone
    assert runner.paused == [("ccc", True)]
    await asyncio.sleep(0.1)  # quantum over: c's turn, a frozen (frozen before c is thawed)
    assert sch.owner == "c" and runner.paused[-2:] == [("aaa", True), ("ccc", False)]
    assert sampler.slots == {"a": "waiting", "c": "running"}
    live["c"] = {"up": True, "running": 0, "waiting": 0}  # c ran out of work: back to a at once, c thawed and idle
    await asyncio.sleep(0.03)
    assert sch.owner == "a" and sampler.slots == {"a": "running"} and runner.paused[-1] == ("aaa", False)
    await sch.stop()
    assert not sch.running and sch.owner is None and sampler.slots == {} and sch.freezer.frozen == set()


def test_settings_switch_starts_and_stops_the_scheduler(env):
    client, s, _ = env
    assert client.get("/api/settings/slots").json() == {"enabled": False, "quantum": 2.0, "active": False, "owner": None, "backend": None}
    r = client.put("/api/settings/slots", json={"enabled": True, "quantum": 3}).json()
    assert r["enabled"] and r["quantum"] == 3.0 and r["active"] and s.slots.running
    assert client.put("/api/settings/slots", json={"quantum": 0.1}).status_code == 422
    r = client.put("/api/settings/slots", json={"enabled": False}).json()
    assert not r["enabled"] and not r["active"] and r["quantum"] == 3.0 and not s.slots.running
    assert s.settings.slots == {"enabled": False, "quantum": 3.0}
    assert [a["detail"] for a in client.get("/api/log").json() if a["action"] == "settings"] == ["GPU slots on, 3 s each", "GPU slots off, 3 s each"]
