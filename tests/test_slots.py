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
    def __init__(self, running, stale=()):
        self.running = running  # name -> container id
        self.paused = []
        self.stale = set(stale)  # names Docker reports as paused when the scheduler starts
        self.docker = SimpleNamespace(containers=SimpleNamespace(get=lambda cid: FakeContainer(self.paused, cid)))

    def status(self):
        return {n: {"state": "running", "id": cid, "engine": "vllm"} for n, cid in self.running.items()}

    def _ours(self):
        return [SimpleNamespace(id=cid, status="paused" if n in self.stale else "running", labels={"dgxkit.model": n},
                                unpause=lambda cid=cid: self.paused.append((cid, False)))
                for n, cid in self.running.items()]


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
    sch = SlotScheduler(s, quantum=0.1, min_requests=1, tick=0.01)
    sch.start()
    await asyncio.sleep(0.03)
    assert sch.owner == "a" and sampler.slots == {"a": "running", "c": "waiting"}  # b idle: left alone
    assert runner.paused == [("ccc", True)]
    await asyncio.sleep(0.1)  # quantum over: c's turn, a frozen (frozen before c is thawed)
    assert sch.owner == "c" and runner.paused[-2:] == [("aaa", True), ("ccc", False)]
    assert sampler.slots == {"a": "waiting", "c": "running"}
    live["c"] = {"up": True, "running": 0, "waiting": 0}  # c ran out of work: a is the only busy model, so nobody is frozen
    await asyncio.sleep(0.03)
    assert not sch.engaged and sch.owner is None and sampler.slots == {} and runner.paused[-1] == ("aaa", False)
    await sch.stop()
    assert not sch.running and sch.owner is None and sampler.slots == {} and sch.freezer.frozen == set()


def test_settings_switch_starts_and_stops_the_scheduler(env):
    client, s, _ = env
    assert client.get("/api/settings/slots").json() == {"enabled": False, "quantum": 2.0, "min_requests": 4, "end_on_finish": False, "min_slot": 0.5, "active": False, "engaged": False, "owner": None, "backend": None, "error": None}
    r = client.put("/api/settings/slots", json={"enabled": True, "quantum": 3, "min_requests": 6}).json()
    assert r["enabled"] and r["quantum"] == 3.0 and r["min_requests"] == 6 and r["active"] and s.slots.running
    assert client.put("/api/settings/slots", json={"quantum": 0.1}).status_code == 422
    assert client.put("/api/settings/slots", json={"min_requests": 0}).status_code == 422
    r = client.put("/api/settings/slots", json={"enabled": False}).json()
    assert not r["enabled"] and not r["active"] and r["quantum"] == 3.0 and not s.slots.running
    assert s.settings.slots == {"enabled": False, "quantum": 3.0, "min_requests": 6, "end_on_finish": False, "min_slot": 0.5}
    r = client.put("/api/settings/slots", json={"end_on_finish": True, "min_slot": 1}).json()
    assert r["end_on_finish"] and r["min_slot"] == 1.0
    assert client.put("/api/settings/slots", json={"min_slot": 0}).status_code == 422
    assert [a["detail"] for a in client.get("/api/log").json() if a["action"] == "settings"] == ["GPU slots on, 3 s each from 6 requests", "GPU slots off, 3 s each from 6 requests",
            "GPU slots off, 3 s each from 6 requests, ending when a request finishes (after 1 s)"]


def test_scheduler_survives_a_failing_tick():
    asyncio.run(_failing_tick())


async def _failing_tick():
    class BrokenRunner(FakeRunner):
        def status(self):
            raise RuntimeError("docker down")
    s = SimpleNamespace(runner=BrokenRunner({}), sampler=SimpleNamespace(latest={"models": {}}, slots={}), root="/nonexistent")
    sch = SlotScheduler(s, quantum=0.1, tick=0.01)
    sch.start()
    await asyncio.sleep(0.03)
    assert sch.running and sch.error == "RuntimeError: docker down"
    await sch.stop()


def test_under_the_threshold_everyone_runs_concurrently():
    asyncio.run(_threshold())


async def _threshold():
    runner = FakeRunner({"a": "aaa", "b": "bbb"})
    sampler = SimpleNamespace(latest={"models": {}}, slots={})
    s = SimpleNamespace(runner=runner, sampler=sampler, root="/nonexistent")
    live = sampler.latest["models"]
    live["a"] = {"up": True, "running": 1}
    live["b"] = {"up": True, "running": 1}
    sch = SlotScheduler(s, quantum=0.1, min_requests=4, tick=0.01)
    sch.start()
    await asyncio.sleep(0.03)
    assert not sch.engaged and sch.owner is None and runner.paused == [] and sampler.slots == {}  # 2 in flight: concurrent
    live["a"] = {"up": True, "running": 3}
    await asyncio.sleep(0.03)
    assert sch.engaged and sch.owner == "a" and runner.paused == [("bbb", True)]  # 4 in flight: slots engage
    live["a"] = {"up": True, "running": 1}
    await asyncio.sleep(0.03)
    assert sch.engaged  # 2 in flight: still above half the threshold, keeps going (hysteresis)
    live["a"] = {"up": True, "running": 0, "waiting": 0}
    await asyncio.sleep(0.03)
    assert not sch.engaged and sch.owner is None and runner.paused[-1] == ("bbb", False) and sampler.slots == {}
    await sch.stop()


def test_end_on_finish_hands_over_at_the_first_finished_request_after_the_minimum():
    asyncio.run(_end_on_finish())


async def _end_on_finish():
    runner = FakeRunner({"a": "aaa", "b": "bbb"})
    sampler = SimpleNamespace(latest={"models": {}}, slots={}, engines={})
    s = SimpleNamespace(runner=runner, sampler=sampler, root="/nonexistent")
    live = sampler.latest["models"]
    live["a"] = {"up": True, "running": 2}
    live["b"] = {"up": True, "running": 2}
    done = {"a": 10.0, "b": 10.0}

    class Sched(SlotScheduler):
        async def _finished(self, name):
            return done[name]

    sch = Sched(s, quantum=1.0, min_requests=1, end_on_finish=True, min_slot=0.05, tick=0.01)
    sch.start()
    await asyncio.sleep(0.03)
    assert sch.owner == "a" and sch.done_at_start == 10.0
    done["a"] = 11.0  # a finishes a request before the minimum: nothing yet
    await asyncio.sleep(0.01)
    assert sch.owner == "a"
    await asyncio.sleep(0.06)  # past the minimum: hands over to b long before the 1 s maximum
    assert sch.owner == "b" and sch.done_at_start == 10.0
    await asyncio.sleep(0.1)  # b finishes nothing: it keeps the slot until the maximum
    assert sch.owner == "b"
    await sch.stop()


def test_a_restarted_dashboard_thaws_what_the_old_one_left_frozen():
    asyncio.run(_stale())


async def _stale():
    runner = FakeRunner({"a": "aaa", "b": "bbb"}, stale={"b"})
    s = SimpleNamespace(runner=runner, sampler=SimpleNamespace(latest={"models": {}}, slots={}, engines={}), root="/nonexistent")
    sch = SlotScheduler(s, quantum=0.1, tick=0.01)
    sch.start()
    await asyncio.sleep(0.03)
    assert ("bbb", False) in runner.paused and sch.freezer.frozen == set()  # b thawed at start, nothing else touched
    await sch.stop()
