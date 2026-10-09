"""GPU time slots: one model's engine runs at a time; the other engines that have work are frozen in place.

On a GB10 every engine shares one memory bus, so two models decoding at once both slow down, and three
lose about a third of the box's output (see the tuning notes in the README). Freezing an engine (the cgroup
freezer, what `docker pause` does) keeps its KV cache and its open streams; it resumes mid-token when
thawed. Each busy model gets `quantum` seconds in turn, the turn ends early when the owner runs out of
work, and idle engines stay thawed so they keep answering health checks and accepting requests.

Which engines are busy comes from the sampler's own /metrics scrape (running + waiting). A frozen engine
can't be scraped, so it keeps the numbers it had, marked `slot: waiting`, and is still counted as busy.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

log = logging.getLogger("dgxkit.slots")


def has_work(live: dict | None) -> bool:
    return bool(live and live.get("up") and ((live.get("running") or 0) + (live.get("waiting") or 0)) > 0)


class Freezer:
    """Freezes and thaws DGX-kit's model containers by id.

    Writes the container's cgroup.freeze file when /sys is mounted writable (microseconds); otherwise asks
    Docker to pause or unpause it (about 15 ms). Docker then reports the container as "paused"; DockerRunner
    shows that as running, because it is.
    """

    def __init__(self, runner, root: str = "/"):
        self.runner, self.root = runner, root
        self.frozen: set[str] = set()
        self.backend: str | None = None

    def _path(self, cid: str) -> str:
        return os.path.join(self.root, "sys/fs/cgroup/system.slice", f"docker-{cid}.scope", "cgroup.freeze")

    def set(self, cid: str, frozen: bool) -> None:
        if (cid in self.frozen) == frozen:
            return
        try:
            with open(self._path(cid), "w") as f:
                f.write("1" if frozen else "0")
            self.backend = "cgroup"
        except OSError:
            c = self.runner.docker.containers.get(cid)
            try:
                (c.pause if frozen else c.unpause)()
            except Exception as e:  # already in that state (someone paused or unpaused it by hand)
                if "already" not in str(e) and "not paused" not in str(e):
                    raise
            self.backend = "docker pause"
        (self.frozen.add if frozen else self.frozen.discard)(cid)

    def thaw_all(self) -> None:
        for cid in list(self.frozen):
            try:
                self.set(cid, False)
            except Exception as e:
                log.warning("could not thaw %s: %s", cid[:12], e)


class SlotScheduler:
    def __init__(self, services, quantum: float, tick: float = 0.25):
        self.s, self.quantum, self.tick = services, quantum, tick
        self.freezer = Freezer(services.runner, services.root)
        self.owner: str | None = None
        self.since = 0.0
        self.task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return bool(self.task and not self.task.done())

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    def _next(self, names: list[str], busy: list[str]) -> str | None:
        """Round robin over the busy models, starting after the current owner."""
        if not busy:
            return None
        i = names.index(self.owner) if self.owner in names else -1
        return next(names[(i + k) % len(names)] for k in range(1, len(names) + 1) if names[(i + k) % len(names)] in busy)

    async def _run(self) -> None:
        ids: dict[str, str] = {}
        ids_at = 0.0
        try:
            while True:
                now = time.monotonic()
                if now - ids_at >= 5 or not ids:  # which of our containers run (ids change on restart)
                    st = await asyncio.to_thread(self.s.runner.status)
                    ids = {n: c["id"] for n, c in st.items() if c["state"] == "running" and c.get("engine")}
                    ids_at = now
                live = self.s.sampler.latest["models"]
                busy = [n for n in ids if ids[n] in self.freezer.frozen or has_work(live.get(n))]
                if self.owner not in busy or now - self.since >= self.quantum:
                    nxt = self._next(list(ids), busy)
                    if nxt != self.owner:
                        log.debug("slot %s -> %s", self.owner, nxt)
                    self.owner, self.since = nxt, now
                for n in ids:  # freeze first, then thaw: never two engines running
                    if n != self.owner and n in busy:
                        await asyncio.to_thread(self.freezer.set, ids[n], True)
                for n in ids:
                    if n == self.owner or n not in busy:
                        await asyncio.to_thread(self.freezer.set, ids[n], False)
                self.s.sampler.slots = {n: "running" if n == self.owner else "waiting" for n in ids
                                        if n == self.owner or ids[n] in self.freezer.frozen}
                await asyncio.sleep(self.tick)
        finally:
            self.s.sampler.slots = {}
            self.owner = None
            await asyncio.to_thread(self.freezer.thaw_all)
