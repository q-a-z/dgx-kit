"""GPU time slots: one model's engine runs at a time; the other engines that have work are frozen in place.

On a GB10 every engine shares one memory bus, so two models decoding at once both slow down, and three
lose about a third of the box's output (see the tuning notes in the README). Freezing an engine (the cgroup
freezer, what `docker pause` does) keeps its KV cache and its open streams; it resumes mid-token when
thawed. Each busy model gets `quantum` seconds in turn, the turn ends early when the owner runs out of
work, and idle engines stay thawed so they keep answering health checks and accepting requests.

Which engines are busy comes from the sampler's own /metrics scrape (running + waiting). A frozen engine
can't be scraped, so it keeps the numbers it had, marked `slot: waiting`, and is still counted as busy.

Slots only pay when a model decodes several requests in a turn, and every new request waits up to two
slots for its first token, so they engage only once at least two models are busy and `min_requests`
requests are in flight between them; below that the engines run concurrently as they would without
the switch. The threshold has hysteresis (engage at the threshold, let go at half of it) so a load that
hovers around it doesn't flap.

With `end_on_finish` a slot also ends as soon as the owner finishes a request (after `min_slot` seconds),
so slots stretch for long answers and shorten for short ones; the scheduler then reads the owner's
finished-requests counter from its /metrics every tick. Off by default: measured against fixed slots it
changes latency, not throughput.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from .engines import prometheus as P

log = logging.getLogger("dgxkit.slots")


def in_flight(live: dict | None) -> int:
    return int((live.get("running") or 0) + (live.get("waiting") or 0)) if live and live.get("up") else 0


def has_work(live: dict | None) -> bool:
    return in_flight(live) > 0


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

    def thaw_stale(self) -> list[str]:
        """Thaw DGX-kit's model containers that are frozen without this scheduler knowing: a dashboard that
        restarted (an update) while it had models frozen. Returns the names thawed."""
        names = []
        for c in self.runner._ours():
            frozen = c.status == "paused"
            if not frozen:
                try:
                    with open(self._path(c.id)) as f:
                        frozen = f.read().strip() == "1"
                except OSError:
                    pass
            if frozen:
                self.frozen.add(c.id)
                self.set(c.id, False)
                names.append(c.labels.get("dgxkit.model", c.id[:12]))
        return names


class SlotScheduler:
    def __init__(self, services, quantum: float, min_requests: int = 4, end_on_finish: bool = False, min_slot: float = 0.5,
                 tick: float = 0.25):
        self.s, self.quantum, self.min_requests, self.tick = services, quantum, min_requests, tick
        self.end_on_finish, self.min_slot = end_on_finish, min_slot
        self.engaged = False  # under the threshold the engines run concurrently
        self.done_at_start: float | None = None  # the owner's finished-requests counter when its slot began
        self.freezer = Freezer(services.runner, services.root)
        self.owner: str | None = None
        self.since = 0.0
        self.error: str | None = None  # the last failed tick, shown in Settings; cleared by the next good one
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

    async def _finished(self, name: str) -> float | None:
        """The engine's lifetime count of finished requests, read now (not the sampler's second-old copy)."""
        ad = self.s.sampler.engines.get(name)
        if ad is None:
            return None
        try:
            text = (await ad.client.get(ad.base_url + "/metrics")).text
            return P.total(P.parse(text), *ad.COUNTERS.get("done", ()))
        except Exception:
            return None

    async def _slot_over(self, now: float) -> bool:
        if now - self.since >= self.quantum:
            return True
        if not self.end_on_finish or self.owner is None or now - self.since < self.min_slot or self.done_at_start is None:
            return False
        done = await self._finished(self.owner)
        return done is not None and done > self.done_at_start

    def _next(self, names: list[str], busy: list[str]) -> str | None:
        """Round robin over the busy models, starting after the current owner."""
        if not busy:
            return None
        i = names.index(self.owner) if self.owner in names else -1
        return next(names[(i + k) % len(names)] for k in range(1, len(names) + 1) if names[(i + k) % len(names)] in busy)

    async def _run(self) -> None:
        ids: dict[str, str] = {}
        ids_at = 0.0
        log.info("GPU slots on: %.1f s each", self.quantum)
        try:
            stale = await asyncio.to_thread(self.freezer.thaw_stale)
            if stale:
                log.warning("GPU slots: thawed %s, left frozen by an earlier dashboard", ", ".join(stale))
            while True:
                try:
                    now = time.monotonic()
                    if now - ids_at >= 5 or not ids:  # which of our containers run (ids change on restart)
                        st = await asyncio.to_thread(self.s.runner.status)
                        ids = {n: c["id"] for n, c in st.items() if c["state"] == "running" and c.get("engine")}
                        ids_at = now
                    live = self.s.sampler.latest["models"]
                    busy = [n for n in ids if ids[n] in self.freezer.frozen or has_work(live.get(n))]
                    load = sum(in_flight(live.get(n)) for n in busy)
                    self.engaged = len(busy) >= 2 and load >= (self.min_requests if not self.engaged else max(1, self.min_requests // 2))
                    if not self.engaged:
                        busy = []  # everyone runs: the loop below thaws what is frozen and marks nothing
                    switched = False
                    if self.owner not in busy or await self._slot_over(now):
                        nxt = self._next(list(ids), busy)
                        if nxt != self.owner:
                            log.debug("slot %s -> %s (busy %s)", self.owner, nxt, busy)
                            switched = True
                        self.owner, self.since = nxt, now
                    for n in ids:  # freeze first, then thaw: never two engines running
                        if n != self.owner and n in busy:
                            await asyncio.to_thread(self.freezer.set, ids[n], True)
                    for n in ids:
                        if n == self.owner or n not in busy:
                            await asyncio.to_thread(self.freezer.set, ids[n], False)
                    if switched or self.done_at_start is None:
                        self.done_at_start = await self._finished(self.owner) if self.end_on_finish and self.owner else None
                    self.s.sampler.slots = {n: "running" if n == self.owner else "waiting" for n in ids
                                            if n == self.owner or ids[n] in self.freezer.frozen}
                    self.error = None
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # Docker hiccup, a container restarting mid-tick: say so and keep going
                    self.error = f"{type(e).__name__}: {str(e)[:160]}"
                    log.warning("GPU slots: %s", self.error)
                    ids = {}
                    await asyncio.sleep(1)
                await asyncio.sleep(self.tick)
        finally:
            self.s.sampler.slots = {}
            self.owner = None
            self.freezer.thaw_all()  # synchronous: this also runs while the dashboard is shutting down
