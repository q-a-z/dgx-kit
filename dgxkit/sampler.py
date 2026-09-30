"""Runs each collector on its own timer and keeps 15 minutes of history in memory.

Separate tasks mean a slow source (a model server that hangs) can't delay the
1-second GPU and CPU samples.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

HISTORY_SECONDS = 15 * 60


class Sampler:
    def __init__(self, gpu, system, engines: dict | None = None, interval: float = 1.0):
        self.gpu, self.system = gpu, system
        self.engines = engines or {}  # model name -> adapter
        self.interval = interval
        self.on_up = None  # called with a model's name the first time it answers after being added
        self.name_process = None  # optional: adds model/container/context to each GPU process
        self.latest: dict = {"gpu": None, "system": None, "models": {}}
        self.history: deque = deque(maxlen=int(HISTORY_SECONDS / interval))
        self._subscribers: set[asyncio.Queue] = set()
        self._tasks: list[asyncio.Task] = []
        self._engine_tasks: dict[str, asyncio.Task] = {}

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def _hardware_loop(self):
        while True:
            # NVML and /proc reads are quick but blocking; keep them off the event loop.
            g = await asyncio.to_thread(self.gpu.sample)
            gd = g.to_dict() if g is not None else None
            if gd and self.name_process:
                try:
                    gd["processes"] = [self.name_process(p) for p in gd["processes"]]
                except Exception:
                    pass  # unnamed is fine
            self.latest["gpu"] = gd
            self.latest["system"] = await asyncio.to_thread(self.system.sample)
            self._publish()
            await asyncio.sleep(self.interval)

    async def _engine_loop(self, name: str, adapter):
        while True:
            try:
                stats = await adapter.scrape()
                first = not self.latest["models"].get(name, {}).get("was_up")
                self.latest["models"][name] = {"up": True, "was_up": True, **adapter.static, **stats}
                if first and self.on_up:
                    self.on_up(name)
            except Exception as e:  # a dead server shows as down, not as a crash; was_up stays, so "never answered yet" reads as still starting
                prev = self.latest["models"].get(name, {})
                self.latest["models"][name] = {**prev, "up": False, "error": type(e).__name__}
            await asyncio.sleep(self.interval)

    def _publish(self):
        snap = {"t": time.time(), **self.latest}
        self.history.append(snap)
        for q in list(self._subscribers):
            if q.full():  # slow browser: drop its oldest frame rather than block
                q.get_nowait()
            q.put_nowait(snap)

    def start(self):
        self._tasks.append(asyncio.create_task(self._hardware_loop()))
        for name, adapter in self.engines.items():
            self._engine_tasks[name] = asyncio.create_task(self._engine_loop(name, adapter))

    def add_engine(self, name: str, adapter) -> None:
        self.remove_engine(name)
        self.engines[name] = adapter
        self._engine_tasks[name] = asyncio.create_task(self._engine_loop(name, adapter))

    def remove_engine(self, name: str) -> None:
        task = self._engine_tasks.pop(name, None)
        if task:
            task.cancel()
        self.engines.pop(name, None)
        self.latest["models"].pop(name, None)

    async def stop(self):
        tasks = self._tasks + list(self._engine_tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.gpu.close()
