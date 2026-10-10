"""Queue mode for GPU slots: one model's requests are admitted at a time; the others wait in a queue here.

Nothing is frozen. LiteLLM is pointed at this proxy for every published model (api_models.sync_gateway),
and the proxy forwards each request to its engine by name. A model owns the GPU while its requests are
admitted; after `quantum` seconds it stops admitting new ones, and once its in-flight requests have
finished the next model with waiting requests takes over (round robin). Streams therefore never pause,
and the wait moves to before the first token. Tiny requests (titles, pings: max_tokens <= FAST_MAX)
skip the queue. The same load threshold as freeze mode applies: under it everything passes straight
through.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

log = logging.getLogger("dgxkit.turns")
QUEUE_PORT = int(os.environ.get("DGXKIT_QUEUE_PORT", "4010"))
FAST_MAX = 16
SKIP_REQ = {"host", "content-length", "accept-encoding", "connection", "transfer-encoding"}
SKIP_RESP = {"content-length", "content-encoding", "transfer-encoding", "connection"}


class Turn:
    """Who may send requests to the GPU now. acquire() returns at once for the owner (or under the threshold)
    and otherwise waits for the model's turn; release() hands over once the owner has drained."""

    def __init__(self, quantum: float, min_requests: int = 4, clock=time.monotonic, on_change=None):
        self.quantum, self.min_requests, self.clock = quantum, min_requests, clock
        self.on_change = on_change  # called with (owner, [models with waiters]) whenever that changes
        self.cnt: dict[str, int] = {}      # model -> in flight
        self.q: dict[str, deque] = {}      # model -> waiting futures
        self.order: list[str] = []
        self.owner: str | None = None
        self.since = 0.0
        self.engaged = False

    def _model(self, m: str) -> None:
        if m not in self.cnt:
            self.cnt[m], self.q[m] = 0, deque()
            self.order.append(m)

    def _busy(self) -> list[str]:
        return [m for m in self.order if self.cnt[m] or self.q[m]]

    def _expired(self) -> bool:
        return any(self.q[m] for m in self.order if m != self.owner) and self.clock() - self.since >= self.quantum

    def _notify(self) -> None:
        if self.on_change:
            self.on_change(self.owner if self.engaged else None, [m for m in self.order if self.q[m]])

    def _admit(self, m: str) -> None:
        while self.q[m]:
            f = self.q[m].popleft()
            if not f.done():
                f.set_result(None)
                self.cnt[m] += 1

    def _settle(self) -> None:
        """Re-evaluate the threshold and hand over when the owner has drained."""
        busy = self._busy()
        load = sum(self.cnt[m] + len(self.q[m]) for m in busy)
        self.engaged = len(busy) >= 2 and load >= (self.min_requests if not self.engaged else max(1, self.min_requests // 2))
        if not self.engaged:  # everyone through
            self.owner = None
            for m in self.order:
                self._admit(m)
        elif self.owner is None or not self.cnt[self.owner]:
            n = len(self.order)
            i = self.order.index(self.owner) if self.owner in self.order else -1
            nxt = next((self.order[(i + k) % n] for k in range(1, n + 1) if self.q[self.order[(i + k) % n]]), None)
            if nxt is None and self.owner is None:  # just engaged, nobody waiting yet: the busiest model keeps going
                nxt = max(busy, key=lambda m: self.cnt[m])
            if nxt is not None:
                self.owner, self.since = nxt, self.clock()
                self._admit(nxt)
        self._notify()

    async def acquire(self, m: str) -> None:
        self._model(m)
        self.cnt[m] += 1  # count it, then see whether it may go now
        self._settle()
        if not self.engaged or (m == self.owner and not self._expired()):
            self._notify()
            return
        self.cnt[m] -= 1  # no: wait for the model's turn
        fut = asyncio.get_running_loop().create_future()
        self.q[m].append(fut)
        self._settle()
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release(m)  # admitted right before the cancel
            elif fut in self.q[m]:
                self.q[m].remove(fut)
                self._settle()
            raise

    def release(self, m: str) -> None:
        self.cnt[m] -= 1
        self._settle()


class QueueProxy:
    """The proxy LiteLLM talks to in queue mode: an HTTP server beside the dashboard, forwarding by model name."""

    def __init__(self, services, quantum: float, min_requests: int = 4, port: int = QUEUE_PORT, host: str = "0.0.0.0",
                 client: httpx.AsyncClient | None = None):
        self.s, self.port, self.host = services, port, host
        self.turn = Turn(quantum, min_requests, on_change=self._mark)
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
        self.ports: dict[str, int] = {}
        self.ports_at = 0.0
        self.app = self._app()
        self.server = None
        self.task: asyncio.Task | None = None
        self.error: str | None = None

    @property
    def owner(self):
        return self.turn.owner if self.turn.engaged else None

    @property
    def engaged(self):
        return self.turn.engaged

    @property
    def running(self) -> bool:
        return bool(self.task and not self.task.done())

    def _mark(self, owner, waiting) -> None:
        self.s.sampler.slots = {**{m: "waiting" for m in waiting}, **({owner: "running"} if owner else {})}

    async def _port(self, model: str) -> int | None:
        now = time.monotonic()
        if now - self.ports_at >= 5 or model not in self.ports:
            try:
                st = await asyncio.to_thread(self.s.runner.status)
                self.ports = {n: c["port"] for n, c in st.items() if c["state"] == "running" and c.get("engine") and c.get("port")}
            except Exception as e:
                self.error = f"{type(e).__name__}: {str(e)[:160]}"
            self.ports_at = now
        return self.ports.get(model)

    def _app(self) -> FastAPI:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.get("/v1/models")
        async def models():
            await self._port("")
            return {"object": "list", "data": [{"id": m, "object": "model"} for m in sorted(self.ports)]}

        @app.get("/health")
        async def health():
            return {"ok": True, "owner": self.owner, "engaged": self.engaged}

        @app.api_route("/v1/{path:path}", methods=["POST"])
        async def forward(path: str, request: Request):
            body = await request.body()
            try:
                req = json.loads(body)
                model = req["model"]
            except (ValueError, KeyError, TypeError):
                return JSONResponse({"error": {"message": "body must be JSON with a 'model' field"}}, status_code=400)
            port = await self._port(model)
            if port is None:
                return JSONResponse({"error": {"message": f"no running model {model!r}"}}, status_code=404)
            mt = req.get("max_tokens") or req.get("max_completion_tokens") or 10**9
            gated = mt > FAST_MAX
            if gated:
                await self.turn.acquire(model)
            released = False

            def release():
                nonlocal released
                if gated and not released:
                    released = True
                    self.turn.release(model)

            headers = {k: v for k, v in request.headers.items() if k.lower() not in SKIP_REQ}
            url = f"http://127.0.0.1:{port}/v1/{path}" + (f"?{request.url.query}" if request.url.query else "")
            try:
                up = await self.client.send(self.client.build_request("POST", url, content=body, headers=headers), stream=True)
            except httpx.HTTPError as e:
                release()
                return JSONResponse({"error": {"message": str(e)}}, status_code=502)

            async def body_iter():
                try:
                    async for chunk in up.aiter_raw():
                        yield chunk
                finally:
                    release()  # also on client disconnect: the upstream closes and the engine aborts

            async def done():
                await up.aclose()
                release()

            return StreamingResponse(body_iter(), status_code=up.status_code,
                                     headers={k: v for k, v in up.headers.items() if k.lower() not in SKIP_RESP},
                                     background=BackgroundTask(done))
        return app

    def start(self) -> None:
        import uvicorn
        self.server = uvicorn.Server(uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", lifespan="off"))
        self.task = asyncio.create_task(self.server.serve())
        log.info("GPU slots (queue): proxy on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self.server:
            self.server.should_exit = True
        if self.task:
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        self.s.sampler.slots = {}
        await self.client.aclose()
