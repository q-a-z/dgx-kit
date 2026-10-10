"""Queue mode: the turn scheduler and the proxy that forwards by model name."""
import asyncio
import json
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from dgxkit import turns
from dgxkit.turns import QueueProxy, Turn


def test_turn_under_threshold_everyone_passes_and_over_it_one_model_at_a_time():
    asyncio.run(_turn())


async def _turn():
    marks = []
    t = Turn(quantum=0.05, min_requests=3, on_change=lambda o, w: marks.append((o, list(w))))
    await t.acquire("a"); await t.acquire("b")          # 2 in flight: under the threshold, both pass
    assert not t.engaged and t.owner is None
    await t.acquire("a")                                 # 3: engaged; a is busiest, so a owns
    assert t.engaged and t.owner == "a"
    w = asyncio.create_task(t.acquire("b"))              # b's next request waits
    await asyncio.sleep(0.01)
    assert not w.done() and marks[-1] == ("a", ["b"])
    await asyncio.sleep(0.06)                            # a's turn expired: new a requests queue too
    w2 = asyncio.create_task(t.acquire("a"))
    await asyncio.sleep(0.01)
    assert not w2.done()
    t.release("a"); t.release("a")                       # a drained -> b's turn, its waiter admitted
    await asyncio.wait_for(w, 1)
    assert t.owner == "b" and t.cnt == {"a": 0, "b": 2} and not w2.done()
    t.release("b"); t.release("b")                       # b drained: only a is busy -> under the threshold, a's waiter admitted
    await asyncio.wait_for(w2, 1)
    assert t.owner is None and not t.engaged and t.cnt == {"a": 1, "b": 0}
    t.release("a")
    assert t.cnt == {"a": 0, "b": 0} and marks[-1] == (None, [])


def test_tiny_requests_and_cancelled_waiters_do_not_wedge_the_turn():
    asyncio.run(_cancel())


async def _cancel():
    t = Turn(quantum=10, min_requests=1)
    await t.acquire("a"); await t.acquire("a")
    w = asyncio.create_task(t.acquire("b"))
    await asyncio.sleep(0.01)
    w.cancel()
    await asyncio.gather(w, return_exceptions=True)
    assert not t.q["b"] and t.cnt["b"] == 0
    t.release("a"); t.release("a")
    assert not t.engaged


class FakeRunner:
    def status(self):
        return {"llama": {"state": "running", "id": "x", "engine": "vllm", "port": 8100}}


def test_proxy_forwards_by_model_name_and_streams_the_answer():
    seen = []

    def engine(request: httpx.Request):
        seen.append((request.url.host, request.url.port, request.url.path, json.loads(request.content)["model"]))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(b"data: {\"a\":1}\n\ndata: [DONE]\n\n"))

    s = SimpleNamespace(runner=FakeRunner(), sampler=SimpleNamespace(slots={}))
    p = QueueProxy(s, quantum=2, min_requests=4, client=httpx.AsyncClient(transport=httpx.MockTransport(engine)))
    with TestClient(p.app) as c:
        r = c.post("/v1/chat/completions", json={"model": "llama", "messages": [], "max_tokens": 50})
        assert r.status_code == 200 and r.text.startswith("data: ") and seen == [("127.0.0.1", 8100, "/v1/chat/completions", "llama")]
        assert p.turn.cnt == {"llama": 0}  # released when the stream ended
        assert c.post("/v1/chat/completions", json={"model": "nope"}).status_code == 404
        assert c.post("/v1/chat/completions", content=b"not json").status_code == 400
        assert c.get("/v1/models").json()["data"] == [{"id": "llama", "object": "model"}]
        assert c.get("/health").json() == {"ok": True, "owner": None, "engaged": False}


def test_queue_mode_points_litellm_at_the_proxy(env, monkeypatch):
    client, s, models = env
    monkeypatch.setattr(turns, "QUEUE_PORT", 0)  # any free port
    client.post("/api/models", json={"name": "llama", "repo": "org/Llama-8B", "config": __import__("tests.test_api", fromlist=["CFG"]).CFG, "weights_bytes": 16 * 2**30})
    (models / "org--Llama-8B").mkdir()
    client.post("/api/models/llama/start")
    assert s.gateway.synced[-1] == {"llama": 8100}
    r = client.put("/api/settings/slots", json={"enabled": True, "mode": "queue"}).json()
    assert r["mode"] == "queue" and r["active"] and r["queue_port"] == 0 and r["backend"] is None
    assert s.gateway.synced[-1] == {"llama": 0}  # every model goes through the queue
    r = client.put("/api/settings/slots", json={"mode": "freeze"}).json()
    assert r["mode"] == "freeze" and r["queue_port"] is None
    assert s.gateway.synced[-1] == {"llama": 8100}  # and back to the engines
    assert client.put("/api/settings/slots", json={"mode": "other"}).status_code == 422
