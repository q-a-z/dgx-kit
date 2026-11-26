import asyncio

import httpx
import pytest

from dgxkit.engines.vllm import VllmAdapter
from dgxkit.sampler import Sampler


def test_a_failed_metrics_request_is_not_a_successful_scrape():
    async def check():
        transport = httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable"))
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = VllmAdapter("http://engine", client=client)
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.scrape()

    asyncio.run(check())


def test_metrics_success_does_not_mark_an_unhealthy_managed_engine_up():
    async def check():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text="") if request.url.path == "/metrics" else httpx.Response(503)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            s = Sampler(None, None)
            s.add_engine("demo", VllmAdapter("http://engine", client=client), check_health=True)
            await s.wait_for_first_samples(["demo"])
            assert s.latest["models"]["demo"]["up"] is False
            s.remove_engine("demo")

    asyncio.run(check())


def test_history_keeps_the_model_sample_that_was_published():
    s = Sampler(None, None)
    s.latest["models"]["demo"] = {"decode_tps": 9}
    s._publish()
    s.latest["models"]["demo"] = {"decode_tps": 1}
    s._publish()
    s.remove_engine("demo")
    assert [frame["models"]["demo"]["decode_tps"] for frame in s.history] == [9, 1]
