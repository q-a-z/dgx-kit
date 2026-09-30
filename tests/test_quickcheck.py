import json

import httpx

from dgxkit import quickcheck


def fake_engine():
    """An OpenAI-style streaming endpoint: every reply is 41 content chunks, then a usage chunk."""
    def handler(req: httpx.Request):
        body = json.loads(req.content)
        big = len(body["messages"][0]["content"]) > 1000
        chunks = [{"choices": [{"delta": {"content": "x"}}]} for _ in range(41)]
        chunks.append({"choices": [], "usage": {"prompt_tokens": 7200 if big else 20, "completion_tokens": 41}})
        text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=text.encode(), headers={"content-type": "text/event-stream"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_quick_check_reports_decode_and_prefill_within_its_budget():
    r = quickcheck.run(8100, "m", client=fake_engine())
    assert len(r["decode_tps"]) == 2 and len(r["prefill"]) == 2
    assert r["decode_mean_tps"] > 0 and r["prefill_mean_tps"] > 0
    assert r["prefill"][0]["tokens"] == 7200 and r["seconds"] <= quickcheck.BUDGET_S


def test_results_are_kept_per_model(tmp_path):
    quickcheck.save(str(tmp_path), "m", {"decode_mean_tps": 90.0})
    assert quickcheck.load(str(tmp_path), "m") == {"decode_mean_tps": 90.0} and quickcheck.load(str(tmp_path), "other") is None
