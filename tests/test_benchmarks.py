import json
import time

import pytest

from dgxkit.benchmarks import Benchmarks

FAKE = """
import argparse, json, os, sys
ap = argparse.ArgumentParser()
for a in ("--port", "--tag", "--conc", "--tests", "--out"):
    ap.add_argument(a)
A = ap.parse_args()
print("running", A.tests)
if A.tag.startswith("bad"):
    sys.exit("ABORT: other models are busy")
os.makedirs(A.out, exist_ok=True)
json.dump({"tag": A.tag, "port": A.port, "conc": A.conc, "decode": {"code_temp0": {"tps": [100.0]}}}, open(f"{A.out}/{A.tag}.json", "w"))
"""


def wait(b, run_id, want):
    for _ in range(100):
        if b.get(run_id)["state"] == want:
            return b.get(run_id)
        time.sleep(0.05)
    raise AssertionError(b.get(run_id))


@pytest.fixture
def bench(tmp_path):
    script = tmp_path / "bench.py"
    script.write_text(FAKE)
    return Benchmarks(str(tmp_path / "state"), script=script)


def test_a_finished_run_has_its_result_and_log(bench):
    run = bench.start("qwen", 8100, "decode,conc", 3)
    done = wait(bench, run["id"], "done")
    assert done["result"]["port"] == "8100" and done["result"]["conc"] == "3"
    assert "running decode,conc" in done["tail"][0]
    assert [r["id"] for r in bench.list("qwen")] == [run["id"]] and bench.list("other") == []


def test_a_run_that_exits_without_a_result_is_failed_and_shows_why(bench):
    run = bench.start("bad", 8100)
    failed = wait(bench, run["id"], "failed")
    assert "ABORT" in failed["tail"][-1] and failed["result"] is None


def test_run_ids_are_names_not_paths(bench):
    assert bench.get("../../etc/passwd") is None
