import importlib.util
import sys
from pathlib import Path

FENCE = "`" * 3


def load_bench(monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    import bench_verify
    return bench_verify


HIDDEN = "from solution import *\ndef t_one():\n    assert True\n"  # the real hidden tests import the solution too


def answer(code):
    return f"here you go\n{FENCE}python\n{code}\n{FENCE}\n"


def test_a_solution_that_does_not_load_says_why(monkeypatch):
    b = load_bench(monkeypatch)
    v = b.verify(answer("class TTLCache:\n    def __init__(self, clock=time.monotonic):\n        pass\n"), HIDDEN)
    assert v["fail_class"] == "crash" and "NameError: name 'time' is not defined" in v["note"]
    assert "JSONDecodeError" not in v["note"]
    v = b.verify(answer("with x:\ny = 1\n"), HIDDEN)
    assert v["fail_class"] == "syntax" and "IndentationError" in v["note"]


def test_an_answer_cut_off_by_the_token_limit_is_not_scored_as_the_models_fault(monkeypatch):
    b = load_bench(monkeypatch)
    cut = b.verify(f"{FENCE}python\nwith lock:\n", HIDDEN, "length")  # the code block never closed
    assert cut["fail_class"] == "truncated" and "cut off at the token limit" in cut["note"]
    ok = b.verify(answer("def f():\n    return 1\n"), HIDDEN, "stop")
    assert ok["correct"] == 1 and "note" not in ok
