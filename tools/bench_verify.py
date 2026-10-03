"""Scoring of a model's code answer against hidden tests, shared by bench.py and its tests."""
import json, re, sys

HARNESS = """
import json, sys, traceback
ns = {}
exec(open(sys.argv[1]).read(), ns)
res = {}
for k, f in list(ns.items()):
    if k.startswith("t_") and callable(f):
        try: f(); res[k] = "ok"
        except BaseException as e: res[k] = (repr(e) or type(e).__name__)[:120]
print("@@" + json.dumps(res))
"""


def why(err, e):
    """What went wrong with a solution that wouldn't even load: the last error line Python printed (IndentationError,
    NameError ...), not the JSONDecodeError of the harness finding no result to read."""
    last = [l.strip() for l in (err or "").splitlines() if re.match(r"^\w*(Error|Exception|Interrupt)\b", l.strip())]
    return last[-1][:160] if last else repr(e)


def verify(content, hidden, finish=None):
    """Run OUR hidden tests against the model's solution (the model's own tests only prove self-consistency).
    finish='length' means the answer was cut off by the token limit: the score is then marked as not a fair one."""
    import subprocess, tempfile
    cut = "the answer was cut off at the token limit, so this is not a fair score; " if finish == "length" else ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", content, re.S)
    if not blocks:
        return dict(correct=0, total=0, fail_class="truncated" if cut else "no_code", note=cut + "no code block in the answer")
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/solution.py", "w").write(blocks[0]); open(f"{d}/hidden.py", "w").write(hidden); open(f"{d}/harness.py", "w").write(HARNESS)
        total = len(re.findall(r"^def t_", hidden, re.M))
        try:
            p = subprocess.run([sys.executable, "harness.py", "hidden.py"], cwd=d, capture_output=True, text=True, timeout=120)
            res = json.loads(p.stdout.split("@@")[-1])
        except Exception as e:
            err = (locals().get("p") and p.stderr) or ""
            cls = "timeout" if isinstance(e, subprocess.TimeoutExpired) else "syntax" if "SyntaxError" in err or "IndentationError" in err else "crash"
            return dict(correct=0, total=total, fail_class="truncated" if cut else cls,
                        note=cut + "solution failed to import/run: " + (why(err, e) if not isinstance(e, subprocess.TimeoutExpired) else "timeout"), stderr=err[-300:] or None)
        own = None
        if len(blocks) > 1:
            open(f"{d}/test_model.py", "w").write(blocks[1])
            try:
                o = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "test_model.py"], cwd=d, capture_output=True, text=True, timeout=120)
                own = "pass" if o.returncode == 0 else "fail" if "passed" in o.stdout or "failed" in o.stdout else "n/a"
            except Exception:
                own = "n/a"
    bad = {k: v for k, v in res.items() if v != "ok"}
    out = dict(correct=total - len(bad), total=total, failed=bad, model_own_tests=own)
    if cut:
        out["note"] = cut.rstrip("; ")
    return out
