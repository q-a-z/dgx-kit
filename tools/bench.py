#!/usr/bin/env python3
"""Fleet benchmark battery for one vLLM endpoint.

  python3 bench.py --port 8082 --tag ornith-256k [--conc 3] [--tests decode,conc,prefill,stall,needle,tools,sanity]

Refuses to run if ANY model on ports 8080-8199 has running requests (contention poisons numbers).
Results are printed and written to <--out>/<tag>.json (default ~/bench_results).
"""
import argparse, json, os, random, re, statistics as st, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--conc", type=int, default=2, help="streams for the concurrency test (= max-num-seqs)")
ap.add_argument("--tests", default="decode,complex,hardcore,conc,prefill,stall,needle,tools,sanity")
ap.add_argument("--needle-frac", type=float, default=0.9, help="fraction of max_model_len for the needle test")
ap.add_argument("--tool-runs", type=int, default=50)
ap.add_argument("--n", type=int, default=0, help="draft tokens per step; needed for the accept label on backends without per-position metrics (llama.cpp, MTPLX, ...)")
ap.add_argument("--ctx", type=int, default=32768, help="context length when /v1/models has no max_model_len (ollama etc.)")
ap.add_argument("--code-think", action="store_true", help="let complex/hardcore think (reasoning can eat the whole token budget -> empty answer -> score 0)")
ap.add_argument("--out", default=os.path.expanduser("~/bench_results"), help="folder for the result json and raw outputs")
ap.add_argument("--force", action="store_true", help="skip the idle check")
A = ap.parse_args()
BASE = f"http://localhost:{A.port}"
OUT = {"tag": A.tag, "port": A.port, "ts": time.strftime("%F %T")}


def http(path, body=None, base=BASE, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def metrics(base=BASE):
    d = {}
    try:
        lines = http("/metrics", base=base, timeout=5).read().decode().splitlines()
    except Exception:
        return d  # ollama & co: no /metrics, acceptance then comes from per-response timings
    for l in lines:
        if l.startswith("#") or "_created" in l:
            continue
        if l.startswith(("vllm:spec_decode", "vllm:num_requests_running")):
            k, v = l.rsplit(" ", 1)
            d[k] = float(v)
    return d


def running(base):
    return sum(v for k, v in metrics(base).items() if k.startswith("vllm:num_requests_running"))


def idle_check():
    busy = {}
    for p in range(8080, 8200):  # llmctl's ports and DGX-kit's
        try:
            r = running(f"http://localhost:{p}")
        except Exception:
            continue  # nothing listening
        if r > 0:
            busy[p] = r
    print("idle check, busy ports:", busy or "none")
    if busy and not A.force:
        sys.exit("ABORT: other models are busy, results would be contaminated (--force to override)")


def spec_delta(a, b):
    dr = ac = 0
    pos = {}
    for k in b:
        dv = b[k] - a.get(k, 0)
        if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            pos[int(k.split('position="')[1].split('"')[0])] = dv
        elif k.startswith("vllm:spec_decode_num_accepted_tokens"):
            ac += dv
        elif k.startswith("vllm:spec_decode_num_drafts"):
            dr += dv
    return dr, ac, pos


def stream(prompt, max_tokens=1500, temperature=None, think=None, on_chunk=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    if temperature is not None:
        body["temperature"] = temperature
    if think is not None:
        body["chat_template_kwargs"] = {"enable_thinking": think}
    t0 = time.time(); tf = None; usage = None; c = []; tm = {}; ms = {}; fin = None
    with http("/v1/chat/completions", body) as resp:
        for raw in resp:
            l = raw.decode().strip()
            if not l.startswith("data:"):
                continue
            p = l[5:].strip()
            if p == "[DONE]":
                break
            j = json.loads(p)
            if j.get("usage"):
                usage = j["usage"]
            tm = j.get("timings") or tm  # llama.cpp server: draft_n / draft_n_accepted
            ms = j.get("mtplx_stats") or ms  # MTPLX: exact verify_calls + accepted_drafts
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                fin = ch.get("finish_reason") or fin
                cc = d.get("content"); rr = d.get("reasoning_content") or d.get("reasoning")
                if cc or rr:
                    now = time.time()
                    tf = tf or now
                    if on_chunk:
                        on_chunk(now)
                if cc:
                    c.append(cc)
    t1 = time.time(); n = usage["completion_tokens"]
    return dict(toks=n, ptoks=usage["prompt_tokens"], ttft=(tf or t1) - t0, tps=(n - 1) / max(t1 - (tf or t0), 1e-6),
                t0=t0, end=t1, finish=fin, content="".join(c), drafted=tm.get("draft_n", 0), accepted=tm.get("draft_n_accepted", 0),
                verifies=ms.get("verify_calls", 0), mt_acc=ms.get("accepted_drafts", 0), depth=ms.get("speculative_depth", 0))


def timing_acc(rs):
    """Accept length from per-response draft counters (no /metrics backends); needs --n."""
    v = sum(r["verifies"] for r in rs)
    if v:  # MTPLX: exact accept length, and it reports its own draft depth
        A.n = A.n or rs[0]["depth"]
        return 1 + sum(r["mt_acc"] for r in rs) / v
    d = sum(r["drafted"] for r in rs); a = sum(r["accepted"] for r in rs)
    return (1 + A.n * a / d) if d and A.n else float("nan")


def one(prompt, **kw):
    a = metrics(); r = stream(prompt, **kw); dr, ac, pos = spec_delta(a, metrics())
    r["acc"] = (1 + ac / dr) if dr else timing_acc([r]); r["pos"] = pos
    return r


CODE = "Spec: implement a class LRUCache with get(key) and put(key, value), capacity set in the constructor, O(1) operations, thread-safe. Return the implementation in one python code block, then 3 pytest tests in a second code block."
CODE_COMPLEX = ("Spec: implement in Python a thread-safe class TTLCache in one file, exact API: TTLCache(capacity, default_ttl_s, clock=time.monotonic); "
                "get(key) -> value or None; put(key, value, ttl_s=None); delete(key); stats() -> dict with int keys hits, misses, evictions, expirations; "
                "get_or_load(key, loader) -> value. Requirements: O(1) get/put with LRU order; expired entries are never returned and are removed lazily (counted as expirations, not evictions); "
                "when full, put() first drops an expired entry if any exists, otherwise evicts the least-recently-used live entry (counted as an eviction); "
                "get_or_load runs loader() at most once per key under concurrent callers, using per-key locks, and must NOT hold any global lock while the loader runs, "
                "so loaders for different keys run in parallel. "
                "Return the implementation in one python code block (it will be saved as solution.py), then 6 pytest tests in a second code block that does `from solution import TTLCache`, "
                "including a fake-clock expiry test and a 16-thread get_or_load test.")
CODE_HARDCORE = ("Spec: write a complete, single-file Python 3 implementation of a small in-memory SQL engine (no sqlite3, no third-party libs). "
                 "Exact API: `db = Database(); res = db.execute(sql)`; res.columns is a list of str, res.rows a list of tuples, res.rowcount an int for INSERT/UPDATE/DELETE; "
                 "syntax errors raise SQLSyntaxError with int attributes .line and .col (1-based). Keywords are case-insensitive, strings use single quotes, a trailing ; is optional. "
                 "Support: CREATE TABLE t (col INT|TEXT|REAL, ...); INSERT INTO t [(cols)] VALUES (...)[, (...)] (omitted columns are NULL); "
                 "SELECT with * or column lists (plain, table-qualified like users.name, and aggregates); WHERE with AND/OR/NOT, parentheses, =, !=, <, <=, >, >=, LIKE (% and _), IS NULL / IS NOT NULL; "
                 "INNER JOIN and LEFT JOIN ... ON a.x = b.y; GROUP BY with COUNT(*)/COUNT(col)/SUM/AVG/MIN/MAX and HAVING; ORDER BY multiple columns with ASC/DESC and NULLS always last; LIMIT n [OFFSET m]; "
                 "UPDATE t SET c = expr[, ...] [WHERE ...]; DELETE FROM t [WHERE ...]; SQL three-valued logic for NULL (NULL = NULL is unknown, unknown rows are not selected; aggregates ignore NULLs); "
                 "BEGIN / COMMIT / ROLLBACK using an undo log. "
                 "Implementation constraints: hand-written tokenizer and recursive-descent parser producing an AST; a planner that uses a hash join instead of a nested loop for equality joins and pushes WHERE predicates below the join when they touch one table only (but never below the null-supplying side of a LEFT JOIN). "
                 "Return the implementation in one python code block (it will be saved as solution.py), then 10 pytest tests in a second code block that does `from solution import Database, SQLSyntaxError`, "
                 "covering joins, NULL logic, HAVING, rollback and error positions.")
PROSE = "Explain the difference between optimistic and pessimistic locking in databases, with a concrete example of when each is the better choice."
NEEDLE = "PELICAN-7734-VIOLET"
WORDS = "alpha bravo cargo delta ember flint garnet harbor ivory jasper kestrel lantern marble nickel opal pepper quartz raven saffron timber umber velvet willow xenon yarrow zephyr copper cobalt basalt cedar dune fjord glacier heron island juniper knoll lagoon meadow nettle orchid prairie quill ridge summit tundra valley".split()


def mkdoc(n_sent, depth, seed=7):
    rng = random.Random(seed)
    s = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 14))).capitalize() + "." for _ in range(n_sent)]
    s.insert(int(len(s) * depth), f"NOTE: The secret deployment code is {NEEDLE}.")
    return "\n".join(s)


Q = "\n\nQuote the secret deployment code from the note above, exactly."


def quality(acc, n):
    """Label from per-token draft acceptance rate (accept_len-1)/n. Thresholds are subjective."""
    if not n or acc != acc:
        return "n/a", None
    r = (acc - 1) / n
    for cut, label in ((0.5, "excellent"), (0.3, "good"), (0.2, "OK"), (0.1, "weak")):
        if r >= cut:
            return label, round(r, 2)
    return "bad", round(r, 2)


def rep(name, rs):
    tps = [round(r["tps"], 1) for r in rs]; acc = [round(r["acc"], 2) for r in rs]
    n = len(rs[0]["pos"]) or A.n; q, rate = quality(st.mean(acc), n)
    print(f"{name:22s} decode t/s {tps} mean={st.mean(tps):.1f} | accept_len {acc} | rate={rate} quality={q}")
    return dict(tps=tps, acc=acc, rate=rate, quality=q, pos=dict(sorted(rs[0]["pos"].items())))


def t_decode():
    o = {}
    rs = [one(CODE, temperature=0.0) for _ in range(2)]; o["code_temp0"] = rep("code temp0 (think)", rs)
    print("   per-pos accepted (run1):", o["code_temp0"]["pos"])
    for t in (0.5, 1.0):
        o[f"code_temp{t:g}"] = rep(f"code temp{t:g} (think)", [one(CODE, temperature=t) for _ in range(2)])
    o["prose"] = rep("prose (think)", [one(PROSE) for _ in range(2)])
    o["code_nothink"] = rep("code non-thinking", [one(CODE, think=False)])
    return o


HIDDEN_COMPLEX = """
import threading, time
from solution import TTLCache
class Clk:
    def __init__(s): s.t = 0.0
    def __call__(s): return s.t
def t_basic():
    c = TTLCache(2, 10, clock=Clk()); c.put('a', 1); assert c.get('a') == 1 and c.get('zz') is None
def t_expiry():
    k = Clk(); c = TTLCache(2, 10, clock=k); c.put('a', 1); k.t = 11
    assert c.get('a') is None and c.stats()['expirations'] == 1
def t_ttl_override():
    k = Clk(); c = TTLCache(2, 100, clock=k); c.put('a', 1, ttl_s=2); k.t = 3; assert c.get('a') is None
def t_lru():
    c = TTLCache(2, 100, clock=Clk()); c.put('a', 1); c.put('b', 2); c.get('a'); c.put('c', 3)
    assert c.get('b') is None and c.get('a') == 1 and c.get('c') == 3 and c.stats()['evictions'] == 1
def t_prefer_expired():
    k = Clk(); c = TTLCache(2, 100, clock=k); c.put('b', 2, ttl_s=5); c.put('a', 1); c.get('b'); k.t = 6; c.put('c', 3)
    assert c.get('a') == 1 and c.get('c') == 3 and c.stats()['evictions'] == 0
def t_delete():
    c = TTLCache(2, 10, clock=Clk()); c.put('a', 1); c.delete('a'); assert c.get('a') is None
def t_stats():
    c = TTLCache(2, 10, clock=Clk()); c.put('a', 1); c.get('a'); c.get('a'); c.get('x'); s = c.stats()
    assert s['hits'] == 2 and s['misses'] == 1
def t_load_once():
    c = TTLCache(4, 10); n = []; out = []
    def ld(): n.append(1); time.sleep(0.05); return 42
    ts = [threading.Thread(target=lambda: out.append(c.get_or_load('k', ld))) for _ in range(16)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(n) == 1 and out == [42] * 16
def t_load_parallel():
    c = TTLCache(8, 10); t0 = time.time()
    ts = [threading.Thread(target=lambda i=i: c.get_or_load(i, lambda: time.sleep(0.2) or i)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert time.time() - t0 < 0.6
"""

HIDDEN_HARDCORE = """
from solution import Database, SQLSyntaxError
def mk():
    d = Database()
    d.execute("CREATE TABLE users (id INT, name TEXT, age INT)")
    d.execute("INSERT INTO users VALUES (1,'ann',30),(2,'bob',25),(3,'cy',NULL),(4,'dee',30)")
    d.execute("CREATE TABLE orders (id INT, user_id INT, amt REAL)")
    d.execute("INSERT INTO orders VALUES (1,1,10.0),(2,1,20.0),(3,2,5.0),(4,9,7.0)")
    return d
def q(sql): return mk().execute(sql).rows
def t_eq(): assert q("select name from users where age = 30") == [('ann',), ('dee',)]
def t_ne_null(): assert q("select name from users where age != 30") == [('bob',)]
def t_is_null(): assert q("select name from users where age is null") == [('cy',)]
def t_not(): assert q("select name from users where not (age = 30)") == [('bob',)]
def t_or_like(): assert q("select name from users where age = 25 or name like 'd%'") == [('bob',), ('dee',)]
def t_null_eq_null(): assert q("select name from users where age = NULL") == []
def t_inner(): assert q("select users.name, orders.amt from users inner join orders on users.id = orders.user_id order by orders.amt") == [('bob', 5.0), ('ann', 10.0), ('ann', 20.0)]
def t_left(): assert q("select users.name, orders.amt from users left join orders on users.id = orders.user_id order by users.name, orders.amt") == [('ann', 10.0), ('ann', 20.0), ('bob', 5.0), ('cy', None), ('dee', None)]
def t_left_where(): assert q("select users.name, orders.amt from users left join orders on users.id = orders.user_id where orders.amt > 8 order by orders.amt") == [('ann', 10.0), ('ann', 20.0)]
def t_group(): assert q("select user_id, count(*), sum(amt) from orders group by user_id order by user_id") == [(1, 2, 30.0), (2, 1, 5.0), (9, 1, 7.0)]
def t_having(): assert q("select user_id, count(*), sum(amt) from orders group by user_id having sum(amt) > 6 order by user_id") == [(1, 2, 30.0), (9, 1, 7.0)]
def t_aggs():
    r = q("select avg(age), min(age), max(age), count(age), count(*) from users")[0]
    assert abs(r[0] - 85 / 3) < 1e-9 and r[1:] == (25, 30, 3, 4)
def t_order_desc(): assert q("select name from users order by age desc, name asc") == [('ann',), ('dee',), ('bob',), ('cy',)]
def t_order_asc_nulls_last(): assert q("select name from users order by age, name") == [('bob',), ('ann',), ('dee',), ('cy',)]
def t_limit(): assert q("select name from users order by id limit 2 offset 1") == [('bob',), ('cy',)]
def t_insert_cols():
    d = mk(); d.execute("insert into users (id, name) values (6,'fay')"); assert d.execute("select age from users where id = 6").rows == [(None,)]
def t_update():
    d = mk(); assert d.execute("update users set age = 31 where age = 30").rowcount == 2 and len(d.execute("select * from users where age = 31").rows) == 2
def t_delete():
    d = mk(); assert d.execute("delete from orders where amt < 8").rowcount == 2 and len(d.execute("select * from orders").rows) == 2
def t_rollback():
    d = mk(); d.execute("begin"); d.execute("insert into users values (5,'eve',40)"); d.execute("delete from orders"); d.execute("rollback")
    assert d.execute("select count(*) from users").rows == [(4,)] and d.execute("select count(*) from orders").rows == [(4,)]
def t_commit():
    d = mk(); d.execute("begin"); d.execute("insert into users values (5,'eve',40)"); d.execute("commit"); assert d.execute("select count(*) from users").rows == [(5,)]
def t_syntax_pos():
    try: mk().execute("SELEC * FROM users")
    except SQLSyntaxError as e: assert e.line == 1 and isinstance(e.col, int) and e.col >= 1
    else: raise AssertionError("no error")
def t_syntax_line3():
    try: mk().execute("SELECT name\\nFROM users\\nWHERE AND")
    except SQLSyntaxError as e: assert e.line == 3
    else: raise AssertionError("no error")
"""

def dump(name, rs):
    d = os.path.join(A.out, "raw"); os.makedirs(d, exist_ok=True)
    for i, r in enumerate(rs):
        open(f"{d}/{A.tag}-{name}-{i}.txt", "w").write(f"finish={r['finish']} toks={r['toks']}\n" + r["content"])


from bench_verify import HARNESS, verify, why  # noqa: E402,F401  (kept in their own file so they can be tested)


def t_complex():
    """Code COMPLEX: TTLCache, verified against hidden tests."""
    o = {}
    for t in (0.0, 0.5, 1.0):
        rs = [one(CODE_COMPLEX, temperature=t, max_tokens=8000, think=A.code_think or False) for _ in range(2)]
        o[f"temp{t:g}"] = rep(f"Code COMPLEX temp{t:g}", rs)
        o[f"temp{t:g}"]["verify"] = [dict(verify(r["content"], HIDDEN_COMPLEX, r["finish"]), finish=r["finish"], toks=r["toks"]) for r in rs]
        dump(f"complex-t{t:g}", rs)
        print("   toks", [r["toks"] for r in rs], "correct:", [f"{v['correct']}/{v['total']}" for v in o[f"temp{t:g}"]["verify"]], "own-tests", [v.get("model_own_tests") for v in o[f"temp{t:g}"]["verify"]])
        for v in o[f"temp{t:g}"]["verify"]:
            if v.get("note") or v.get("failed"): print("      ", v.get("note") or v["failed"])
    return o


def t_hardcore():
    """Code HARDCORE: SQL engine, verified against hidden tests. Passing = actually works, not just written."""
    rs = [one(CODE_HARDCORE, temperature=0.0, max_tokens=32000, think=A.code_think or False) for _ in range(2)]
    o = rep("Code HARDCORE temp0", rs)
    o["completion_toks"] = [x["toks"] for x in rs]
    o["verify"] = [dict(verify(r["content"], HIDDEN_HARDCORE, r["finish"]), finish=r["finish"], toks=r["toks"]) for r in rs]
    dump("hardcore", rs)
    for r, v in zip(rs, o["verify"]):
        print(f"   {r['toks']} toks -> correct {v['correct']}/{v['total']} own-tests={v.get('model_own_tests')} {v.get('note', '')}")
        for k, e in (v.get("failed") or {}).items():
            print("      FAIL", k, e)
    return o


def t_conc():
    out = [None] * A.conc
    def w(i): out[i] = stream(CODE)
    a = metrics(); t0 = time.time()
    ts = [threading.Thread(target=w, args=(i,)) for i in range(A.conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.time() - t0; dr, ac, pos = spec_delta(a, metrics())
    agg = sum(o["toks"] for o in out) / wall
    acc = (1 + ac / dr) if dr else timing_acc(out)
    r = dict(streams=A.conc, per_stream=[round(o["tps"], 1) for o in out], aggregate=round(agg, 1), acc=round(acc, 2) if acc == acc else None)
    r["quality"], r["rate"] = quality(r["acc"] or float("nan"), len(pos) or A.n)
    print(f"{A.conc} concurrent".ljust(22), r)
    return r


def t_prefill():
    rs = []
    for i in range(2):  # distinct doc per run so prefix caching can't skip the prefill
        r = stream(mkdoc(1900, 0.4, seed=100 + i) + Q, 256, think=False)
        rs.append(dict(ptoks=r["ptoks"], ttft=round(r["ttft"], 2), prefill_tps=round(r["ptoks"] / r["ttft"]), needle_ok=NEEDLE in r["content"]))
        print("  prefill", rs[-1])
    return rs


def t_stall():
    res = []
    for rep_i in range(2):
        doc = mkdoc(1900, 0.4, seed=200 + rep_i)
        stamps = []
        ta = threading.Thread(target=lambda: stream("Write a detailed essay of about 900 words on the history of the internet.", 1500, on_chunk=stamps.append))
        ta.start()
        while not stamps:
            time.sleep(0.05)
        time.sleep(3.0)
        tb0 = time.time(); stream(doc + Q, 48, think=False); tb1 = time.time(); ta.join()
        gaps = [(stamps[i] - stamps[i - 1], stamps[i]) for i in range(1, len(stamps))]
        pre = [g for g, t in gaps if t < tb0]; win = [g for g, t in gaps if tb0 <= t <= tb1 + 0.5]
        r = dict(prefill_window_s=round(tb1 - tb0, 1), gap_before_ms=round(st.median(pre) * 1000), gap_during_max_ms=round(max(win) * 1000) if win else None)
        print("  stall", r); res.append(r)
    return res


def t_needle():
    maxlen = json.load(http("/v1/models"))["data"][0].get("max_model_len") or A.ctx
    target = int(maxlen * A.needle_frac) - 400
    n_sent = int(target / 20.0)  # ~20 tok/sentence with this word list; actual count is reported below
    res = []
    for depth in (0.25, 0.75):
        r = stream(mkdoc(n_sent, depth) + Q, 256, think=False)
        res.append(dict(depth=depth, max_len=maxlen, ptoks=r["ptoks"], ttft=round(r["ttft"], 1), ok=NEEDLE in r["content"]))
        print("  needle", res[-1])
    return res


TOOLS = [
    {"type": "function", "function": {"name": "shell", "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]
TOOL_PROMPTS = ["List the files in /var/log.", "What's the weather in {c}?", "Show me the contents of /etc/hostname.",
                "How much disk space is free? Use the shell.", "Read ~/notes/{c}.txt", "Check the weather in {c} in fahrenheit."]
CITIES = ["Oslo", "Lima", "Kyoto", "Cairo", "Perth"]


def t_tools():
    ok = 0; fails = []
    for i in range(A.tool_runs):
        p = TOOL_PROMPTS[i % len(TOOL_PROMPTS)].format(c=CITIES[i % len(CITIES)])
        body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "tools": TOOLS, "tool_choice": "auto", "max_tokens": 1024, "temperature": 0.7}
        m = json.load(http("/v1/chat/completions", body))["choices"][0]
        calls = m["message"].get("tool_calls") or []
        good = False
        try:
            good = bool(calls) and all(isinstance(json.loads(c["function"]["arguments"]), dict) and c["function"]["name"] in {t["function"]["name"] for t in TOOLS} for c in calls)
        except Exception:
            pass
        if good:
            ok += 1
        else:
            fails.append(dict(prompt=p, finish=m["finish_reason"], message=m["message"]))
    print(f"  tool calls valid: {ok}/{A.tool_runs}")
    return dict(valid=ok, total=A.tool_runs, failures=fails)


def t_sanity():
    r = stream(PROSE + " Write at least 700 words.", 1200, think=False)
    w = re.findall(r"\w+", r["content"].lower()); grams = [" ".join(w[i:i + 8]) for i in range(len(w) - 7)]
    dup = 1 - len(set(grams)) / max(len(grams), 1)
    res = dict(words=len(w), repeated_8gram_ratio=round(dup, 3), empty=not r["content"].strip())
    print("  sanity", res, "(ratio >0.1 or empty = suspect)")
    return res


idle_check()
MODEL = json.load(http("/v1/models"))["data"][0]["id"]
print(f"model={MODEL} tag={A.tag}")
for name in A.tests.split(","):
    print(f"\n== {name} ==")
    try:
        OUT[name] = globals()["t_" + name]()
    except Exception as e:
        OUT[name] = {"error": repr(e)}; print("  FAILED:", repr(e))
os.makedirs(A.out, exist_ok=True)
path = os.path.join(A.out, f"{A.tag}.json")
json.dump(OUT, open(path, "w"), indent=1, default=str)
print("\nsaved", path)
