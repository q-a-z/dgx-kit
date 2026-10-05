"""An HTTP server for Bekko System One v0 (400M), which ships only a Python class.

It speaks the same /v1/systemone as Laya: a `state` and typed `questions` in, an answer per question out with
calibrated probabilities, so a client can switch between the two by changing the address. The model is read from the
Hugging Face cache folder mounted at /models/bekko (nothing is fetched), and the GPU is used when there is one.

Request:  {"state": text or object, "questions": {name: {"type": "noul" | "choice" | "score", "instructions": ...,
           "criteria": ...}}, "min_confidence": 0.8 (optional)}
          noul    criteria optional: {"yes": "...", "no": "..."} says what each outcome means
          choice  criteria: {option: description or null}, or a list of option names
          score   criteria: the ordered level labels, a list; level i is worth i
Batch:    POST /v1/systemone/batch  {"states": [...], "questions": {...}}
Health:   GET /health
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

MODEL = "hotchpotch/bekko-system-one-v0-400m"
CACHE = Path(os.environ.get("BEKKO_CACHE", "/models/bekko/hub"))
KEY = os.environ.get("BEKKO_API_KEY") or None
PORT = int(os.environ.get("BEKKO_PORT", "8202"))
HOST = os.environ.get("BEKKO_HOST", "0.0.0.0")
MAX_STATES = 256
MAX_QUESTIONS = 64

device = os.environ.get("BEKKO_DEVICE", "cuda")
# A tiny GPU call first: CUDA needs truly free memory to open, and the page cache can hold nearly all of it on the GB10.
# Then run on the CPU (400M parameters: about a second per call) and say so, instead of not starting.
if device.startswith("cuda"):
    try:
        torch.zeros(1, device="cuda")
    except Exception as e:
        print(f"bekko: the GPU is not available ({str(e).splitlines()[0]}); running on the CPU. "
              "Free memory (drop the page cache), then restart it to get the GPU back.", flush=True)
        device = "cpu"


def snapshot() -> str:
    found = sorted((CACHE / ("models--" + MODEL.replace("/", "--")) / "snapshots").glob("*"))
    if not found:
        sys.exit(f"bekko: {MODEL} is not in {CACHE}")
    return str(found[-1])


def load():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    path = snapshot()
    cls = get_class_from_dynamic_module("inference_v0.BekkoSentenceTransformer", path)
    return cls(path, trust_remote_code=True, device=device), Path(path).name


# ---- request -> the model's input

def decision(name: str, q: dict) -> dict:
    """One Laya-style question as one Bekko decision."""
    kind = q.get("type")
    crit = q.get("criteria")
    instructions = str(q.get("instructions") or "")
    if kind == "noul":
        said = crit if isinstance(crit, dict) else {}
        rows = [("false", said.get("no") or "No: the condition does not hold."), ("true", said.get("yes") or "Yes: the condition holds.")]
        values = [None, None]
    elif kind == "choice":
        if isinstance(crit, dict):
            rows = [(str(k), v or str(k)) for k, v in crit.items()]
        elif isinstance(crit, list):
            rows = [(str(k), str(k)) for k in crit]
        else:
            raise ValueError(f"question {name!r}: a choice needs criteria (option names, or option: description)")
        values = [None] * len(rows)
        if len(rows) < 2:
            raise ValueError(f"question {name!r}: a choice needs at least two options")
    elif kind == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError(f"question {name!r}: a score needs a list of at least two level labels")
        rows = [(str(i), str(label)) for i, label in enumerate(crit)]
        values = [float(i) for i in range(len(rows))]
    else:
        raise ValueError(f"question {name!r}: type must be noul, choice or score")
    return {"id": name, "kind": "judgment", "type": kind, "instructions_json": json.dumps(instructions), "system_prompt": "",
            "criteria": [{"id": i, "description_json": json.dumps(d), "value": v} for (i, d), v in zip(rows, values)],
            "documents": [], "scoring": None}


def request_for(state, questions: dict) -> dict:
    return {"state_json": json.dumps(state), "decisions": [decision(n, q) for n, q in questions.items()]}


# ---- the model's output -> Laya-shaped answers

def answer(name: str, q: dict, r: dict, min_confidence: float | None) -> dict:
    kind = q["type"]
    probs = {k: round(float(v), 6) for k, v in r["probabilities"].items()}
    if kind == "noul":
        p = float(r["probability_yes"])
        out = {"type": "noul", "noul": round(p, 6), "confidence": round(max(p, 1 - p), 6)}
    elif kind == "choice":
        sel = r["selected_id"]
        out = {"type": "choice", "choice": sel, "probabilities": probs, "confidence": probs[sel]}
    else:
        out = {"type": "score", "score": round(float(r["score"]), 6), "normalized_score": round(float(r["normalized_score"]), 6),
               "legend": {str(i): str(label) for i, label in enumerate(q["criteria"])}, "probabilities": probs,
               "confidence": max(probs.values())}
    out["answer_confidence"] = out["confidence"]
    if min_confidence is not None:
        low = out["answer_confidence"] < min_confidence
        out.update({"low_confidence": low, "abstention": "abstained" if low else "passed", "abstention_threshold": min_confidence})
    return out


def create_app():
    from fastapi import FastAPI, Header, HTTPException

    model, revision = load()
    lock = threading.Lock()  # one forward at a time
    app = FastAPI(title="bekko-system-one")

    def check(authorization):
        if KEY and authorization != f"Bearer {KEY}":
            raise HTTPException(401, "missing or wrong bearer key")

    def prepared(states, questions, body):
        if not isinstance(questions, dict) or not questions or len(questions) > MAX_QUESTIONS:
            raise HTTPException(400, f"questions must be an object with 1 to {MAX_QUESTIONS} questions")
        if not isinstance(states, list) or not states or len(states) > MAX_STATES:
            raise HTTPException(400, f"states must be a list of 1 to {MAX_STATES}")
        mc = body.get("min_confidence")
        if mc is not None and not (isinstance(mc, (int, float)) and 0 <= mc <= 1):
            raise HTTPException(422, "min_confidence must be a number from 0 to 1")
        try:
            return [request_for(s, questions) for s in states], mc
        except ValueError as e:
            raise HTTPException(422, str(e))

    def run(requests, questions, mc):
        t = time.perf_counter()
        with lock:
            results = model.predict(requests, show_progress_bar=False)
        ms = round((time.perf_counter() - t) * 1000, 1)
        return [{"model": "bekko-system-one-v0-400m", "answers": {n: answer(n, q, r[n], mc) for n, q in questions.items()},
                 "usage": {"output_tokens": 0, "inference_ms": ms},
                 "routing": {"model": "bekko-system-one-v0-400m", "reason": "only checkpoint"}} for r in results]

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": ["bekko-system-one-v0-400m"], "revision": revision, "device": device}

    @app.post("/v1/systemone")
    def systemone(body: dict, authorization: str | None = Header(default=None)):
        check(authorization)
        if "questions" not in body:
            raise HTTPException(400, "request body must be an object with a 'questions' field")
        requests, mc = prepared([body.get("state")], body["questions"], body)
        return run(requests, body["questions"], mc)[0]

    @app.post("/v1/systemone/batch")
    def batch(body: dict, authorization: str | None = Header(default=None)):
        check(authorization)
        if "questions" not in body or "states" not in body:
            raise HTTPException(400, "request body must be an object with 'states' and 'questions' fields")
        requests, mc = prepared(body["states"], body["questions"], body)
        results = run(requests, body["questions"], mc)
        return {"results": results, "total_usage": {"output_tokens": 0}}

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")
