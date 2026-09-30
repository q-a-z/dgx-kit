"""Benchmark runs of a running model, using tools/bench.py.

Each run is its own process, logged to state/bench/<id>.log; the result is <id>.json when it finishes.
Runs are found again from those files after a dashboard restart.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "bench.py"
TESTS = "decode,complex,hardcore,conc,prefill,stall,needle,tools,sanity"


class Benchmarks:
    def __init__(self, state_dir: str, script: Path = SCRIPT):
        self.dir = Path(state_dir) / "bench"
        self.script = script
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, model: str, port: int, tests: str = TESTS, conc: int = 2) -> dict:
        if any(r["model"] == model and r["state"] == "running" for r in self.list()):
            raise ValueError(f"{model} is already being benchmarked")
        self.dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{model}-{time.strftime('%Y%m%d-%H%M%S')}"
        cmd = [sys.executable, str(self.script), "--port", str(port), "--tag", run_id, "--conc", str(conc),
               "--tests", tests, "--out", str(self.dir)]
        with open(self.dir / f"{run_id}.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self._procs[run_id] = proc
        meta = {"id": run_id, "model": model, "port": port, "tests": tests, "conc": conc, "started": time.time(), "pid": proc.pid}
        (self.dir / f"{run_id}.meta").write_text(json.dumps(meta))
        return {**meta, "state": "running"}

    def _alive(self, meta: dict) -> bool:
        proc = self._procs.get(meta["id"])
        if proc is not None:
            return proc.poll() is None
        try:
            os.kill(meta["pid"], 0)
            return True
        except OSError:
            return False

    def _view(self, meta_file: Path) -> dict:
        meta = json.loads(meta_file.read_text())
        done = (self.dir / f"{meta['id']}.json").exists()
        state = "done" if done else "running" if self._alive(meta) else "failed"
        return {**meta, "state": state}

    def list(self, model: str | None = None) -> list[dict]:
        runs = [self._view(f) for f in self.dir.glob("*.meta")] if self.dir.exists() else []
        return sorted((r for r in runs if model in (None, r["model"])), key=lambda r: -r["started"])

    def get(self, run_id: str) -> dict | None:
        f = self.dir / f"{Path(run_id).name}.meta"  # a name, never a path
        if not f.exists():
            return None
        run = self._view(f)
        log = self.dir / f"{run['id']}.log"
        run["tail"] = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
        res = self.dir / f"{run['id']}.json"
        run["result"] = json.loads(res.read_text()) if res.exists() else None
        return run

    def stop(self, run_id: str) -> bool:
        run = self.get(run_id)
        if not run or run["state"] != "running":
            return False
        try:
            os.killpg(run["pid"], signal.SIGTERM)
        except OSError:
            return False
        return True
