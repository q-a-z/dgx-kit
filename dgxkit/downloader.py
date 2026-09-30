"""Background Hugging Face downloads with a disk check, progress, cancel and resume.

Each download runs in its own subprocess so cancel is a clean kill;
huggingface_hub resumes partial files on the next attempt. A marker file stays
in the model folder until the download completes, and model control refuses
to start a model while it's there.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .control import model_dir

MARKER = ".dgxkit-incomplete"
RESERVE_BYTES = 20 * 2**30


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def fits_on_disk(models_root: str, needed: int, already: int = 0, reserve: int = RESERVE_BYTES) -> bool:
    free = shutil.disk_usage(models_root).free
    return needed - already <= free - reserve


@dataclass
class Job:
    repo: str
    total_bytes: int
    dest: Path
    allow_patterns: list[str] | None = None
    revision: str | None = None
    state: str = "queued"  # queued | running | paused | done | failed | cancelled
    error: str | None = None
    started: float = field(default_factory=time.time)
    proc: asyncio.subprocess.Process | None = None
    _last: tuple[float, int] | None = None

    def progress(self) -> dict:
        have = dir_bytes(self.dest) if self.dest.exists() else 0
        now = time.time()
        speed = None
        if self._last and now > self._last[0]:
            speed = max(0.0, (have - self._last[1]) / (now - self._last[0]))
        self._last = (now, have)
        return {"repo": self.repo, "state": self.state, "error": self.error,
                "bytes": have, "total_bytes": self.total_bytes,
                "pct": round(100 * have / self.total_bytes, 1) if self.total_bytes else None,
                "bytes_per_s": speed}


class Downloader:
    def __init__(self, models_root: str, token: str | None = None):
        self.models_root = models_root
        self.token = token
        self.jobs: dict[str, Job] = {}

    async def start(self, repo: str, total_bytes: int, allow_patterns: list[str] | None = None,
                    revision: str | None = None) -> Job:
        dest = Path(model_dir(self.models_root, repo))
        already = dir_bytes(dest) if dest.exists() else 0
        if not fits_on_disk(self.models_root, total_bytes, already):
            raise RuntimeError("not enough free disk for this download")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / MARKER).touch()
        job = Job(repo, total_bytes, dest, allow_patterns, revision)
        self.jobs[repo] = job
        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: Job):
        args = [sys.executable, "-m", "dgxkit.downloader", job.repo, str(job.dest), job.revision or "",
                ",".join(job.allow_patterns or [])]
        env = {**os.environ, **({"HF_TOKEN": self.token} if self.token else {})}
        job.state = "running"
        job.proc = await asyncio.create_subprocess_exec(*args, env=env, stderr=asyncio.subprocess.PIPE)
        _, err = await job.proc.communicate()
        if job.state in ("cancelled", "paused"):
            return
        if job.proc.returncode == 0:
            (job.dest / MARKER).unlink(missing_ok=True)
            job.state = "done"
        else:
            lines = err.decode(errors="replace").strip().splitlines()
            job.state, job.error = "failed", lines[-1] if lines else "download failed"

    def cancel(self, repo: str):
        job = self.jobs.get(repo)
        if job and job.state == "paused":
            job.state = "cancelled"
        elif job and job.proc and job.state == "running":
            job.state = "cancelled"
            job.proc.kill()  # partial files stay; the next start resumes them

    def pause(self, repo: str) -> bool:
        """Stop the transfer but keep the job; the files so far stay and resume picks up from them."""
        job = self.jobs.get(repo)
        if not (job and job.proc and job.state == "running"):
            return False
        job.state = "paused"
        job.proc.kill()
        return True

    def resume(self, repo: str) -> bool:
        job = self.jobs.get(repo)
        if not (job and job.state == "paused"):
            return False
        job.state = "queued"
        asyncio.create_task(self._run(job))
        return True


def _worker(repo: str, dest: str, revision: str, patterns: str):
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=dest, revision=revision or None,
                      allow_patterns=patterns.split(",") if patterns else None)


if __name__ == "__main__":
    _worker(*sys.argv[1:5])
