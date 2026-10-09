"""Update DGX-kit from GitHub, from the page.

The dashboard runs in a container with the Docker socket, so it can do what `dgx-kit update` does: fetch the latest source,
build the image, keep the previous one as `:previous`, and restart itself (the service restarts the container on the new image).
Models and the gateway are separate containers and keep running. Needs no git: the source comes as a tarball.
"""
from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_REPO = "https://github.com/q-a-z/dgx-kit.git"
MAX_SOURCE_BYTES = 200 * 2**20
CHECK_EVERY = {"hourly": 3600, "daily": 24 * 3600, "weekly": 7 * 24 * 3600}  # seconds; "never" is absent


def due(frequency: str, checked: float | None, now: float | None = None) -> bool:
    """Is it time for an automatic look at GitHub? Never when set to never; at once when there has been no look yet."""
    every = CHECK_EVERY.get(frequency)
    if every is None:
        return False
    return not checked or (now or time.time()) - checked >= every


def slug_of(repo: str) -> str:
    m = re.match(r"^https://github\.com/([\w.-]+/[\w.-]+?)(?:\.git)?/?$", repo.strip())
    if not m:
        raise ValueError(f"updates come from a github.com repository, not {repo!r}")
    return m.group(1)


def parse_version(text: str) -> str | None:
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def newer(latest: str | None, current: str | None) -> bool:
    def key(v: str | None):
        return tuple(int(x) for x in re.findall(r"\d+", v or "")[:4])
    return bool(latest and current and key(latest) > key(current))


def first_notes(text: str) -> str:
    """The newest section of RELEASE_NOTES.md (up to the next version heading)."""
    parts = re.split(r"^## ", text, flags=re.M)
    return ("## " + parts[1]).strip() if len(parts) > 1 else ""


@dataclass
class UpdateJob:
    version: str | None
    state: str = "running"  # running | restarting | failed
    step: str = "Starting"
    error: str | None = None
    lines: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def add(self, line: str) -> None:
        line = line.rstrip()
        if line:
            self.lines = (self.lines + [line])[-12:]

    def view(self) -> dict:
        return {"version": self.version, "state": self.state, "step": self.step, "error": self.error,
                "tail": self.lines[-6:], "started": self.started}


class Updater:
    def __init__(self, docker=None, repo: str | None = None, service: str | None = None, state_dir: str | None = None):
        self._docker = docker
        self._file = Path(state_dir) / "update-check.json" if state_dir else None
        self.repo = repo or os.environ.get("DGXKIT_UPDATE_REPO") or DEFAULT_REPO
        self.service = service or os.environ.get("DGXKIT_SERVICE", "dgx-kit")
        self.job: UpdateJob | None = None
        self.cache: dict = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self._file.read_text()) if self._file else {}
        except (OSError, ValueError):
            return {}  # the last look is remembered across restarts, so a restart doesn't cause a new one

    @property
    def docker(self):
        if self._docker is None:
            import docker
            self._docker = docker.from_env()
        return self._docker

    def check(self, timeout: float = 15.0) -> dict:
        """The version on GitHub's main branch, its commit and what's new. Cached in self.cache."""
        slug = slug_of(self.repo)
        out: dict = {"checked": time.time(), "repo": f"https://github.com/{slug}"}
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as c:
                raw = f"https://raw.githubusercontent.com/{slug}/main"
                fresh = {"t": str(int(time.time()))}  # GitHub's CDN keeps raw files up to 5 min; a new query string skips its copy
                p = c.get(f"{raw}/pyproject.toml", params=fresh)
                p.raise_for_status()
                out["version"] = parse_version(p.text)
                n = c.get(f"{raw}/RELEASE_NOTES.md", params=fresh)
                out["notes"] = first_notes(n.text) if n.status_code == 200 else ""
                g = c.get(f"https://api.github.com/repos/{slug}/commits/main", headers={"Accept": "application/vnd.github+json"})
                if g.status_code == 200:
                    out["sha"] = g.json().get("sha", "")[:7]
        except Exception as e:
            out["error"] = f"Couldn't reach GitHub: {type(e).__name__}: {str(e)[:120]}"
        if "error" in out and self.cache.get("version"):  # a failed look keeps the last good answer, and says why
            out = {**self.cache, "error": out["error"], "checked": out["checked"]}
        self.cache = out
        if self._file:
            try:
                self._file.parent.mkdir(parents=True, exist_ok=True)
                self._file.write_text(json.dumps(out))
            except OSError:
                pass
        return out

    def start(self, version: str | None = None) -> UpdateJob:
        if self.job and self.job.state in ("running", "restarting"):
            raise ValueError("an update is already running")
        self.job = UpdateJob(version or self.cache.get("version"))
        threading.Thread(target=self._guard, args=(self.job,), daemon=True).start()
        return self.job

    def _guard(self, job: UpdateJob) -> None:
        try:
            self._run(job)
        except Exception as e:
            job.state, job.error = "failed", str(e)[:400]

    def _own_image_tag(self) -> str:
        try:
            tags = self.docker.containers.get(self.service).image.tags
        except Exception:
            raise RuntimeError(f"can't find this dashboard's own container ({self.service}); update it from a terminal with `dgx-kit update`")
        return tags[0] if tags else "dgx-kit:latest"

    def _run(self, job: UpdateJob) -> None:
        slug = slug_of(self.repo)
        tag = self._own_image_tag()
        with tempfile.TemporaryDirectory(prefix="dgxkit-update-") as tmp:
            job.step = "Downloading the source"
            tar = Path(tmp) / "src.tar.gz"
            got = 0
            with httpx.stream("GET", f"https://codeload.github.com/{slug}/tar.gz/refs/heads/main", timeout=60, follow_redirects=True) as r:
                r.raise_for_status()
                with open(tar, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        got += len(chunk)
                        if got > MAX_SOURCE_BYTES:
                            raise RuntimeError("the source is larger than expected; refusing it")
                        f.write(chunk)
            job.add(f"downloaded {got / 1e6:.1f} MB")
            with tarfile.open(tar) as t:
                t.extractall(tmp, filter="data")  # no paths outside the folder, no links out of it
            roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
            ctx = next((p for p in roots if (p / "Dockerfile").exists()), None)
            if ctx is None:
                raise RuntimeError("the download has no Dockerfile")
            new_version = parse_version((ctx / "pyproject.toml").read_text()) if (ctx / "pyproject.toml").exists() else None
            job.version = new_version or job.version
            job.step = "Keeping the current image for a rollback"
            repo_name, _, _ = tag.rpartition(":")
            try:
                self.docker.images.get(tag).tag(repo_name or tag, "previous")
            except Exception:
                job.add("no current image to keep")
            job.step = "Building the new image"
            for ev in self.docker.api.build(path=str(ctx), tag=tag, rm=True, decode=True, pull=True):
                if "error" in ev:
                    raise RuntimeError(ev["error"])
                if "stream" in ev:
                    job.add(ev["stream"])
        job.step = "Restarting"
        job.state = "restarting"
        threading.Timer(2.0, self._restart).start()  # after this answer has gone out

    def _restart(self) -> None:
        try:  # the service (Restart=always) starts the container again on the new image
            self.docker.containers.get(self.service).stop(timeout=20)
        except Exception as e:
            if self.job:
                self.job.state, self.job.error = "failed", f"built, but couldn't restart: {e}"[:300]
