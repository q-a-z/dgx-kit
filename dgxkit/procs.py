"""Names the processes on the GPU: which container and model each PID belongs to.

A PID maps to its Docker container through /proc/<pid>/cgroup; the container's
labels or command line give the model and context length. Read-only.
"""
from __future__ import annotations

import os
import re

_CONTAINER_ID = re.compile(r"(?:docker[-/]|/)([0-9a-f]{64})")


def container_id(root: str, pid: int) -> str | None:
    try:
        with open(os.path.join(root, "proc", str(pid), "cgroup")) as f:
            m = _CONTAINER_ID.search(f.read())
    except OSError:
        return None
    return m.group(1) if m else None


def command(root: str, pid: int) -> list[str]:
    try:
        with open(os.path.join(root, "proc", str(pid), "cmdline"), "rb") as f:
            return [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return []


class ProcessNamer:
    def __init__(self, root: str = "/"):
        self.root = root
        self.containers: dict[str, dict] = {}  # container id -> {key, model, container, ctx, managed}
        self._pids: dict[int, str | None] = {}

    def set_containers(self, containers: dict[str, dict]) -> None:
        self.containers = containers
        self._pids.clear()  # containers came or went; look PIDs up again

    def __call__(self, proc: dict) -> dict:
        pid = proc["pid"]
        if pid not in self._pids:
            self._pids[pid] = container_id(self.root, pid)
        c = self.containers.get(self._pids[pid] or "")
        if c:
            return {**proc, **c}
        from .discover import _context, parse
        args = command(self.root, pid)
        info = parse(args) if args else None
        model = info and (info["served_name"] or (info["model"] or "").rstrip("/").rsplit("/", 1)[-1] or None)
        return {**proc, "key": None, "model": model, "container": None, "ctx": _context(args) if args else None,
                "managed": False}
