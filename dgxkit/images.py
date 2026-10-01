"""Images DGX-kit runs: one per engine plus the LiteLLM gateway.

Each has a pinned default. The user can switch any of them to another tag
(kept in the state dir), pull it, or build llama.cpp locally for the GB10
(sm_121). Pulls and builds run in the background and report their state.
Only images DGX-kit set up are ever removed, and never one a container uses.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BUILD_DIR = Path(__file__).resolve().parent.parent / "images"

# Pinned upstream tags, all with arm64 builds (checked on the registries 2026-09-29; vllm-openai v0.29.0 too).
# Whether each runs on the GB10 as-is is part of the M1 test on a Spark.
DEFAULTS = {
    "vllm": os.environ.get("DGXKIT_IMAGE_VLLM", "vllm/vllm-openai:v0.30.0"),
    "sglang": os.environ.get("DGXKIT_IMAGE_SGLANG", "lmsysorg/sglang:v0.5.20-cu130"),
    "llamacpp": os.environ.get("DGXKIT_IMAGE_LLAMACPP", "ghcr.io/ggml-org/llama.cpp:server-cuda"),
    "litellm": os.environ.get("DGXKIT_IMAGE_LITELLM", "ghcr.io/berriai/litellm:main-stable"),
}
CATALOG = DEFAULTS  # kept for callers that only need the default names

FLASHINFER = os.environ.get("DGXKIT_FLASHINFER", "0.7.0")  # what the vLLM builds install; empty keeps the release's own

# Local builds for the GB10. Each uses images/<dir>/Dockerfile with build args and produces one tag.
# vLLM builds are an upstream release plus the patches under images/vllm/patches.
BUILDS: dict[str, dict] = {
    "llamacpp-gb10": {"engine": "llamacpp", "dir": "llamacpp", "tag": "dgx-kit/llamacpp:gb10", "args": {},
                      "about": "llama.cpp compiled for the GB10 (sm_121)"},
    "vllm-0.30-gb10": {"engine": "vllm", "dir": "vllm", "tag": "dgx-kit/vllm:0.30-gb10",
                       "args": {"BASE": os.environ.get("DGXKIT_VLLM_030_BASE", "vllm/vllm-openai:v0.30.0"), "SERIES": "0.30", "FLASHINFER": FLASHINFER},
                       "about": "vLLM 0.30 with the GB10 patches and FlashInfer " + (FLASHINFER or "as shipped")},
    "vllm-0.29-gb10": {"engine": "vllm", "dir": "vllm", "tag": "dgx-kit/vllm:0.29-gb10",
                       "args": {"BASE": os.environ.get("DGXKIT_VLLM_029_BASE", "vllm/vllm-openai:v0.29.0"), "SERIES": "0.29", "FLASHINFER": FLASHINFER},
                       "about": "vLLM 0.29 with the GB10 patches and FlashInfer " + (FLASHINFER or "as shipped")},
}


def build_for(image: str) -> str | None:
    """The local vLLM build that stands in for an image this box doesn't have, matched by release
    (an llmctl conf naming vllm-spark:0.29-pfxtest gets the 0.29 build). None if no build fits."""
    import re
    m = re.search(r"(\d+\.\d+)", image.rpartition(":")[2] or image)
    key = f"vllm-{m.group(1)}-gb10" if m else None
    return key if key in BUILDS else None


def patches_for(build: dict) -> list[str]:
    """Patch files a build would apply: common ones first, then its release's own."""
    if build["dir"] != "vllm":
        return []
    root = BUILD_DIR / "vllm" / "patches"
    names: list[str] = []
    for sub in ("common", build["args"].get("SERIES", "")):
        if sub:
            names += sorted(f"{sub}/{p.name}" for p in (root / sub).glob("*.patch"))
    return names


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


@dataclass
class Job:
    engine: str
    kind: str  # pull | build
    image: str
    state: str = "running"  # running | done | failed
    error: str | None = None
    lines: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    build: str | None = None
    make_default: bool = False
    progress: float | None = None  # 0..1 of the bytes to download, when Docker reports sizes

    def add(self, line: str) -> None:
        line = line.rstrip()
        if line:
            self.lines = (self.lines + [line])[-40:]

    def view(self) -> dict:
        return {"kind": self.kind, "image": self.image, "state": self.state, "error": self.error,
                "tail": self.lines[-8:], "started": self.started, "progress": self.progress}


class ImageManager:
    def __init__(self, client=None, catalog: dict | None = None, state_dir: str | None = None):
        self._client = client
        self.defaults = dict(catalog or DEFAULTS)
        self._file = Path(state_dir) / "images.yaml" if state_dir else None
        self.overrides: dict[str, str] = {}
        if self._file and self._file.exists():
            self.overrides = {k: v for k, v in (yaml.safe_load(self._file.read_text()) or {}).items()
                              if k in self.defaults}
        self.jobs: dict[str, Job] = {}

    @property
    def docker(self):
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    @property
    def catalog(self) -> dict[str, str]:
        return {e: self.current(e)[0] for e in self.defaults}

    def current(self, engine: str) -> tuple[str, str]:
        """(image, why): the user's choice, else the pinned default if it's here, else a matching image already on the box."""
        if engine in self.overrides:
            return self.overrides[engine], "chosen"
        default = self.defaults[engine]
        if self.is_ready(default):
            return default, "default"
        found = self.found(engine)
        return (found, "found") if found else (default, "default")

    def found(self, engine: str) -> str | None:
        """found_now(), remembered for half a minute: every page poll asks, and each answer costs Docker calls."""
        now = time.time()
        cache = self.__dict__.setdefault("_found", {})
        if engine in cache and now - cache[engine][0] < 30:
            return cache[engine][1]
        cache[engine] = (now, self.found_now(engine))
        return cache[engine][1]

    def forget(self) -> None:
        """Drop cached Docker reads after a pull, build or tag change."""
        self.__dict__.pop("_found", None)
        self.__dict__.pop("_tag_cache", None)

    def found_now(self, engine: str) -> str | None:
        """The best image of this engine someone already pulled or built:
        one of the default's release first (vllm-spark:0.30 for vllm-openai:v0.30.0),
        then one a running container uses, then the newest."""
        local = self.local(engine)
        if not local:
            return None
        try:
            # Config.Image is in the list reply; c.image would ask Docker again for every container.
            used = {(c.attrs.get("Config") or {}).get("Image", "") for c in self.docker.containers.list()}
        except Exception:
            used = set()
        import re
        m = re.search(r"(\d+\.\d+)", self.defaults[engine].rpartition(":")[2])
        release = m.group(1) if m else None

        def rank(t: str):
            created = getattr(self, "_created", {}).get(t, "")
            return (bool(release and re.search(rf"(?<![\d.]){re.escape(release)}(?![\d])", t.rpartition(":")[2])), t in used, created)
        return max(local, key=rank)

    def image_for(self, engine: str) -> str:
        return self.catalog[engine]

    def set_image(self, engine: str, image: str | None) -> None:
        """Use another tag for an engine; None goes back to the pinned default."""
        self.forget()
        if engine not in self.defaults:
            raise KeyError(engine)
        if image and (" " in image or len(image) > 256):
            raise ValueError("not an image name")
        if image and image != self.defaults[engine]:
            self.overrides[engine] = image
        else:
            self.overrides.pop(engine, None)
        if self._file:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._file.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump(self.overrides))
            tmp.replace(self._file)

    def is_ready(self, image: str) -> bool:
        try:
            if image in self._tags():
                return True
            self.docker.images.get(image)  # an id or digest the tag list doesn't show
            return True
        except Exception:  # not pulled yet, or Docker is down
            return False

    def _tags(self) -> list[str]:
        """Every image tag on the box; listed once per few seconds, since a page asks many times per load."""
        now = time.time()
        cached = getattr(self, "_tag_cache", None)
        if cached and now - cached[0] < 30:
            return cached[1]
        imgs = self.docker.images.list()
        tags = [t for i in imgs for t in (i.tags or [])]
        self._created = {t: i.attrs.get("Created", "") for i in imgs for t in (i.tags or [])}
        self._tag_cache = (now, tags)
        return tags

    def local(self, engine: str) -> list[str]:
        """Tags on this box that look like this engine, whoever pulled or built them."""
        key = {"llamacpp": "llama"}.get(engine, engine)
        try:
            tags = self._tags()
        except Exception:
            return []
        return sorted(t for t in tags if key in t.lower().rpartition(":")[0])

    def list(self) -> list[dict]:
        out = []
        for e, img in self.catalog.items():
            job = self.jobs.get(e)
            builds = [{"id": b, "image": d["tag"], "about": d["about"], "ready": self.is_ready(d["tag"]),
                       "patches": patches_for(d), "job": self.jobs[b].view() if b in self.jobs else None}
                      for b, d in BUILDS.items() if d["engine"] == e]
            out.append({"engine": e, "image": img, "default": self.defaults[e], "ready": self.is_ready(img),
                        "source": self.current(e)[1],
                        "builds": builds, "job": job.view() if job else None, "local": self.local(e)})
        return out

    def choices(self, engine: str) -> list[str]:
        """Images a model on this engine can run: the engine's current one, its default, its local builds."""
        seen = [self.image_for(engine), self.defaults[engine]] + [d["tag"] for d in BUILDS.values() if d["engine"] == engine]
        return list(dict.fromkeys(seen))

    def busy(self, engine: str) -> bool:
        j = self.jobs.get(engine)
        return bool(j and j.state == "running")

    # Pull and build run on a thread each; the API returns at once and the page polls list().

    def pull(self, engine: str) -> None:
        """Blocking pull of the engine's current image."""
        repo, _, tag = self.image_for(engine).rpartition(":")
        self.docker.images.pull(repo, tag=tag or "latest")
        self.forget()

    def build_for(self, image: str) -> str | None:
        """The local build that makes this tag, if it is one (such as dgx-kit/llamacpp:gb10): those exist only once built here."""
        return next((b for b, d in BUILDS.items() if d["tag"] == image), None)

    def start_pull(self, engine: str) -> Job:
        build = self.build_for(self.image_for(engine))
        if build:  # not in any registry: building is how it gets here
            return self.start_build(build)
        return self._spawn(Job(engine, "pull", self.image_for(engine)), self._pull_job)

    def start_pull_image(self, image: str) -> Job:
        """Pull any tag, such as the one picked in a model's settings; tracked by the tag."""
        if self.build_for(image):
            raise ValueError(f"{image} is built on this box, not pulled from a registry: use Build")
        return self._spawn(Job("", "pull", image), self._pull_job, key=f"image:{image}")

    def image_status(self, image: str) -> dict:
        """Is this tag on the box, and is it being pulled or built (from the model picker or as an engine's image) right now.
        `build` is set when the tag is a local build, which has to be built here rather than pulled."""
        build = self.build_for(image)
        job = self.jobs.get(f"image:{image}") or (self.jobs.get(build) if build else None) \
            or next((j for j in self.jobs.values() if j.image == image), None)
        return {"image": image, "ready": self.is_ready(image), "build": build, "job": job.view() if job else None}

    def start_build(self, build: str, make_default: bool = False) -> Job:
        """Build a local image in the background; make_default also switches its engine to it."""
        if build not in BUILDS:
            raise ValueError(f"no local build called {build}")
        b = BUILDS[build]
        job = Job(b["engine"], "build", b["tag"], build=build, make_default=make_default)
        return self._spawn(job, self._build_job, key=build)

    def _spawn(self, job: Job, target, key: str | None = None) -> Job:
        key = key or job.engine
        if self.busy(key):
            raise ValueError(f"{key} is already being pulled or built")
        self.jobs[key] = job
        threading.Thread(target=self._guard, args=(job, target), daemon=True).start()
        return job

    def _guard(self, job: Job, target) -> None:
        try:
            target(job)
            job.state = "done"
        except Exception as e:
            job.state, job.error = "failed", str(e)
        finally:
            self.forget()

    def _pull_job(self, job: Job) -> None:
        repo, _, tag = job.image.rpartition(":")
        layers: dict[str, str] = {}
        sizes: dict[str, tuple[int, int]] = {}  # layer -> (bytes downloaded, bytes in all)
        for ev in self.docker.api.pull(repo, tag=tag or "latest", stream=True, decode=True):
            if "error" in ev:
                raise RuntimeError(ev["error"])
            if not (ev.get("id") and ev.get("status")):
                continue
            layers[ev["id"]] = ev["status"]
            d = ev.get("progressDetail") or {}
            if ev["status"] == "Downloading" and d.get("total"):
                sizes[ev["id"]] = (d.get("current", 0), d["total"])
            elif ev["status"] in ("Download complete", "Extracting", "Pull complete", "Already exists") and ev["id"] in sizes:
                sizes[ev["id"]] = (sizes[ev["id"]][1], sizes[ev["id"]][1])
            done = sum(1 for st in layers.values() if st in ("Pull complete", "Already exists"))
            text = f"{done} of {len(layers)} layers"
            total = sum(t for _, t in sizes.values())
            if total:
                got = sum(c for c, _ in sizes.values())
                job.progress = round(got / total, 3)
                text = f"{_gb(got)} of {_gb(total)} downloaded · {text}"
            job.lines = [text]
        job.progress = 1.0

    def _build_job(self, job: Job) -> None:
        b = BUILDS[job.build]
        ctx = BUILD_DIR / b["dir"]
        for ev in self.docker.api.build(path=str(ctx), tag=job.image, rm=True, decode=True, pull=True,
                                        buildargs=b["args"] or None):
            if "error" in ev:
                raise RuntimeError(ev["error"])
            if "stream" in ev:
                job.add(ev["stream"])
        if job.make_default:
            self.set_image(job.engine, job.image)

    def pull_missing(self) -> list[str]:
        """Pull every image that isn't on the box yet; returns what failed. Never raises."""
        failed = []
        for engine, image in self.catalog.items():
            if self.is_ready(image):
                continue
            try:
                self.pull(engine)
            except Exception:
                failed.append(image)
        return failed

    def remove_unused(self, in_use: set[str]) -> list[str]:
        """Remove images DGX-kit pulled or built earlier that no engine points at any more."""
        keep = set(self.catalog.values()) | in_use
        ours = set(self.defaults.values()) | {d["tag"] for d in BUILDS.values()} | set(self._history())
        removed = []
        for img in sorted(ours - keep):
            try:
                self.docker.images.remove(img)
                removed.append(img)
            except Exception:
                continue  # not on the box, or still used by some container
        return removed

    def _history(self) -> list[str]:
        return [j.image for j in self.jobs.values()]
