import time

import pytest

from dgxkit.images import ImageManager


class FakeApi:
    def __init__(self, fail=None):
        self.fail = fail

    def pull(self, repo, tag, stream, decode):
        yield {"id": "a", "status": "Downloading"}
        if self.fail:
            yield {"error": self.fail}
        yield {"id": "a", "status": "Pull complete"}

    def build(self, path, tag, rm, decode, pull, buildargs=None):
        self.built = (path, tag, buildargs)
        yield {"stream": "Step 1/9\n"}
        yield {"stream": "Successfully built\n"}


class FakeDocker:
    def __init__(self, have=(), fail=None):
        self.have, self.removed = set(have), []
        self.api = FakeApi(fail)
        outer = self

        class Images:
            def get(self, image):
                if image not in outer.have:
                    raise LookupError(image)

            def remove(self, image):
                if image not in outer.have:
                    raise LookupError(image)
                outer.removed.append(image)

        self.images = Images()


CAT = {"vllm": "vllm/vllm-openai:v1", "llamacpp": "ghcr.io/ggml-org/llama.cpp:server-cuda"}


def wait(m, engine):
    for _ in range(100):
        if not m.busy(engine):
            return m.jobs[engine]
        time.sleep(0.01)
    raise AssertionError("job never finished")


def test_choosing_a_tag_persists_and_default_clears_it(tmp_path):
    m = ImageManager(FakeDocker(), CAT, str(tmp_path))
    m.set_image("vllm", "vllm/vllm-openai:v2")
    assert ImageManager(FakeDocker(), CAT, str(tmp_path)).image_for("vllm") == "vllm/vllm-openai:v2"
    m.set_image("vllm", None)
    assert ImageManager(FakeDocker(), CAT, str(tmp_path)).image_for("vllm") == "vllm/vllm-openai:v1"
    with pytest.raises(ValueError):
        m.set_image("vllm", "bad name; rm -rf")


def test_pull_runs_in_the_background_and_reports(tmp_path):
    m = ImageManager(FakeDocker(), CAT, str(tmp_path))
    m.start_pull("vllm")
    job = wait(m, "vllm")
    assert job.state == "done" and job.lines == ["1 of 1 layers"]
    m = ImageManager(FakeDocker(fail="manifest unknown"), CAT, str(tmp_path))
    m.start_pull("vllm")
    assert wait(m, "vllm").error == "manifest unknown"


def test_build_can_switch_the_engine_to_the_built_image(tmp_path):
    m = ImageManager(FakeDocker(), CAT, str(tmp_path))
    m.start_build("llamacpp-gb10", make_default=True)
    assert wait(m, "llamacpp-gb10").state == "done"
    assert m.image_for("llamacpp") == "dgx-kit/llamacpp:gb10"
    with pytest.raises(ValueError):
        m.start_build("vllm")


def test_patched_vllm_builds_leave_the_engine_default_and_are_choices(tmp_path):
    d = FakeDocker()
    m = ImageManager(d, CAT, str(tmp_path))
    m.start_build("vllm-0.29-gb10")
    assert wait(m, "vllm-0.29-gb10").state == "done"
    path, tag, args = d.api.built
    assert path.endswith("images/vllm") and tag == "dgx-kit/vllm:0.29-gb10" and args["SERIES"] == "0.29"
    assert m.image_for("vllm") == "vllm/vllm-openai:v1"  # only this model's choice changes, not every vLLM model
    assert m.choices("vllm") == ["vllm/vllm-openai:v1", "dgx-kit/vllm:0.30-gb10", "dgx-kit/vllm:0.29-gb10"]
    vllm = next(i for i in m.list() if i["engine"] == "vllm")
    assert [b["id"] for b in vllm["builds"]] == ["vllm-0.30-gb10", "vllm-0.29-gb10"]


def test_clean_only_removes_our_images_nobody_points_at(tmp_path):
    d = FakeDocker(have={"vllm/vllm-openai:v1", "vllm/vllm-openai:v2", "someone/else:1"})
    m = ImageManager(d, CAT, str(tmp_path))
    m.set_image("vllm", "vllm/vllm-openai:v2")
    assert m.remove_unused(in_use=set()) == ["vllm/vllm-openai:v1"]
    assert "someone/else:1" not in d.removed


def test_an_image_already_on_the_box_is_used_instead_of_pulling(tmp_path):
    """A Spark can have vllm-spark:0.30 and a running litellm, not the pinned tags."""
    from types import SimpleNamespace as NS

    class Docker:
        tags = ["vllm-spark:0.29-pfxtest", "vllm-spark:0.30", "vllm-spark:0.29", "eugr/spark-vllm-b12x:latest",
                "docker.litellm.ai/berriai/litellm-database:latest", "docker.litellm.ai/berriai/litellm-database:main-latest",
                "postgres:16"]

        class images:
            @staticmethod
            def list():
                return [NS(tags=[t], attrs={"Created": "2026"}) for t in Docker.tags]

            @staticmethod
            def get(t):
                if t not in Docker.tags:
                    raise LookupError(t)
                return NS(attrs={"Created": "2026"})

        class containers:
            @staticmethod
            def list():
                return [NS(attrs={"Config": {"Image": "vllm-spark:0.29-pfxtest"}}),
                        NS(attrs={"Config": {"Image": "docker.litellm.ai/berriai/litellm-database:main-latest"}})]

    m = ImageManager(Docker(), {"vllm": "vllm/vllm-openai:v0.30.0", "litellm": "ghcr.io/berriai/litellm:main-stable"}, str(tmp_path))
    assert m.current("vllm") == ("vllm-spark:0.30", "found")
    assert m.current("litellm") == ("docker.litellm.ai/berriai/litellm-database:main-latest", "found")
    row = {r["engine"]: r for r in m.list()}
    assert row["vllm"]["ready"] and row["vllm"]["source"] == "found"
    assert "postgres:16" not in row["litellm"]["local"]
    calls = []
    real = Docker.images.list
    Docker.images.list = staticmethod(lambda: calls.append(1) or real())
    for _ in range(5):
        m.list()
    assert len(calls) <= 1  # page polls reuse one Docker listing
    m.set_image("vllm", "vllm-spark:0.29")
    assert m.current("vllm") == ("vllm-spark:0.29", "chosen")


def test_an_image_from_another_machine_maps_to_the_local_build_of_the_same_release():
    from dgxkit.images import BUILDS, FLASHINFER, build_for
    assert build_for("vllm-spark:0.29-pfxtest") == "vllm-0.29-gb10"
    assert build_for("vllm/vllm-openai:v0.30.0") == "vllm-0.30-gb10"
    assert build_for("vllm-spark:0.31") is None and build_for("nothing") is None
    assert FLASHINFER == "0.7.0" and BUILDS["vllm-0.29-gb10"]["args"]["FLASHINFER"] == FLASHINFER


def test_any_tag_can_be_pulled_and_followed_by_its_name(tmp_path):
    m = ImageManager(FakeDocker(), CAT, str(tmp_path))
    assert m.image_status("org/engine:1") == {"image": "org/engine:1", "ready": False, "build": None, "job": None}
    m.start_pull_image("org/engine:1")
    wait(m, "image:org/engine:1")
    st = m.image_status("org/engine:1")
    assert st["job"]["state"] == "done" and st["job"]["kind"] == "pull"
    with pytest.raises(ValueError):  # a second pull of the same tag while one runs is refused
        m.jobs["image:org/engine:1"].state = "running"
        m.start_pull_image("org/engine:1")


def test_a_pull_reports_bytes_and_progress(tmp_path):
    seen = []

    class Api:
        def pull(self, repo, tag, stream, decode):
            yield {"id": "a", "status": "Downloading", "progressDetail": {"current": 500_000_000, "total": 1_000_000_000}}
            yield {"id": "b", "status": "Downloading", "progressDetail": {"current": 0, "total": 1_000_000_000}}
            j = m.jobs["vllm"]
            seen.append((j.progress, list(j.lines)))  # the state after both layers were reported
            yield {"id": "a", "status": "Pull complete"}

    d = FakeDocker()
    d.api = Api()
    m = ImageManager(d, CAT, str(tmp_path))
    job = m.start_pull("vllm")
    wait(m, "vllm")
    assert seen[0] == (0.25, ["0.5 GB of 2.0 GB downloaded · 0 of 2 layers"])
    assert job.state == "done" and job.progress == 1.0


def test_a_local_build_is_built_not_pulled(tmp_path):
    from dgxkit.images import BUILDS
    build, d = next(iter(BUILDS.items()))
    m = ImageManager(FakeDocker(), CAT, str(tmp_path))
    st = m.image_status(d["tag"])
    assert st["build"] == build and st["ready"] is False
    with pytest.raises(ValueError, match="Build"):
        m.start_pull_image(d["tag"])
    m.set_image(d["engine"], d["tag"])
    assert m.start_pull(d["engine"]).kind == "build"  # the engine's own Pull button builds it instead
