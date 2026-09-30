import yaml

from dgxkit.gateway import Gateway, bundled_key, litellm_config


class FakeContainer:
    def __init__(self, status="running"):
        self.status, self.restarts, self.removed = status, 0, False

    def restart(self, timeout=10):
        self.restarts += 1

    def remove(self, force=False):
        self.removed = True


class FakeDocker:
    def __init__(self, container=None, image_present=True):
        self.container, self.image_present, self.runs = container, image_present, []
        outer = self

        class Containers:
            def list(self, all=False, filters=None):
                return [outer.container] if outer.container else []

            def run(self, image, cmd, **kw):
                outer.runs.append((image, cmd, kw))
                outer.container = FakeContainer()

        class Images:
            def get(self, image):
                if not outer.image_present:
                    raise LookupError(image)

        self.containers, self.images = Containers(), Images()


def test_config_routes_each_model_to_its_engine_port():
    cfg = litellm_config({"qwen": 8101, "llama": 8100})
    assert [m["model_name"] for m in cfg["model_list"]] == ["llama", "qwen"]
    assert cfg["model_list"][1]["litellm_params"] == {
        "model": "openai/qwen", "api_base": "http://127.0.0.1:8101/v1", "api_key": "none"}
    assert cfg["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"


def test_first_sync_starts_the_container_with_the_config_folder(tmp_path):
    d = FakeDocker()
    g = Gateway(str(tmp_path), port=45999, master_key="sk-test", client=d)
    g.sync({"llama": 8100})
    image, cmd, kw = d.runs[0]
    assert cmd == ["--config", "/app/dgxkit/config.yaml", "--host", "0.0.0.0", "--port", "45999"]
    assert kw["volumes"] == {str(tmp_path / "litellm"): {"bind": "/app/dgxkit", "mode": "ro"}}
    assert kw["environment"] == {"LITELLM_MASTER_KEY": "sk-test", "NUM_WORKERS": "1", "LITELLM_LOG": "ERROR",
                              "LITELLM_DISABLE_NO_REDIS_WARNING": "true"}
    assert kw["mem_limit"] == "4g"
    written = yaml.safe_load((tmp_path / "litellm" / "config.yaml").read_text())
    assert written["model_list"][0]["model_name"] == "llama"
    assert g.problem is None


def test_restarts_only_when_the_model_list_changes(tmp_path):
    c = FakeContainer()
    g = Gateway(str(tmp_path), master_key="k", client=FakeDocker(c))
    g.sync({"llama": 8100})
    g.sync({"llama": 8100})
    assert c.restarts == 1
    g.sync({})
    assert c.restarts == 2


def test_explains_why_it_cannot_start(tmp_path):
    g = Gateway(str(tmp_path), master_key=None, client=FakeDocker())
    g.sync({})
    assert "key" in g.problem
    g = Gateway(str(tmp_path), port=45998, master_key="k", client=FakeDocker(image_present=False))
    g.sync({})
    assert "isn't pulled" in g.problem


def test_status_finds_a_litellm_started_outside_dgx_kit(tmp_path):
    class Ext:
        name = "litellm-proxy"
        status = "running"
        image = type("I", (), {"tags": ["ghcr.io/berriai/litellm:main-v1.80"]})()
        attrs = {"NetworkSettings": {"Ports": {"4000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "4001"}]}}}

    class Docker:
        class containers:
            @staticmethod
            def list(all=False, filters=None):
                return [] if filters else [Ext()]

    g = Gateway(str(tmp_path), port=4000, client=Docker())
    g.problem = "not started"
    s = g.status()
    assert (s["state"], s["external"], s["port"], s["problem"]) == ("running", "litellm-proxy", 4001, None)


def test_publishing_to_an_existing_litellm_touches_only_our_entries():
    import httpx
    from dgxkit.gateway import sync_remote

    models = [
        {"model_name": "theirs", "litellm_params": {"api_base": "http://x:1/v1"}, "model_info": {"id": "abc"}},
        {"model_name": "old", "litellm_params": {"api_base": "http://h:8100/v1"}, "model_info": {"id": "dgxkit-old"}},
        {"model_name": "keep", "litellm_params": {"api_base": "http://h:8101/v1"}, "model_info": {"id": "dgxkit-keep"}},
    ]
    calls = []

    def handler(req: httpx.Request):
        assert req.headers["authorization"] == "Bearer sk-1"
        calls.append((req.method, req.url.path, req.content.decode()))
        return httpx.Response(200, json={"data": models} if req.url.path == "/model/info" else {})

    problem = sync_remote("http://gw:4000/v1", "sk-1", {"keep": 8101, "new": 8102}, "h",
                          client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert problem is None
    posted = [(p, b) for m, p, b in calls if m == "POST"]
    assert [p for p, _ in posted] == ["/model/delete", "/model/new"]
    assert '"dgxkit-old"' in posted[0][1] and '"dgxkit-new"' in posted[1][1] and "http://h:8102/v1" in posted[1][1]


def test_rejected_key_is_reported():
    import httpx
    from dgxkit.gateway import sync_remote
    c = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    assert sync_remote("http://gw:4000/v1", "bad", {}, "h", client=c) == "the gateway rejected the key"


def test_bundled_key_is_made_once_and_kept_private(tmp_path):
    k = bundled_key(str(tmp_path))
    assert k.startswith("sk-dgxkit-") and bundled_key(str(tmp_path)) == k
    assert oct((tmp_path / "litellm" / "master.key").stat().st_mode & 0o777) == "0o600"


def test_config_turns_off_the_caches_that_grow():
    ls = litellm_config({})["litellm_settings"]
    assert ls["disable_cache"] and ls["disable_spend_logs"]


class DbDocker:
    """Two containers by label: LiteLLM's and its Postgres."""

    def __init__(self, gateway=None):
        self.have = {"dgxkit.gateway": gateway, "dgxkit.gateway-db": None}
        self.runs = []
        outer = self

        class Ready(FakeContainer):
            def exec_run(self, cmd):
                return (0, b"")

        class Containers:
            def list(self, all=False, filters=None):
                c = outer.have.get((filters or {}).get("label"))
                return [c] if c else []

            def run(self, image, cmd, **kw):
                outer.runs.append((kw["name"], kw))
                c = Ready()
                outer.have[next(iter(kw["labels"]))] = c
                return c

        class Images:
            def get(self, image):
                pass

        self.containers, self.images = Containers(), Images()


def test_postgres_starts_first_and_litellm_gets_its_address(tmp_path):
    d = DbDocker()
    Gateway(str(tmp_path), port=45997, master_key="k", client=d, db=True).sync({})
    assert [n for n, _ in d.runs] == ["dgxkit-gateway-db", "dgxkit-gateway"]
    db, gw = d.runs[0][1], d.runs[1][1]
    assert db["volumes"] and db["mem_limit"] == "1g"
    pw = (tmp_path / "litellm" / "db.password").read_text()
    assert gw["environment"]["DATABASE_URL"] == f"postgresql://litellm:{pw}@127.0.0.1:5433/litellm"


def test_a_litellm_made_before_the_database_is_replaced(tmp_path):
    old = FakeContainer()
    old.labels = {"dgxkit.gateway": "1"}
    d = DbDocker(gateway=old)
    import socket
    held = socket.socket(); held.bind(("0.0.0.0", 0)); held.listen()  # the old container's port is taken
    Gateway(str(tmp_path), port=held.getsockname()[1], master_key="k", client=d, db=True).sync({})
    assert old.removed and [n for n, _ in d.runs] == ["dgxkit-gateway-db", "dgxkit-gateway"]


def test_status_reports_each_part_of_the_bundled_stack(tmp_path):
    d = DbDocker()
    g = Gateway(str(tmp_path), port=45995, master_key="k", client=d, db=True)
    assert g.status()["db"] == "missing" and g.status()["key_ready"]
    g.sync({})
    assert g.status()["db"] == "running" and g.status()["state"] == "running"
