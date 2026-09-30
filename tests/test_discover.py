from dgxkit.control import LABEL
from dgxkit.discover import discover, parse


def test_vllm_with_dotted_speculative_config_like_q_spark():
    args = ["vllm", "serve", "/models/vllm/laguna-xs-2.1-abliterix-nvfp4", "--port", "8095",
            "--served-model-name", "laguna", "--speculative_config.model", "/models/vllm/laguna-xs-2.1-dflash-nvfp4",
            "--speculative_config.method", "dflash"]
    assert parse(args) == {"engine": "vllm", "port": 8095, "model": "/models/vllm/laguna-xs-2.1-abliterix-nvfp4",
                           "served_name": "laguna", "draft": "/models/vllm/laguna-xs-2.1-dflash-nvfp4"}


def test_json_speculative_config_sglang_llamacpp_and_non_servers():
    v = parse(["python3", "-m", "vllm.entrypoints.openai.api_server", "--model=/m",
               '--speculative-config={"model": "/d", "method": "eagle3"}'])
    assert v["engine"] == "vllm" and v["port"] == 8000 and v["model"] == "/m" and v["draft"] == "/d"
    s = parse(["python3", "-m", "sglang.launch_server", "--model-path", "/m", "--port", "30001"])
    assert (s["engine"], s["port"], s["model"]) == ("sglang", 30001, "/m")
    ll = parse(["/app/llama-server", "-m", "/g/m.gguf", "--port", "8090"])
    assert (ll["engine"], ll["port"], ll["model"]) == ("llamacpp", 8090, "/g/m.gguf")
    assert parse(["litellm", "--port", "4000"]) is None
    assert parse(["postgres"]) is None


class C:
    def __init__(self, name, args, labels=None, network="host", ports=None):
        self.name, self.labels, self.ports = name, labels or {}, ports or {}
        self.id = name.encode().hex()[:64].ljust(64, "0")
        self.attrs = {"Path": args[0], "Args": args[1:], "Config": {"Image": "img"},
                      "HostConfig": {"NetworkMode": network}}


class Client:
    def __init__(self, cs):
        self.containers = self
        self.cs = cs

    def list(self):
        return self.cs


def test_discover_skips_our_containers_and_maps_bridge_ports():
    cs = [C("llm-laguna", ["vllm", "serve", "/m", "--port", "8095"]),
          C("dgxkit-mine", ["vllm", "serve", "/m"], labels={LABEL: "mine"}),
          C("bridged", ["sh", "-c", "vllm serve /x --port 8000"], network="bridge",
            ports={"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8081"}]}),
          C("litellm", ["litellm", "--port", "4000"])]
    found = discover(Client(cs))
    assert [(f["name"], f["port"]) for f in found] == [("bridged", 8081), ("llm-laguna", 8095)]
    assert all(f["managed"] is False for f in found)


def test_gpu_processes_get_model_and_context(tmp_path):
    from dgxkit.discover import container_models
    from dgxkit.procs import ProcessNamer
    cs = [C("llm-laguna", ["vllm", "serve", "/m/laguna-xs", "--served-model-name", "laguna", "--max-model-len", "262144"]),
          C("mine", ["/app/llama-server", "-m", "/x.gguf"], labels={LABEL: "qwen", "dgxkit.meta": '{"context_tokens": 32768}'})]
    namer = ProcessNamer(str(tmp_path))
    namer.set_containers(container_models(Client(cs)))
    for pid, c in ((10, cs[0]), (11, cs[1])):
        (tmp_path / "proc" / str(pid)).mkdir(parents=True)
        (tmp_path / "proc" / str(pid) / "cgroup").write_text(f"0::/system.slice/docker-{c.id}.scope\n")
    (tmp_path / "proc" / "12").mkdir()
    (tmp_path / "proc" / "12" / "cmdline").write_bytes(b"python3\0-m\0sglang.launch_server\0--model-path\0/m/gemma\0--context-length\08192\0")
    a, b, c = namer({"pid": 10, "mem_mib": 1}), namer({"pid": 11, "mem_mib": 2}), namer({"pid": 12, "mem_mib": 3})
    assert (a["model"], a["ctx"], a["container"], a["key"]) == ("laguna", 262144, "llm-laguna", "llm-laguna")
    assert (b["model"], b["ctx"], b["managed"]) == ("qwen", 32768, True)
    assert (c["model"], c["ctx"], c["container"]) == ("gemma", 8192, None)
    assert namer({"pid": 99, "mem_mib": 0})["model"] is None


def test_launch_settings_of_a_container_started_elsewhere():
    from types import SimpleNamespace as NS
    from dgxkit.discover import launch
    c = NS(attrs={"Path": "vllm", "Args": ["serve", "/models/ornith", "--max-model-len", "262144", "--enable-prefix-caching", "--port=8000"],
                  "Config": {"Image": "vllm-spark:0.29-pfxtest", "Env": ["HF_TOKEN=hf_x", "PATH=/bin", "VLLM_USE_V1=1"]},
                  "HostConfig": {"NetworkMode": "host", "RestartPolicy": {"Name": "unless-stopped"}},
                  "Mounts": [{"Source": "/home/user/models", "Destination": "/models"}]})
    l = launch(c)
    assert l["image"] == "vllm-spark:0.29-pfxtest" and l["ports"] == ["host network"]
    assert ["--max-model-len", "262144"] in l["options"] and ["--enable-prefix-caching", None] in l["options"] and ["--port", "8000"] in l["options"]
    assert ["HF_TOKEN", "(hidden)"] in l["env"] and not any(k == "PATH" for k, _ in l["env"])
