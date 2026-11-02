import pytest

from dgxkit import cluster
from dgxkit.control import DockerRunner, docker_options, engine_command
from dgxkit.recipes import Recipe
from dgxkit.sizing import Plan
from dgxkit.store import ModelStore

P = Plan(context_tokens=65536, kv_pool_tokens=200000, kv_bytes=10 * 2**30, concurrency=3.05, fits=True)
CL = {"workers": [{"host": "q@gx10-2", "ip": "192.168.100.11"}], "head_ip": "192.168.100.10",
      "interface": "enp1s0f1np1", "hca": "rocep1s0f1", "gid_index": 3}


def recipe(**kw):
    return Recipe(name="m", repo="org/M", image="img:1", cluster=CL, **kw)


def test_head_and_worker_commands_share_everything_but_the_rank():
    cmd = engine_command(recipe(), P, 8100, "/models")
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2" and cmd[cmd.index("--nnodes") + 1] == "2"
    assert cmd[cmd.index("--node-rank") + 1] == "0" and cmd[cmd.index("--master-addr") + 1] == "192.168.100.10"
    w = cluster.worker_command(cmd, 1)
    assert w[w.index("--node-rank") + 1] == "1" and w[-1] == "--headless"
    assert [a for a in w if a not in ("--headless",)][:len(cmd)] == cmd[:cmd.index("--node-rank") + 1] + ["1"] + cmd[cmd.index("--node-rank") + 2:]


def test_a_plain_recipe_is_untouched_and_other_engines_refuse():
    assert "--nnodes" not in engine_command(Recipe(name="m", repo="org/M"), P, 8100, "/models")
    with pytest.raises(ValueError):
        engine_command(recipe(engine="llamacpp", gguf_file="g.gguf"), P, 8100, "/models")


def test_each_node_gets_its_own_address_and_the_rdma_device():
    head = docker_options(recipe(env={"X": "1"}), "/models")
    worker = docker_options(recipe(), "/models", "192.168.100.11")
    assert head["environment"]["VLLM_HOST_IP"] == "192.168.100.10" and head["environment"]["X"] == "1"
    assert worker["environment"]["VLLM_HOST_IP"] == "192.168.100.11"
    assert worker["environment"]["NCCL_SOCKET_IFNAME"] == "enp1s0f1np1" and worker["environment"]["NCCL_IB_HCA"] == "rocep1s0f1"
    assert worker["devices"] == ["/dev/infiniband:/dev/infiniband:rwm"]
    assert "devices" not in docker_options(Recipe(name="m", repo="org/M"), "/models")


GOOD = "state=up\nmtu=9000\nspeed=200000\nip={ip}\nib=yes\nhca=4: ACTIVE\ndocker=yes\nimage=yes\nweights=yes"


def fake_run(overrides=None, ssh_ok=True):
    def run(host, script, timeout=20):
        if script == "true":
            return (0 if ssh_ok else 255), ""
        ip = "192.168.100.10" if host is None else "192.168.100.11"
        text = GOOD.format(ip=ip)
        for k, v in (overrides or {}).get(host, {}).items():
            text = "\n".join(f"{k}={v}" if ln.startswith(k + "=") else ln for ln in text.splitlines())
        return 0, text
    return run


def test_a_healthy_cluster_has_no_problems():
    checks = cluster.check(recipe(), "/models", run=fake_run(), ping=lambda ip: True, which=lambda x: "/usr/bin/ssh")
    assert cluster.problems(checks) == [] and cluster.report(checks)["ok"]


def test_failures_say_what_to_fix_and_soft_ones_only_warn():
    bad = {"q@gx10-2": {"ip": "", "mtu": "1500"}}
    checks = cluster.check(recipe(), "/models", run=fake_run(bad), ping=lambda ip: False, which=lambda x: None)
    probs = "\n".join(cluster.problems(checks))
    assert "ssh client" in probs and "enp1s0f1np1 has 192.168.100.11" in probs and "fabric reachable" in probs
    assert "MTU" not in probs  # a warning, not a blocker
    assert any(c.name.endswith("MTU 9000") and not c.ok and not c.hard for c in checks)
    assert "fix:" in cluster.render(checks)


def test_no_ssh_login_stops_that_node_s_other_checks():
    checks = cluster.check(recipe(), "/models", run=fake_run(ssh_ok=False), ping=lambda ip: True, which=lambda x: "ssh")
    assert any(c.name == "ssh to q@gx10-2" and not c.ok for c in checks)
    assert not any("Docker" in c.name for c in checks)


class Box:
    """A fake Docker: records what ran and what was removed."""
    def __init__(self, fail=False):
        self.containers, self.ran, self.fail, self.removed = self, [], fail, []

    def run(self, image, command, **kw):
        if self.fail:
            raise RuntimeError("boom")
        c = type("C", (), {"remove": lambda me, force=False: self.removed.append(kw["name"])})()
        self.ran.append((kw["name"], command, kw["labels"]))
        return c

    def list(self, all=False, filters=None):
        return []


def test_workers_start_before_the_head_and_a_failed_head_cleans_them_up():
    r, remote, local = recipe(), Box(), Box(fail=True)
    runner = DockerRunner(local)
    runner.remote = lambda host: remote
    runner._ours = lambda name=None: []
    cmd = engine_command(r, P, 8100, "/models")
    with pytest.raises(RuntimeError):
        runner.start(r, cmd, 8100, "/models")
    assert remote.ran[0][0] == "dgxkit-m-w1" and remote.ran[0][1][-1] == "--headless"
    assert remote.removed == ["dgxkit-m-w1"]  # the head failed, so the worker was removed again


def test_the_cluster_survives_a_save(tmp_path):
    st = ModelStore(str(tmp_path))
    st.save(recipe())
    assert st.get("m").cluster == CL
