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
    return Recipe(**({"name": "m", "repo": "org/M", "image": "img:1", "cluster": CL} | kw))


def test_head_and_worker_commands_share_everything_but_the_rank():
    cmd = engine_command(recipe(), P, 8100, "/models")
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2" and cmd[cmd.index("--nnodes") + 1] == "2"
    assert cmd[cmd.index("--node-rank") + 1] == "0" and cmd[cmd.index("--master-addr") + 1] == "192.168.100.10"
    w = cluster.worker_command(cmd, 1)
    assert w[w.index("--node-rank") + 1] == "1" and w[-1] == "--headless"
    assert [a for a in w if a not in ("--headless",)][:len(cmd)] == cmd[:cmd.index("--node-rank") + 1] + ["1"] + cmd[cmd.index("--node-rank") + 2:]


def test_nccl_preflight_uses_cluster_transport_and_rank_settings():
    head = cluster.nccl_preflight_command(recipe(), 0)
    worker = cluster.nccl_preflight_command(recipe(), 1)
    assert head[-2:] == ["-c", cluster.nccl_preflight_code()]
    assert "--env" in head and "RANK=0" in head and "WORLD_SIZE=2" in head
    assert "--ulimit" in head and "memlock=-1" in head and "/dev/infiniband" in head
    assert "RANK=1" in worker and "VLLM_HOST_IP=192.168.100.11" in worker
    assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in worker and "NCCL_IB_HCA=rocep1s0f1" in worker


def test_nccl_preflight_rejects_non_cluster_and_missing_worker_ip():
    with pytest.raises(ValueError, match="requires a cluster"):
        cluster.nccl_preflight_command(Recipe(name="m", repo="org/M"), 0)
    r = recipe(cluster={**CL, "workers": [{"host": "q@gx10-2"}]})
    with pytest.raises(ValueError, match="no fabric address"):
        cluster.nccl_preflight_command(r, 1)


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
    assert worker["devices"] == [f"/dev/infiniband/{name}:/dev/infiniband/{name}:rwm" for name in
                                 ("rdma_cm", "uverbs0", "uverbs1", "uverbs2", "uverbs3")]
    assert worker["cap_add"] == ["IPC_LOCK"]
    assert "devices" not in docker_options(Recipe(name="m", repo="org/M"), "/models")


GOOD = "state=up\nmtu=9000\nspeed=200000\nip={ip}\nib=yes\nhca=4: ACTIVE\nimage_id=sha256:test-image\ndocker=yes\nimage=yes\nmem_kib=83886080"


def fake_run(overrides=None, ssh_ok=True):
    def run(host, script, timeout=20):
        if script == "true":
            return (0 if ssh_ok else 255), ""
        if script.startswith("docker run"):
            failure = (overrides or {}).get(host, {}).get("checkpoint_error")
            return (1, failure) if failure else (0, "")
        ip = "192.168.100.10" if host is None else "192.168.100.11"
        text = GOOD.format(ip=ip)
        for k, v in (overrides or {}).get(host, {}).items():
            text = "\n".join(f"{k}={v}" if ln.startswith(k + "=") else ln for ln in text.splitlines())
        return 0, text
    return run


def test_a_healthy_cluster_has_no_problems():
    checks = cluster.check(recipe(), "/models", run=fake_run(), ping=lambda ip: True, which=lambda x: "/usr/bin/ssh")
    assert cluster.problems(checks) == [] and cluster.report(checks)["ok"]


def test_worker_preflight_reports_checkpoint_permission_and_memory():
    overrides = {"q@gx10-2": {"checkpoint_error": "Permission denied: /models/draft/model.safetensors",
                             "mem_kib": str(8 * 2**20)}}
    checks = cluster.check(recipe(), "/models", run=fake_run(overrides),
                           ping=lambda ip: True, which=lambda x: "ssh", required_bytes=10 * 2**30)
    problems = cluster.problems(checks)
    assert any("Permission denied: /models/draft/model.safetensors" in p for p in problems)
    assert any("8.0 GiB available, 10.0 GiB needed" in p for p in problems)


def test_checkpoint_probe_uses_engine_image_user_and_mounts(tmp_path):
    import shlex
    import subprocess
    import sys

    target = tmp_path / "draft"
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"weights")
    r = recipe(path=str(target), draft_path=str(target), docker={
        "user": "1000:1000",
        "volumes": ["/home/q/vllm_as_owner.py:/usr/local/bin/vllm:ro"],
    })
    args = shlex.split(cluster.checkpoint_probe(r, str(tmp_path)))
    assert args[args.index("--user") + 1] == "1000:1000"
    assert "type=bind,source=/home/q/vllm_as_owner.py,target=/usr/local/bin/vllm,readonly" in args
    assert args[args.index("img:1") + 3] == "/usr/local/bin/vllm"
    probe = [sys.executable, "-c", cluster.WEIGHT_READ_PROBE, "", str(target)]
    assert subprocess.run(probe, capture_output=True).returncode == 0
    (target / ".dgxkit-incomplete").touch()
    failed = subprocess.run(probe, capture_output=True, text=True)
    assert failed.returncode and "download has not finished" in failed.stderr


def test_worker_fabric_ip_can_be_derived_from_ssh_target():
    r = recipe(cluster={**CL, "workers": [{"host": "q@192.168.100.11"}]})
    checks = cluster.check(r, "/models", run=fake_run(), ping=lambda ip: ip == "192.168.100.11",
                           which=lambda x: "/usr/bin/ssh")
    assert cluster.worker_ip(r.cluster["workers"][0]) == "192.168.100.11"
    assert cluster.problems(checks) == []


def test_a_worker_with_a_different_image_id_is_blocked():
    checks = cluster.check(recipe(), "/models", run=fake_run({"q@gx10-2": {"image_id": "sha256:other"}}),
                           ping=lambda ip: True, which=lambda x: "/usr/bin/ssh")
    mismatch = next(c for c in checks if c.name == "q@gx10-2: image matches head")
    assert not mismatch.ok and mismatch.hard and "sha256:other" in mismatch.detail
    assert any("image matches head" in p for p in cluster.problems(checks))


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


def test_start_will_not_replace_an_active_worker():
    from types import SimpleNamespace

    r, remote, local = recipe(), Box(), Box()
    remote.list = lambda **kw: [SimpleNamespace(status="running")]
    runner = DockerRunner(local)
    runner.remote = lambda host: remote
    with pytest.raises(ValueError, match="still active"):
        runner.start(r, engine_command(r, P, 8100, "/models"), 8100, "/models")
    assert not remote.ran and not local.ran


def test_worker_start_derives_fabric_ip_from_literal_ssh_target():
    r = recipe(cluster={**CL, "workers": [{"host": "q@192.168.100.11"}]})
    remote, local = Box(), Box()
    runner = DockerRunner(local)
    runner.remote = lambda host: remote
    runner._ours = lambda name=None: []
    runner.start(r, engine_command(r, P, 8100, "/models"), 8100, "/models")
    assert remote.ran and remote.ran[0][2]["dgxkit.worker"] == "m"


def test_the_cluster_survives_a_save(tmp_path):
    st = ModelStore(str(tmp_path))
    st.save(recipe())
    assert st.get("m").cluster == CL


def test_ssh_is_pointed_at_the_users_keys_only_when_there_are_some(tmp_path):
    conf = tmp_path / "conf.d" / "50.conf"
    assert not cluster.ensure_ssh(str(tmp_path), str(conf)) and not conf.exists()  # no ~/.ssh at all
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "id_ed25519").write_text("k")
    (ssh / "id_ed25519.pub").write_text("p")
    (ssh / "authorized_keys").write_text("a")
    assert cluster.ensure_ssh(str(tmp_path), str(conf))
    text = conf.read_text()
    assert f"IdentityFile {ssh}/id_ed25519\n" in text and "id_ed25519.pub" not in text and "authorized_keys" not in text
    assert f"UserKnownHostsFile {ssh}/known_hosts" in text and "BatchMode yes" in text


def test_this_nodes_missing_infiniband_device_is_not_a_warning():
    checks = cluster.check(recipe(), "/models", run=fake_run({None: {"ib": "no"}}), ping=lambda ip: True, which=lambda x: "ssh")
    assert all(c.ok for c in checks if "RDMA" in c.name)
    checks = cluster.check(recipe(), "/models", run=fake_run({"q@gx10-2": {"ib": "no"}}), ping=lambda ip: True, which=lambda x: "ssh")
    assert any("RDMA" in c.name and not c.ok for c in checks)  # but a worker without it is


def test_a_worker_connection_error_says_which_node_and_why(monkeypatch):
    import docker

    def boom(*a, **k):
        raise docker.errors.DockerException("Install paramiko package to enable ssh:// support")
    monkeypatch.setattr(docker, "DockerClient", boom)
    with pytest.raises(ValueError, match="q@gx10-2.*paramiko"):
        DockerRunner(Box()).remote("q@gx10-2")
