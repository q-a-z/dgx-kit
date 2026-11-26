"""Two or more Sparks serving one model: vLLM's own multi-node tensor parallelism over the fabric link.

The head (rank 0) runs here and serves the API, so the gateway publishes it like any other model. Each worker is a headless
vLLM container on another Spark, started over SSH. Setting up the SSH key and the fiber link is the user's job
(docs/CLUSTER.md); this module builds the commands and checks that the requirements hold before anything starts.

A recipe's `cluster` looks like:
    workers: [{host: "q@gx10-2", ip: "192.168.100.11"}]   # SSH target, and its address on the fabric link
    head_ip: 192.168.100.10    interface: enp1s0f1np1    hca: rocep1s0f1    gid_index: 3    master_port: 29501
"""
from __future__ import annotations

import ipaddress
import os
import shlex
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
MIN_MTU, MIN_SPEED_MBIT = 9000, 100_000


def workers(r) -> list[dict]:
    return list((r.cluster or {}).get("workers") or [])


def worker_ip(worker: dict) -> str:
    """Return a worker's fabric address, accepting the deployed short form.

    Older/live recipes commonly use ``host: q@192.168.100.11`` and omit the
    redundant ``ip`` key.  Cluster checks should not fail before they can
    validate the link.  An explicit ``ip`` remains authoritative; otherwise
    only a literal address embedded in the SSH target is accepted.
    """
    explicit = str(worker.get("ip") or "").strip()
    if explicit:
        return explicit
    host = str(worker.get("host") or "").rsplit("@", 1)[-1].strip()
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return ""


def is_cluster(r) -> bool:
    return bool(workers(r))


def size(r) -> int:
    """Nodes in the cluster, this one included (one GPU per Spark)."""
    return 1 + len(workers(r))


def head_args(r) -> list[str]:
    c, n = r.cluster, size(r)
    return ["--tensor-parallel-size", str(n), "--nnodes", str(n), "--node-rank", "0",
            "--master-addr", c["head_ip"], "--master-port", str(c.get("master_port") or 29501)]


def worker_command(head_cmd: list[str], rank: int) -> list[str]:
    cmd = list(head_cmd)
    cmd[cmd.index("--node-rank") + 1] = str(rank)
    return cmd + ["--headless"]


def env(r, node_ip: str) -> dict:
    """What NCCL and vLLM need to find the fabric link, per node."""
    c = r.cluster
    out = {"VLLM_HOST_IP": node_ip, "NCCL_SOCKET_IFNAME": c["interface"], "GLOO_SOCKET_IFNAME": c["interface"]}
    if c.get("hca"):
        # The RoCE-v2 GID index can differ across the two hosts even when
        # they share the same fabric interface and HCA. Keep gid_index as the
        # backward-compatible fallback for older recipes.
        if node_ip == c.get("head_ip"):
            gid_index = c.get("head_gid_index", c.get("gid_index", 3))
        else:
            gid_index = c.get("worker_gid_index", c.get("gid_index", 3))
        out.update(NCCL_IB_HCA=c["hca"], NCCL_IB_GID_INDEX=str(gid_index))
    return out


def nccl_preflight_code() -> str:
    """Small torch.distributed probe used by the disposable cluster preflight.

    It deliberately does one all-reduce rather than importing vLLM: this isolates
    the fabric/NCCL path from model load time and gives a useful transport check
    before a large multi-node process is started.
    """
    return ("import os,time,torch,torch.distributed as d; "
            "d.init_process_group('nccl'); "
            "x=torch.ones(1,device='cuda'); d.barrier(); t=time.perf_counter(); "
            "d.all_reduce(x); d.barrier(); dt=(time.perf_counter()-t)*1000; "
            "print(f'NCCL_PREFLIGHT rank={d.get_rank()} world={d.get_world_size()} ms={dt:.3f} value={x.item():.0f}',flush=True); "
            "d.destroy_process_group()")


def nccl_preflight_command(r, rank: int) -> list[str]:
    """Return the disposable container command for one cluster rank.

    The caller launches one command per node with host networking and the same
    image/RDMA device used by the model containers.  Keeping this as a pure
    builder makes it safe to test and easy for the service/CLI to reuse.
    """
    if not is_cluster(r):
        raise ValueError("NCCL preflight requires a cluster recipe")
    n = size(r)
    c = r.cluster
    node_ip = c["head_ip"] if rank == 0 else worker_ip(workers(r)[rank - 1])
    if not node_ip:
        raise ValueError(f"worker rank {rank} has no fabric address")
    e = env(r, node_ip)
    e.update(MASTER_ADDR=c["head_ip"], MASTER_PORT=str(c.get("master_port") or 29501),
             WORLD_SIZE=str(n), RANK=str(rank))
    cmd = ["python3", "-c", nccl_preflight_code()]
    # Environment is represented separately so DockerRunner/remote launchers can
    # pass it without shell interpolation.
    docker_env = ["--network", "host", "--ipc", "host", "--device", "/dev/infiniband",
                  "--ulimit", "memlock=-1"]
    for k, v in e.items():
        docker_env += ["--env", f"{k}={v}"]
    return docker_env + cmd


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    hard: bool = True  # a soft failure is a warning: it runs, but slower or less safely


def ensure_ssh(home: str | None = None, conf: str = "/etc/ssh/ssh_config.d/50-dgxkit.conf") -> bool:
    """Point the container's ssh at the user's keys. The service mounts /home read-only, so the keys and known_hosts are
    already visible at DGXKIT_HOME/.ssh; ssh just looks in /root/.ssh. Written once per start of the container."""
    d = Path(home or os.environ.get("DGXKIT_HOME") or "") / ".ssh"
    keys = sorted(p for p in d.glob("id_*") if p.is_file() and p.suffix != ".pub") if d.is_dir() else []
    if not keys:
        return False
    text = "Host *\n" + "".join(f"  IdentityFile {k}\n" for k in keys) + f"  UserKnownHostsFile {d / 'known_hosts'}\n  BatchMode yes\n"
    try:
        if not os.path.exists(conf) or Path(conf).read_text() != text:
            Path(conf).parent.mkdir(parents=True, exist_ok=True)
            Path(conf).write_text(text)
        return True
    except OSError:
        return False


def run(host: str | None, script: str, timeout: int = 20) -> tuple[int, str]:
    """A shell script here (host None) or on another node over SSH."""
    if host is not None:
        ensure_ssh()
    cmd = ["sh", "-c", script] if host is None else [*SSH, "--", host, script]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stderr if p.returncode and p.stderr else p.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 255, str(e)


def probe(host: str | None, r, models_root: str, run=run) -> dict[str, str]:
    """Facts about one node, in one SSH round trip."""
    c, local = r.cluster, host is None
    i, h = shlex.quote(c["interface"]), shlex.quote(c.get("hca") or "")
    image = shlex.quote(r.image)
    lines = [f"echo state=$(cat /sys/class/net/{i}/operstate 2>/dev/null)",
             f"echo mtu=$(cat /sys/class/net/{i}/mtu 2>/dev/null)",
             f"echo speed=$(cat /sys/class/net/{i}/speed 2>/dev/null)",
             f"echo ip=$(ip -4 -o addr show dev {i} 2>/dev/null | awk '{{print $4}}' | cut -d/ -f1 | head -1)",
             "echo ib=$([ -e /dev/infiniband ] && echo yes || echo no)",
             f"echo hca=$(cat /sys/class/infiniband/{h}/ports/1/state 2>/dev/null)"]
    # Inspect the selected image on every node. Comparing the local image ID
    # catches a mutable tag resolving to different patched vLLM contents.
    lines += [f"echo image_id=$(docker image inspect --format '{{{{.Id}}}}' {image} 2>/dev/null)"]
    if not local:  # this box's Docker and weights are the ordinary start checks
        lines += ["echo docker=$(docker info >/dev/null 2>&1 && echo yes || echo no)",
                  f"echo image=$(docker image inspect {image} >/dev/null 2>&1 && echo yes || echo no)",
                  "echo mem_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"]
    rc, out = run(host, "; ".join(lines))
    return {k: v for k, _, v in (ln.partition("=") for ln in out.splitlines())} if rc == 0 else {"error": out or f"exit {rc}"}


WEIGHT_READ_PROBE = """\
import os, runpy, sys
from pathlib import Path

launcher, *paths = sys.argv[1:]
if launcher:
    root = os.geteuid() == 0
    runpy.run_path(launcher)
    if root and os.geteuid() == 0:
        sys.exit("owner launcher did not drop root")
weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
for arg in paths:
    path = Path(arg)
    try:
        if path.is_dir():
            if (path / ".dgxkit-incomplete").exists():
                sys.exit(f"{arg}: download has not finished")
            files = sorted(f for f in path.iterdir() if f.suffix.lower() in
                           weight_suffixes | {".json", ".py", ".jinja", ".model", ".tiktoken"})
            if not any(f.suffix.lower() in weight_suffixes for f in files):
                sys.exit(f"{arg}: no checkpoint shards found")
        else:
            files = [path]
        for file in files:
            with file.open("rb") as stream:
                if not stream.read(1):
                    sys.exit(f"{file}: empty file")
    except OSError as exc:
        sys.exit(str(exc))
"""


def checkpoint_probe(r, models_root: str) -> str:
    from .control import draft_path, mounts, recipe_volumes, weights_path

    paths = [p for p in (weights_path(r, models_root), draft_path(r, models_root)) if p]
    volumes = {**mounts(r, models_root), **recipe_volumes(r)}
    launcher = next((v["bind"] for source, v in volumes.items()
                     if v["bind"] == "/usr/local/bin/vllm" and Path(source).name == "vllm_as_owner.py"), "")
    cmd = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--read-only", "--tmpfs", "/tmp",
           "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HOME=/tmp", "--entrypoint", "python3"]
    if user := (r.docker or {}).get("user"):
        cmd += ["--user", str(user)]
    for source, v in volumes.items():
        target = v["bind"]
        if target == launcher or any(p == target or Path(target) in Path(p).parents for p in paths):
            cmd += ["--mount", f"type=bind,source={source},target={target},readonly"]
    return shlex.join([*cmd, r.image, "-c", WEIGHT_READ_PROBE, launcher, *paths])


def reachable(ip: str, port: int = 22, timeout: float = 2) -> bool:
    try:
        socket.create_connection((ip, port), timeout).close()
        return True
    except OSError:
        return False


def check(r, models_root: str, run=run, ping=reachable, which=shutil.which, *, required_bytes: int = 0) -> list[Check]:
    c = r.cluster
    out = [Check("ssh client", bool(which("ssh")), "found" if which("ssh") else "not installed in this container",
                 "rebuild the dgx-kit image (the Dockerfile installs openssh-client and iproute2)")]
    nodes = [("this node", None, c["head_ip"])]
    for w in workers(r):
        # Keep the check actionable when a hostname is used without a
        # separate fabric address: the SSH check still runs, while the
        # reachability/link checks report the missing address instead of a
        # Python KeyError.
        nodes.append((w.get("host") or "worker", w.get("host"), worker_ip(w)))
    head_image_id = ""
    for label, host, ip in nodes:
        if host is not None:
            rc, _ = run(host, "true")
            out.append(Check(f"ssh to {label}", rc == 0, "" if rc == 0 else "key login failed",
                             f"set up key login from this container to {label}, and add its host key to known_hosts (docs/CLUSTER.md)"))
            if rc != 0:
                continue
        f = probe(host, r, models_root, run)
        if "error" in f:
            out.append(Check(f"{label}: probe", False, f["error"], "the node didn't run the check script"))
            continue
        i = c["interface"]
        if label == "this node":
            head_image_id = f.get("image_id") or ""
        elif head_image_id:
            worker_image_id = f.get("image_id") or ""
            out.append(Check(f"{label}: image matches head", bool(worker_image_id) and worker_image_id == head_image_id,
                             f"head={head_image_id or 'unknown'} worker={worker_image_id or 'unknown'}",
                             f"install the exact image ID used by the head ({r.image}) on {label}"))
        out.append(Check(f"{label}: {i} has {ip or 'an address'}", bool(ip) and f.get("state") == "up" and f.get("ip") == ip,
                         f"state={f.get('state') or '?'} ip={f.get('ip') or 'none'}",
                         f"cable the fiber link, then give {i} the address {ip + '/24' if ip else 'on the fabric link'} (docs/CLUSTER.md, fiber setup)"))
        mtu = int(f.get("mtu") or 0)
        out.append(Check(f"{label}: MTU {MIN_MTU}", mtu >= MIN_MTU, f"mtu={mtu}", f"set mtu: {MIN_MTU} on {i} on both nodes", hard=False))
        speed = int(f.get("speed") or 0) if (f.get("speed") or "").lstrip("-").isdigit() else 0
        out.append(Check(f"{label}: link speed", speed >= MIN_SPEED_MBIT, f"{speed} Mb/s" if speed > 0 else "unknown",
                         "the link negotiated below 100G: check the cable and the port", hard=False))
        if c.get("hca"):
            # /dev/infiniband is mounted into the engine containers, not into this one, so only another node can show it
            ok = (host is None or f.get("ib") == "yes") and "ACTIVE" in (f.get("hca") or "")
            out.append(Check(f"{label}: RDMA {c['hca']}", ok, f"/dev/infiniband={f.get('ib')} port={f.get('hca') or '?'}",
                             "NCCL falls back to TCP sockets, far slower; check `rdma link` and the hca name", hard=False))
        if host is not None:
            checks = [("docker", "Docker works", "give the SSH user access to Docker"),
                      ("image", f"image {r.image}", "pull or load the same image on that node")]
            for key, what, fix in checks:
                out.append(Check(f"{label}: {what}", f.get(key) == "yes", f.get(key) or "?", fix))
            if r.engine == "vllm" and f.get("docker") == f.get("image") == "yes":
                rc, output = run(host, checkpoint_probe(r, models_root), timeout=45)
                out.append(Check(f"{label}: engine can read checkpoint and draft", rc == 0,
                                 output[-800:] if rc else "readable in the engine image",
                                 "check weight and draft mounts, download status, and the engine user's read access"))
            if required_bytes:
                kib = f.get("mem_kib") or ""
                available = int(kib) * 1024 if kib.isdigit() else None
                out.append(Check(f"{label}: available memory", available is not None and available >= required_bytes,
                                 f"{available / 2**30:.1f} GiB available, {required_bytes / 2**30:.1f} GiB needed"
                                 if available is not None else "could not read MemAvailable",
                                 "free memory on this node or reduce the model's context or KV cache"))
            if ip:
                out.append(Check(f"{label}: fabric reachable from here", ping(ip), ip, f"{ip} didn't answer on port 22 over the fabric link"))
    return out


def problems(checks: list[Check]) -> list[str]:
    return [f"{c.name}: {c.detail}. {c.fix}" for c in checks if not c.ok and c.hard]


def render(checks: list[Check]) -> str:
    return "\n".join(f"[{'ok' if c.ok else ('FAIL' if c.hard else 'warn')}] {c.name}"
                     + (f" ({c.detail})" if c.detail else "") + ("" if c.ok else f"\n       fix: {c.fix}") for c in checks)


def report(checks: list[Check]) -> dict:
    return {"ok": not problems(checks), "checks": [asdict(c) for c in checks], "text": render(checks)}
