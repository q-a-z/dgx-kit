"""Two or more Sparks serving one model: vLLM's own multi-node tensor parallelism over the fabric link.

The head (rank 0) runs here and serves the API, so the gateway publishes it like any other model. Each worker is a headless
vLLM container on another Spark, started over SSH. Setting up the SSH key and the fiber link is the user's job
(docs/CLUSTER.md); this module builds the commands and checks that the requirements hold before anything starts.

A recipe's `cluster` looks like:
    workers: [{host: "q@gx10-2", ip: "192.168.100.11"}]   # SSH target, and its address on the fabric link
    head_ip: 192.168.100.10    interface: enp1s0f1np1    hca: rocep1s0f1    gid_index: 3    master_port: 29501
"""
from __future__ import annotations

import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
MIN_MTU, MIN_SPEED_MBIT = 9000, 100_000


def workers(r) -> list[dict]:
    return list((r.cluster or {}).get("workers") or [])


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
        out.update(NCCL_IB_HCA=c["hca"], NCCL_IB_GID_INDEX=str(c.get("gid_index", 3)))
    return out


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    hard: bool = True  # a soft failure is a warning: it runs, but slower or less safely


def run(host: str | None, script: str, timeout: int = 20) -> tuple[int, str]:
    """A shell script here (host None) or on another node over SSH."""
    cmd = ["sh", "-c", script] if host is None else [*SSH, host, script]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 255, str(e)


def probe(host: str | None, r, models_root: str, run=run) -> dict[str, str]:
    """Facts about one node, in one SSH round trip."""
    c, local = r.cluster, host is None
    i, h = c["interface"], c.get("hca") or ""
    lines = [f"echo state=$(cat /sys/class/net/{i}/operstate 2>/dev/null)",
             f"echo mtu=$(cat /sys/class/net/{i}/mtu 2>/dev/null)",
             f"echo speed=$(cat /sys/class/net/{i}/speed 2>/dev/null)",
             f"echo ip=$(ip -4 -o addr show dev {i} 2>/dev/null | awk '{{print $4}}' | cut -d/ -f1 | head -1)",
             "echo ib=$([ -e /dev/infiniband ] && echo yes || echo no)",
             f"echo hca=$(cat /sys/class/infiniband/{h}/ports/1/state 2>/dev/null)"]
    if not local:  # this box's Docker, image and weights are the ordinary start checks
        from .control import weights_path
        lines += ["echo docker=$(docker info >/dev/null 2>&1 && echo yes || echo no)",
                  f"echo image=$(docker image inspect '{r.image}' >/dev/null 2>&1 && echo yes || echo no)",
                  f"echo weights=$([ -e '{weights_path(r, models_root)}' ] && echo yes || echo no)"]
    rc, out = run(host, "; ".join(lines))
    return {k: v for k, _, v in (ln.partition("=") for ln in out.splitlines())} if rc == 0 else {"error": out or f"exit {rc}"}


def reachable(ip: str, port: int = 22, timeout: float = 2) -> bool:
    try:
        socket.create_connection((ip, port), timeout).close()
        return True
    except OSError:
        return False


def check(r, models_root: str, run=run, ping=reachable, which=shutil.which) -> list[Check]:
    c = r.cluster
    out = [Check("ssh client", bool(which("ssh")), "found" if which("ssh") else "not installed in this container",
                 "rebuild the dgx-kit image (the Dockerfile installs openssh-client and iproute2)")]
    nodes = [("this node", None, c["head_ip"])] + [(w["host"], w["host"], w["ip"]) for w in workers(r)]
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
        out.append(Check(f"{label}: {i} has {ip}", f.get("state") == "up" and f.get("ip") == ip,
                         f"state={f.get('state') or '?'} ip={f.get('ip') or 'none'}",
                         f"cable the fiber link, then give {i} the address {ip}/24 (docs/CLUSTER.md, fiber setup)"))
        mtu = int(f.get("mtu") or 0)
        out.append(Check(f"{label}: MTU {MIN_MTU}", mtu >= MIN_MTU, f"mtu={mtu}", f"set mtu: {MIN_MTU} on {i} on both nodes", hard=False))
        speed = int(f.get("speed") or 0) if (f.get("speed") or "").lstrip("-").isdigit() else 0
        out.append(Check(f"{label}: link speed", speed >= MIN_SPEED_MBIT, f"{speed} Mb/s" if speed > 0 else "unknown",
                         "the link negotiated below 100G: check the cable and the port", hard=False))
        if c.get("hca"):
            ok = f.get("ib") == "yes" and "ACTIVE" in (f.get("hca") or "")
            out.append(Check(f"{label}: RDMA {c['hca']}", ok, f"/dev/infiniband={f.get('ib')} port={f.get('hca') or '?'}",
                             "NCCL falls back to TCP sockets, far slower; check `rdma link` and the hca name", hard=False))
        if host is not None:
            for key, what, fix in (("docker", "Docker works", "give the SSH user access to Docker"),
                                   ("image", f"image {r.image}", "pull or load the same image on that node"),
                                   ("weights", "weights at the same path", "copy the model folder to the same path on that node")):
                out.append(Check(f"{label}: {what}", f.get(key) == "yes", f.get(key) or "?", fix))
            out.append(Check(f"{label}: fabric reachable from here", ping(ip), ip, f"{ip} didn't answer on port 22 over the fabric link"))
    return out


def problems(checks: list[Check]) -> list[str]:
    return [f"{c.name}: {c.detail}. {c.fix}" for c in checks if not c.ok and c.hard]


def render(checks: list[Check]) -> str:
    return "\n".join(f"[{'ok' if c.ok else ('FAIL' if c.hard else 'warn')}] {c.name}"
                     + (f" ({c.detail})" if c.detail else "") + ("" if c.ok else f"\n       fix: {c.fix}") for c in checks)


def report(checks: list[Check]) -> dict:
    return {"ok": not problems(checks), "checks": [asdict(c) for c in checks], "text": render(checks)}
