"""What this machine is made of, and how its firmware and software versions change over time.

Versions come from the firmware (DMI), the kernel, NVML and Docker. Firmware devices and the updates on offer come from
fwupd (see fwupd_probe). Every reading is remembered in <state>/system.json, so a change of version shows up with its date.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

FILE = "system.json"
DBUS = "/run/dbus/system_bus_socket"
KEEP_CHANGES = 200


def _read(root: str, rel: str) -> str:
    try:
        return (Path(root) / rel).read_text().strip()
    except OSError:
        return ""


def inventory(s) -> list[dict]:
    """The versions worth tracking, each {id, group, label, value}."""
    items: list[dict] = []

    def add(id_: str, group: str, label: str, value) -> None:
        if value:
            items.append({"id": id_, "group": group, "label": label, "value": str(value)})

    m = s.sampler.system.machine() or {}
    add("machine", "Machine", "Model", m.get("name"))
    add("family", "Machine", "Family", m.get("family"))
    add("vendor", "Machine", "Vendor", m.get("vendor"))
    add("bios", "Firmware", "BIOS version", m.get("bios"))
    add("bios_date", "Firmware", "BIOS date", m.get("bios_date"))
    add("bios_vendor", "Firmware", "BIOS vendor", m.get("bios_vendor"))
    v = _read(s.root, "proc/version").split()
    add("kernel", "Software", "Kernel", v[2] if len(v) > 2 else "")
    gv = getattr(s.sampler.gpu, "versions", lambda: {})()
    add("driver", "Software", "NVIDIA driver", gv.get("driver"))
    add("cuda", "Software", "CUDA (driver supports)", gv.get("cuda"))
    add("vbios", "Firmware", "GPU VBIOS", gv.get("vbios"))
    try:
        add("docker", "Software", "Docker engine", s.runner.docker.version().get("Version"))
    except Exception:
        pass
    from .app import app_version
    add("dgxkit", "Software", "DGX-kit", app_version())
    return items


def load(state_dir: str) -> dict:
    try:
        return json.loads((Path(state_dir) / FILE).read_text())
    except (OSError, ValueError):
        return {}


def save(state_dir: str, data: dict) -> None:
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    tmp = Path(state_dir) / (FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(Path(state_dir) / FILE)


def track(data: dict, now: dict[str, tuple[str, str]], t: float | None = None) -> dict:
    """Compare what is installed now ({key: (label, value)}) with what was last seen, and note each change.
    Something seen for the first time is only a baseline, not a change."""
    t = t or time.time()
    seen = data.setdefault("values", {})
    changes = data.setdefault("changes", [])
    data.setdefault("since", t)
    for key, (label, value) in now.items():
        was = seen.get(key)
        if was is not None and was["value"] != value:  # something first seen now is a baseline, not a change
            changes.append({"t": t, "key": key, "label": label, "from": was["value"], "to": value})
        seen[key] = {"label": label, "value": value}
    del changes[:-KEEP_CHANGES]
    return data


def refresh(s, firmware: list[dict] | None = None) -> dict:
    """Read the versions now, update the history, and return what is stored."""
    data = load(s.state_dir)
    now = {i["id"]: (i["label"], i["value"]) for i in inventory(s)}
    if firmware is not None:
        data["firmware"] = {"checked": time.time(), "devices": firmware, "error": None}
        now.update({f"fw:{d['id']}": (d["name"], d["version"]) for d in firmware if d.get("id") and d.get("version")})
    track(data, now)
    save(s.state_dir, data)
    return data


def probe_firmware(s) -> list[dict]:
    """fwupd's devices and updates. Starts a throw-away container from DGX-kit's own image with the system D-Bus socket
    (the dashboard container has no access to it); from a source checkout it asks directly."""
    probe = getattr(s, "firmware_probe", None)
    if probe:
        return probe()
    if not os.path.exists(DBUS):
        raise RuntimeError("fwupd isn't reachable: no system D-Bus socket on this machine")
    client = s.runner.docker
    try:
        image = client.containers.get(os.environ.get("DGXKIT_SERVICE", "dgx-kit")).image.id
    except Exception:
        image = None
    if image is None:  # not running in a container: ask from here
        out = subprocess.run([sys.executable, "-m", "dgxkit.fwupd_probe"], capture_output=True, text=True, timeout=90, check=True).stdout
    else:
        out = client.containers.run(
            image, ["python", "-m", "dgxkit.fwupd_probe"], remove=True, network_mode="none",
            security_opt=["apparmor=unconfined"],  # the stock Docker profile may not talk to the system bus
            volumes={DBUS: {"bind": DBUS, "mode": "rw"}}, stdout=True, stderr=False).decode()
    return json.loads(out)


def report(s, data: dict) -> dict:
    """What the System tab shows."""
    fw = data.get("firmware") or {}
    devices = fw.get("devices") or []
    return {
        "items": inventory(s),
        "firmware": {"checked": fw.get("checked"), "error": fw.get("error"), "devices": devices,
                     "updates": sum(1 for d in devices if d.get("updates"))},
        "changes": list(reversed(data.get("changes", [])))[:50],
        "since": data.get("since"),
    }
