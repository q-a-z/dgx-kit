"""CPU, memory, storage, network and sensor stats straight from /proc and /sys.

Rates (CPU %, MB/s) need two readings, so SystemCollector keeps the previous
raw counters and returns deltas. `root` lets tests point at fixture files.
"""
from __future__ import annotations

import glob
import re
import os
import time
from pathlib import Path


def parse_meminfo(text: str) -> dict[str, int]:
    """Return /proc/meminfo fields in bytes."""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            out[key] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    return out


def parse_cpu_stat(text: str) -> dict[str, tuple[int, int]]:
    """Return {cpu name: (busy jiffies, total jiffies)} from /proc/stat."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("cpu"):
            continue
        name, *vals = line.split()
        v = [int(x) for x in vals]
        idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
        total = sum(v[:8])  # guest time is already counted in user
        out[name] = (total - idle, total)
    return out


def parse_net_dev(text: str) -> dict[str, tuple[int, int]]:
    """Return {interface: (rx bytes, tx bytes)}, skipping loopback."""
    out = {}
    for line in text.splitlines()[2:]:
        iface, _, rest = line.partition(":")
        iface = iface.strip()
        f = rest.split()
        if iface and iface != "lo" and len(f) >= 9:
            out[iface] = (int(f[0]), int(f[8]))
    return out


def parse_diskstats(text: str) -> dict[str, tuple[int, int, int]]:
    """Return {whole disk: (sectors read, sectors written, ms busy)}."""
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) < 14:
            continue
        dev = f[2]
        # Whole NVMe namespaces and sdX disks only, not partitions.
        is_nvme = dev.startswith("nvme") and "p" not in dev.split("n", 2)[-1]
        is_sd = dev.startswith("sd") and not dev[-1].isdigit()
        if is_nvme or is_sd:
            out[dev] = (int(f[5]), int(f[9]), int(f[12]))
    return out


def rate(now: dict, before: dict, seconds: float, index: int, scale: float = 1.0) -> dict:
    return {
        k: max(0.0, (now[k][index] - before[k][index]) * scale / seconds)
        for k in now if k in before and seconds > 0
    }


def machine_name(vendor: str, product: str) -> str:
    """A short name for the box: "ASUS GX10" from ASUSTeK COMPUTER INC. / GX10."""
    v = re.sub(r"\b(computer|inc|corporation|corp|co|ltd)\b\.?,?", "", vendor, flags=re.I).strip(" ,.")
    v = {"ASUSTeK": "ASUS"}.get(v, v)
    return f"{v} {product}".strip() if v and v.lower() not in product.lower() else product


class SystemCollector:
    def __init__(self, root: str = "/"):
        self.root = Path(root)
        self._prev = None
        self._prev_t = None

    def _read(self, rel: str) -> str:
        return (self.root / rel).read_text()

    def _cpu_freqs(self) -> dict[str, int]:
        freqs = {}
        for p in sorted(glob.glob(str(self.root / "sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))):
            cpu = p.split("/cpufreq")[0].rsplit("/", 1)[-1]
            freqs[cpu] = int(Path(p).read_text()) // 1000  # kHz to MHz
        return freqs

    def _sensors(self) -> list[dict]:
        sensors = []
        for d in sorted(glob.glob(str(self.root / "sys/class/hwmon/hwmon*"))):
            name_file = Path(d, "name")
            chip = name_file.read_text().strip() if name_file.exists() else os.path.basename(d)
            for f in sorted(glob.glob(f"{d}/*_input")):
                kind = os.path.basename(f).split("_")[0]
                label_file = Path(f.replace("_input", "_label"))
                label = label_file.read_text().strip() if label_file.exists() else kind
                try:
                    raw = int(Path(f).read_text())
                except (OSError, ValueError):
                    continue
                if kind.startswith("temp"):
                    sensors.append({"chip": chip, "label": label, "unit": "C", "value": raw / 1000})
                elif kind.startswith("power"):
                    sensors.append({"chip": chip, "label": label, "unit": "W", "value": raw / 1e6})
                elif kind.startswith("fan"):
                    sensors.append({"chip": chip, "label": label, "unit": "rpm", "value": raw})
        return sensors

    def _default_iface(self) -> str | None:
        """Interface of the default route (the box's real uplink, not docker/veth bridges)."""
        try:
            for line in self._read("proc/net/route").splitlines()[1:]:
                f = line.split()
                if len(f) > 2 and f[1] == "00000000":
                    return f[0]
        except OSError:
            pass
        return None

    def machine(self) -> dict | None:
        """Which box this is, from the firmware (DMI): vendor, product, family and BIOS. Read once."""
        if not hasattr(self, "_machine"):
            def dmi(name: str) -> str:
                try:
                    v = self._read(f"sys/class/dmi/id/{name}").strip()
                except OSError:
                    return ""
                return "" if v.lower() in ("", "default string", "to be filled by o.e.m.", "system product name", "none") else v
            vendor, product = dmi("sys_vendor"), dmi("product_name")
            self._machine = {"name": machine_name(vendor, product), "vendor": vendor, "product": product,
                             "family": dmi("product_family"), "bios": dmi("bios_version")} if product else None
        return self._machine

    def _cpu_model(self) -> str | None:
        try:
            for line in self._read("proc/cpuinfo").splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return None

    def sample(self) -> dict:
        t = time.monotonic()
        raw = {
            "cpu": parse_cpu_stat(self._read("proc/stat")),
            "net": parse_net_dev(self._read("proc/net/dev")),
            "disk": parse_diskstats(self._read("proc/diskstats")),
        }
        mem = parse_meminfo(self._read("proc/meminfo"))
        out = {
            "memory": {
                "total_bytes": mem.get("MemTotal"),
                "available_bytes": mem.get("MemAvailable"),
                "used_bytes": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
                "cached_bytes": mem.get("Cached"),
                "swap_used_bytes": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
            },
            "load": [float(x) for x in self._read("proc/loadavg").split()[:3]],
            "cpu_freq_mhz": self._cpu_freqs(),
            "net_iface": self._default_iface(),
            "cpu_model": self._cpu_model(),
            "machine": self.machine(),
            "sensors": self._sensors(),
        }
        if self._prev is not None:
            dt = t - self._prev_t
            busy = {k: v for k, v in raw["cpu"].items()}
            out["cpu_pct"] = {
                k: round(100 * (busy[k][0] - self._prev["cpu"][k][0]) /
                         max(1, busy[k][1] - self._prev["cpu"][k][1]), 1)
                for k in busy if k in self._prev["cpu"]
            }
            out["net_rx_bps"] = rate(raw["net"], self._prev["net"], dt, 0)
            out["net_tx_bps"] = rate(raw["net"], self._prev["net"], dt, 1)
            out["disk_read_bps"] = rate(raw["disk"], self._prev["disk"], dt, 0, 512)
            out["disk_write_bps"] = rate(raw["disk"], self._prev["disk"], dt, 1, 512)
            out["disk_busy_pct"] = {k: min(100.0, v / 10) for k, v in rate(raw["disk"], self._prev["disk"], dt, 2).items()}
        self._prev, self._prev_t = raw, t
        return out
