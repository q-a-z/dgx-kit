from pathlib import Path

from dgxkit.collectors.gpu import FakeGpu, decode_events
from dgxkit.collectors.system import SystemCollector, parse_diskstats

ROOT = Path(__file__).parent / "fixtures" / "root"


def test_system_first_sample_has_memory_and_sensors():
    s = SystemCollector(str(ROOT)).sample()
    assert s["memory"]["total_bytes"] == 127535852 * 1024
    assert s["memory"]["used_bytes"] == (127535852 - 30000000) * 1024
    assert s["load"] == [1.5, 1.2, 0.9]
    assert s["cpu_freq_mhz"] == {"cpu0": 2400}
    assert s["sensors"] == [{"chip": "acpitz", "label": "temp1", "unit": "C", "value": 47.5}]
    assert "cpu_pct" not in s  # rates need two samples
    assert s["net_iface"] == "wlP9s9" and s["cpu_model"] is None


def test_system_second_sample_has_rates():
    c = SystemCollector(str(ROOT))
    c.sample()
    s = c.sample()
    assert s["cpu_pct"]["cpu"] == 0.0  # same counters, no change
    assert set(s["net_rx_bps"]) == {"enP7s7"}  # loopback skipped


def test_diskstats_keeps_whole_disks_only():
    text = (ROOT / "proc/diskstats").read_text()
    assert list(parse_diskstats(text)) == ["nvme0n1"]


def test_fake_gpu_matches_gb10_gaps():
    g = FakeGpu().sample()
    assert g.sm_clock_max_mhz == 3003
    assert g.mem_clock_mhz is None and g.fan_pct is None and g.power_limit_w is None


def test_event_bits():
    assert decode_events(0x40 | 0x80) == ["hw_thermal_slowdown", "hw_power_brake"]
    assert decode_events(0) == []


class FakeNvml:
    """Just enough of pynvml to behave like q's GB10 (2026-09-29 check output)."""
    NVML_ERROR_NOT_SUPPORTED = 3
    NVML_CLOCK_SM, NVML_CLOCK_MEM, NVML_TEMPERATURE_GPU = 1, 2, 0

    class NVMLError(Exception):
        def __init__(self, value):
            self.value = value

    def __init__(self):
        self.calls, self.energy = [], 1_000_000

    def _unsupported(self, *a):
        self.calls.append("unsupported")
        raise self.NVMLError(3)

    nvmlInit = nvmlShutdown = lambda self: None
    nvmlDeviceGetHandleByIndex = lambda self, i: "h"
    nvmlDeviceGetName = lambda self, h: "NVIDIA GB10"
    nvmlDeviceGetMaxClockInfo = lambda self, h, c: 3003
    nvmlDeviceGetUtilizationRates = lambda self, h: type("U", (), {"gpu": 5})()
    nvmlDeviceGetTemperature = lambda self, h, t: 42
    nvmlDeviceGetPowerUsage = lambda self, h: 9999
    nvmlDeviceGetCurrentClocksEventReasons = lambda self, h: 0
    nvmlDeviceGetComputeRunningProcesses = lambda self, h: []

    def nvmlDeviceGetClockInfo(self, h, c):
        return 2398 if c == self.NVML_CLOCK_SM else self._unsupported()

    nvmlDeviceGetEnforcedPowerLimit = nvmlDeviceGetFanSpeed = _unsupported

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        self.energy += 30_000  # 30 J per sample
        return self.energy


def test_nvml_power_from_energy_and_unsupported_calls_skipped(monkeypatch):
    from dgxkit.collectors import gpu as G
    fake = FakeNvml()
    g = G.NvmlGpu(nvml=fake)
    t = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(G.time, "monotonic", lambda: next(t))
    first = g.sample()
    assert first.power_w == 9.999  # no energy delta yet: instant reading
    assert first.mem_clock_mhz is None and first.fan_pct is None and first.sm_clock_mhz == 2398
    unsupported_after_first = fake.calls.count("unsupported")
    second = g.sample()
    assert second.power_w == 30.0  # 30 J over 1 s
    assert fake.calls.count("unsupported") == unsupported_after_first  # not retried


def test_missing_nvml_leaves_the_rest_running(monkeypatch):
    from dgxkit.collectors import gpu as G
    monkeypatch.delenv("DGXKIT_FAKE_GPU", raising=False)
    monkeypatch.setattr(G, "NvmlGpu", lambda *a: (_ for _ in ()).throw(OSError("no libnvidia-ml")))
    g = G.open_gpu()
    assert g.sample() is None and "libnvidia-ml" in g.reason


def test_the_idle_power_governor_blip_is_not_throttling():
    assert decode_events(0x4, util_pct=0) == []  # sw_power_cap at idle: normal on the GB10
    assert decode_events(0x4, util_pct=95) == ["sw_power_cap"]  # busy and capped: worth showing
    assert decode_events(0x4) == ["sw_power_cap"]  # load unknown: don't hide it
    assert decode_events(0x40 | 0x4, util_pct=0) == ["hw_thermal_slowdown"]  # a real slowdown always shows
