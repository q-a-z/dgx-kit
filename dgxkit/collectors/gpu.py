"""GPU stats from one NVML session held open for the life of the service.

Never shells out to nvidia-smi. Every read is tolerant: a value GB10 doesn't
support (memory info, memory clock, fan, power limit) comes back as None and
the rest of the sweep carries on.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass, field

# Clock event reason bits we surface as events (NVML's nvmlClocksEventReason*).
EVENT_BITS = {
    0x0000000000000008: "hw_slowdown",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake",
    0x0000000000000004: "sw_power_cap",
}


@dataclass
class GpuSample:
    name: str | None = None
    sm_clock_mhz: int | None = None
    sm_clock_max_mhz: int | None = None
    mem_clock_mhz: int | None = None
    util_pct: int | None = None
    temp_c: int | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    fan_pct: int | None = None
    events: list[str] = field(default_factory=list)
    processes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decode_events(mask: int | None, util_pct: int | None = None) -> list[str]:
    """Names of the active clock-event reasons.

    The GB10's power governor flags sw_power_cap for a moment even at 4 W idle. That is only worth
    showing while the GPU is busy, where it means the power limit is really holding the clocks down.
    """
    if not mask:
        return []
    return [name for bit, name in EVENT_BITS.items()
            if mask & bit and not (name == "sw_power_cap" and util_pct is not None and util_pct < 30)]


class NvmlGpu:
    """Real NVML reader. Opens NVML once; call close() on shutdown."""

    def __init__(self, index: int = 0, nvml=None):
        if nvml is None:
            import pynvml as nvml
        N = self.N = nvml
        self._unsupported: set = set()  # calls that said NotSupported once are never retried
        self._energy: tuple[float, int] | None = None
        N.nvmlInit()
        self.h = N.nvmlDeviceGetHandleByIndex(index)
        self.name = self._opt(N.nvmlDeviceGetName, self.h)
        # Max clock doesn't change at runtime; read once.
        self.sm_max = self._opt(N.nvmlDeviceGetMaxClockInfo, self.h, N.NVML_CLOCK_SM)

    def _opt(self, fn, *args):
        key = (getattr(fn, "__name__", fn), args[1:])
        if key in self._unsupported:
            return None
        try:
            return fn(*args)
        except self.N.NVMLError as e:
            if getattr(e, "value", None) == getattr(self.N, "NVML_ERROR_NOT_SUPPORTED", 3):
                self._unsupported.add(key)
            return None

    def _power_w(self):
        """Average power since the last sample, from the energy counter.

        On q's GB10 the instant reading came back as exactly 9999 mW with models
        loaded, which looks like a placeholder, while the energy counter moves;
        the counter also averages over the whole interval instead of one instant.
        """
        now = time.monotonic()
        mj = self._opt(self.N.nvmlDeviceGetTotalEnergyConsumption, self.h)
        if mj is not None:
            prev, self._energy = self._energy, (now, mj)
            if prev and now > prev[0] and mj >= prev[1]:
                return round((mj - prev[1]) / 1000 / (now - prev[0]), 1)
        power = self._opt(self.N.nvmlDeviceGetPowerUsage, self.h)
        return power / 1000 if power is not None else None

    def _events(self):
        fn = getattr(self.N, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            self.N.nvmlDeviceGetCurrentClocksThrottleReasons
        return self._opt(fn, self.h)

    def _processes(self):
        procs = self._opt(self.N.nvmlDeviceGetComputeRunningProcesses, self.h) or []
        return [{"pid": p.pid, "mem_mib": (p.usedGpuMemory or 0) // 2**20} for p in procs]

    def sample(self) -> GpuSample:
        N, h = self.N, self.h
        util = self._opt(N.nvmlDeviceGetUtilizationRates, h)
        limit = self._opt(N.nvmlDeviceGetEnforcedPowerLimit, h)
        return GpuSample(
            name=self.name,
            sm_clock_mhz=self._opt(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_SM),
            sm_clock_max_mhz=self.sm_max,
            mem_clock_mhz=self._opt(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_MEM),
            util_pct=util.gpu if util else None,
            temp_c=self._opt(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU),
            power_w=self._power_w(),
            power_limit_w=limit / 1000 if limit is not None else None,
            fan_pct=self._opt(N.nvmlDeviceGetFanSpeed, h),
            events=decode_events(self._events(), util.gpu if util else None),
            processes=self._processes(),
        )

    def close(self):
        self._opt(self.N.nvmlShutdown)


class FakeGpu:
    """Simulated GB10 for development away from real hardware.

    Mirrors what GB10 reports: memory clock, power limit and fan are None.
    """

    name = "NVIDIA GB10 (simulated)"

    def sample(self) -> GpuSample:
        util = random.randint(0, 100)
        return GpuSample(
            name=self.name,
            sm_clock_mhz=2400 + random.randint(-80, 80),
            sm_clock_max_mhz=3003,
            util_pct=util,
            temp_c=45 + util // 5,
            power_w=round(25 + util * 0.8, 1),
        )

    def close(self):
        pass


class NoGpu:
    """Stands in when NVML can't be loaded, so the rest of the dashboard still runs."""

    def __init__(self, reason: str):
        self.reason = reason

    def sample(self):
        return None

    def close(self):
        pass


def open_gpu():
    if os.environ.get("DGXKIT_FAKE_GPU") == "1":
        return FakeGpu()
    try:
        return NvmlGpu(int(os.environ.get("DGXKIT_GPU_INDEX", "0")))
    except Exception as e:  # no driver, no libnvidia-ml, or no GPU
        return NoGpu(str(e))
