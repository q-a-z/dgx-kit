#!/usr/bin/env python3
"""Check which GPU and system stats the Spark dashboard can read, and how fast.

Read-only: it sets nothing. It uses nvidia-ml-py; when that isn't installed it
fetches the package's single pynvml.py file from PyPI into /tmp and uses it from
there, so nothing on the system changes.
Run on the Spark:  python3 tools/spark_gpu_check.py
"""
import glob, io, json, os, shutil, statistics, subprocess, sys, time, urllib.request, zipfile


def load_pynvml():
    try:
        import pynvml
        return pynvml
    except ImportError:
        pass
    cache = "/tmp/dgxkit-nvml"
    if not os.path.exists(os.path.join(cache, "pynvml.py")):
        print("pynvml not installed; fetching nvidia-ml-py from PyPI into", cache)
        meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/nvidia-ml-py/json", timeout=30))
        wheel = next(u["url"] for u in meta["urls"] if u["filename"].endswith(".whl"))
        data = urllib.request.urlopen(wheel, timeout=60).read()
        os.makedirs(cache, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with open(os.path.join(cache, "pynvml.py"), "wb") as f:
                f.write(z.read("pynvml.py"))
    sys.path.insert(0, cache)
    import pynvml
    return pynvml


N = load_pynvml()

N.nvmlInit()
h = N.nvmlDeviceGetHandleByIndex(0)

def events(h):
    fn = getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None) or N.nvmlDeviceGetCurrentClocksThrottleReasons
    return hex(fn(h))

def procs(h):
    return [(p.pid, (p.usedGpuMemory or 0) // 2**20) for p in N.nvmlDeviceGetComputeRunningProcesses(h)]

def mem(h):
    m = N.nvmlDeviceGetMemoryInfo(h)
    return f"used {m.used // 2**20} MiB of {m.total // 2**20} MiB"

CHECKS = [
    ("name", lambda: N.nvmlDeviceGetName(h)),
    ("SM clock now (MHz)", lambda: N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)),
    ("graphics clock now (MHz)", lambda: N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_GRAPHICS)),
    ("SM clock max (MHz)", lambda: N.nvmlDeviceGetMaxClockInfo(h, N.NVML_CLOCK_SM)),
    ("memory clock now (MHz)", lambda: N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM)),
    ("supported memory clocks", lambda: N.nvmlDeviceGetSupportedMemoryClocks(h)),
    ("GPU / memory util (%)", lambda: (lambda u: (u.gpu, u.memory))(N.nvmlDeviceGetUtilizationRates(h))),
    ("temperature (C)", lambda: N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)),
    ("power draw (mW)", lambda: N.nvmlDeviceGetPowerUsage(h)),
    ("energy since load (mJ)", lambda: N.nvmlDeviceGetTotalEnergyConsumption(h)),
    ("power limit (mW)", lambda: N.nvmlDeviceGetEnforcedPowerLimit(h)),
    ("performance state", lambda: N.nvmlDeviceGetPerformanceState(h)),
    ("clock event reasons (bitmask)", lambda: events(h)),
    ("memory info", lambda: mem(h)),
    ("processes (pid, MiB)", lambda: procs(h)),
    ("fan speed (%)", lambda: N.nvmlDeviceGetFanSpeed(h)),
]

def timed(fn, n=20):
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1000)
    return statistics.median(ms), max(ms)

print(f"{'NVML call':32} {'result':45} {'median ms':>9} {'max ms':>7}")
ok = []
for label, fn in CHECKS:
    try:
        val = fn()
        med, worst = timed(fn)
        ok.append(fn)
        print(f"{label:32} {str(val)[:45]:45} {med:9.2f} {worst:7.2f}")
    except N.NVMLError as e:
        print(f"{label:32} {'FAILS: ' + str(e):45}")

sweep = timed(lambda: [f() for f in ok], n=10)
print(f"\nOne full sweep of the {len(ok)} working calls: median {sweep[0]:.1f} ms, max {sweep[1]:.1f} ms")

if shutil.which("nvidia-smi"):
    q = ["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader"]
    med, worst = timed(lambda: subprocess.run(q, capture_output=True), n=5)
    print(f"For comparison, one nvidia-smi query: median {med:.0f} ms, max {worst:.0f} ms")

print("\nSystem sources")
for path in ["/proc/meminfo", "/proc/stat", "/proc/diskstats", "/proc/net/dev",
             "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"]:
    print(f"  {path:55} {'ok' if os.path.exists(path) else 'MISSING'}")
for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
    name = open(f"{d}/name").read().strip() if os.path.exists(f"{d}/name") else "?"
    kinds = sorted({os.path.basename(f).split("_")[0].rstrip("0123456789") for f in glob.glob(f"{d}/*_input")})
    print(f"  hwmon {name:20} {', '.join(kinds) or 'no inputs'}")
cg = glob.glob("/sys/fs/cgroup/system.slice/docker-*.scope/memory.current")
print(f"  docker cgroup memory files: {len(cg)} found")

print("\nDCGM")
if shutil.which("dcgmi"):
    r = subprocess.run(["dcgmi", "dmon", "-e", "1005,1002", "-c", "3"], capture_output=True, text=True, timeout=30)
    print(r.stdout or r.stderr)
else:
    print("  dcgmi not installed")

N.nvmlShutdown()
