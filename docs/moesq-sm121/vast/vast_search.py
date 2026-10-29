import json, urllib.request, sys
URL = "https://console.vast.ai/api/v0/bundles/"
# (gpu_name as vast lists it, assumed speedup vs the Spark for this workload, min VRAM GB)
CANDS = [("H100 SXM", 10), ("H100 NVL", 10), ("H100 PCIE", 8), ("H200", 10), ("A100 SXM4", 5), ("A100 PCIE", 4.5),
         ("L40S", 3.5), ("RTX PRO 6000 S", 6.5), ("RTX PRO 6000 WS", 6.5), ("RTX 5090", 3.5), ("RTX 4090", 3.5), ("B200", 18)]
SPARK_H = 52.0   # estimated Spark-hours for the 35B medium config
OVERHEAD = 1.4   # setup, download, pilot, retry margin
def query(name):
    body = {"verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False},
            "gpu_name": {"eq": name}, "num_gpus": {"eq": 1}, "type": "on-demand",
            "reliability2": {"gte": 0.98}, "inet_down": {"gte": 500}, "disk_space": {"gte": 200},
            "cpu_ram": {"gte": 64000}, "cuda_max_good": {"gte": 12.8}, "allocated_storage": 200,
            "order": [["dph_total", "asc"]], "limit": 4}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=40)).get("offers", [])
rows = []
for name, sp in CANDS:
    try:
        offers = query(name)
    except Exception as e:
        print(name, "ERR", e); continue
    for o in offers[:3]:
        hrs = SPARK_H / sp
        rows.append((o["dph_total"] * hrs * OVERHEAD, name, o, hrs, sp))
rows.sort(key=lambda r: r[0])
print(f"{'est $':>6} {'gpu':16} {'$/h':>6} {'VRAM':>5} {'RAM':>5} {'disk':>5} {'down Mbps':>9} {'rel':>5} {'cpu':>4} {'cuda':>5} {'est h':>6}  id / location")
for cost, name, o, hrs, sp in rows:
    print(f"{cost:6.1f} {name:16} {o['dph_total']:6.2f} {o['gpu_ram']/1024:5.0f} {o['cpu_ram']/1024:5.0f} {o['disk_space']:5.0f} {o['inet_down']:9.0f} {o['reliability2']:5.3f} {o['cpu_cores_effective']:4.0f} {o['cuda_max_good']:5.1f} {hrs:6.1f}  {o['id']} {o.get('geolocation','')}")
