#!/usr/bin/env bash
# MoESQ on Qwen3.6-35B-A3B, "medium" config (2 refinement epochs, 1024 samples x 4096 tokens), on a rented GPU box.
# Run ON the rented instance, in this order (each phase is a separate command, so you can stop between them):
#   bash run_on_box.sh setup                      # venv, MoESQ at a pinned commit, BF16 model download + check (~15-25 min)
#   PRICE=0.79 bash run_on_box.sh probe           # one layer of the real config; prints projected hours and dollars
#   PRICE=0.79 CAP=16 bash run_on_box.sh run      # the full run; aborts itself if the projected total passes the cap
#   bash run_on_box.sh status                     # progress and money spent so far
# PRICE is this instance's $/h from the listing. CAP is the most you want the full run to cost (default 16).
# Nothing here uses your vast.ai account key. At the end it prints the copy command; the instance keeps billing until you destroy it.
set -euo pipefail
PHASE="${1:-}"
WORK="${WORK:-/workspace}"; [ -d "$WORK" ] || WORK="$HOME/work"; mkdir -p "$WORK"
REPO_URL=https://github.com/IST-DASLab/MoESQ.git
REPO_COMMIT=584d78c3d24905c24a2574c29317c985328c9fef      # the commit this was prepared and tested against (2026-10-05)
VENV="$WORK/venv"; SRC="$WORK/MoESQ"; MODEL="$WORK/model"; CKPT="$WORK/ckpt"; OUT="$WORK/out"; CFG="$SRC/configs/medium35.yaml"
PRICE="${PRICE:-}"; CAP="${CAP:-16}"; LAYERS=40
export WANDB_MODE=disabled HF_HOME="$WORK/hf" PYTHONUNBUFFERED=1

need_price() { [ -n "$PRICE" ] || { echo "set PRICE=<dollars per hour of this instance>, e.g. PRICE=0.79"; exit 2; }; }
activate() { [ -f "$VENV/bin/activate" ] || { echo "run setup first"; exit 2; }; . "$VENV/bin/activate"; }
self_stop() {   # only on an abort, never after success (a stopped instance cannot be copied from)
  if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] && command -v vastai >/dev/null; then
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" || true
  else echo "stop or destroy this instance yourself now (billing continues)"; fi
}

write_config() {
  cat > "$CFG" <<'YAML'
# Qwen3.5-397B-A17B, ours: paired-4:8 NVFP4 W4A4, gate-weighted refinement p=2.
# MEDIUM config on Qwen3.6-35B-A3B: 2 refinement epochs, 1024 samples x 4096 tokens, GPTQ on 512 samples (paper dataset mix)

model:
  name: @MODEL@
  device: cuda
  dtype: bfloat16
  dummy: false

data:
  dataset_name: mixed
  num_samples: 1024
  val_samples: 32
  batch_size: 8
  max_length: 4096
  num_workers: 2
  seed: 0
  shuffle_seed: 1234
  shuffle_buffer_size: 100000
  open_thoughts_max_samples: 10000
  mixed_source_weights: [0.1, 0.45, 0.45]

compression:
  prunen: 4
  prunem: 8
  quant_type: nvfp4
  groupsize: 16
  learn_weight_values: true
  fake_quantize_activations: true

init:
  method: gptq
  wbits: 4
  nsamples: 512
  sym: true
  trits: false
  percdamp: 0.1
  blocksize: 128
  static_groups: false
  hessian_chunk_tokens: 65536

refine:
  enabled: true
  start_layer: 0
  self_attn: false
  num_epochs: 2
  device_microbatch_size: 2
  warmup_steps: 0
  lr_decay_type: cosine
  scheduler_min_lr: 0.1
  masks_lr: 2.0e-4
  weights_lr: 2.5e-5
  weight_decay: 1
  lion_betas: [0.9, 0.95]
  temperature: [2.0, 0.05]
  scale: [100, 500]
  strength: 4
  std: 0.01
  logits_dtype: bfloat16
  gate_weight_exponent: 2
  divergence_warn_ratio: 1.5

training:
  checkpoint_dir: @CKPT@
  log_dir: @WORK@/logs
  eval_baseline: false
  ppl_eval_every_n_layers: 1000
  meta_init_std: 0.02
  act_cache_dir: @WORK@/act_cache
  act_cache_mmap_threshold_gb: 2.0

distributed:
  timeout_hours: 4.0

wandb:
  enabled: false
  project: moe-sq
  entity: ''
YAML
  sed -i "s#@MODEL@#$MODEL#; s#@CKPT@#$CKPT#; s#@WORK@#$WORK#" "$CFG"
}

setup() {
  command -v nvidia-smi >/dev/null || { echo "no GPU driver visible in this container"; exit 1; }
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)
  major=${cuda% *}; minor=${cuda#* }
  if [ "$major" -ge 13 ]; then CU=cu130; elif [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; then CU=cu128
  else echo "driver supports CUDA $cuda, need 12.8 or newer: destroy this instance and pick another"; exit 1; fi
  echo "[setup] driver CUDA $cuda -> torch wheels $CU"; df -h "$WORK" | tail -1
  [ -f "$VENV/bin/activate" ] || python3 -m venv "$VENV"
  . "$VENV/bin/activate"
  pip install -q --upgrade pip wheel setuptools
  pip install -q torch==2.11.0 --index-url "https://download.pytorch.org/whl/$CU"
  [ -d "$SRC/.git" ] || git clone -q "$REPO_URL" "$SRC"
  git -C "$SRC" checkout -q "$REPO_COMMIT"
  VENV_DIR="$VENV" bash "$SRC/scripts/setup_venv.sh"
  pip install -q flash-linear-attention einops vastai      # fla: without it the 30 linear-attention layers use a very slow path
  python -c "import torch; assert torch.__version__.startswith('2.11.0'), torch.__version__; assert torch.cuda.is_available(); print('[setup] torch', torch.__version__, torch.cuda.get_device_name(0))"
  sed -i 's/"flash_attention_2"/"sdpa"/g' "$SRC/src/models/qwen35_moe.py"     # flash-attn is not installed
  write_config
  python - <<PY
import json, os, struct
from huggingface_hub import snapshot_download
d = snapshot_download("Qwen/Qwen3.6-35B-A3B", local_dir="$MODEL", max_workers=16)
idx = json.load(open(f"{d}/model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values())); bad = []
for s in shards:
    p = f"{d}/{s}"; sz = os.path.getsize(p)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    if 8 + n + max(v["data_offsets"][1] for v in h.values()) != sz: bad.append(s)
print("[setup] model shards", len(shards), "bad:", bad); assert not bad
PY
  echo "[setup] done. Next: PRICE=<\$/h> bash run_on_box.sh probe"
}

probe() {
  need_price; activate; cd "$SRC"
  rm -f "$WORK/gpumem.log"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$WORK/gpumem.log"; sleep 5; done ) & SAMP=$!
  python main.py --config "$CFG" --max-layers 1 2>&1 | tee "$WORK/probe.log" || true
  kill $SAMP 2>/dev/null || true
  S=$(grep -E 'sec/layer: +[0-9.]+s' "$WORK/probe.log" | grep -v avg | tail -1 | sed -E 's/.*sec\/layer: +([0-9.]+)s.*/\1/')
  [ -n "$S" ] || { echo "PROBE FAILED: no timing line found; read $WORK/probe.log"; exit 1; }
  HOURS=$(awk -v s="$S" -v L=$LAYERS 'BEGIN{printf "%.1f", s*L/3600}')
  COST=$(awk -v h="$HOURS" -v p="$PRICE" 'BEGIN{printf "%.1f", h*p}')
  PEAK=$(sort -n "$WORK/gpumem.log" | tail -1)
  printf 'SEC_PER_LAYER=%s\nHOURS=%s\nCOST=%s\n' "$S" "$HOURS" "$COST" > "$WORK/PROBE_RESULT"
  echo "=============================================================="
  echo "one layer: ${S}s | projected full run: ${HOURS} h = \$${COST} at \$${PRICE}/h | GPU memory peak: ${PEAK} MiB"
  if awk -v c="$COST" -v cap="$CAP" 'BEGIN{exit !(c>cap)}'; then
    echo "OVER THE CAP (\$${CAP}). Do not run. Destroy this instance."
  else echo "within the cap. To start: PRICE=$PRICE CAP=$CAP bash run_on_box.sh run"; fi
  echo "=============================================================="
}

run() {
  need_price; activate; cd "$SRC"
  [ -f "$WORK/PROBE_RESULT" ] || { echo "run the probe first"; exit 2; }
  . "$WORK/PROBE_RESULT"
  if awk -v c="$COST" -v cap="$CAP" 'BEGIN{exit !(c>cap)}' && [ "${FORCE:-0}" != 1 ]; then
    echo "projected \$$COST is over the cap \$$CAP (FORCE=1 overrides)"; exit 1; fi
  ARGS=(--config "$CFG"); [ "${RESUME:-0}" = 1 ] && ARGS+=(--resume)
  nohup python main.py "${ARGS[@]}" >> "$WORK/run.log" 2>&1 &
  PID=$!; START=$(date +%s); echo "[run] started pid $PID; log $WORK/run.log"
  while kill -0 $PID 2>/dev/null; do
    sleep 60
    spent=$(awk -v n="$(date +%s)" -v s=$START -v p="$PRICE" 'BEGIN{printf "%.2f", (n-s)/3600*p}')
    avg=$(grep -oE 'avg sec/layer: +[0-9.]+' "$WORK/run.log" | tail -1 | grep -oE '[0-9.]+$' || true)
    layer=$(grep Throughput "$WORK/run.log" | tail -1 | grep -oE 'Layer [0-9]+' | grep -oE '[0-9]+' || true)
    if [ -n "$avg" ] && [ -n "$layer" ]; then
      proj=$(awk -v a="$avg" -v l="$layer" -v L=$LAYERS -v p="$PRICE" -v sp="$spent" 'BEGIN{printf "%.2f", sp+(L-l)*a/3600*p}')
      echo "[run] layer $layer/$LAYERS | spent \$$spent | projected total \$$proj (cap \$$CAP)"
      if awk -v pr="$proj" -v cap="$CAP" 'BEGIN{exit !(pr>cap*1.15)}'; then
        kill $PID; echo "[run] ABORT: projected \$$proj is over 115% of the \$$CAP cap"; self_stop; exit 3; fi
    fi
  done
  wait $PID || { echo "[run] main.py FAILED:"; tail -30 "$WORK/run.log"; exit 1; }
  pack
}

pack() {
  activate; cd "$SRC"
  RID=$(ls -1t "$CKPT" | head -1); echo "[pack] run id $RID"
  python save_model.py --config "$CFG" --run-id "$RID" --out-dir "$OUT"
  ( cd "$OUT" && sha256sum * > SHA256SUMS && du -sh . && ls )
  echo "=============================================================="
  echo "DONE. Copy $OUT to the Spark NOW, then destroy this instance (it is still billing)."
  echo "on the Spark:  rsync -avP -e 'ssh -p <PORT>' root@<HOST>:$OUT/ ~/models/_work/moesq-35b/"
  echo "=============================================================="
}

status() {
  for f in probe.log run.log; do [ -f "$WORK/$f" ] && { echo "--- $f"; grep -E 'Throughput|sec/layer' "$WORK/$f" | tail -4; }; done
  [ -f "$WORK/PROBE_RESULT" ] && cat "$WORK/PROBE_RESULT"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
}

case "$PHASE" in setup) setup ;; probe) probe ;; run) run ;; pack) pack ;; status) status ;; *) sed -n '2,12p' "$0"; exit 2 ;; esac
