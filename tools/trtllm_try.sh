#!/usr/bin/env bash
# Try TensorRT-LLM on the Spark beside DGX-kit: one model, a spare port, nothing registered with the dashboard.
# Stop the model you compare against first (or keep both inside memory), then benchmark with tools/bench.py
# pointed at the port. Remove with: docker rm -f trtllm-try
#   tools/trtllm_try.sh ~/models/nvidia--Some-NVFP4 [port] [gpu-memory-fraction]
set -euo pipefail
model=${1:?usage: trtllm_try.sh MODEL_DIR [port] [gpu-memory-fraction]}
port=${2:-8355}
frac=${3:-0.5}
image=${DGXKIT_IMAGE_TRTLLM:-nvcr.io/nvidia/tensorrt-llm/release:1.2.1}  # arm64; 1.3.0rc* has newer Blackwell kernels

docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
docker rm -f trtllm-try >/dev/null 2>&1 || true
# Flag names are from the TRT-LLM 1.2 docs; `docker run --rm $image trtllm-serve --help` is the check if one is refused.
docker run -d --name trtllm-try --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$model:$model:ro" \
  "$image" trtllm-serve "$model" --host 0.0.0.0 --port "$port" --backend pytorch \
  --max_batch_size 4 --max_num_tokens 32768 --kv_cache_free_gpu_memory_fraction "$frac"
echo "started; follow with: docker logs -f trtllm-try   then bench against http://localhost:$port"
