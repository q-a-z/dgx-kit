#!/bin/sh
# Serve Clover through the vLLM plugin. Usage: start_vllm.sh [model dir] [extra vllm args]. Run only when the box is free.
MODEL=${1:-/home/q/models/_work/clover-bf16}; shift
exec docker run --rm --name clover-vllm -e CLOVER_DEBUG=1 --gpus all --ipc host -p 8110:8000 \
  -v /home/q/models:/home/q/models:ro -v /home/q/clover-port:/clover:ro -e PYTHONPATH=/clover \
  -v /home/q/.cache/vllm-jit:/root/.cache/vllm -v /home/q/.cache/flashinfer:/root/.cache/flashinfer \
  --entrypoint vllm dgx-kit/vllm:nightly-gb10 serve "$MODEL" --served-model-name clover --trust-remote-code \
  --hf-overrides '{"canvas_length":32}' --diffusion-config '{"canvas_length":32}' \
  --attention-backend TRITON_ATTN --mamba-ssm-cache-dtype float32 \
  --max-model-len 8192 --max-num-seqs 2 --max-num-batched-tokens 2048 --kv-cache-memory-bytes 3000000000 \
  --gpu-memory-utilization 0.5 --no-enable-prefix-caching --language-model-only "$@"
