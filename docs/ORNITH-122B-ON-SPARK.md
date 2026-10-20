# Ornith-1.5-122B on a DGX Spark

A 122B mixture-of-experts model (256 experts per layer, 8 used per token) that runs on one DGX Spark: routed experts
in NVFP4, linear attention in FP8, a DFlash2 draft for speculative decoding. Nobody publishes this exact checkpoint
(the official one is FP8, 125 GB, which does not fit), so you convert it yourself. That takes about 5 minutes of GPU
time once the download is done.

## What you get (measured on a GB10, vLLM nightly, one stream unless noted)

| | |
|---|---|
| Weights | 73 GB (+ 1.9 GB draft). With the KV cache the box shows about 91-97 GB in use. |
| Decode, code, temperature 0 | about 54 t/s, 4.7-5.8 draft tokens accepted per step (7 draft tokens) |
| Decode, prose | about 31 t/s (acceptance is only about 3) |
| Without any draft | 23.4 t/s, flat |
| 2 / 4 streams, total | 53 / 74 t/s (measured with 15 draft tokens; not re-measured with 7) |
| Load time | about 9 minutes the first time (cold read of 73 GB) |
| Role benchmark, the 13 hardest tasks, 1 run each | 13 / 13 passed (developer 3/3, reviewer 3/3, analyst 3/3, project-manager 4/4) |

It is slow next to the 35B models (about 160 t/s) and it takes the whole machine: run it alone. It is the first model
we tested that passes the cross-shard-bug review task and the project-manager triage tasks that every 35B model fails.

Quality caveat: the FP8 original (117 GB) cannot run on a Spark, so we could not measure the conversion against it.
We compared the two conversions with each other (same mean NLL within 0.1 %, top-1 agreement 91.6 % against a
noise floor of 93.5 %) and ran the role benchmark.

## What you need

- A DGX Spark (121 GB unified memory) with nothing else big loaded.
- vLLM nightly with GB10 support and DFlash2 drafts. We used `0.31.1rc1.dev376`, built from DGX-kit's
  Settings -> Build GB10 engine images (image name `dgx-kit/vllm:nightly-gb10`).
- Disk: 125 GB for the download + 73 GB for the result = about 200 GB free while converting. Delete the FP8 source
  afterwards if you are short (it downloads again in about an hour).
- A Hugging Face connection; no token is needed, the repos are open.

## 1. Download

| What | Link | Size |
|---|---|---|
| Model, FP8 original (MIT) | https://huggingface.co/ornith-ai/Ornith-1.5-122B-FP8 | 125 GB |
| Draft, DFlash2 (Apache-2.0) | https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2 | 1.9 GB |
| DGX-kit (vLLM image build, this guide, the converter) | https://github.com/q-a-z/dgx-kit | |
| vLLM | https://github.com/vllm-project/vllm | |

```bash
mkdir -p ~/models/_work && cd ~/models/_work
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("ornith-ai/Ornith-1.5-122B-FP8", local_dir="ornith-122b-fp8", max_workers=16)   # 125 GB, MIT
snapshot_download("incoai/Ornith-1.5-122B-DFlash2", local_dir="ornith-122b-dflash2")               # 1.9 GB, Apache-2.0
EOF
```

The speed from Hugging Face varies a lot (2-60 MB/s for us). It resumes if you stop it. Check afterwards that all 62
`model-*.safetensors` files are there and have the sizes the repo page shows.

## 2. Convert

`tools/ornith122_convert.py` reads one source shard at a time, quantizes it on the GPU and writes the matching output shard.
Download the script first (you do not need a clone of the repo), then run it from any folder:

```bash
curl -fsSL -o ~/ornith122_convert.py \
  https://raw.githubusercontent.com/q-a-z/dgx-kit/main/tools/ornith122_convert.py
ls -l ~/ornith122_convert.py        # must be a file of about 7 KB, not a folder

docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v ~/models:/m -v ~/ornith122_convert.py:/b.py:ro \
  --entrypoint python3 dgx-kit/vllm:nightly-gb10 \
  /b.py /m/_work/ornith-122b-fp8 /m/vllm/ornith-122b-nvfp4 B
```

The last argument (`B`) is the variant. If the mounted file does not exist, Docker silently creates an empty
folder in its place and Python then says `can't find '__main__' module in '/b.py'`; if you see that, check the path
of the file you mount and remove any folder Docker created there (`sudo rm -rf` it, it is owned by root).

`B` is the recommended variant: experts NVFP4, linear attention FP8. `A` leaves linear attention in BF16 (4 GB bigger,
about 20 % slower, same quality as far as we could measure). The script prints the quantization error per shard;
expect about 9.3 % relative error on the experts and 2.6 % on the FP8 linear attention. The counter
"zero-tensors" was 0 for us; any all-zero tensor it finds is stored as zeros.

Then copy the small files the script does not write (tokenizer, chat template, preprocessor configs) and the draft:

```bash
cd ~/models/_work/ornith-122b-fp8
cp chat_template.jinja tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json \
   preprocessor_config.json processor_config.json video_preprocessor_config.json ../../vllm/ornith-122b-nvfp4/
cp -r ../ornith-122b-dflash2 ../../vllm/ornith-122b-nvfp4/draft
```

## 3. Serve

Plain vLLM:

```bash
vllm serve ~/models/vllm/ornith-122b-nvfp4 --served-model-name ornith-122b \
  --max-model-len 131072 --kv-cache-memory-bytes 10000000000 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
  --speculative-config '{"method":"dflash","model":"'$HOME'/models/vllm/ornith-122b-nvfp4/draft","num_speculative_tokens":7}' \
  --mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode align --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --language-model-only \
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20,"max_new_tokens":32768}' \
  --max-num-seqs 8 --long-prefill-token-threshold 1024 --attention-backend FLASHINFER
```

With DGX-kit: create a recipe with repo `local/ornith-122b-nvfp4`, engine vllm, draft method `dflash`, 7
speculative tokens, KV cache dtype fp8, and the same extra options as above (one per line). Give the recipe the
model's `config.json` (the kit sizes the KV cache from it) and keep the folder name lowercase.

## How many draft tokens

The draft card uses 15. On a Spark that is too many: verifying a long block reads many more experts from memory,
and the late positions are mostly rejected. Same model, same machine, decode t/s (mean of 2 runs per test):

| Draft tokens | code t=0 | code t=0.5 | code t=1 | prose | code, no thinking |
|---|---|---|---|---|---|
| none | 23.5 | 23.3 | 23.4 | 23.4 | 23.4 |
| 5 | 45.7 | 47.5 | 48.8 | 32.5 | 55.6 |
| **7** | **54.1** | 48.9 | 51.0 | 30.9 | 56.6 |
| 11 | 41.9 | 47.7 | 37.7 | 26.2 | 57.0 |
| 15 | 47.5-50.5 | 42.2 | 48.1 | 24.1 | 45.3 |
| the model's own MTP head, 3 tokens | 29-31 | 32 | 30 | 25.8 | 32.7 |

Run-to-run noise is about +-5 t/s, so 5 and 7 are close; 7 had the best average. MTP is clearly worse.
The `bench.py` in `tools/` refuses to start while the model has a running request, so wait for the server to go idle
before benchmarking.

## Things that went wrong for us

- **Do not use `--load-format fastsafetensors`.** It works on the 35B models and 3x speeds up their load, but on this
  one it ran the machine out of memory at shard 35 of 62 and the kernel killed the container (and restarted the
  gateway). The default loader peaks at about 97 GB used and finishes.
- **Do not quantize the lm_head.** vLLM fails at the end of a 9-minute load with
  `no module or parameter named 'lm_head.weight_scale'`. The converter leaves it in BF16.
- **Leave the MTP head alone.** It has no FP8 scales; the converter copies it unchanged and keeps it in the ignore list.
- **Memory:** nothing else big fits next to it. With the kit, stop your other models first.
- **First start is slow:** about 9 minutes to read 73 GB cold, then about a minute of compile and warm-up.

## Status

The converted files in this guide are not published anywhere. The converter in this repo was tidied after our first
run (it no longer tries the FP8 lm_head that vLLM rejects). We re-ran the tidied script on two shards (one with
experts, one with the lm_head) and the output files are byte-identical to the ones we loaded, and the config it
writes is identical to ours. A full end-to-end run of the tidied script (all 62 shards, index included) has not been
repeated.
