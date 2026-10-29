# MoESQ kernels on SM121 (DGX Spark / GB10)

Status 2026-10-21: the `grouped-sparse-GEMM` kernels (paired-4:8 sparse NVFP4 grouped GEMM, commit 7fdfa49)
build for `sm_121a` and pass the full test suite on the Spark: 161 passed, 7 skipped (skip reasons not checked). The sparse GEMM matches the dequantized reference (`test_group_mm_matches_dequant_reference`).
Speed is not measured yet.

Apply: `git apply grouped-sparse-gemm-sm121.patch` in a checkout of the repo, then in the nightly-gb10 image
(CUDA 13.0, torch 2.13):

    PAIRED_NVFP4_ARCHS=121a MAX_JOBS=6 pip wheel . --no-deps --no-build-isolation -w dist

What was needed (4 edits + 1 new header):
- `sm_120f` (family target) does NOT work: ptxas rejects "Sparse mma with block scale" / `.kind::mxf4nvf4 with .sp`
  there. Only the arch-specific targets (`120a`, `121a`) have the instruction. A `120a` build does not run on SM121.
- CUTLASS enables its SM120 kernels only when `__CUDA_ARCH__ == 1200`; `csrc/sm121_compat.h` (force-included with
  `-include`) sets `CUTLASS_ARCH_MMA_SM120_ENABLED` / `..._SM120A_ENABLED` when `__CUDA_ARCH__ == 1210`. Setting them
  with a plain `-D` breaks the host pass (device-only code pulled in).
- `setup.py`: `121a` arch entry, built from the `csrc/sm120/` sources.
- `paired_nvfp4_kernels/__init__.py`: device capability 12.1 maps to the SM120 module.
- `csrc/ops.cu` and `csrc/testing/selftest.cu`: the runtime arch checks accept 12.1 for an SM120 module.

Not done: the vLLM side (`integrations/vllm/moe-sq-v0.30.0.patch`, 5367 lines, targets v0.30.0; the Spark image is
0.31.1rc1). Its arch gate rejects SM121 on purpose (`test_arch_check_rejects_sm121`). The patch must be rebased
onto 0.31 and the gate widened; the released MoESQ checkpoints are Kimi-K2.5 and Qwen3.8-Flash-Next only, so a
Qwen3.6-35B or Ornith checkpoint would also have to be produced with MoESQ's own `main.py` first.
