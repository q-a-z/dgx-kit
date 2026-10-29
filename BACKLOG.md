# Backlog

Rewritten 2026-10-29; entries older than that were audited against the tree and closed.
Resolved then: TRT-LLM shipped as the fifth engine (0.1.41–0.1.43). Qwen3.8-27B 3-bit GSQ,
Qwen3-8B NVFP4 + dflash2, and GLM-4.7-Flash Eagle3/DFlash trials are done or abandoned (not pending).
NIMs were checked 10-14: none covers the fleet, not pursued.

## Active

### Clover-35B-A3B on vLLM (route B) — started 2026-10-29
- Plan and findings live in `tools/clover_vllm/PLAN.md`; converter in `tools/clover_nvfp4.py`.
- M1 partly done (ops + CPU tests pass, model imports on CPU). Next: M2 weight load + logits check vs HF, needs a free GPU (box busy: ornith, qwen3.6, minicpm). Ask before any GPU launch.
- Fallback route A: SGLang PR 46927 image.

### MoESQ kernels on SM121 — speed unmeasured
- `grouped-sparse-GEMM` builds for sm_121a and passes its suite on the Spark (2026-10-21, `docs/moesq-sm121/`).
- Next: bench `bench_group_mm.py` numbers vs the dense NVFP4 path, then decide whether the recipe uses it.

### Ternary Bonsai 2 27B — NVFP4 route
- `tools/bonsai27_nvfp4.py` (2026-10-23) converts the unpacked ternary BF16 to NVFP4 weight-only for vLLM; no llama.cpp fork needed anymore.
- Conversion quality and Spark decode speed are not recorded here; bench before any further use.

### FlashMap drafter for Qwen3.5-35B-A3B — plan only
- `docs/flashmap/` (2026-10-28): recipe to train a FlashMap drafter via the SpecForge fork; nothing run.

## Gaps in role benchmarks
- No Reviewer test anywhere; PM management outputs never judged.
- qwen3.6-35b and kat-coder-v2.5-dev have no quality eval.
- Complex-code harness reports 0/N on ornith and nemotron (import/timeouts), unreliable.
