# Clover-35B-A3B on vLLM (route B), started 2026-10-29
Reference: HF modeling_clover_moe.py / sampling_clover_moe.py (copies in /tmp on the Mac peek dir; weights in ~/models/_work/clover-bf16).
State layouts match: conv_state (dim, width-1) DS, ssm_state [H,V,K] -> HF bd_gdn_* ops usable on the vLLM state pools with cache_indices.
Per-request modes: COMMIT (prefill / finished block replay: standard causal vLLM path, writes state+KV), PREVIEW (32-token mask block:
  full-attn layers non-causal via per-request causal tensor (FlashAttention backend), GDN layers use bd_gdn_conv1d + bd_gdn_local_memory
  reading clean state, writing none; MoE router logits masked by coreset(top_p 0.9) per block).
Hook for GDN: subclass QwenGatedDeltaNetAttention._forward_core; point preview requests state index at null block 0 for the standard path
  and overwrite their rows with preview output (needs ssm cache fp32: mamba_ssm_cache_dtype float32, no prefix caching at first).
Sampler/state machine: model_state like DiffusionGemmaModelState; LowConfidence threshold 0.95 (fill conf>thr else argmax conf), penalties
  (presence .9, repetition 1.05, length 10, think-end 10) on committed response history; commit emits block, stop only after full block.
Milestones: M1 ops+tests (CPU) -> M2 model class + weight load (check logits on a short prompt vs HF, needs GPU) -> M3 state/sampler
  -> M4 end-to-end parity vs HF greedy -> M5 NVFP4 convert + recipe + bench. Fallback route A: SGLang PR 46927 image.
Rules: ask the user before any GPU launch. Box is busy (ornith, qwen3.6, minicpm loaded).

## Finding 2026-10-29 (after reading mamba_hybrid model state + GDN layer)
Diffusion in vLLM rides the speculative-decoding path (canvas = draft tokens). For a hybrid GDN model, preview and commit rows go
through the GDN spec branch (spec_state_indices_tensor [seq, 1+num_spec], num_accepted_tokens, per-position state slots), not the
non_spec path that CloverGDN._forward_core patches today. So CloverGDN must patch the spec_* metadata too (preview = read slot[accepted-1],
write nothing; commit = all canvas tokens accepted). The model state must subclass MambaHybridModelState (not the Gemma DefaultModelState
pattern), conv state width grows by num_spec, and DiffusionConfig.canvas_length drives num_spec.
Written: ops.py (CPU test passes), model.py (non_spec hook, CoresetGate, layer swap, arch registered, imports on CPU), ctx.py.
NOT written: model state, sampler, spec-path GDN patch. No GPU run yet. Go/no-go: one preview block logits == HF reference (free box needed).
