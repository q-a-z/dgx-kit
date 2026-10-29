# FlashMap drafter for Qwen3.5-35B-A3B (plan, nothing run yet)

FlashMap ([paper](https://arxiv.org/abs/2610.24365), [code](https://github.com/al-jshen/flashmap)) drafts 16 tokens in one
pass from the target's hidden states. Published drafters exist only for dense Qwen3-8B/4B; this is the recipe to train one for
`Qwen/Qwen3.5-35B-A3B`. Training uses the [SpecForge fork](https://github.com/al-jshen/SpecForge/tree/flashmap) (branch `flashmap`).

Why it should work: FlashMap reads the same hidden states as DFlash, and the fork already has a working
`qwen3.5-35b-a3b-dflash` config and recipe (layers 1,10,19,28,37; embedding key `model.language_model.embed_tokens.weight`;
chat template `qwen3.5`). [qwen3.5-35b-a3b-flashmap.json](qwen3.5-35b-a3b-flashmap.json) is the 8B FlashMap config with those
target fields swapped in.

Unknown until measured: acceptance length on an MoE target, and whether verifying 16 tokens costs too much on A3B (many experts
touched). Paper numbers are dense, concurrency 1, one GPU only.

## Steps (rented node, 8x H200/B200, one node)

1. Setup
   ```bash
   git clone -b flashmap https://github.com/al-jshen/SpecForge && cd SpecForge && uv pip install -e .
   cp <dgx-kit>/docs/flashmap/qwen3.5-35b-a3b-flashmap.json configs/
   # sglang JIT needs CUDA_HOME with nvcc, and ninja on PATH
   ```
2. Data: regenerate answers with the target itself (500K OpenPerfectBlend conversations in the paper; use ~5K for the smoke run).
   The fork's `scripts/` has the regen script used for `perfectblend_qwen3-8b_regen.jsonl`; point it at Qwen3.5-35B-A3B.
3. Capture hidden states
   ```bash
   torchrun --nproc_per_node=8 scripts/prepare_hidden_states.py --strategy flashmap \
     --target-model-path Qwen/Qwen3.5-35B-A3B --draft-model-config configs/qwen3.5-35b-a3b-flashmap.json \
     --data-path ./cache/dataset/perfectblend_qwen3.5-35b-a3b_regen.jsonl \
     --output-path ./cache/hidden_states/qwen3.5-35b-a3b-flashmap --chat-template qwen3.5 --max-length 4096
   ```
4. Train: copy the two `qwen3-8b-flashmap-{flow,map}-offline.yaml` files and change only
   `target_model_path` -> `Qwen/Qwen3.5-35B-A3B`, `draft_model_config`, `hidden_states_path`, `chat_template: qwen3.5`,
   `embedding_key: model.language_model.embed_tokens.weight`, `run_id`/`output_dir`, and the map's `flashmap_flow_checkpoint`.
   Keep the paper's hyperparameters (flow: 20 epochs, lr 6e-4, wd 0.1; map: 12 epochs, lr 5e-5, `num_anchors: 128`).
   `specforge train --config <flow.yaml>`, then `<map.yaml>`.
   - **Smoke run first:** ~5K samples, a few hundred flow steps; check `flashmap/ce_loss@tau*` falls and nothing OOMs.
   - Vocab is 248,320 (vs 151,936), so the map stage is heavier; lower `num_anchors` if it OOMs.
   - **AdamW vs Muon A/B on the smoke run** (pick the optimizer before paying for the full run). Muon
     ([Moonlight](https://github.com/MoonshotAI/Moonlight), [paper](https://arxiv.org/abs/2502.16982)) matched AdamW with
     ~52% of the FLOPs when training from scratch; the draft layers are trained from scratch, so it may carry over.
     Unmeasured for drafts.
     1. Apply the patch (against fork commit `78c0cdd`; adds `training.optimizer: adamw|muon`):
        `git apply <dgx-kit>/docs/flashmap/specforge-muon.patch`. Muon takes the hidden weight matrices, AdamW keeps
        norms, biases and anything named `embed`/`lm_head`; Muon updates are scaled by `0.2·√max(A,B)` so the same
        `learning_rate` and `weight_decay` apply. Needs `fsdp_sharding: NO_SHARD` (the flow config already uses it).
     2. Run the flow smoke twice on the same 5K samples and steps, changing only `optimizer` and `run_id`/`output_dir`:
        `adamw` (as is) and `muon`.
     3. Compare `flashmap/ce_loss@tau*` at equal steps and steps-to-equal-loss, and step time (Newton-Schulz adds a few
        matmuls per matrix). Then export both, serve each, and compare acceptance length: lower loss is not the goal.
     4. Muon clearly ahead (e.g. AdamW's final loss reached in ≤70% of the steps, acceptance no worse): use it for flow
        **and** map; the paper found switching optimizers between stages loses the gain. Otherwise keep AdamW.
5. Export: `specforge export --to hf --checkpoint ./outputs/<map run> --draft-config configs/qwen3.5-35b-a3b-flashmap.json --output-dir ./exports/FlashMap-Qwen3.5-35B-A3B`
6. Serve + benchmark: `flashmap[sglang]` needs `sglang==0.5.20` exactly, TP=1, `--speculative-algorithm DSPARK`
   (or `flashmap[vllm]` with `vllm==0.31.*`). Compare against your existing no-draft / MTP / DFlash numbers for this model at the
   concurrency you actually serve, not just 1.

## Cost (paper, dense 8B, scaled loosely)
Flow ~800 GPU-h + map ~290 GPU-h on H200 for 500K samples; expect the same order or more here. The smoke run is a few GPU-hours.
Decide after the smoke run whether the full run is worth it.
