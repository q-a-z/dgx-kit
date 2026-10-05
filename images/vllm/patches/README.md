# vLLM patches for the GB10

The Images tab builds two patched vLLM images from `images/vllm/Dockerfile`:

| Build | Base | Patches applied |
|---|---|---|
| `dgx-kit/vllm:0.30-gb10` | `vllm/vllm-openai:v0.30.0` | `common/*.patch`, then `0.30/*.patch` |
| `dgx-kit/vllm:0.29-gb10` | `vllm/vllm-openai:v0.29.0` | `common/*.patch`, then `0.29/*.patch` |

Put a patch in `common/` when it applies cleanly to both releases, otherwise in the release's own folder.
Files apply in name order, so prefix them (`010-…`, `020-…`).

Each patch is a unified diff against the installed site-packages, with paths like `a/vllm/…` and `b/vllm/…`
(applied with `patch -p1` from the site-packages directory). A patch that doesn't apply fails the build,
so a broken patch never ships silently.

Both builds also install **FlashInfer 0.7.0** (the release the GB10 fleet was validated on) in place of the older one
the upstream image ships, remove that image's prebuilt FlashInfer kernel packages (they must match it exactly), and set
`TORCH_CUDA_ARCH_LIST` and `FLASHINFER_CUDA_ARCH_LIST` to `12.1a` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Kernels then compile on first use, which is why the containers share `~/.cache/flashinfer`. `DGXKIT_FLASHINFER=` (empty)
builds with the upstream FlashInfer instead.

When an imported llmctl conf names an image this box doesn't have (for example `vllm-spark:0.29-pfxtest`), the import uses
the build of the same vLLM release and says so; build it first in Settings, Engine images.

A build never changes the engine's default image. Pick it per model in the model's settings (Image),
so one model (for example ornith on 0.29) can run a different vLLM from the rest.

Override the base tags with `DGXKIT_VLLM_029_BASE` and `DGXKIT_VLLM_030_BASE`.

## Current patches

Each patch was checked against clean vLLM 0.29.0 and 0.30.0 sources (the release wheels). The full set
(`common/*` then the series folder) applies with no fuzz or offset, and every touched file compiles.
The 0.29 and 0.30 copies of a patch make the same change; only their line numbers differ.

| Patch | What it fixes | Models that need it | 0.29 | 0.30 |
|---|---|---|---|---|
| `common/010-nemotron-v3-content` | The Nemotron v3 reasoning parser returned `content: null` when generation stopped inside `<think>` and the client sent no `chat_template_kwargs` | nemotron-3.5 behind LiteLLM | yes | yes |
| `0.29/`, `0.30/020-eagle-quantized-lm-head` | The EAGLE draft/target `lm_head` sharing check crashed on a quantized `ParallelLMHead` that has no `.weight`. It now skips sharing instead | EAGLE3 drafts on NVFP4 targets, for example qwen3-coder | yes | yes |
| `common/030-mla-decode-smem-gb10` | Triton MLA decode (Lk=576) at `num_stages=2` needed more shared memory than GB10's 101376-byte limit. It now uses `num_stages=1` | MLA models such as GLM-4.x-Flash on GB10 | yes | yes |
| `0.29/`, `0.30/040-glm4-moe-lite-eagle3` | `glm4_moe_lite` had no EAGLE3 support. The patch adds the aux hidden-state capture and `SupportsEagle3` | EAGLE3 for GLM-4.x-Flash | yes | yes |
| `common/050-mamba-align-block-size` | With prefix caching in align mode, the Mamba state index was computed from the KV-cache block size instead of the Mamba block size. A cache hit then went out of bounds | ornith (Mamba-hybrid) with prefix caching | yes | yes |
| `0.29/`, `0.30/060-dflash-prefix-cache` | The DFlash draft attended to context KV that was never written for prefix-cache-restored tokens. The patch tracks the restored count and masks those blocks out of the draft's seq_lens and block table. It backports the unmerged upstream PR #47926 (issue #47930) | ornith (Mamba-hybrid + DFlash) with prefix caching | yes | yes (port not runtime-tested yet) |

`050` and `060` are both needed for ornith with prefix caching. Neither change is in stock 0.29.0 or 0.30.0.
Drop `060` once upstream merges PR #47926 into a release.
