"""Clover-35B-A3B on vLLM: Qwen3.5-MoE backbone plus block-diffusion preview in the GDN layers and expert-subset routing.

COMMIT steps (prompt, finished block replay) use vLLM's normal causal path. PREVIEW steps (a 32-token block of mask tokens)
- full attention: non-causal, chosen per request by the model state (attention metadata `causal` tensor);
- linear attention: Clover's bd_gdn_* ops read the clean conv/ssm state and write nothing;
- MoE: router logits are cut to the block's expert subset (coreset, top_p) before routing.
"""
import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM
from vllm.model_executor.utils import set_weight_attrs

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from . import ctx as _ctx

logger = init_logger(__name__)


def _p(fmt, *a):
    import sys
    print(fmt % a, file=sys.stderr, flush=True)
from .ops import bd_gdn_conv1d as _conv, bd_gdn_local_memory as _mem, coreset_torch as _coreset


def _compiled(fn):
    """torch.compile fuses the ~25 small kernels of each op into a few (the step is launch-bound); falls back to eager."""
    if os.environ.get("CLOVER_NO_COMPILE"):
        return fn
    c = torch.compile(fn, dynamic=True)

    def run(*a, **k):
        try:
            return c(*a, **k)
        except Exception as e:                # a compile failure must not take the server down
            _p("CLOVER compile fallback for %s: %s", fn.__name__, str(e)[:200])
            return fn(*a, **k)
    return run


bd_gdn_conv1d, bd_gdn_local_memory, coreset_torch = _compiled(_conv), _compiled(_mem), _compiled(_coreset)


_DBG = bool(os.environ.get('CLOVER_DEBUG'))


class CloverGDN(QwenGatedDeltaNetAttention):
    _dbg = 0
    _dbg2 = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tp_size == 1, "Clover preview ops are written for tensor parallel 1"
        hv, c, w = self.num_v_heads, self.conv_dim, self.conv_kernel_size
        self.conv_right_weight = nn.Parameter(torch.zeros(c, 1, w - 1), requires_grad=False)
        self.bd_softmax_logit_scale = nn.Parameter(torch.full((hv,), 10.0), requires_grad=False)
        self.bd_softmax_history_bias = nn.Parameter(torch.full((hv,), 5.0), requires_grad=False)
        self.bd_softmax_history_c = nn.Parameter(torch.ones(hv), requires_grad=False)

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        step = _ctx.CTX
        if step is None:
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)

        # The model state already pointed the preview requests' normal-path state slots at a scratch block, so this runs
        # for everyone without touching their clean states; the preview rows are then overwritten from the clean states.
        if step.rows.numel() != mixed_qkv.shape[0]:
            # mixed batch (prompt chunks or commit replays next to denoise blocks): the normal path serves those rows. When the
            # whole batch is denoise blocks there is nothing for it to do, and skipping it saves most of the per-layer CPU time.
            super()._forward_core(mixed_qkv, b, a, core_attn_out)
        clean_idx = get_forward_context().attn_metadata[self.prefix].clover_clean_idx
        if _DBG and CloverGDN._dbg < 8:
            CloverGDN._dbg += 1
            ssm, cv = self.kv_cache[1], self.kv_cache[0]
            ci = clean_idx.tolist()
            _p("CLOVER gdn layer=%d ssm_ptr=%d conv_ptr=%d clean_idx=%s ssm_clean_norm=%.3f ssm_other_norms=%s conv_clean_norm=%.3f std_out_row_norm=%.3f has_nan_rows=%s",
               self.layer_idx, ssm.data_ptr(), cv.data_ptr(), ci, float(ssm[ci[0]].float().norm()),
               [round(float(ssm[j].float().norm()), 2) for j in range(1, 5)], float(cv[ci[0]].float().norm()),
               float(core_attn_out.index_select(0, step.rows).float().nan_to_num().norm()), bool(core_attn_out.index_select(0, step.rows).isnan().any()))

        conv_pool = self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
        ssm_pool = self.kv_cache[1]
        rows, block = step.rows, step.block
        x = mixed_qkv.index_select(0, rows)
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0), 1, -1)
        mixed = bd_gdn_conv1d(x.contiguous(), clean_conv_states=conv_pool, cache_indices=clean_idx, conv_weights=conv_w,
                              block_size=block, active_valid_mask=valid, conv_right_weight=self.conv_right_weight)
        kd, vd = self.key_dim, self.value_dim
        q, k, v = mixed.split((kd, kd, vd), dim=-1)
        n = mixed.shape[0]
        q = q.reshape(1, n, self.num_k_heads, self.head_k_dim)
        k = k.reshape(1, n, self.num_k_heads, self.head_k_dim)
        v = v.reshape(1, n, self.num_v_heads, self.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(a.index_select(0, rows).float() + self.dt_bias.float())
        y = bd_gdn_local_memory(q, k, v, g.reshape(1, n, -1), clean_ssm_states=ssm_pool, cache_indices=clean_idx,
                                active_valid_mask=valid, scale=self.head_k_dim ** -0.5,
                                softmax_logit_scale=self.bd_softmax_logit_scale,
                                softmax_history_bias=self.bd_softmax_history_bias,
                                softmax_history_c=self.bd_softmax_history_c)
        if _DBG and CloverGDN._dbg2 < 8:
            CloverGDN._dbg2 += 1
            _p("CLOVER gdn layer=%d preview_y_norm=%.3f mixed_norm=%.3f x_norm=%.3f finite=%s", self.layer_idx, float(y.float().norm()),
               float(mixed.float().norm()), float(x.float().norm()), bool(torch.isfinite(y).all()))
        core_attn_out.index_copy_(0, rows, y.reshape(n, self.num_v_heads, self.head_v_dim).to(core_attn_out.dtype))


def install_coreset(runner, num_experts, top_k, top_p):
    """Cut the router logits of preview blocks down to the block's expert subset right before dispatch.

    The MoE runner computes router logits itself (one fused GEMM: experts + shared-expert gate columns), so the hook sits on its
    _maybe_dispatch. Only the first `num_experts` columns are routable experts; the last column is the shared-expert gate."""
    orig = runner._maybe_dispatch

    def hooked(hidden_states, router_logits):
        step = _ctx.CTX
        if step is not None:
            blk = router_logits.index_select(0, step.rows)[:, :num_experts].reshape(-1, step.block, num_experts)
            valid = torch.ones(blk.shape[:2], dtype=torch.bool, device=blk.device)
            cut = coreset_torch(blk, valid, top_k, top_p).reshape(-1, num_experts).to(router_logits.dtype)
            router_logits = router_logits.clone()
            router_logits[step.rows, :num_experts] = cut
        return orig(hidden_states, router_logits)

    runner._maybe_dispatch = hooked


_orig_gdn, _orig_layer_init = qwen3_5.QwenGatedDeltaNetAttention, qwen3_5.Qwen3_5DecoderLayer.__init__


def _clover_layer_init(self, vllm_config, layer_type, prefix=""):
    cfg = vllm_config.model_config.hf_text_config
    real = cfg.model_type
    cfg.model_type = "qwen3_5_moe_text"        # the vLLM layer picks its MoE block by this name
    qwen3_5.QwenGatedDeltaNetAttention = CloverGDN
    try:
        _orig_layer_init(self, vllm_config, layer_type, prefix)
    finally:
        cfg.model_type = real
        qwen3_5.QwenGatedDeltaNetAttention = _orig_gdn
    if getattr(cfg, "moe_block_coreset_enabled", False):
        install_coreset(self.mlp.experts, cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_block_coreset_top_p)


class CloverMoeModelLM(Qwen3_5MoeForCausalLM):
    """Architecture name in the checkpoint's config.json: CloverMoeModelLM."""

    @staticmethod
    def get_model_state_cls():
        from .state import CloverModelState
        return CloverModelState

    def __init__(self, *, vllm_config, prefix=""):
        qwen3_5.Qwen3_5DecoderLayer.__init__ = _clover_layer_init
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_5.Qwen3_5DecoderLayer.__init__ = _orig_layer_init
