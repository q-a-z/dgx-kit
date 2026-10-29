"""Clover model state and block-diffusion sampler for vLLM's V2 GPU runner.

Per request the sampler runs a small state machine, mirrored on the host in `phase` (no GPU sync to know it):
  COMMIT (0): prompt chunks and the replay of a resolved block. Attention is causal, GDN/KV state is written.
              A commit step also emits the block's tokens (num_sampled = block) and re-arms a fresh all-mask block.
  DENOISE (1): the block (mask tokens, partly filled) is the query. Attention is bidirectional, GDN states are read-only
              (see model.CloverGDN), nothing is accepted (num_rejected = query length), so the position does not advance.
Each denoise step is Clover's LowConfidence rule: mask token excluded, penalties on committed response history, prediction
by argmax (or sampling), fill every pending position whose confidence is above the threshold, else the single most
confident one. When no position is pending the block is resolved and the next step is its commit.
"""
import functools
from types import SimpleNamespace

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.gpu.model_states import mamba_hybrid as _mh
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from . import ctx as _ctx

logger = init_logger(__name__)


def _p(fmt, *a):
    import sys
    print(fmt % a, file=sys.stderr, flush=True)
COMMIT, DENOISE = 0, 1
import os
_DBG = bool(os.environ.get('CLOVER_DEBUG'))
SCRATCH_SLOT = 0   # vLLM's null block: nobody's state lives there, so preview rows can dump their (unused) standard-path writes
_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)


class CloverModelState(MambaHybridModelState):
    num_new_sampled_tokens_per_step = 0   # diffusion has no bonus token after the block (as DiffusionGemma's state)

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        hf = vllm_config.model_config.hf_text_config
        self.CL = vllm_config.diffusion_config.canvas_length
        self.mask_id = int(hf.mask_token_id)
        self.threshold = float(hf.denoising_config.get("threshold", 0.95))
        self.gen = dict(self.model_config.try_get_generation_config() or {})
        R, V = self.max_num_reqs, self.model_config.get_vocab_size()
        self.phase = np.full(R, COMMIT, dtype=np.int8)
        self.canvas = torch.full((R, self.CL), self.mask_id, dtype=torch.int64, device=device)
        self.pending = torch.zeros((R, self.CL), dtype=torch.bool, device=device)
        self.counts = torch.zeros((R, V), dtype=torch.int32, device=device)   # committed response tokens, per slot
        self.resp_len = np.zeros(R, dtype=np.int64)
        self._causal_buf = torch.zeros(R, dtype=torch.int32, device=device)
        self._dbg_n = 0

    @staticmethod
    def get_model_state_cls():
        return CloverModelState

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        self.reset_slot(req_index)

    def reset_slot(self, i):
        self.phase[i] = COMMIT          # prompt first
        self.canvas[i].fill_(self.mask_id)
        self.pending[i].zero_()
        self.counts[i].zero_()
        self.resp_len[i] = 0

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler):
        return CloverSampler(sampler, self), None

    # ---- attention + GDN metadata -------------------------------------------------
    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                     for_capture=False, ubatch_idx=0):
        n, CL = input_batch.num_reqs, self.CL
        pos_np = np.zeros(0, dtype=np.int64)
        if not for_capture and n:
            slots = input_batch.idx_mapping_np[:n]
            qlen = np.diff(input_batch.query_start_loc_np[: n + 1])
            pos_np = np.nonzero((self.phase[slots] == DENOISE) & (qlen == CL))[0]
        if _DBG and self._dbg_n < 40 and not for_capture and n:
            self._dbg_n += 1
            _p("CLOVER prepare_attn n=%d phases=%s qlen=%s denoise_pos=%s", n, self.phase[input_batch.idx_mapping_np[:n]].tolist(),
                        np.diff(input_batch.query_start_loc_np[: n + 1]).tolist(), pos_np.tolist())
        causal = self._causal_buf
        causal[:n].fill_(1)
        if len(pos_np):
            causal[torch.from_numpy(pos_np).to(self.device)] = 0       # denoise blocks attend both ways
        orig = _mh.build_attn_metadata
        _mh.build_attn_metadata = functools.partial(orig, causal=causal[:n])
        try:
            md = super().prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                                      kv_cache_config, for_capture, ubatch_idx)
        finally:
            _mh.build_attn_metadata = orig

        if not len(pos_np):
            _ctx.set_ctx(None)
            return md
        pos = torch.from_numpy(pos_np).to(self.device)
        qsl = input_batch.query_start_loc[:n].to(torch.long)
        rows = (qsl[pos].unsqueeze(1) + torch.arange(CL, device=self.device)).reshape(-1)
        patched = {}
        for name, m in md.items():
            if not isinstance(m, GDNAttentionMetadata):
                continue
            if id(m) not in patched:                       # one metadata object per mamba KV-cache group, each with its own slots
                idx = m.non_spec_state_indices_tensor
                assert idx is not None and idx.dim() == 1, "Clover needs one mamba state slot per request (no mamba prefix caching)"
                hidden = idx.clone()
                hidden[pos] = SCRATCH_SLOT
                kw = {"non_spec_state_indices_tensor": hidden}
                if m.prefill_state_indices is not None:
                    kw["prefill_state_indices"] = hidden if m.prefill_state_indices.shape[0] == hidden.shape[0] else hidden[m.num_decodes:]
                pm = type(m)(**{**m.__dict__, **kw})
                pm.clover_clean_idx = idx[pos].to(torch.long)   # this group's clean slots of the preview requests
                patched[id(m)] = pm
            md[name] = patched[id(m)]
        _ctx.set_ctx(_ctx.StepCtx(rows=rows, req_pos=pos, block=CL))
        return md


class CloverSampler:
    """Replaces vLLM's sampler for Clover (the runner calls it like Sampler)."""

    compute_nans = False

    def __init__(self, base, state: CloverModelState):
        self.st = state
        self.sampling_states = base.sampling_states
        self.logprob_token_ids_state = base.logprob_token_ids_state
        self.req_states = base.req_states
        self.logprobs_mode = base.logprobs_mode
        R, CL = state.max_num_reqs, state.CL
        dev = state.device
        self._sampled = torch.zeros(R, CL, dtype=torch.int32, device=dev)
        self._num_sampled = torch.zeros(R, dtype=torch.int32, device=dev)
        self.params: dict[int, dict] = {}
        self._last_end, self._period, self._body = 0.0, [], []
        self._dbg = 0
        g = state.gen
        self.stop_ids = sorted({*(g.get("eos_token_id") if isinstance(g.get("eos_token_id"), list) else [g.get("eos_token_id")]),
                                g.get("pad_token_id")} - {None})

    @property
    def penalties_state(self):
        return _NO_PENALTIES_STATE

    def add_request(self, i, sp):
        self.sampling_states.add_request(i, sp)
        self.logprob_token_ids_state.add_request(i, sp)
        g = self.st.gen
        pick = lambda v, k, d: v if v else g.get(k, d)    # a request value of 0 means "use the model default"
        self.params[i] = dict(
            temperature=float(sp.temperature or 0.0) if g.get("do_sample") else 0.0,   # the model's generation_config is greedy;
             top_k=int(sp.top_k or 0), top_p=float(sp.top_p if sp.top_p is not None else 1.0),
            presence=float(pick(sp.presence_penalty, "presence_penalty", 0.0)),
            frequency=float(pick(sp.frequency_penalty, "frequency_penalty", 0.0)),
            repetition=float(sp.repetition_penalty if sp.repetition_penalty not in (None, 1.0) else g.get("repetition_penalty", 1.0)),
            len_pen=float(g.get("dllm_length_penalty", 0.0)), len_start=int(g.get("dllm_length_penalty_start", 8192)),
            len_win=int(g.get("dllm_length_penalty_window", 8192)),
            think_bias=float(g.get("dllm_think_end_bias", 0.0)), think_start=int(g.get("dllm_think_end_start", 4096)),
            think_win=int(g.get("dllm_think_end_window", 4096)), think_id=g.get("dllm_think_end_token_id"))

    def _default_params(self):
        g = self.st.gen
        return dict(temperature=0.0, top_k=0, top_p=1.0, presence=float(g.get("presence_penalty", 0.0)),
                    frequency=float(g.get("frequency_penalty", 0.0)), repetition=float(g.get("repetition_penalty", 1.0)),
                    len_pen=float(g.get("dllm_length_penalty", 0.0)), len_start=int(g.get("dllm_length_penalty_start", 8192)),
                    len_win=int(g.get("dllm_length_penalty_window", 8192)), think_bias=float(g.get("dllm_think_end_bias", 0.0)),
                    think_start=int(g.get("dllm_think_end_start", 4096)), think_win=int(g.get("dllm_think_end_window", 4096)),
                    think_id=g.get("dllm_think_end_token_id"))

    def apply_staged_writes(self):
        self.sampling_states.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    # ---- the penalties and confidence rule of the reference (sampling_clover_moe.py) ---------------
    def _penalize(self, lg, slot, block, p):
        st = self.st
        counts, seen = st.counts[slot], st.counts[slot] > 0
        if p["frequency"]:
            lg.sub_(counts.float() * p["frequency"])
        if p["presence"]:
            lg.sub_(seen * p["presence"])
        if p["repetition"] != 1.0:
            v = lg[:, seen]
            lg[:, seen] = torch.where(v < 0, v * p["repetition"], v / p["repetition"])
        positions = int(st.resp_len[slot]) + torch.arange(1, lg.shape[0] + 1, device=lg.device)
        ramp = lambda s, w, k: ((positions.float() - s) / w).clamp(0, 1) * k
        if p["len_pen"] and self.stop_ids:
            lg[:, self.stop_ids] += ramp(p["len_start"], p["len_win"], p["len_pen"])[:, None]
        if p["think_bias"] and p["think_id"] is not None:
            c = p["think_id"]
            visible = ((block != st.mask_id) & (block == c)).any()
            lg[:, c] += (~(seen[c] | visible)) * ramp(p["think_start"], p["think_win"], p["think_bias"])
        return lg

    def _predict(self, lg, p):
        if p["temperature"] > 0:
            s = lg / p["temperature"]
            if 0 < p["top_k"] < s.shape[-1]:
                s = s.masked_fill(s < s.topk(p["top_k"], -1).values[:, -1:], -float("inf"))
            if p["top_p"] < 1:
                sl, order = s.sort(-1, descending=True)
                pr = sl.softmax(-1)
                sl = sl.masked_fill((pr.cumsum(-1) - pr) > p["top_p"], -float("inf"))
                s = torch.empty_like(s).scatter(-1, order, sl)
            pred = torch.multinomial(s.softmax(-1), 1).squeeze(-1)
        else:
            pred = lg.argmax(-1)
        return pred, lg.softmax(-1).gather(-1, pred[:, None]).squeeze(-1)

    def __call__(self, logits, input_batch, draft_logits=None):
        if _DBG:
            import time
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if self._last_end:
                self._period.append(t0 - self._last_end)
        out = self._call(logits, input_batch, draft_logits)
        if _DBG:
            torch.cuda.synchronize()
            n_calls = len(self._body)
            if os.environ.get('CLOVER_PROF'):
                if n_calls == 40:
                    from torch.profiler import profile, ProfilerActivity
                    self._prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); self._prof.__enter__()
                elif n_calls == 44 and getattr(self, '_prof', None):
                    self._prof.__exit__(None, None, None)
                    _p("CLOVER PROFILE (4 steps) by CUDA time:\n%s", self._prof.key_averages().table(sort_by='cuda_time_total', row_limit=22, max_name_column_width=60))
                    _p("CLOVER PROFILE by CPU time:\n%s", self._prof.key_averages().table(sort_by='cpu_time_total', row_limit=14, max_name_column_width=60))
                    self._prof = None
            self._last_end = time.perf_counter()
            self._body.append(self._last_end - t0)
            if len(self._body) % 20 == 0:
                _p("CLOVER timing: sampler body %.1f ms avg, gap between samplers (forward+sched) %.1f ms avg over last 20 steps",
                   1000 * sum(self._body[-20:]) / 20, 1000 * sum(self._period[-20:]) / max(1, len(self._period[-20:])))
        return out

    def _call(self, logits, input_batch, draft_logits=None):
        st, CL = self.st, self.st.CL
        n = input_batch.num_reqs
        slots = input_batch.idx_mapping_np[:n]
        nl = np.diff(input_batch.cu_num_logits_np[: n + 1])
        sampled, num_sampled = self._sampled[:n], self._num_sampled[:n]
        sampled.zero_()
        num_sampled.zero_()
        draft = self.req_states.draft_tokens
        row = 0
        done_flags = []
        for i in range(n):
            slot = int(slots[i])
            if nl[i] == 0:                                   # a prompt chunk: no logits
                self._maybe_finish_prefill(input_batch, i, slot)
                continue
            W = int(nl[i])
            lg = logits[row: row + W].float()
            row += W
            if st.phase[slot] == COMMIT:                     # replay done: emit the block, arm the next one
                blk = st.canvas[slot]
                sampled[i, :CL] = blk.to(torch.int32)
                num_sampled[i] = CL
                st.counts[slot].index_add_(0, blk, torch.ones_like(blk, dtype=torch.int32))
                st.resp_len[slot] += CL
                self._arm(slot)
                continue
            p = self.params.get(slot) or self._default_params()
            lg[:, st.mask_id] = -float("inf")
            self._penalize(lg, slot, st.canvas[slot], p)
            pred, conf = self._predict(lg, p)
            pend = st.pending[slot]
            conf = conf.masked_fill(~pend, -float("inf"))
            take = (conf > st.threshold) & pend
            best = torch.zeros_like(take)
            best[conf.argmax()] = True
            take = torch.where(take.any(), take, best) & pend
            st.canvas[slot] = torch.where(take, pred, st.canvas[slot])
            st.pending[slot] = pend & ~take
            draft[slot, :CL] = st.canvas[slot]
            if _DBG and self._dbg < 50:
                self._dbg += 1
                top = lg.softmax(-1).max(-1).values
                _p("CLOVER denoise slot=%d resp_len=%d took=%d pending=%d conf_max=%.3f conf_mean=%.3f pred=%s", slot,
                            int(st.resp_len[slot]), int(take.sum()), int(st.pending[slot].sum()), float(conf.max()),
                            float(top.mean()), pred[take].tolist()[:8])
            done_flags.append((slot, ~st.pending[slot].any()))
        if done_flags:
            flags = torch.stack([f for _, f in done_flags]).cpu().tolist()
            for (slot, _), done in zip(done_flags, flags):
                if done:
                    st.phase[slot] = COMMIT                   # next step feeds the resolved block causally
        num_logits = torch.from_numpy(nl.astype(np.int32)).to(logits.device)
        qlen = torch.from_numpy(np.diff(input_batch.query_start_loc_np[: n + 1]).astype(np.int32)).to(logits.device)
        denoised = (num_logits > 0) & (num_sampled == 0)
        rejected = torch.where(denoised, qlen, num_logits - num_sampled)
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                             num_sampled=num_sampled, num_rejected=rejected)

    def _arm(self, slot):
        st = self.st
        st.canvas[slot].fill_(st.mask_id)
        st.pending[slot].fill_(True)
        st.phase[slot] = DENOISE
        self.req_states.draft_tokens[slot, : st.CL] = st.canvas[slot]

    def _maybe_finish_prefill(self, input_batch, i, slot):
        done = (input_batch.num_computed_prefill_tokens_np[i] + input_batch.num_scheduled_tokens[i]
                >= input_batch.prefill_len_np[i])
        if done and self.st.phase[slot] == COMMIT and self.st.resp_len[slot] == 0 and not self.st.pending[slot].any():
            self._arm(slot)
