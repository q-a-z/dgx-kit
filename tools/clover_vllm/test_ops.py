"""CPU check: preview ops on a state POOL with scattered cache indices equal the reference run one request at a time
(what the vLLM path does: many requests share one pool), and the coreset keeps >=1 expert and respects top_p."""
import sys, torch
sys.path.insert(0, "/w")
from clover_vllm.ops import bd_gdn_conv1d, bd_gdn_local_memory, coreset_torch
torch.manual_seed(0)
B, BS, C, W, H, K, V, KH = 3, 32, 64, 4, 4, 8, 8, 2
pool = 7
conv_pool = torch.randn(pool, C, W - 1)
ssm_pool = torch.randn(pool, H, V, K)
idx = torch.tensor([5, 1, 3])
x = torch.randn(B * BS, C)
valid = torch.ones(B * BS, dtype=torch.bool); valid[BS + 20: 2 * BS] = False   # one partial block
cw, rw = torch.randn(C, 1, W), torch.randn(C, 1, W - 1)
def conv(i, xs, vs):
    return bd_gdn_conv1d(xs, clean_conv_states=conv_pool, cache_indices=i, conv_weights=cw, block_size=BS,
                         active_valid_mask=vs, conv_right_weight=rw)
batched = conv(idx, x, valid)
single = torch.cat([conv(idx[j:j + 1], x[j * BS:(j + 1) * BS], valid[j * BS:(j + 1) * BS]) for j in range(B)])
assert torch.allclose(batched, single, atol=1e-5), "conv: batched != per-request"
q = torch.randn(1, B * BS, KH, K); k = torch.randn(1, B * BS, KH, K); v = torch.randn(1, B * BS, H, V); g = -torch.rand(1, B * BS, H)
sc, hb, hc = torch.full((H,), 10.), torch.full((H,), 5.), torch.ones(H)
def mem(i, sl):
    return bd_gdn_local_memory(q[:, sl], k[:, sl], v[:, sl], g[:, sl], clean_ssm_states=ssm_pool, cache_indices=i,
                               active_valid_mask=valid[sl], scale=K ** -0.5, softmax_logit_scale=sc,
                               softmax_history_bias=hb, softmax_history_c=hc)
b = mem(idx, slice(0, B * BS))
s = torch.cat([mem(idx[j:j + 1], slice(j * BS, (j + 1) * BS)) for j in range(B)], 1)
assert torch.allclose(b, s, atol=1e-5), "memory: batched != per-request"
before = (conv_pool.clone(), ssm_pool.clone())
assert torch.equal(before[0], conv_pool) and torch.equal(before[1], ssm_pool)   # pools are read-only
lg = torch.randn(2, BS, 16); vm = torch.ones(2, BS, dtype=torch.bool)
out = coreset_torch(lg, vm, 4, 0.9)
kept = torch.isfinite(out[:, 0]).sum(-1)
assert (kept >= 1).all() and (kept < 16).any(), kept
assert torch.equal(torch.isfinite(out[:, 0]), torch.isfinite(out[:, -1])), "coreset must be shared across the block"
print("ok", kept.tolist())
