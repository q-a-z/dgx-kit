"""Ternary-Bonsai-27B-unpacked (BF16, every weight in {0, +-s} per 128 group) -> NVFP4 weight-only (compressed-tensors nvfp4-pack-quantized).

A ternary group is {0, +-s}, so each 16-group needs only one E2M1 code c (|x| = c * fp8scale * ws2) and an FP8 scale; the code is picked
per group from {0.5,1,1.5,2,3,4,6} to make the FP8 rounding of s/c as small as possible. The global scale is one constant for the whole
model, so fused projections (qkv, gate_up) share it, as vLLM requires.
Linear layers of the text model and lm_head become NVFP4; embed_tokens, the vision tower, conv1d, norms and in_proj_a/b stay BF16.

usage: bonsai27_nvfp4.py SRC DST [--stats] [--keep-lm-head] [--only 1,2,..]
  --stats   read only: print the range of the group scales (decides the global scale), write nothing
  --only    shard numbers, for tests; no index/config written then
"""
import sys, os, json, shutil, re, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]
STATS = '--stats' in sys.argv
KEEP_HEAD = '--keep-lm-head' in sys.argv
ONLY = {int(x) for x in sys.argv[sys.argv.index('--only') + 1].split(',')} if '--only' in sys.argv else None
dev = 'cuda'
CODES = torch.tensor([0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=dev)   # E2M1 magnitudes 1..7
LIN = re.compile(r'^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)\.weight$')
TARGET = lambda k: bool(LIN.match(k)) or (k == 'lm_head.weight' and not KEEP_HEAD)


def group_amax(w, g=16):
    return w.reshape(w.shape[0], -1, g).abs().amax(-1)


def quantize(w, ws2, g=16):
    w = w.to(dev, torch.float32)
    O, I = w.shape
    a = group_amax(w, g)                                           # O x I/g, = s for a live group, 0 for a dead one
    cand = (a.unsqueeze(-1) / CODES / ws2).clamp(max=448.0).to(torch.float8_e4m3fn).float()   # O x I/g x 7
    err = (cand * CODES * ws2 - a.unsqueeze(-1)).abs()
    best = err.argmin(-1)
    s8 = cand.gather(-1, best.unsqueeze(-1)).squeeze(-1).to(torch.float8_e4m3fn)
    code = (best + 1).to(torch.uint8)
    dead = a == 0
    s8 = torch.where(dead, torch.ones_like(s8), s8)
    code = torch.where(dead, torch.zeros_like(code), code)
    nz = w.reshape(O, I // g, g) != 0
    nib = (code.unsqueeze(-1) * nz.to(torch.uint8)) | ((w.reshape(O, I // g, g) < 0).to(torch.uint8) << 3)
    nib = nib.reshape(O, I)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    mag = torch.cat([torch.zeros(1, device=dev), CODES])[code.long()]
    deq = torch.sign(w.reshape(O, I // g, g)) * mag.unsqueeze(-1) * (s8.float() * ws2).unsqueeze(-1)
    return packed.cpu(), s8.cpu(), deq.reshape(O, I), w


idx = json.load(open(f'{SRC}/model.safetensors.index.json'))
shards = sorted(set(idx['weight_map'].values()))

# pass 1: range of the live group scales over every tensor that will be quantized
amax_all, amin_all = 0.0, float('inf')
for sh in shards:
    with safe_open(f'{SRC}/{sh}', 'pt') as f:
        for k in f.keys():
            if TARGET(k):
                a = group_amax(f.get_tensor(k).to(dev, torch.float32))
                live = a[a > 0]
                amax_all = max(amax_all, live.max().item()); amin_all = min(amin_all, live.min().item())
print(f'group scale s: min {amin_all:.6g} max {amax_all:.6g} ratio {amax_all/amin_all:.1f}', flush=True)
WS2 = amax_all / (6.0 * 448.0)
if STATS:
    sys.exit(0)

assert not os.path.exists(DST) or ONLY, 'destination exists'
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    p = os.path.join(SRC, f)
    if os.path.isfile(p) and not f.endswith('.safetensors') and f not in ('model.safetensors.index.json', 'config.json') and not f.startswith('.'):
        shutil.copy(p, os.path.join(DST, f))

new_wm, n, sum_rel, max_rel = {}, 0, 0.0, 0.0
G = torch.tensor([1.0 / WS2], dtype=torch.float32)
t0 = time.time()
for sh in shards:
    num = int(re.search(r'model-(\d+)-of', sh).group(1))
    if ONLY and num not in ONLY:
        continue
    out = {}
    with safe_open(f'{SRC}/{sh}', 'pt') as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            if TARGET(k):
                assert t.dtype == torch.bfloat16 and t.shape[1] % 16 == 0, (k, t.dtype, t.shape)
                parts = [quantize(t[i:i + 8192], WS2) for i in range(0, t.shape[0], 8192)]   # row chunks keep the GPU temporaries small
                pk, s8 = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
                rel = (sum(((p[2] - p[3]).norm() ** 2).item() for p in parts) / sum((p[3].norm() ** 2).item() for p in parts)) ** 0.5
                del parts
                n += 1; sum_rel += rel; max_rel = max(max_rel, rel)
                base = k[:-len('.weight')]
                out[base + '.weight_packed'] = pk
                out[base + '.weight_scale'] = s8
                out[base + '.weight_global_scale'] = G.clone()
            else:
                out[k] = t
    for k in out:
        new_wm[k] = sh
    save_file(out, f'{DST}/{sh}', metadata=meta or {'format': 'pt'})
    print(f'[{time.time()-t0:5.0f}s] {sh}: {os.path.getsize(f"{DST}/{sh}")/1e9:.2f} GB | quantized {n} tensors, mean rel err {sum_rel/max(n,1):.5f} max {max_rel:.5f}', flush=True)
    torch.cuda.empty_cache()

if ONLY:
    print('test run: no index/config written')
    sys.exit(0)

idx['weight_map'] = new_wm
json.dump(idx, open(f'{DST}/model.safetensors.index.json', 'w'))
cfg = json.load(open(f'{SRC}/config.json'))
cfg['quantization_config'] = {
    'config_groups': {'NVFP4A16': {
        'format': 'nvfp4-pack-quantized', 'input_activations': None, 'output_activations': None,
        'targets': ['Linear'],
        'weights': {'actorder': None, 'block_structure': None, 'dynamic': False, 'group_size': 16, 'num_bits': 4, 'observer': None,
                    'observer_kwargs': {}, 'scale_dtype': 'torch.float8_e4m3fn', 'strategy': 'tensor_group', 'symmetric': True,
                    'type': 'float', 'zp_dtype': None}}},
    'format': 'nvfp4-pack-quantized', 'global_compression_ratio': None,
    'ignore': ['re:.*embed_tokens.*', 're:.*visual.*', 're:.*conv1d.*', 're:.*norm.*', 're:.*mtp.*', r're:.*linear_attn\.in_proj_[ab]$']
              + (['re:.*lm_head$'] if KEEP_HEAD else []),
    'kv_cache_scheme': None, 'quant_method': 'compressed-tensors', 'quantization_status': 'compressed', 'transform_config': {}, 'version': '0.19.0'}
json.dump(cfg, open(f'{DST}/config.json', 'w'), indent=2)
print('built', DST)
