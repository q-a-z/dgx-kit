"""Clover-35B-A3B-Preview BF16 -> mixed checkpoint for a DGX Spark, quality-first.

Routed experts (fused 3-D tensors in the source) are split per expert and become NVFP4 weight-only (compressed-tensors
nvfp4-pack-quantized, same layout as the Ornith converter). Attention, linear-attention in/out projections and the shared expert
become FP8 per-channel weight-only. Everything numerically delicate stays BF16: embeddings, lm_head, the router and shared-expert
gate, in_proj_a/b (feed the GDN gate), norms, conv weights, A_log, dt_bias and Clover's own bd_* / conv_right_weight tensors.

usage: clover_nvfp4.py SRC DST [--only 1,2]   (--only = shard numbers, for tests; no index/config written then)
Runs on the CPU unless DEV=cuda. Output is flushed every ~3 GB into model-sNNN.safetensors.
"""
import sys, os, json, shutil, re, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]
ONLY = {int(x) for x in sys.argv[sys.argv.index('--only') + 1].split(',')} if '--only' in sys.argv else None
dev = os.environ.get('DEV', 'cpu')
ZERO = [0]
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=dev)
MIDS = (E2M1[1:] + E2M1[:-1]) / 2
FUSED = re.compile(r'^(model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$')
FP8_LIN = re.compile(r'\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.shared_expert\.(gate|up|down)_proj)\.weight$')


def nvfp4_quantize(w, g=16):
    w = w.to(dev, torch.float32)
    O, I = w.shape
    assert I % g == 0
    amax = w.abs().max()
    if amax == 0:
        ZERO[0] += 1
        return torch.zeros(O, I // 2, dtype=torch.uint8), torch.zeros(O, I // g, dtype=torch.float8_e4m3fn), torch.tensor([6.0 * 448.0]), torch.zeros(O, I, device=dev)
    ws2 = amax / (6.0 * 448.0)
    wb = w.reshape(O, I // g, g)
    s8 = ((wb.abs().amax(-1) / 6.0) / ws2).clamp(max=448.0).to(torch.float8_e4m3fn)
    sd = s8.float() * ws2
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    x = wb / sd.unsqueeze(-1)
    code = torch.bucketize(x.abs(), MIDS)
    nib = (code.to(torch.uint8) | ((x < 0).to(torch.uint8) << 3)).reshape(O, I)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    deq = (torch.sign(x) * E2M1[code] * sd.unsqueeze(-1)).reshape(O, I)
    return packed.cpu(), s8.cpu(), (1.0 / ws2).reshape(1).float().cpu(), deq


def fp8_row(w):
    w = w.to(dev, torch.float32)
    sc = (w.abs().amax(1, keepdim=True) / 448.0).clamp(min=1e-12)
    q = (w / sc).to(torch.float8_e4m3fn)
    return q.cpu(), sc.cpu(), q.float() * sc


assert not os.path.exists(DST) or ONLY, 'destination exists'
os.makedirs(DST, exist_ok=True)
idx = json.load(open(f'{SRC}/model.safetensors.index.json'))
wm = idx['weight_map']
shards = sorted({v for v in wm.values() if re.match(r'model-\d+-of-\d+\.safetensors$', v)})
extra_files = sorted({v for v in wm.values()} - set(shards))          # bd_gdn_*.safetensors: copied unchanged
for f in os.listdir(SRC):
    p = os.path.join(SRC, f)
    if os.path.isfile(p) and not f.endswith('.safetensors') and f not in ('model.safetensors.index.json', 'config.json') and not f.startswith('.'):
        shutil.copy(p, os.path.join(DST, f))
if not ONLY:
    for f in extra_files:
        shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))

new_wm = {k: v for k, v in wm.items() if v in extra_files}
stats = {'expert': [0, 0.0, 0.0], 'fp8': [0, 0.0, 0.0]}
t0 = time.time()
out, osize, nout = {}, 0, [0]


def flush():
    global out, osize
    if not out:
        return
    nout[0] += 1
    name = f'model-s{nout[0]:03d}.safetensors'
    save_file(out, f'{DST}/{name}', metadata={'format': 'pt'})
    for k in out:
        new_wm[k] = name
    print(f'[{time.time()-t0:5.0f}s] {name} {os.path.getsize(f"{DST}/{name}")/1e9:.2f} GB zero-tensors {ZERO[0]} | '
          + ' | '.join(f'{k} n={v[0]} mean-rel-err={v[1]/max(v[0],1):.4f} max={v[2]:.4f}' for k, v in stats.items() if v[0]), flush=True)
    out, osize = {}, 0


def put(k, t):
    global osize
    out[k] = t
    osize += t.numel() * t.element_size()
    if osize > 3e9:
        flush()


def expert(base, w):
    ref = w.to(dev, torch.float32)
    pk, s8, G, deq = nvfp4_quantize(ref)
    rel = ((deq - ref).norm() / ref.norm().clamp(min=1e-30)).item()
    s = stats['expert']; s[0] += 1; s[1] += rel; s[2] = max(s[2], rel)
    put(base + '.weight_packed', pk); put(base + '.weight_scale', s8); put(base + '.weight_global_scale', G)


for sh in shards:
    if ONLY and int(re.search(r'model-(\d+)-of', sh).group(1)) not in ONLY:
        continue
    with safe_open(f'{SRC}/{sh}', 'pt') as f:
        for k in list(f.keys()):
            t = f.get_tensor(k)
            m = FUSED.match(k)
            if m:
                base, kind = m.groups()
                assert t.dtype == torch.bfloat16 and t.dim() == 3, (k, t.dtype, t.shape)
                for e in range(t.shape[0]):
                    if kind == 'gate_up_proj':               # [E, 2I, H]: first I rows gate, last I rows up
                        half = t.shape[1] // 2
                        expert(f'{base}.{e}.gate_proj', t[e, :half])
                        expert(f'{base}.{e}.up_proj', t[e, half:])
                    else:                                    # [E, H, I]
                        expert(f'{base}.{e}.down_proj', t[e])
            elif FP8_LIN.search(k):
                assert t.dtype == torch.bfloat16, (k, t.dtype)
                q, sc, deq = fp8_row(t)
                ref = t.to(dev, torch.float32)
                rel = ((deq - ref).norm() / ref.norm()).item()
                s = stats['fp8']; s[0] += 1; s[1] += rel; s[2] = max(s[2], rel)
                put(k, q); put(k[:-len('.weight')] + '.weight_scale', sc)
            else:
                put(k, t)
flush()

if ONLY:
    print('test run: no index/config written')
    sys.exit(0)

idx['weight_map'] = new_wm
json.dump(idx, open(f'{DST}/model.safetensors.index.json', 'w'))
cfg = json.load(open(f'{SRC}/config.json'))
fp8 = {'format': 'float-quantized', 'input_activations': None, 'output_activations': None,
       'targets': [r're:.*(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.shared_expert\.(gate|up|down)_proj)$'],
       'weights': {'num_bits': 8, 'type': 'float', 'strategy': 'channel', 'dynamic': False, 'symmetric': True,
                   'observer': 'memoryless_minmax', 'observer_kwargs': {}, 'group_size': None, 'block_structure': None,
                   'actorder': None, 'scale_dtype': None, 'zp_dtype': None}}
nv = {'format': 'nvfp4-pack-quantized', 'input_activations': None, 'output_activations': None,
      'targets': [r're:.*\.mlp\.experts\.\d+\.(gate|up|down)_proj'],
      'weights': {'num_bits': 4, 'type': 'float', 'strategy': 'tensor_group', 'group_size': 16, 'dynamic': False, 'symmetric': True,
                  'observer': 'minmax', 'scale_dtype': 'torch.float8_e4m3fn', 'zp_dtype': None}}
cfg['quantization_config'] = {'quant_method': 'compressed-tensors', 'format': 'mixed-precision', 'quantization_status': 'compressed',
                              'config_groups': {'config_group_0': fp8, 'config_group_1': nv},
                              'ignore': ['lm_head', r're:.*mlp\.gate$', r're:.*shared_expert_gate$', r're:.*linear_attn\.(in_proj_a|in_proj_b)$'],
                              'kv_cache_scheme': None, 'sparsity_config': {}, 'transform_config': {}, 'global_compression_ratio': None}
json.dump(cfg, open(f'{DST}/config.json', 'w'), indent=2)
print('built', DST, 'zero tensors', ZERO[0])
