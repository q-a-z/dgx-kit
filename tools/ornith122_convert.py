"""Ornith-1.5-122B FP8 (compressed-tensors, per-channel) -> mixed checkpoint for a DGX Spark.

Routed experts become NVFP4 weight-only (compressed-tensors nvfp4-pack-quantized layout); the shared experts and
the full-attention projections stay FP8 as in the source; the MTP head, embeddings, lm_head and router stay BF16.
Variant B (recommended) also turns the BF16 linear-attention projections into FP8 per-channel, weight-only.
(The lm_head is deliberately left alone: vLLM has no parameter for a quantized lm_head in this model class.)

usage: ornith122_convert.py SRC DST VARIANT(A|B) [--only 1,2,..]   (--only = shard numbers, for tests; no index/config written then)
Streams one source shard -> one output shard, on the GPU (about 5 minutes for the whole model).
"""
import sys, os, json, shutil, copy, re, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST, VAR = sys.argv[1], sys.argv[2], sys.argv[3]
ONLY = None
if '--only' in sys.argv:
    ONLY = {int(x) for x in sys.argv[sys.argv.index('--only') + 1].split(',')}
assert VAR in ('A', 'B')
dev = 'cuda'
ZERO = [0]
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=dev)
MIDS = (E2M1[1:] + E2M1[:-1]) / 2
EXPERT = re.compile(r'\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$')
LINATT = re.compile(r'\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$')


def nvfp4_quantize(w, g=16):
    w = w.to(dev, torch.float32)
    O, I = w.shape
    assert I % g == 0
    amax = w.abs().max()
    if amax == 0:                                          # dead (all-zero) tensor: store zeros with a harmless scale
        ZERO[0] += 1
        z = torch.zeros(O, I // 2, dtype=torch.uint8)
        return z, torch.zeros(O, I // g, dtype=torch.float8_e4m3fn), torch.tensor([6.0 * 448.0]), torch.zeros(O, I, device=dev)
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
shards = sorted(set(wm.values()))
for f in os.listdir(SRC):
    p = os.path.join(SRC, f)
    if os.path.isfile(p) and not f.endswith('.safetensors') and f not in ('model.safetensors.index.json', 'config.json') and not f.startswith('.'):
        shutil.copy(p, os.path.join(DST, f))
    elif os.path.isdir(p) and f == 'assets':
        pass

new_wm = {}
stats = {'expert': [0, 0.0, 0.0], 'lin': [0, 0.0, 0.0]}   # count, sum rel err, max rel err
t0 = time.time()
for sh in shards:
    n = int(re.search(r'model-(\d+)-of', sh).group(1))
    if ONLY and n not in ONLY:
        continue
    out = {}
    with safe_open(f'{SRC}/{sh}', 'pt') as f:
        meta = f.metadata()
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if k.startswith('mtp.'):
                out[k] = t                                 # MTP head stays as in the source (BF16, in the ignore list)
            elif EXPERT.search(k):
                base = k[:-len('.weight')]
                scale = f.get_tensor(base + '.weight_scale')
                assert t.dtype == torch.float8_e4m3fn and scale.shape == (t.shape[0], 1), (k, t.dtype, scale.shape)
                ref = t.to(dev, torch.float32) * scale.to(dev, torch.float32)
                pk, s8, G, deq = nvfp4_quantize(ref)
                rel = ((deq - ref).norm() / ref.norm().clamp(min=1e-30)).item()
                s = stats['expert']; s[0] += 1; s[1] += rel; s[2] = max(s[2], rel)
                out[base + '.weight_packed'] = pk
                out[base + '.weight_scale'] = s8
                out[base + '.weight_global_scale'] = G
            elif k.endswith('.weight_scale') and EXPERT.search(k[:-len('_scale')]):
                continue                                   # replaced above
            elif VAR == 'B' and LINATT.search(k):
                assert t.dtype == torch.bfloat16, (k, t.dtype)
                q, sc, deq = fp8_row(t)
                ref = t.to(dev, torch.float32)
                rel = ((deq - ref).norm() / ref.norm()).item()
                s = stats['lin']; s[0] += 1; s[1] += rel; s[2] = max(s[2], rel)
                base = k[:-len('.weight')]
                out[k] = q
                out[base + '.weight_scale'] = sc
            else:
                out[k] = t
    for k in out:
        new_wm[k] = sh
    save_file(out, f'{DST}/{sh}', metadata=meta or {'format': 'pt'})
    print(f'[{time.time()-t0:5.0f}s] zero-tensors so far {ZERO[0]} | {sh}: {os.path.getsize(f"{DST}/{sh}")/1e9:.2f} GB '
          + ' | '.join(f'{k} n={v[0]} mean-rel-err={v[1]/max(v[0],1):.4f} max={v[2]:.4f}' for k, v in stats.items() if v[0]), flush=True)
    torch.cuda.empty_cache()

if ONLY:
    print('test run: no index/config written')
    sys.exit(0)

idx['weight_map'] = new_wm
json.dump(idx, open(f'{DST}/model.safetensors.index.json', 'w'))
cfg = json.load(open(f'{SRC}/config.json'))
q = cfg['quantization_config']
g0 = q['config_groups']['config_group_0']
g1 = copy.deepcopy(g0)
g1['weights'] = {'num_bits': 4, 'type': 'float', 'strategy': 'tensor_group', 'group_size': 16, 'dynamic': False, 'symmetric': True,
                 'observer': 'minmax', 'scale_dtype': 'torch.float8_e4m3fn', 'zp_dtype': None}
g1['input_activations'] = None
g1['format'] = 'nvfp4-pack-quantized'
g1['targets'] = [r're:.*\.mlp\.experts\.\d+\.(gate|up|down)_proj']
q['config_groups']['config_group_1'] = g1
if VAR == 'B':
    q['ignore'] = [x for x in q['ignore'] if x != 're:.*linear_attn.*'] + [r're:.*linear_attn\.(in_proj_a|in_proj_b)']   # lm_head stays in the ignore list
json.dump(cfg, open(f'{DST}/config.json', 'w'), indent=2)
print('ignore:', q['ignore'])
print('group_0:', json.dumps(g0))
print('group_1:', json.dumps(g1))
print('built', DST, VAR)
