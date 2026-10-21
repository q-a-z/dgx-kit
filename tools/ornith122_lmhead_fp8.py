"""Optional step after ornith122_convert.py: a twin of the converted checkpoint with an FP8 per-channel lm_head.

usage: ornith122_lmhead_fp8.py SRC DST      (needs a GPU and torch; reads and rewrites only the shard that holds lm_head)
Every other file in DST is a symlink to SRC, so SRC must stay where it is and be mounted at the same path when serving.
The config edit has two parts, and both are needed: take lm_head out of `ignore`, and add `re:.*lm_head$` to the FP8
group's targets (the group targets the class name `Linear`, which `ParallelLMHead` does not match; vLLM's prefix for the
head can carry a `language_model.` part, so the regex is not anchored at the start).
Measured on a GB10: code decode 56.9 vs about 54 t/s (inside the run-to-run noise), prose unchanged, 0.8 GB less
weight, weight rounding error 2.6 %. Accuracy against BF16 was not measured."""
import sys, os, json, torch
from safetensors import safe_open
from safetensors.torch import save_file
SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
for n in os.listdir(SRC):
    if not os.path.lexists(f'{DST}/{n}'):
        os.symlink(f'{SRC}/{n}', f'{DST}/{n}')
idx = json.load(open(f'{SRC}/model.safetensors.index.json'))
sh = idx['weight_map']['lm_head.weight']
out = {}
with safe_open(f'{SRC}/{sh}', 'pt', device='cpu') as f:
    meta = f.metadata()
    for k in f.keys():
        t = f.get_tensor(k)
        if k == 'lm_head.weight':
            assert t.dtype == torch.bfloat16
            w = t.to('cuda', torch.float32)
            sc = (w.abs().amax(1, keepdim=True) / 448.0).clamp(min=1e-12)
            q = (w / sc).to(torch.float8_e4m3fn)
            rel = ((q.float() * sc - w).norm() / w.norm()).item()
            print('lm_head rel err', rel)
            out[k] = q.cpu()
            out['lm_head.weight_scale'] = sc.to(torch.bfloat16).cpu()
        else:
            out[k] = t
os.remove(f'{DST}/{sh}')
save_file(out, f'{DST}/{sh}', metadata=meta or {'format': 'pt'})
idx['weight_map']['lm_head.weight_scale'] = sh
os.remove(f'{DST}/model.safetensors.index.json')
json.dump(idx, open(f'{DST}/model.safetensors.index.json', 'w'))
cfg = json.load(open(f'{SRC}/config.json'))
q = cfg['quantization_config']
q['ignore'] = [x for x in q['ignore'] if x != 'lm_head']
g0 = q['config_groups']['config_group_0']
g0['targets'] = [t for t in g0['targets'] if t != 're:.*lm_head$'] + ['re:.*lm_head$']
os.remove(f'{DST}/config.json')
json.dump(cfg, open(f'{DST}/config.json', 'w'), indent=2)
print('ok', DST, q['ignore'])
