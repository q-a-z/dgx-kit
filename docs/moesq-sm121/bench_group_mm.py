"""Decode-size benchmark of the paired-4:8 sparse NVFP4 grouped GEMM on a Spark (SM121).

Reports time per call and the bytes the kernel must read (compressed weights + metadata + scales) as GB/s,
next to the GPU's plain device-to-device copy speed. Run in the nightly-gb10 image with the 121a wheel installed:
    python3 bench_group_mm.py
Shapes are Ornith-122B experts: w13 = [2*1024, 3072], w2 = [3072, 1024]; E = distinct experts touched by a
spec-decode step (8 tokens -> ~58, so 64).
"""
import torch
import paired_nvfp4_kernels as pnk
import paired_nvfp4_kernels.reference as ref

ops = torch.ops.paired_nvfp4
dev = "cuda"


def timeit(fn, iters=200, warm=20):
    for _ in range(warm):
        fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def copy_gbps(nbytes=1 << 30):
    a = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    b = torch.empty_like(a)
    us = timeit(lambda: b.copy_(a), iters=20, warm=3)
    return 2 * nbytes / us / 1e3  # read + write, GB/s


def build(M, K, E, counts, max_n):
    torch.manual_seed(0)
    chunks = []
    w_packed, w_sf = [], []
    for i in range(0, E, 8):  # build in slices to keep fp32 temporaries small
        w = ref.prune_paired_4of8(torch.randn(min(8, E - i), M, K, device=dev)).to(torch.bfloat16)
        p, s, _ = ref.quantize_weight(w)
        w_packed.append(p)
        w_sf.append(s)
    w_packed, w_sf = torch.cat(w_packed), torch.cat(w_sf)
    a_comp, e_meta, sfa = ops.compress(w_packed, w_sf, K)
    cnt = torch.full((E,), counts, device=dev, dtype=torch.int32)
    x = torch.randn(E, max_n, K, device=dev).to(torch.bfloat16)
    gscale = torch.ones(E, device=dev, dtype=torch.float32)
    b_act, sfb = ops.quant_act(x, gscale, cnt, M)
    alphas = torch.ones(E, device=dev, dtype=torch.float32)
    return dict(a_comp=a_comp, e_meta=e_meta, sfa=sfa, b_act=b_act, sfb=sfb, alphas=alphas, cnt=cnt)


def main():
    print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "available", pnk.is_available())
    cp = copy_gbps()
    print(f"device copy: {cp:.0f} GB/s (read+write)")
    E = 64
    for name, M, K in (("w13", 2048, 3072), ("w2", 3072, 1024)):
        for tokens in (1, 2, 8):
            max_n = 64
            o = build(M, K, E, tokens, max_n)
            out = torch.zeros(E, max_n, M, device=dev, dtype=torch.bfloat16)
            best = None
            for tac in pnk.all_tactics():
                try:
                    us = timeit(lambda: ops.group_mm(out, o["a_comp"], o["e_meta"], o["sfa"], o["b_act"], o["sfb"],
                                                      o["alphas"], o["cnt"], M, K, *tac), iters=50, warm=5)
                except RuntimeError:
                    continue
                if best is None or us < best[0]:
                    best = (us, tac)
            nbytes = sum(o[k].numel() * o[k].element_size() for k in ("a_comp", "e_meta", "sfa"))
            dense = E * M * K // 2 + E * M * (K // 16)  # dense NVFP4: fp4 + group-16 fp8 scales
            us, tac = best
            print(f"{name} M={M} K={K} E={E} tokens/expert={tokens}: {us:.0f} us (tactic {tac}) | reads {nbytes/1e6:.0f} MB "
                  f"= {nbytes/us/1e3:.0f} GB/s | dense NVFP4 would read {dense/1e6:.0f} MB ({nbytes/dense:.2f}x)")
            del o, out
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
