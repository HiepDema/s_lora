#!/usr/bin/env python
"""Gather dang chay o bao nhieu phan tram bang thong dinh?

Neu index_select dat gan dinh thi 1.673 ms la gia phai tra, het cach. Neu no
chi dat mot phan nho thi viet mot kernel gather rieng se an duoc — va do la
kernel DE viet hon GEMM nhieu, vi khong phai canh tranh voi cuBLAS.
"""
import time

import torch


def timeit(fn, n=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


dev, dt = "cuda", torch.bfloat16
m, k, BT = 14336, 4096, 8 * 512
g = torch.Generator(device="cpu").manual_seed(0)
perm = torch.randperm(m, generator=g).to(dev)
sel, rest = perm[:k].contiguous(), perm[k:].contiguous()
x = torch.randn(BT, m, device=dev, dtype=dt)

B = 2                                     # bf16
print(f"x = {BT} x {m} bf16 = {BT*m*B/1e6:.0f} MB\n")


def gbps(byt, sec):
    return byt / sec / 1e9


# Tran: copy thuan tuy, doc het x ghi het x
t = timeit(lambda: x.clone())
peak = gbps(2 * BT * m * B, t)
print(f"  clone ca x            {t*1e3:7.3f} ms   {peak:6.1f} GB/s   <- coi la tran")

for nm, idx in (("gather sel  (k cot)", sel), ("gather rest (m-k cot)", rest),
                ("gather ca m cot", perm)):
    nc = idx.numel()
    t = timeit(lambda: x.index_select(-1, idx))
    bw = gbps(2 * BT * nc * B, t)          # doc nc cot + ghi nc cot
    print(f"  {nm:<22}{t*1e3:7.3f} ms   {bw:6.1f} GB/s   {100*bw/peak:4.0f}% cua tran")

t_sel = timeit(lambda: x.index_select(-1, sel))
t_rest = timeit(lambda: x.index_select(-1, rest))
tong = (t_sel + t_rest) * 1e3
ly_thuyet = 2 * BT * m * B / (peak * 1e9) * 1e3
print()
print(f"  gather sel + rest hien tai : {tong:7.3f} ms")
print(f"  neu dat toc do clone       : {ly_thuyet:7.3f} ms")
print(f"  du dia                     : {tong/ly_thuyet:7.2f}x")
print()
print("  Khoang cach S-LoRA/LoRA la 1.639 ms. Neu gather dat toc do clone thi")
print(f"  khoang do con {tong - ly_thuyet:.3f} ms bot di, tuc con "
      f"{1.639 - (tong - ly_thuyet):.3f} ms.")
