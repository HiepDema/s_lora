#!/usr/bin/env python
"""20% cham cua S-LoRA nam o dau, neu gather chi chiem 13%?

Gia thuyet con lai: dang phan ra tach mot GEMM to thanh HAI GEMM phu thuoc
nhau. Cung tong so phep nhan, nhung hai kernel noi tiep thi it song song hon
mot kernel, va GEMM thu hai phai doi GEMM thu nhat xong.

    S-LoRA:  gather -> (x_rest @ X.T) -> (h @ C)        hai GEMM noi tiep
    LoRA  :  (x @ W)  ||  (x @ A) @ B                   mot GEMM to + hai GEMM ti
"""
import time

import torch


def timeit(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


dev, dt = "cuda", torch.bfloat16
n, m, BT, r = 4096, 14336, 8 * 512, 64
k = n
g = torch.Generator(device="cpu").manual_seed(0)
perm = torch.randperm(m, generator=g).to(dev)
sel, rest = perm[:k].contiguous(), perm[k:].contiguous()

x = torch.randn(BT, m, device=dev, dtype=dt)
W = torch.randn(m, k, device=dev, dtype=dt)          # x @ W -> (BT, k)
XT = torch.randn(m - k, k, device=dev, dtype=dt)     # x_rest @ XT -> (BT, k)
C = torch.randn(k, k, device=dev, dtype=dt)
A = torch.randn(m, r, device=dev, dtype=dt)
B = torch.randn(r, k, device=dev, dtype=dt)
Ak = torch.randn(k, r, device=dev, dtype=dt)
Bk = torch.randn(r, k, device=dev, dtype=dt)

print(f"gate_proj: x({BT}x{m}) -> ({BT}x{k}),  rank {r}")
print(f"  so phep nhan cua ca hai duong deu la {BT*m*k/1e12:.2f} T\n")


def slora():
    h = x.index_select(-1, sel) + x.index_select(-1, rest) @ XT
    return h @ C + ((h @ Ak) @ Bk)


def lora():
    return x @ W + ((x @ A) @ B)


def chi_gemm_to():
    return x @ W


def chi_hai_gemm_noi_tiep():
    h = x[:, :m - k] @ XT
    return h @ C


rows = [
    ("LoRA: x@W + (x@A)@B", lora),
    ("S-LoRA: gather + hai GEMM", slora),
    ("chi mot GEMM to (x@W)", chi_gemm_to),
    ("chi hai GEMM noi tiep", chi_hai_gemm_noi_tiep),
    ("chi gather sel+rest", lambda: (x.index_select(-1, sel),
                                     x.index_select(-1, rest))),
]
base = None
for nm_, fn in rows:
    t = timeit(fn)
    if base is None:
        base = t
    print(f"  {nm_:<32}{t:7.3f} ms   {t/base:5.2f}x so voi LoRA")

print()
print("  Neu 'chi hai GEMM noi tiep' da cham hon 'chi mot GEMM to' mot khoang")
print("  bang phan lon chenh lech LoRA/S-LoRA, thi nguyen nhan la CAU TRUC hai")
print("  GEMM phu thuoc, khong phai gather — va khong kernel nao sua duoc, vi")
print("  hai GEMM do that su phu thuoc nhau.")
