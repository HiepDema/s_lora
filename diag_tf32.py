#!/usr/bin/env python
"""Sai lech fp32 cua Triton la LOI KERNEL hay chi la TF32?

tl.dot tren Ampere mac dinh dung TF32 cho dau vao fp32 (10 bit mantissa), con
torch mac dinh dung fp32 that. Neu bat TF32 cho ca torch ma hai ben khop nhau
thi kernel dung, chi la dat do chinh xac khac.
"""
import torch

import slora_kernels as K

dev = "cuda"
BT, m, k = 512, 4096, 1024
g = torch.Generator(device="cpu").manual_seed(0)
perm = torch.randperm(m, generator=g).to(dev)
sel, rest = perm[:k].contiguous(), perm[k:].contiguous()

Xd = torch.randn(k, m - k, dtype=torch.float64, device=dev)
xd = torch.randn(BT, m, dtype=torch.float64, device=dev)
chuan = xd[:, sel] + xd[:, rest] @ Xd.t()
s = chuan.abs().max().item()

X, x = Xd.float(), xd.float()
XT = X.t().contiguous()
tri = K.PreInProj.apply(x, sel, rest, XT).double()

for tf32 in (False, True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    cub = (x[:, sel] + x[:, rest] @ X.t()).double()
    print(f"  torch TF32={str(tf32):<5}  sai so cuBLAS {(cub-chuan).abs().max().item()/s:.2e}"
          f"   lech voi triton {(tri-cub).abs().max().item()/s:.2e}")
print(f"  {'':<19} sai so triton {(tri-chuan).abs().max().item()/s:.2e}")
print()
print("  Neu dong TF32=True cho 'lech voi triton' nho hon han dong TF32=False")
print("  thi kernel DUNG, chi khac o che do do chinh xac.")
