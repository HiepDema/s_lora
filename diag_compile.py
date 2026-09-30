#!/usr/bin/env python
"""Muc 2: torch.compile co fuse duoc gather vao prologue cua GEMM khong?

Da biet: toan bo 18.4% chenh lech la gather (1.67 ms tren 9.73 ms), va hai
GEMM noi tiep khong ton them gi. Da thu hai cach, ca hai hong:

  muc 1 (mot gather + slice) : CHAM hon ban goc, 7.49 vs 6.56 ms
  muc 3 (Triton gop vao GEMM): cham gap 32 lan, vi GEMM tu viet thua cuBLAS

Con lai mot kha nang: de inductor tu lo. No dung template GEMM rieng va co the
gan gather lam prologue, tuc van giu duoc GEMM toi uu ma bo duoc ban sao.
"""
import time

import torch


def timeit(fn, n=30, warmup=15):
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
XT = torch.randn(m - k, k, device=dev, dtype=dt)
C = torch.randn(k, k, device=dev, dtype=dt)
Ak = torch.randn(k, r, device=dev, dtype=dt)
Bk = torch.randn(r, k, device=dev, dtype=dt)
W = torch.randn(m, k, device=dev, dtype=dt)
A = torch.randn(m, r, device=dev, dtype=dt)
B = torch.randn(r, k, device=dev, dtype=dt)


def slora(x):
    h = x.index_select(-1, sel) + x.index_select(-1, rest) @ XT
    return h @ C + ((h @ Ak) @ Bk)


def lora(x):
    return x @ W + ((x @ A) @ B)


print(f"gate_proj: x({BT}x{m}) -> ({BT}x{k}),  bf16, A10\n")
t_lora = timeit(lambda: lora(x))
t_eager = timeit(lambda: slora(x))
print(f"  LoRA (eager)                 {t_lora:7.3f} ms")
print(f"  S-LoRA (eager)               {t_eager:7.3f} ms   {t_eager/t_lora:5.2f}x")

for mode in ("default", "max-autotune"):
    try:
        cs = torch.compile(slora, mode=mode, dynamic=False)
        t0 = time.perf_counter()
        cs(x)
        torch.cuda.synchronize()
        biendich = time.perf_counter() - t0
        t = timeit(lambda: cs(x))
        print(f"  S-LoRA (compile {mode:<12}) {t:7.3f} ms   {t/t_lora:5.2f}x"
              f"   [bien dich {biendich:.0f}s]")
    except Exception as e:                                   # noqa: BLE001
        print(f"  S-LoRA (compile {mode}) LOI: {type(e).__name__}: {str(e)[:70]}")

cl = torch.compile(lora, mode="max-autotune", dynamic=False)
cl(x)
torch.cuda.synchronize()
print(f"  LoRA (compile max-autotune)  {timeit(lambda: cl(x)):7.3f} ms   (de doi chieu)")
