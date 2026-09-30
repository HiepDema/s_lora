#!/usr/bin/env python
"""Kiem slora_kernels truoc khi dung vao bat cu run nao.

    python test_kernels.py            # phan chay duoc o CPU
    python test_kernels.py --cuda     # them phan Triton, can GPU

Vi sao phai ky den the: mot backward sai se lam S-LoRA hoc kem di ma KHONG
bao loi. Ca paper dua tren phep so sanh S-LoRA voi LoRA, nen mot kernel sai
thang lang se bien thanh mot ket luan sai ma khong ai phat hien duoc. Moi
duong nhanh o day deu phai khop voi ban goc den muc sai so lam tron.
"""
from __future__ import annotations

import argparse
import sys

import torch

import slora_kernels as K


def ref_pre(x, sel, rest, X):
    """Dung nguyen van RowSpaceLinear._in_proj."""
    return x.index_select(-1, sel) + x.index_select(-1, rest) @ X.t()


def ref_post(h, X, inv):
    """Dung nguyen van RowSpaceLinear._out_proj."""
    return torch.cat([h, h @ X.t()], dim=-1).index_select(-1, inv)


def make_case(n, m, device, dtype, seed=0):
    """Tao mot phep hoan vi va X giong cai phan ra that sinh ra.

    Bat doi xung phai nho: huong 'wide' (m>n) cho X la k x (m-k), con huong
    'tall' (n>m) cho X la (n-k) x k. Dat nham chieu thi thuong bao loi ngay,
    nhung neu hai chieu tinh co bang nhau thi no chay im ru va cho so sai.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    k, big = min(n, m), max(n, m)
    perm = torch.randperm(big, generator=g).to(device)
    sel, rest = perm[:k].contiguous(), perm[k:].contiguous()
    inv = torch.argsort(torch.cat([sel, rest]))
    shape = (k, big - k) if m > n else (big - k, k)
    X = torch.randn(*shape, generator=g, dtype=dtype).to(device)
    return k, sel, rest, inv, X


def check(name, got, exp, tol):
    err = (got - exp).abs().max().item()
    ok = err <= tol
    print(f"  {'OK ' if ok else 'SAI'} {name:<34} sai lech {err:.3e}  (nguong {tol:.0e})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    ok = True

    print("=== MUC 1: duong nhanh thuan PyTorch (CPU) ===")
    dt = torch.float64        # fp64 de sai so lam tron khong che mat loi that
    BT = 37                   # co le, khong chia het cho block size nao

    # Hai huong PHAI dung case rieng: wide cho X la k x (m-k), tall cho X la
    # (n-k) x k. Dung chung mot case la sai chieu.
    n, m = 64, 208                                   # wide: m > n
    k, sel, rest, _, X = make_case(n, m, "cpu", dt)
    perm = torch.cat([sel, rest])
    XT = X.t().contiguous()

    x = torch.randn(BT, m, dtype=dt, requires_grad=True)
    exp = ref_pre(x, sel, rest, X)
    ok &= check("pre: forward", K.pre_in_proj_fast(x, perm, k, XT), exp, 1e-12)

    # so GRADIENT chu khong chi so gia tri: mot backward sai van co the cho
    # forward dung, va do moi la kieu hong nguy hiem.
    gx = torch.autograd.grad(exp.square().sum(), x)[0]
    x2 = x.detach().clone().requires_grad_(True)
    gx2 = torch.autograd.grad(
        K.pre_in_proj_fast(x2, perm, k, XT).square().sum(), x2)[0]
    ok &= check("pre: gradient theo x", gx2, gx, 1e-12)

    n, m = 208, 64                                   # tall: n > m
    k, sel, rest, inv, X = make_case(n, m, "cpu", dt, seed=2)
    XT = X.t().contiguous()

    h = torch.randn(BT, k, dtype=dt, requires_grad=True)
    exp = ref_post(h, X, inv)
    ok &= check("post: forward",
                K.post_out_proj_fast(h, sel, rest, XT, n), exp, 1e-12)

    gh = torch.autograd.grad(exp.square().sum(), h)[0]
    h2 = h.detach().clone().requires_grad_(True)
    gh2 = torch.autograd.grad(
        K.post_out_proj_fast(h2, sel, rest, XT, n).square().sum(), h2)[0]
    ok &= check("post: gradient theo h", gh2, gh, 1e-12)

    # Hinh dang that cua Mistral, van o CPU nhung nho lai cho chay duoc
    for nm, nn, mm in (("k_proj", 512, 128), ("gate_proj", 256, 896),
                       ("down_proj", 896, 256)):
        kk, s2, r2, i2, X2 = make_case(nn, mm, "cpu", dt, seed=1)
        if mm > nn:
            xx = torch.randn(8, mm, dtype=dt)
            ok &= check(f"pre: ty le canh {nm}",
                        K.pre_in_proj_fast(xx, torch.cat([s2, r2]), kk,
                                           X2.t().contiguous()),
                        ref_pre(xx, s2, r2, X2), 1e-12)
        else:
            hh = torch.randn(8, kk, dtype=dt)
            ok &= check(f"post: ty le canh {nm}",
                        K.post_out_proj_fast(hh, s2, r2, X2.t().contiguous(), nn),
                        ref_post(hh, X2, i2), 1e-12)

    if not a.cuda:
        print("\n  (bo qua phan Triton — them --cuda khi co GPU)")
        print("  => " + ("phan CPU dung het" if ok else "CO LOI"))
        return 0 if ok else 1

    if not K.HAVE_TRITON or not torch.cuda.is_available():
        print("\n  LOI: yeu cau --cuda nhung khong co triton hoac GPU")
        return 1

    print("\n=== MUC 3: kernel Triton (GPU) ===")
    for dt, tol in ((torch.float32, 2e-3), (torch.bfloat16, 3e-1)):
        for BT, n, m in ((37, 64, 208), (256, 512, 1536), (128, 1024, 4096)):
            k, sel, rest, inv, X = make_case(n, m, "cuda", dt)
            x = torch.randn(BT, m, device="cuda", dtype=dt, requires_grad=True)
            XT = X.t().contiguous()
            exp = ref_pre(x, sel, rest, X)
            got = K.PreInProj.apply(x, sel, rest, XT)
            ok &= check(f"triton pre {dt} {BT}x{m}", got, exp, tol)

            g = torch.randn_like(exp)
            ge = torch.autograd.grad(exp, x, g, retain_graph=True)[0]
            x2 = x.detach().clone().requires_grad_(True)
            gg = torch.autograd.grad(K.PreInProj.apply(x2, sel, rest, XT), x2, g)[0]
            ok &= check(f"triton pre grad {dt}", gg, ge, tol)

    print("\n=== toc do (gate_proj cua Mistral: 4096x14336) ===")
    import time
    n, m, BT = 4096, 14336, 8 * 512
    k, sel, rest, inv, X = make_case(n, m, "cuda", torch.bfloat16)
    x = torch.randn(BT, m, device="cuda", dtype=torch.bfloat16)
    XT = X.t().contiguous()
    perm = torch.cat([sel, rest])
    for nm, fn in (("goc  (hai index_select)", lambda: ref_pre(x, sel, rest, X)),
                   ("muc 1 (mot gather)", lambda: K.pre_in_proj_fast(x, perm, k, XT)),
                   ("muc 3 (triton)", lambda: K.PreInProj.apply(x, sel, rest, XT))):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        print(f"  {nm:<26} {(time.perf_counter()-t)/20*1e3:7.3f} ms")

    print("  => " + ("tat ca dung" if ok else "CO LOI, khong duoc dung"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
