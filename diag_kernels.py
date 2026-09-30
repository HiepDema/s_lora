#!/usr/bin/env python
"""Chan doan: kernel sai that hay chi khac lam tron, va cho nao cham.

    python diag_kernels.py

Lan do dau cho ba ket qua deu dang ngo, nen phai tach bach:

  1. "Sai lech" cua Triton co the chi la TF32. tl.dot tren Ampere mac dinh
     dung TF32 cho dau vao fp32, con cuBLAS cua torch thi khong. Muon biet
     kernel co SAI khong thi phai so ca hai voi mot chuan fp64, roi xem sai so
     cua Triton co cung bac voi cuBLAS hay khong.

  2. Muc 1 do duoc CHAM hon ban goc. Phai xem mot gather that su re hon hai
     gather khong, va .contiguous() minh them vao co phai la thu pham khong.

  3. Triton cham gap 33 lan. Phai tach thoi gian autotune ra khoi thoi gian
     chay, va xem truy cap cot theo chi so ngau nhien co pha vo coalescing hay
     khong — neu co thi y tuong gop gather vao GEMM la sai tu goc.
"""
import time

import torch

import slora_kernels as K


def timeit(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    dev = "cuda"
    n, m, BT = 4096, 14336, 8 * 512
    k = n
    g = torch.Generator(device="cpu").manual_seed(0)
    perm_all = torch.randperm(m, generator=g).to(dev)
    sel, rest = perm_all[:k].contiguous(), perm_all[k:].contiguous()
    perm = torch.cat([sel, rest])

    print(f"gate_proj cua Mistral: x({BT} x {m}) -> h({BT} x {k})")
    print(f"  sel {sel.numel()} cot, rest {rest.numel()} cot")

    # ---------------------------------------------------------------- dung sai
    print()
    print("=== 1. Sai that hay chi khac lam tron ===")
    Xd = torch.randn(k, m - k, dtype=torch.float64, device=dev)
    xd = torch.randn(BT, m, dtype=torch.float64, device=dev)
    chuan = xd[:, sel] + xd[:, rest] @ Xd.t()          # chuan fp64

    for dt in (torch.float32, torch.bfloat16):
        X = Xd.to(dt)
        x = xd.to(dt)
        XT = X.t().contiguous()
        cub = (x[:, sel] + x[:, rest] @ X.t()).double()
        tri = K.PreInProj.apply(x, sel, rest, XT).double()
        s = chuan.abs().max().item()
        e_cub = (cub - chuan).abs().max().item() / s
        e_tri = (tri - chuan).abs().max().item() / s
        e_pair = (tri - cub).abs().max().item() / s
        verdict = ("cung bac -> KHONG phai loi kernel"
                   if e_tri < 10 * max(e_cub, 1e-8) else "LECH HAN -> kernel SAI")
        print(f"  {str(dt):<16} cuBLAS {e_cub:.2e}  triton {e_tri:.2e}  "
              f"lech nhau {e_pair:.2e}   {verdict}")

    # -------------------------------------------------------------- toc do
    print()
    print("=== 2. Tung manh mot (bf16) ===")
    dt = torch.bfloat16
    x = xd.to(dt)
    X = Xd.to(dt)
    XT = X.t().contiguous()

    parts = [
        ("gather sel  (BT x 1024)", lambda: x.index_select(-1, sel)),
        ("gather rest (BT x 3072)", lambda: x.index_select(-1, rest)),
        ("gather perm (BT x 4096)", lambda: x.index_select(-1, perm)),
        ("GEMM tren ban sao lien tuc", None),
        ("GEMM tren view co stride", None),
    ]
    xr = x.index_select(-1, rest).contiguous()
    xp = x.index_select(-1, perm)
    parts[3] = ("GEMM tren ban sao lien tuc", lambda: xr @ XT)
    parts[4] = ("GEMM tren view co stride", lambda: xp[:, k:] @ XT)
    for nm_, fn in parts:
        print(f"  {nm_:<30} {timeit(fn):7.3f} ms")

    print()
    print("=== 3. Ca duong forward ===")
    def goc():
        return x.index_select(-1, sel) + x.index_select(-1, rest) @ X.t()

    def muc1_co_contiguous():
        xp_ = x.index_select(-1, perm)
        return xp_[:, :k].contiguous() + xp_[:, k:] @ XT

    def muc1_khong_contiguous():
        xp_ = x.index_select(-1, perm)
        return xp_[:, :k] + xp_[:, k:] @ XT

    def muc1_gemm_lien_tuc():
        xp_ = x.index_select(-1, perm)
        return xp_[:, :k] + xp_[:, k:].contiguous() @ XT

    base = timeit(goc)
    for nm_, fn in (("goc (hai index_select)", goc),
                    ("muc 1 nhu da viet", muc1_co_contiguous),
                    ("muc 1 bo .contiguous()", muc1_khong_contiguous),
                    ("muc 1 + GEMM lien tuc", muc1_gemm_lien_tuc)):
        t = timeit(fn)
        print(f"  {nm_:<30} {t:7.3f} ms   {base/t:5.2f}x")

    print()
    print("=== 4. Triton: autotune hay ban than kernel cham ===")
    t0 = time.perf_counter()
    K.PreInProj.apply(x, sel, rest, XT)
    torch.cuda.synchronize()
    print(f"  lan goi dau (gom autotune) {(time.perf_counter()-t0)*1e3:8.1f} ms")
    print(f"  sau khi da autotune        {timeit(lambda: K.PreInProj.apply(x, sel, rest, XT)):8.3f} ms")

    # Coalescing: doc cot theo chi so NGAU NHIEN so voi doc cot LIEN TIEP
    print()
    print("  doc cot ngau nhien pha vo coalescing den muc nao:")
    lien_tiep = torch.arange(m - k, device=dev)
    print(f"    gather chi so ngau nhien {timeit(lambda: x.index_select(-1, rest)):7.3f} ms")
    print(f"    gather chi so lien tiep  {timeit(lambda: x.index_select(-1, lien_tiep)):7.3f} ms")


if __name__ == "__main__":
    main()
