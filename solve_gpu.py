#!/usr/bin/env python
"""
Giai AX = B tren GPU.   A: n x n,  X: n x m,  B: n x m

Muc tieu: do xem tren GPU co fp64 bi cat (A10, RTX 30xx/40xx, L4...) thi
mixed-precision iterative refinement nhanh hon fp64 thuan bao nhieu lan,
ma van dat do chinh xac tuong duong.

Cac phuong phap duoc do:
  1. fp64 LU  (lu_factor + lu_solve)          <- chuan vang, cham tren A10
  2. fp32 LU  (cast xuong, giai, cast len)    <- nhanh, chi ~1e-6 chinh xac
  3. mixed fp32 -> fp64 iterative refinement  <- nhanh VA chinh xac  [KHUYEN DUNG]
  4. fp64 Cholesky (neu --spd)                <- re bang nua LU
  5. inv(A) @ B                               <- cach SAI, de doi chung

Vi du:
  python solve_gpu.py --n 4000 --m 10000
  python solve_gpu.py --n 4000 --m 10000 --spd
  python solve_gpu.py --n 8000 --m 8000 --cond 1e8 --ir-steps 5

Cai torch cho CUDA 12.4:
  pip install torch --index-url https://download.pytorch.org/whl/cu124
"""
import argparse
import sys
import time

import torch

# --------------------------------------------------------------------------- do gio


def sync():
    torch.cuda.synchronize()


class Timer:
    """Dem gio co sync CUDA. Khong sync => moi so do deu vo nghia vi kernel bat dong bo."""

    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        sync()
        self.dt = time.perf_counter() - self.t0


def timeit(fn, repeats, warmup=1):
    """Chay warmup roi lay thoi gian NHO NHAT qua cac lan lap."""
    for _ in range(warmup):
        out = fn()
    del out
    torch.cuda.empty_cache()
    best = float("inf")
    for _ in range(repeats):
        with Timer() as t:
            out = fn()
        best = min(best, t.dt)
    return out, best


RESULTS = []


def report(name, dt, res, fwd, gflops=None, note=""):
    RESULTS.append(dict(name=name, dt=dt, res=res, fwd=fwd))
    speed = f"{gflops:7.0f} GF/s" if gflops else " " * 12
    print(f"  {name:<38s} {dt*1e3:9.1f} ms  {speed}  res={res:.2e}  fwd={fwd:.2e}  {note}")


# --------------------------------------------------------------------------- thiet lap


def describe_gpu(dtype):
    p = torch.cuda.get_device_properties(0)
    print(f"GPU : {p.name}   sm_{p.major}{p.minor}   {p.total_memory / 2**30:.1f} GiB")
    print(f"torch {torch.__version__} / CUDA {torch.version.cuda}")

    # GA102 (A10, A40, RTX 3090) va AD102 (L40, RTX 4090) deu bi cat fp64 nang.
    crippled = p.major == 8 and p.minor == 6 or p.major == 8 and p.minor == 9
    if crippled and dtype == "float64":
        print(
            "  [!] Kien truc nay co fp64 bi cat con ~1/32-1/64 toc do fp32.\n"
            "      => fp64 thuan se rat cham. Xem dong 'mixed fp32->fp64 IR' ben duoi."
        )
    print()
    return p


def build_problem(a, dev):
    """Tra ve (A, B, X_true) o fp64. B = A @ X_true nen ta biet dap an dung."""
    n, m = a.n, a.m
    torch.manual_seed(a.seed)
    f64 = torch.float64

    if a.spd:
        # A = M M^T + I  -> doi xung xac dinh duong, dung duoc Cholesky
        M = torch.randn(n, n, device=dev, dtype=f64) / n**0.5
        A = M @ M.T
        A.diagonal().add_(1.0)
        del M
        kind = "SPD (M M^T + I)"
    elif a.cond is not None:
        # A = Q1 diag(s) Q2^T voi s tu 1 giam den 1/cond -> cond(A) = a.cond chinh xac
        import math

        s = torch.logspace(0, -math.log10(a.cond), n, device=dev, dtype=f64)
        Q1, _ = torch.linalg.qr(torch.randn(n, n, device=dev, dtype=f64))
        Q2, _ = torch.linalg.qr(torch.randn(n, n, device=dev, dtype=f64))
        A = (Q1 * s) @ Q2.T
        del Q1, Q2, s
        kind = f"cond(A) = {a.cond:.0e}"
    else:
        A = torch.randn(n, n, device=dev, dtype=f64)
        kind = "gaussian (cond ~ O(n))"

    X_true = torch.randn(n, m, device=dev, dtype=f64)
    with Timer() as t:
        B = A @ X_true
    print(f"Bai toan: A = {kind}")
    print(f"  B = A @ X_true : {t.dt * 1e3:.0f} ms  ({2 * n * n * m / t.dt / 1e9:.0f} GFLOP/s fp64)\n")
    return A, B, X_true


# --------------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description="Giai AX=B tren GPU")
    p.add_argument("--n", type=int, default=4000, help="A la n x n")
    p.add_argument("--m", type=int, default=10000, help="so cot cua X va B")
    p.add_argument("--spd", action="store_true", help="A doi xung xac dinh duong")
    p.add_argument("--cond", type=float, default=None, help="ep condition number cua A")
    p.add_argument("--ir-steps", type=int, default=3, help="so vong iterative refinement")
    p.add_argument("--repeats", type=int, default=3, help="so lan lap khi do gio")
    p.add_argument("--tf32", action="store_true", help="cho phep TF32 trong matmul fp32")
    p.add_argument("--skip-inv", action="store_true", help="bo qua phep inv(A) @ B")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("Khong thay CUDA. Cai: pip install torch --index-url "
                 "https://download.pytorch.org/whl/cu124")

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32

    n, m = a.n, a.m
    lu_flop, tri_flop = (2 / 3) * n**3, 2 * n * n * m

    print(f"\n=== AX = B    A: {n}x{n}    X, B: {n}x{m} ===\n")
    describe_gpu("float64")
    print(f"FLOP  : LU factor {lu_flop / 1e9:.0f} GF | solve {m} RHS {tri_flop / 1e9:.0f} GF"
          f" | ty le solve/factor = {tri_flop / lu_flop:.1f}x")
    print(f"VRAM  : A {8 * n * n / 2**30:.2f} GiB | B {8 * n * m / 2**30:.2f} GiB"
          f" | uoc tinh tong ~{8 * (n * n + 4 * n * m) / 2**30:.2f} GiB (fp64)\n")

    A, B, X_true = build_problem(a, dev)

    # --- ham cham diem, luon tinh o fp64 du nghiem den tu fp32
    nrm = torch.linalg.norm
    nA, nXt = nrm(A), nrm(X_true)

    def score(X):
        X = X.double()
        res = (nrm(B - A @ X) / (nA * nrm(X))).item()   # backward error
        fwd = (nrm(X - X_true) / nXt).item()            # forward error
        return res, fwd

    total_flop = lu_flop + tri_flop
    print("Ket qua  (res = backward error, fwd = forward error):")

    # --- 1. fp64 LU: phan ra mot lan, giai m ve phai
    def f64_lu():
        LU, piv = torch.linalg.lu_factor(A)
        return torch.linalg.lu_solve(LU, piv, B)

    X64, t64 = timeit(f64_lu, a.repeats)
    report("fp64 LU (lu_factor + lu_solve)", t64, *score(X64), gflops=total_flop / t64 / 1e9)
    del X64

    # tach rieng factor / solve de thay phan nao ton thoi gian
    with Timer() as tf:
        LU, piv = torch.linalg.lu_factor(A)
    with Timer() as ts:
        _ = torch.linalg.lu_solve(LU, piv, B)
    print(f"  {'  |- tach ra':<38s} {tf.dt * 1e3:9.1f} ms factor + {ts.dt * 1e3:.1f} ms solve")

    # tai su dung factorization cho ve phai khac -> khong ton lai chi phi n^3
    B2 = torch.randn(n, m, device=dev, dtype=torch.float64)
    _, t_reuse = timeit(lambda: torch.linalg.lu_solve(LU, piv, B2), a.repeats)
    print(f"  {'  |- RHS moi, tai dung LU':<38s} {t_reuse * 1e3:9.1f} ms"
          f"                nhanh hon {t64 / t_reuse:.2f}x")
    del B2, LU, piv
    torch.cuda.empty_cache()

    # --- 2. fp64 Cholesky (chi khi A la SPD): ton mot nua so FLOP cua LU
    if a.spd:
        def f64_chol():
            L = torch.linalg.cholesky(A)
            return torch.cholesky_solve(B, L)

        Xc, tc = timeit(f64_chol, a.repeats)
        report("fp64 Cholesky", tc, *score(Xc),
               gflops=(lu_flop / 2 + tri_flop) / tc / 1e9, note=f"nhanh hon LU {t64 / tc:.2f}x")
        del Xc
        torch.cuda.empty_cache()

    # --- 3. fp32 LU thuan: nhanh nhung chi dat ~1e-6
    A32, B32 = A.float(), B.float()

    def f32_lu():
        LU, piv = torch.linalg.lu_factor(A32)
        return torch.linalg.lu_solve(LU, piv, B32)

    X32, t32 = timeit(f32_lu, a.repeats)
    report("fp32 LU", t32, *score(X32),
           gflops=total_flop / t32 / 1e9, note=f"nhanh hon fp64 {t64 / t32:.1f}x")
    del X32
    torch.cuda.empty_cache()

    # --- 4. Mixed precision iterative refinement: phan ra o fp32, tinh residual o fp64
    def mixed_ir():
        LU, piv = torch.linalg.lu_factor(A32)
        X = torch.linalg.lu_solve(LU, piv, B32).double()
        for _ in range(a.ir_steps):
            R = B - A @ X                                    # residual BAT BUOC o fp64
            X += torch.linalg.lu_solve(LU, piv, R.float()).double()
        return X

    Xir, tir = timeit(mixed_ir, a.repeats)
    report(f"mixed fp32->fp64 IR ({a.ir_steps} vong)", tir, *score(Xir),
           note=f"nhanh hon fp64 {t64 / tir:.1f}x   <-- KHUYEN DUNG")

    # hoi tu tung vong: cho thay bao nhieu vong la du
    LU, piv = torch.linalg.lu_factor(A32)
    X = torch.linalg.lu_solve(LU, piv, B32).double()
    print(f"  {'  |- hoi tu theo vong (fwd err)':<38s} vong 0: {score(X)[1]:.2e}", end="")
    for k in range(a.ir_steps):
        X += torch.linalg.lu_solve(LU, piv, (B - A @ X).float()).double()
        print(f" -> {k + 1}: {score(X)[1]:.2e}", end="")
    print()
    del LU, piv, X, Xir, A32, B32
    torch.cuda.empty_cache()

    # --- 5. Cach SAI: nghich dao tuong minh
    if not a.skip_inv:
        def via_inv():
            return torch.linalg.inv(A) @ B

        Xi, tinv = timeit(via_inv, a.repeats)
        report("inv(A) @ B   [CACH SAI]", tinv, *score(Xi),
               gflops=(2 * n**3 + tri_flop) / tinv / 1e9, note=f"cham hon LU {tinv / t64:.2f}x")
        del Xi
        torch.cuda.empty_cache()

    print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("\nres = ||B-AX|| / (||A||.||X||)  -> ~1e-16 nghia la backward stable")
    print("fwd = ||X-X_true|| / ||X_true||  -> ~ cond(A) * eps")


if __name__ == "__main__":
    main()
