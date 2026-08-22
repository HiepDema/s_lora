#!/usr/bin/env python
"""
Giai  W2 @ X = W1  de co phan tich chinh xac  W = W2 [X | I]  (interpolative decomposition).

W: n x m (m > n)  ->  tach thanh W1: n x (m-n)  va  W2: n x n
Nghiem X = W2^-1 W1 la CHINH XAC (khong phai xap xi) mien la W2 kha nghich.

Hai che do tach cot:
  --mode fixed    lay n cot cuoi lam W2  (dung nhu de bai)
  --mode pivoted  chon n cot "tot nhat" bang column-pivoted QR  [KHUYEN DUNG]

Chi so suc khoe quan trong nhat KHONG phai residual ma la  max|X| :
neu no bung len 1e6 thi phan tich vo dung khi chay o fp32/bf16, du residual dep.

  python split_solve.py --n 1024 --m 4096 --spectrum decay
  python split_solve.py --n 1024 --m 4096 --mode fixed
  python split_solve.py --from-gpt2 h.0.mlp.c_proj        # can transformers
"""
import argparse
import sys

import numpy as np

try:
    from scipy.linalg import qr as sp_qr, solve_triangular as sp_solve_tri
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------------- pivoted QR (fallback)

def qr_pivot_numpy(A):
    """Householder QR voi column pivoting. Tra ve (R, piv) sao cho A[:, piv] = Q R.

    Khong dung Q vi ta khong can no (xem giai thich trong solve_split).
    scipy.linalg.qr(pivoting=True) goi LAPACK geqp3, nhanh va chuan hon; day chi la fallback.
    """
    A = np.array(A, dtype=np.float64, order="F", copy=True)
    n, m = A.shape
    piv = np.arange(m)
    cn = np.einsum("ij,ij->j", A, A)          # binh phuong norm tung cot
    cn0 = cn.copy()

    for k in range(min(n, m)):
        j = k + int(np.argmax(cn[k:]))
        if j != k:
            A[:, [k, j]] = A[:, [j, k]]
            piv[[k, j]] = piv[[j, k]]
            cn[[k, j]] = cn[[j, k]]
            cn0[[k, j]] = cn0[[j, k]]

        x = A[k:, k]
        nx = np.linalg.norm(x)
        if nx == 0.0:
            continue
        alpha = -nx if x[0] >= 0 else nx      # chon dau de tranh cancellation
        v = x.copy()
        v[0] -= alpha
        nv = np.linalg.norm(v)
        if nv == 0.0:
            continue
        v /= nv
        A[k:, k:] -= 2.0 * np.outer(v, v @ A[k:, k:])
        A[k, k] = alpha
        A[k + 1:, k] = 0.0

        # downdate norm cot; tinh lai tu dau neu mat qua nhieu chu so
        if k + 1 < m:
            cn[k + 1:] -= A[k, k + 1:] ** 2
            bad = cn[k + 1:] < 1e-10 * cn0[k + 1:]
            if bad.any():
                idx = np.where(bad)[0] + k + 1
                cn[idx] = np.einsum("ij,ij->j", A[k + 1:, idx], A[k + 1:, idx])
                cn0[idx] = cn[idx]
            np.maximum(cn[k + 1:], 0.0, out=cn[k + 1:])

    return np.triu(A), piv


def solve_upper(R, B):
    """Giai R X = B voi R tam giac tren. Back substitution = backward stable."""
    if HAVE_SCIPY:
        return sp_solve_tri(R, B, lower=False)
    # LU-with-partial-pivoting tren ma tran tam giac tren khong doi hang nao,
    # nen np.linalg.solve o day chinh la back substitution -> ket qua giong het.
    return np.linalg.solve(R, B)


# ---------------------------------------------------------------- loi giai

def solve_split(W, mode="pivoted", ir_steps=2, verbose=True):
    """Tra ve dict voi X, chi so cot cua W2/W1, va cac chi so chan doan."""
    W = np.asarray(W, dtype=np.float64)
    n, m = W.shape
    assert m > n, "can m > n"
    k = m - n

    if mode == "pivoted":
        # ------------------------------------------------------------------
        # Meo chinh: MOT lan pivoted QR lam CA hai viec (chon cot + giai he).
        #   W[:, piv] = Q [R11 | R12],  R11: n x n tam giac tren
        #   W2 = Q R11,  W1 = Q R12
        #   X  = W2^-1 W1 = R11^-1 Q^T Q R12 = R11^-1 R12     <- Q trieu tieu!
        # Nen khong bao gio phai dung Q, va chi con mot triangular solve.
        # ------------------------------------------------------------------
        if HAVE_SCIPY:
            _, R, piv = sp_qr(W, mode="economic", pivoting=True)
        else:
            R, piv = qr_pivot_numpy(W)
        cols2, cols1 = piv[:n], piv[n:]          # geqp3 xep cot duoc chon len truoc
        X = solve_upper(R[:, :n], R[:, n:])
        diag = np.abs(np.diag(R[:, :n]))
        rrqr_ratio = float(diag[0] / max(diag[-1], np.finfo(float).tiny))
    else:
        cols1, cols2 = np.arange(k), np.arange(k, m)
        rrqr_ratio = None
        W2, W1 = W[:, cols2], W[:, cols1]
        X = np.linalg.solve(W2, W1)              # LU + partial pivoting

    W2, W1 = W[:, cols2], W[:, cols1]

    # --- iterative refinement: residual tinh lai o fp64 roi hieu chinh
    res_hist = [rel_res(W2, X, W1)]
    for _ in range(ir_steps):
        X = X + np.linalg.solve(W2, W1 - W2 @ X)
        res_hist.append(rel_res(W2, X, W1))

    out = dict(X=X, cols2=cols2, cols1=cols1, W2=W2, W1=W1,
               res_hist=res_hist, rrqr_ratio=rrqr_ratio)
    if verbose:
        diagnose(W, out, mode)
    return out


def rel_res(A, X, B):
    return float(np.linalg.norm(A @ X - B) / np.linalg.norm(B))


def diagnose(W, out, mode):
    X, W2 = out["X"], out["W2"]
    n, m = W.shape
    cond2 = float(np.linalg.cond(W2))
    condW = float(np.linalg.cond(W))
    recon = W.copy()
    recon[:, out["cols1"]] = W2 @ X                    # dung lai W1 tu X
    err = float(np.linalg.norm(recon - W) / np.linalg.norm(W))

    print(f"  mode = {mode}")
    print(f"    cond(W)             = {condW:.3e}")
    print(f"    cond(W2)            = {cond2:.3e}"
          f"   <- {'ON' if cond2 < 1e8 else 'QUA LON, nghiem se mat chu so'}")
    if out["rrqr_ratio"]:
        print(f"    R11 diag ratio      = {out['rrqr_ratio']:.3e}  (uoc luong cond(W2))")
    print(f"    max|X|              = {np.abs(X).max():.3e}"
          f"   <- {'ON cho fp32/bf16' if np.abs(X).max() < 1e3 else 'NGUY HIEM o fp32/bf16'}")
    print(f"    ||X||_F             = {np.linalg.norm(X):.3e}")
    print(f"    residual sau IR     = " + " -> ".join(f"{r:.2e}" for r in out["res_hist"]))
    print(f"    ||W2[X|I] - W||/||W|| = {err:.3e}   (sai so tai tao toan bo W)")
    print(f"    fp32 round-trip     = {fp32_roundtrip(W2, X, out):.3e}"
          f"   <- do do bung so khi ha xuong fp32")


def fp32_roundtrip(W2, X, out):
    """Ha W2 va X xuong fp32, dung lai W1, do sai so tuong doi (tinh o fp64)."""
    a = W2.astype(np.float32).astype(np.float64) @ X.astype(np.float32).astype(np.float64)
    return float(np.linalg.norm(a - out["W1"]) / np.linalg.norm(out["W1"]))


# ---------------------------------------------------------------- du lieu

def make_W(n, m, spectrum, seed):
    rng = np.random.default_rng(seed)
    if spectrum == "flat":
        return rng.standard_normal((n, m)) * 0.02
    # pho suy giam - giong weight transformer da train hon nhieu so voi gaussian thuan
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((m, n)))
    s = np.logspace(0, -4, n)
    return (U * s) @ V.T * 0.02


def load_gpt2(name):
    from transformers import GPT2LMHeadModel
    sd = GPT2LMHeadModel.from_pretrained("gpt2-medium").state_dict()
    key = f"transformer.{name}.weight"
    Wc = sd[key].numpy().astype(np.float64)       # HF Conv1D luu [in, out]
    W = Wc.T                                      # -> [out, in], tuc y = W x
    print(f"  {key}: Conv1D {Wc.shape} -> W (out x in) {W.shape}")
    if W.shape[1] <= W.shape[0]:
        W = W.T
        print(f"  m <= n, chuyen vi -> {W.shape}")
    return W


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Giai W2 X = W1")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--spectrum", choices=["flat", "decay"], default="decay")
    p.add_argument("--mode", choices=["both", "fixed", "pivoted"], default="both")
    p.add_argument("--ir-steps", type=int, default=2)
    p.add_argument("--from-gpt2", default=None,
                   help="vd: h.0.mlp.c_proj  (can pip install transformers)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.from_gpt2:
        W = load_gpt2(a.from_gpt2)
    else:
        W = make_W(a.n, a.m, a.spectrum, a.seed)
    n, m = W.shape

    print(f"\n=== W: {n} x {m}   ->   W2: {n} x {n},  W1: {n} x {m - n} ===")
    print(f"scipy: {'co (LAPACK geqp3)' if HAVE_SCIPY else 'KHONG -> dung pivoted QR tu viet'}\n")

    modes = ["fixed", "pivoted"] if a.mode == "both" else [a.mode]
    for mo in modes:
        solve_split(W, mode=mo, ir_steps=a.ir_steps)
        print()

    print("Doc ket qua:")
    print("  residual nho  = giai he dung.  KHONG dam bao phan tich dung duoc.")
    print("  max|X| nho    = phan tich ON DINH, con song sot khi ha xuong fp32/bf16.")
    print("  Neu fixed cho max|X| lon hon pivoted nhieu bac -> phai chon cot bang pivoted QR.")


if __name__ == "__main__":
    sys.exit(main())
