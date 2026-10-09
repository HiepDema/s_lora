"""Sinh cac con so trong Appendix A (kiem chung cac dinh ly o Section 3.6).

    python paper/gap_check.py          (chi can numpy)

Moi dinh ly duoc doi chieu voi mot phep toi uu DOC LAP: alternating least
squares, 8 lan khoi tao ngau nhien. ALS chi dung hai phep binh phuong toi thieu
luan phien va khong bao gio tinh mot gia tri ky di nao, nen neu cong thuc pho
sai thi sai lech se lo ra. Thuc te khop toi ~1e-15.

Hai chi tiet cua setup la BAT BUOC, ban dau viet sai ca hai:
  * X phai duoc whiten CHINH XAC (Sigma = T T^T dung bang gia tri dat ra).
    Sai so mau huu han o muc 3% con lon hon chinh cai gap can do, va no pha
    hong hoan toan du doan cua gradient flow.
  * Y phai sinh tu teacher W0 + Dtrue. Neu lay Y doc lap voi X thi viec duy
    nhat can lam la triet tieu W0 — ma huong do nam san trong row(W0), tuc
    setup tu dong thien vi cho S-LoRA va gap gan nhu bang 0.
"""
import numpy as np


def orth(M):
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > s[0] * 1e-12]


def isqrtm(S):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-14))) @ V.T


def setup(n, m, N, cond=1.0, noise=0.5, seed=0):
    """y = (W0 + Dtrue) x + nhieu.  Tra ve Sigma CHINH XAC bang T T^T."""
    g = np.random.default_rng(seed)
    W0 = g.standard_normal((n, m)) / np.sqrt(m)
    Dtrue = g.standard_normal((n, m)) / np.sqrt(m)        # update "that" can hoc
    if cond == 1.0:
        T = np.eye(m)
    else:
        Q, _ = np.linalg.qr(g.standard_normal((m, m)))
        T = Q @ np.diag(np.linspace(1.0, np.sqrt(cond), m)) @ Q.T
    Z = g.standard_normal((m, N))
    Z = isqrtm(Z @ Z.T / N) @ Z                           # cov(Z) = I CHINH XAC
    X = T @ Z
    Y = (W0 + Dtrue) @ X + noise * g.standard_normal((n, N))
    R = Y - W0 @ X
    return W0, X, Y, R, T @ T.T, R @ X.T / N


def als(R, Z, r, restarts=8, iters=300):
    """min_{B,A} ||BAZ - R||^2/(2N) bang ALS — doc lap voi cong thuc pho."""
    n, N = R.shape
    Czz, Crz = Z @ Z.T / N, R @ Z.T / N
    crr = np.sum(R * R) / N
    Czzi = np.linalg.pinv(Czz)
    best = np.inf
    for t in range(restarts):
        g = np.random.default_rng(1000 + t)
        B = orth(g.standard_normal((n, r)))
        for _ in range(iters):
            A = B.T @ Crz @ Czzi
            B = orth(Crz @ A.T @ np.linalg.pinv(A @ Czz @ A.T))
            if B.shape[1] < r:
                break
        else:
            A = B.T @ Crz @ Czzi
            best = min(best, 0.5 * (np.trace(B @ (A @ Czz @ A.T) @ B.T)
                                    - 2 * np.trace(B @ (A @ Crz.T)) + crr))
    return best


def closed_form_gap(G, Sig, U, r):
    w, V = np.linalg.eigh(Sig)
    w = np.maximum(w, 1e-14)
    Sh = V @ np.diag(np.sqrt(w)) @ V.T
    Gt = G @ (V @ np.diag(1 / np.sqrt(w)) @ V.T)
    Q = orth(Sh @ U)
    sg = np.linalg.svd(Gt, compute_uv=False)
    sp = np.linalg.svd(Gt @ (Q @ Q.T), compute_uv=False)
    return 0.5 * (np.sum(sg[:r] ** 2) - np.sum(sp[:r] ** 2))


def banner(s):
    print(f"\n{'=' * 76}\n{s}\n{'=' * 76}")


# ====================================================================== CHECK A
banner("CHECK A - cong thuc gap tai toi uu (setup khong thien vi)")
print(f"  {'seed':>5} {'r':>3} {'LoRA*':>10} {'S-LoRA*':>10} {'gap do':>10} "
      f"{'cong thuc':>11} {'|lech|':>10}")
for seed in (1, 2, 3):
    for r in (2, 4):
        W0, X, Y, R, Sig, G = setup(8, 32, 20000, seed=seed)
        U = orth(W0.T)
        lo, sl = als(R, X, r), als(R, U.T @ X, r)
        p = closed_form_gap(G, Sig, U, r)
        print(f"  {seed:5d} {r:3d} {lo:10.5f} {sl:10.5f} {sl - lo:+10.5f} "
              f"{p:+11.5f} {abs(sl - lo - p):10.2e}")

# ====================================================================== CHECK B
banner("CHECK B - Sigma bat ky (du lieu khong whiten)")
print(f"  {'cond':>8} {'LoRA*':>10} {'S-LoRA*':>10} {'gap do':>10} "
      f"{'cong thuc':>11} {'|lech|':>10}")
for cond in (10.0, 100.0, 1000.0):
    W0, X, Y, R, Sig, G = setup(8, 32, 20000, cond=cond, seed=7)
    U = orth(W0.T)
    lo, sl = als(R, X, 4), als(R, U.T @ X, 4)
    p = closed_form_gap(G, Sig, U, 4)
    print(f"  {cond:8.0f} {lo:10.5f} {sl:10.5f} {sl - lo:+10.5f} "
          f"{p:+11.5f} {abs(sl - lo - p):10.2e}")

# ====================================================================== CHECK C
banner("CHECK C - W0 VUONG kha nghich -> hai tap trung nhau -> gap = 0")
W0, X, Y, R, Sig, G = setup(24, 24, 20000, seed=5)
U = orth(W0.T)
print(f"  dim row(W0) = {U.shape[1]} = m")
for r in (2, 6, 12):
    lo, sl = als(R, X, r), als(R, U.T @ X, r)
    print(f"  r={r:2d}: LoRA*={lo:.6f}  S-LoRA*={sl:.6f}  gap={sl - lo:+.2e}  "
          f"cong thuc={closed_form_gap(G, Sig, U, r):+.2e}")

# ====================================================================== CHECK D
banner("CHECK D - GRADIENT FLOW che do tuyen tinh (Sigma = I chinh xac)")
print("  Du doan: gap(t) = 1/2||G(I-P)||^2 (1 - e^{-2t}), tang don dieu -> bao hoa")
W0, X, Y, R, Sig, G = setup(8, 32, 20000, cond=1.0, seed=11)
U = orth(W0.T); P = U @ U.T
sat = 0.5 * np.linalg.norm(G - G @ P) ** 2
N = X.shape[1]
L = lambda D: 0.5 * np.sum(((W0 + D) @ X - Y) ** 2) / N
print(f"  muc bao hoa = 1/2||G(I-P)||_F^2 = {sat:.6f}\n")
print(f"  {'t':>6} {'L_LoRA':>11} {'L_S-LoRA':>11} {'gap do':>11} "
      f"{'du doan':>11} {'|lech|':>10}")
for t in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    c = 1 - np.exp(-t)
    a, b = L(c * G), L(c * (G @ P))
    p = sat * (1 - np.exp(-2 * t))
    print(f"  {t:6.2f} {a:11.6f} {b:11.6f} {b - a:+11.6f} {p:+11.6f} "
          f"{abs(b - a - p):10.2e}")

# ====================================================================== CHECK E
banner("CHECK E - GD THAT tren tham so hoa factored (B,A) va (B,A,C), hang r")
print("  Day moi la thu ban quan sat: hai run that, cung init (Delta=0), cung lr.")
W0, X, Y, R, Sig, G = setup(8, 32, 20000, cond=1.0, noise=0.5, seed=3)
U = orth(W0.T); C = U.T                                   # k x m
Np = X.shape[1]
crr = np.sum(R * R) / Np
r, lr, steps = 4, 0.05, 20000


def run(use_C):
    g = np.random.default_rng(0)
    k = C.shape[0] if use_C else G.shape[1]
    B = np.zeros((G.shape[0], r))                         # lora_up = 0 -> Delta=0
    A = g.standard_normal((r, k)) / np.sqrt(k)            # lora_down ngau nhien
    hist = {}
    for s in range(steps + 1):
        D = B @ A @ C if use_C else B @ A
        E = D @ Sig - G                                   # grad theo Delta
        if s in (0, 10, 50, 200, 1000, 5000, 10000, 20000):
            hist[s] = 0.5 * (np.trace(D @ Sig @ D.T) - 2 * np.sum(D * G) + crr)
        M = A @ C if use_C else A
        gB, gA = E @ M.T, (B.T @ E @ C.T if use_C else B.T @ E)
        B, A = B - lr * gB, A - lr * gA
    return hist


hL, hS = run(False), run(True)
opt_gap = closed_form_gap(G, Sig, U, r)
print(f"\n  gap tai toi uu (cong thuc) = {opt_gap:.6f}\n")
print(f"  {'buoc':>7} {'L_LoRA':>11} {'L_S-LoRA':>11} {'gap':>11} "
      f"{'% gap toi uu':>13}")
for s in sorted(hL):
    gp = hS[s] - hL[s]
    print(f"  {s:7d} {hL[s]:11.6f} {hS[s]:11.6f} {gp:+11.6f} "
          f"{100 * gp / opt_gap:12.1f}%")
