#!/usr/bin/env python
"""Kiem DAI SO cua slora_kernels bang numpy, khong can torch.

    python test_kernel_algebra.py

May nay khong co torch nen test_kernels.py chua chay duoc. Nhung ba dieu duoi
day la dai so thuan tuy, va cung la cho de sai nhat — mot dau chuyen vi dat sai
cho hay mot phep hoan vi nguoc chieu se lam ket qua sai ma van chay tron tru.
Numpy du de bat het.

Con lai phai cho GPU: kernel Triton co viet dung khong, va nhanh hon bao nhieu.
"""
import sys

import numpy as np

rng = np.random.default_rng(0)
ok = True


def chk(name, got, exp, tol=1e-12):
    global ok
    e = np.abs(got - exp).max()
    good = e <= tol
    ok &= good
    print(f"  {'OK ' if good else 'SAI'} {name:<46} sai lech {e:.2e}")


print("=== 1. pre: mot gather + slice  ==  hai index_select ===")
for BT, n, m in ((37, 64, 208), (8, 256, 896), (128, 1024, 4096)):
    k = n
    perm = rng.permutation(m)
    sel, rest = perm[:k], perm[k:]
    X = rng.standard_normal((k, m - k))
    x = rng.standard_normal((BT, m))

    goc = x[:, sel] + x[:, rest] @ X.T          # ban goc trong rowspace_peft
    xp = x[:, np.concatenate([sel, rest])]      # MOT lan gather
    nhanh = xp[:, :k] + xp[:, k:] @ X.T
    chk(f"pre {BT}x{m} -> {k}", nhanh, goc)

print()
print("=== 2. post: ghi thang vao vi tri  ==  cat roi index_select ===")
for BT, n, m in ((37, 208, 64), (8, 896, 256), (128, 4096, 1024)):
    k = m
    perm = rng.permutation(n)
    sel, rest = perm[:k], perm[k:]
    inv = np.argsort(np.concatenate([sel, rest]))
    # Chu y bat doi xung: huong 'wide' co X la k x (m-k), huong 'tall' nay co
    # X la (n-k) x k. Dat nham chieu thi h @ X.T bao loi ngay, nhung neu n-k
    # tinh co bang k thi no chay im ru va cho ket qua sai.
    X = rng.standard_normal((n - k, k))
    h = rng.standard_normal((BT, k))

    goc = np.concatenate([h, h @ X.T], axis=-1)[:, inv]
    nhanh = np.zeros((BT, n))
    nhanh[:, sel] = h
    nhanh[:, rest] = h @ X.T
    chk(f"post {BT}x{k} -> {n}", nhanh, goc)

print()
print("=== 3. backward: cong thuc viet tay  ==  vi phan that ===")
# h = x[:,sel] + x[:,rest] @ X.T.  X dong bang nen chi can dL/dx.
#   dL/dx[:,sel]  = dh
#   dL/dx[:,rest] = dh @ X
BT, n, m = 13, 32, 96
k = n
perm = rng.permutation(m)
sel, rest = perm[:k], perm[k:]
X = rng.standard_normal((k, m - k))
x = rng.standard_normal((BT, m))
dh = rng.standard_normal((BT, k))               # gradient tu tren xuong

viet_tay = np.zeros_like(x)
viet_tay[:, sel] = dh
viet_tay[:, rest] = dh @ X

# vi phan so: dL/dx[i,j] voi L = sum(dh * h)
eps = 1e-6
so = np.zeros_like(x)
for i in range(BT):
    for j in range(m):
        xp_ = x.copy(); xp_[i, j] += eps
        xm_ = x.copy(); xm_[i, j] -= eps
        fp = ((xp_[:, sel] + xp_[:, rest] @ X.T) * dh).sum()
        fm = ((xm_[:, sel] + xm_[:, rest] @ X.T) * dh).sum()
        so[i, j] = (fp - fm) / (2 * eps)
chk("gradient theo x", viet_tay, so, 1e-6)

print()
print("=== 4. chi so khong trung nhau (scatter khong can atomic) ===")
for n_, k_ in ((208, 64), (4096, 1024)):
    perm = rng.permutation(n_)
    hop = np.concatenate([perm[:k_], perm[k_:]])
    good = len(np.unique(hop)) == n_
    ok &= good
    print(f"  {'OK ' if good else 'SAI'} n={n_:<6} sel va rest phu kin, khong lap")

print()
print("  => " + ("dai so dung het — con lai phai cho GPU de kiem Triton"
                 if ok else "CO LOI DAI SO, phai sua truoc"))
sys.exit(0 if ok else 1)
