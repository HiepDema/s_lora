"""Exact row-space / column-space factorization of non-square weight matrices.

Given W₀ ∈ ℝⁿˣᵐ with n ≠ m, split it into:

  - A square factor  W₂ ∈ ℝᵏˣᵏ  (k = min(n,m)), retaining pretrained values
  - A frozen rectangular factor containing an identity block

Wide (m > n):  W₀Π = W₂ [X̄ | I]      X̄ = W₂⁻¹ W₁
Tall (n > m):  Π W₀ = [X̄ ; I] W₂     X̄ = W₁ W₂⁻¹

The factorization is exact (not an approximation).  Column-pivoted QR selects
which columns/rows form W₂ to minimise conditioning.  The Q-cancellation trick
(X̄ = R₁₁⁻¹ R₁₂) avoids forming Q entirely — one triangular solve suffices.

Two bases
---------
colperm  Structured [I | X̄].  Preserves parameter count exactly.
         Forward error ≈ 5.9× the original matmul (GPT-2 medium).
svd      Dense orthonormal P.  Forward error ≈ 1.29×, but costs an extra
         k² parameters per layer (the identity block becomes dense).
"""
from __future__ import annotations

import os
import sys

import torch


# ======================================================================= core
def pivoted_qr(A):
    """Column-pivoted QR via LAPACK geqp3:  A[:, piv] = Q R.

    Returns ``(R, piv)`` as numpy arrays.  *Q is never formed* — only the
    upper-triangular R and the column permutation are needed downstream.
    """
    import numpy as np
    try:
        from scipy.linalg import qr
        _, R, piv = qr(A, mode="economic", pivoting=True)
        return np.ascontiguousarray(R), np.ascontiguousarray(piv)
    except ImportError:
        pass
    try:
        from split_solve import qr_pivot_numpy
    except ImportError:
        sys.exit("Need `pip install scipy` or split_solve.py in the same directory.")
    return qr_pivot_numpy(np.asarray(A))


# ----------------------------------------------------------------------- wide
def _factorize_wide(W, basis):
    """W ∈ ℝⁿˣᵐ with m > n.   W = W₂ · P.

    colperm → dict(kind='struct', W2[n,n], X[n,m-n], sel[n], rest[m-n])
    svd     → dict(kind='dense',  W2[n,n], P[n,m])
    """
    n, m = W.shape

    if basis == "svd":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return dict(kind="dense", W2=U * S, P=Vh,
                    maxP=Vh.abs().max().item())

    # W[:, piv] = Q [R₁₁ | R₁₂]
    # W₂ = W[:, piv[:n]] = Q R₁₁
    # X̄  = W₂⁻¹ W₁ = (Q R₁₁)⁻¹ (Q R₁₂) = R₁₁⁻¹ R₁₂   (Q cancels)
    R, piv = pivoted_qr(W.cpu().numpy())
    R   = torch.as_tensor(R,   dtype=W.dtype, device=W.device)
    piv = torch.as_tensor(piv, dtype=torch.long, device=W.device)

    R11 = R[:, :n]
    X   = torch.linalg.solve_triangular(R11, R[:, n:], upper=True)
    diag = R11.diagonal().abs()

    return dict(
        kind="struct",
        W2=W[:, piv[:n]].contiguous(),
        X=X.contiguous(),
        sel=piv[:n].contiguous(),
        rest=piv[n:].contiguous(),
        maxX=X.abs().max().item(),
        condW2=(diag.max() / diag.clamp_min(torch.finfo(W.dtype).tiny).min()).item(),
    )


# ----------------------------------------------------------------------- tall
def _factorize_tall(W, basis):
    """W ∈ ℝⁿˣᵐ with n > m.   W = P · W₂.

    Dual of the wide case: transpose, factorise, transpose back.
    """
    if basis == "svd":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return dict(kind="dense", W2=S[:, None] * Vh, P=U,
                    maxP=U.abs().max().item())

    fac = _factorize_wide(W.T.contiguous(), basis)
    fac["W2"] = fac["W2"].T.contiguous()
    fac["X"]  = fac["X"].T.contiguous()
    return fac


# ================================================================== public API
def factorize(weight, basis="colperm", dtype=torch.float64):
    """Factorise a non-square weight into square + rectangular factors.

    Parameters
    ----------
    weight : Tensor [d_in, d_out]
        Weight in Conv1D layout.  For ``nn.Linear`` pass ``weight.T``.
    basis : ``'colperm'`` | ``'svd'``
    dtype : torch.dtype
        Compute precision for the factorization (float64 recommended).

    Returns
    -------
    fac : dict      — factorization tensors (W2, X/P, sel/rest, …)
    orient : str    — ``'pre'`` (wide, input mixing) or ``'post'`` (tall, output expansion)
    info : dict     — diagnostics (residual, max|X|, cond(W₂))

    Raises
    ------
    ValueError  if the weight is square (factorization gives no constraint).
    """
    d_in, d_out = weight.shape
    if d_in == d_out:
        raise ValueError("Square matrix: factorization imposes no constraint")

    W = weight.T.to(dtype)          # [d_out, d_in] = [n, m],  y = W x
    n, m = W.shape

    if m > n:
        fac, orient = _factorize_wide(W, basis), "pre"
        recon = fac["W2"] @ reconstruct_P(fac, n, m, W)
    else:
        fac, orient = _factorize_tall(W, basis), "post"
        recon = reconstruct_P(fac, n, m, W) @ fac["W2"]

    info = {k: v for k, v in fac.items() if isinstance(v, float)}
    info["res"] = (torch.linalg.norm(recon - W) / torch.linalg.norm(W)).item()
    return fac, orient, info


def reconstruct_P(fac, n, m, ref):
    """Build the full dense P from a structured factorization.

    Only needed for verification and merging — the forward pass uses the
    structured form directly (identity block = index gather, not matmul).
    """
    if fac["kind"] == "dense":
        return fac["P"]

    P = torch.zeros(n, m, dtype=ref.dtype, device=ref.device)
    k = fac["W2"].shape[0]
    if fac["X"].shape[0] == k:                  # wide: P = [I | X̄] permuted
        P[:, fac["sel"]]  = torch.eye(k, dtype=ref.dtype, device=ref.device)
        P[:, fac["rest"]] = fac["X"]
    else:                                       # tall: P = [I ; X̄] permuted
        P[fac["sel"], :]  = torch.eye(k, dtype=ref.dtype, device=ref.device)
        P[fac["rest"], :] = fac["X"]
    return P


# ============================================================= pivot caching
def rebuild_from_pivots(W, sel, rest, orient):
    """Reconstruct (W₂, X̄) from cached pivot indices without re-running QR.

    Pivoted QR is the bottleneck (~111 min for Mistral-7B × 7 layers).
    Once we have the pivot indices, a single k×k solve recovers W₂ and X̄
    almost instantly on GPU.  Saving the indices costs ~25 MB; saving X̄
    would cost ~32 GB.
    """
    if orient == "pre":                         # wide
        W2 = W[:, sel].contiguous()
        X  = torch.linalg.solve(W2, W[:, rest])
    else:                                       # tall
        W2 = W[sel, :].contiguous()
        X  = torch.linalg.solve(W2.T, W[rest, :].T).T
    return W2, X.contiguous()


def factorize_cached(weight, basis, dtype, cache_path):
    """Like :func:`factorize`, but caches pivot indices to disk.

    ``cache_path = None`` falls back to :func:`factorize` (no caching).
    """
    if not cache_path or basis != "colperm":
        return factorize(weight, basis, dtype)

    d_in, d_out = weight.shape
    if d_in == d_out:
        raise ValueError("Square matrix: factorization imposes no constraint")

    W = weight.T.to(dtype)
    n, m = W.shape
    orient = "pre" if m > n else "post"

    # Try both .pt (torch) and .npz (numpy, from factorize_cpu.py) formats
    npz_path = cache_path.replace(".pt", ".npz") if cache_path.endswith(".pt") else cache_path + ".npz"
    for cp in (cache_path, npz_path):
        if not os.path.exists(cp):
            continue
        if cp.endswith(".npz"):
            import numpy as np
            z = np.load(cp)
            sel  = torch.as_tensor(z["sel"], dtype=torch.long, device=W.device)
            rest = torch.as_tensor(z["rest"], dtype=torch.long, device=W.device)
        else:
            z    = torch.load(cp, map_location="cpu", weights_only=True)
            sel  = z["sel"].to(W.device)
            rest = z["rest"].to(W.device)
        W2, X = rebuild_from_pivots(W, sel, rest, orient)
        fac = dict(kind="struct", W2=W2, X=X, sel=sel, rest=rest,
                   maxX=X.abs().max().item(), condW2=float("nan"))
        return fac, orient, dict(res=float("nan"), cached=1.0)

    fac, orient, info = factorize(weight, basis, dtype)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = cache_path + ".tmp"
    torch.save({"sel": fac["sel"].cpu(), "rest": fac["rest"].cpu()}, tmp)
    os.replace(tmp, cache_path)
    info["cached"] = 0.0
    return fac, orient, info
