#!/usr/bin/env python
"""Kernel nhanh cho RowSpaceLinear, theo loi Unsloth.

    import slora_kernels as K
    K.patch(model)                      # doi forward cua moi RowSpaceLinear
    K.unpatch(model)                    # tra lai ban goc

CHUA DO TREN GPU. Viet khi khong co may, nen moi con so toc do trong file nay
la suy luan tu so phep truy cap bo nho, khong phai do duoc. Chay
test_kernels.py truoc khi tin bat cu dieu gi o day.

--------------------------------------------------------------------- van de
S-LoRA cham hon LoRA 20% moi buoc (0.090 vs 0.108 it/s, do tren H100 SXM5) du
IT tham so hon. Cho cham KHONG phai so phep tinh: dang phan ra co dung so phep
nhan nhu ma tran day. Voi gate_proj 4096x14336,

    x[:,rest] @ X.T : BT x 10240 x 4096
    h @ C           : BT x  4096 x 4096
    cong lai        : BT x 4096 x 14336  = dung bang  x @ W

Cho cham la BO NHO. Ban goc, moi lop moi buoc lam:

  pre  : hai lan index_select (moi lan cap phat mot ban sao cua x)
  post : mot torch.cat (cap phat tensor day du) roi mot index_select nua

--------------------------------------------------------------------- cach sua
Ba muc, muc sau kho hon va rui ro hon muc truoc.

MUC 1 — hoan vi mot lan, roi dung SLICE (thuan PyTorch, rui ro thap)
    xp = x[:, perm]           mot lan gather duy nhat
    h  = xp[:, :k] + xp[:, k:] @ X.T
  xp[:, :k] va xp[:, k:] la VIEW chu khong phai ban sao — cuBLAS nhan duoc
  leading-dimension stride nen khong phai copy. Hai gather thanh mot.

  Ben post: cap phat san output roi index_copy_ vao dung cho, bo duoc ca
  torch.cat lan index_select cuoi.

MUC 2 — torch.compile de inductor tu fuse gather voi epilogue.

MUC 3 — Triton: mot kernel GEMM doc thang cot cua x theo vector chi so, va lay
  x[:,sel] lam gia tri khoi tao cua accumulator. Khong con trung gian nao.

X, sel, rest la BUFFER dong bang, khong can gradient. Nho vay backward chi phai
tra ve dL/dx, tuc mot phep scatter-add — de hon nhieu so voi truong hop X cung
duoc train.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                        # CPU hoac chua cai
    HAVE_TRITON = False


# ===================================================================== MUC 1
# Thuan PyTorch. Khong can Triton, chay duoc o moi noi, va la ban THAM CHIEU
# ma cac muc sau phai khop.

def pre_in_proj_fast(x, perm, k, XT):
    """h = x[:,sel] + x[:,rest] @ X.T, mot lan gather thay vi hai.

    perm = cat([sel, rest]); XT = X.T da chuyen vi san (X la buffer, chuyen vi
    mot lan luc patch chu khong phai moi buoc).
    """
    xp = x.index_select(-1, perm)
    return xp[..., :k].contiguous() + xp[..., k:] @ XT


def post_out_proj_fast(h, sel, rest, XT, n_out):
    """out[:,sel] = h; out[:,rest] = h @ X.T — bo torch.cat va index_select.

    Ban goc cap phat tensor cat (BT x n) roi gather them mot lan nua ra tensor
    thu hai. Ban nay cap phat dung mot tensor va ghi thang vao vi tri.
    """
    out = h.new_empty(*h.shape[:-1], n_out)
    out.index_copy_(-1, sel, h)
    out.index_copy_(-1, rest, h @ XT)
    return out


# ===================================================================== MUC 3
# Triton. Mot kernel lam ca gather lan GEMM lan cong x[:,sel].

if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_stages=4, num_warps=4),
            triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_stages=4, num_warps=4),
            triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_stages=4, num_warps=4),
            triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_stages=3, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _gather_gemm(X_ptr, W_ptr, SEL_ptr, REST_ptr, OUT_ptr,
                     M, N, K,
                     sx_m, sx_k, sw_k, sw_n, so_m, so_n,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """OUT[m,n] = X[m, SEL[n]] + sum_k X[m, REST[k]] * W[k,n].

        Cot cua X duoc doc theo vector chi so ngay trong vong lap GEMM, nen ban
        sao cua x[:,rest] khong bao gio ton tai. Phan x[:,sel] vao thang
        accumulator lam gia tri khoi tao, nen khong ton mot phep cong rieng.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mask_m = rm < M
        mask_n = rn < N

        # accumulator khoi tao bang x[:, sel] — khoi identity, khong phai matmul
        sel = tl.load(SEL_ptr + rn, mask=mask_n, other=0)
        acc = tl.load(X_ptr + rm[:, None] * sx_m + sel[None, :] * sx_k,
                      mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            mask_k = rk < K
            cols = tl.load(REST_ptr + rk, mask=mask_k, other=0)
            a = tl.load(X_ptr + rm[:, None] * sx_m + cols[None, :] * sx_k,
                        mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(W_ptr + rk[:, None] * sw_k + rn[None, :] * sw_n,
                        mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)

        tl.store(OUT_ptr + rm[:, None] * so_m + rn[None, :] * so_n,
                 acc.to(OUT_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _scatter_add_cols(G_ptr, IDX_ptr, DST_ptr, M, N,
                          sg_m, sg_n, sd_m, sd_n,
                          BM: tl.constexpr, BN: tl.constexpr):
        """DST[m, IDX[n]] += G[m, n].  Dung cho backward cua phan gather.

        Cac chi so trong IDX doi mot khong trung nhau (sel va rest la mot phep
        hoan vi), nen cong thang duoc, khong can atomic.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        idx = tl.load(IDX_ptr + rn, mask=rn < N, other=0)
        g = tl.load(G_ptr + rm[:, None] * sg_m + rn[None, :] * sg_n,
                    mask=mask, other=0.0)
        p = DST_ptr + rm[:, None] * sd_m + idx[None, :] * sd_n
        tl.store(p, tl.load(p, mask=mask, other=0.0) + g, mask=mask)


class PreInProj(torch.autograd.Function):
    """h = x[:,sel] + x[:,rest] @ XT, co backward viet tay.

    X dong bang nen backward chi tra ve dL/dx:
        dL/dx[:,sel]  += dh
        dL/dx[:,rest] += dh @ XT.T
    Khong luu x cho backward — chi luu chi so va XT, deu la buffer co san. Do
    la cho tiet kiem bo nho lon nhat: x co kich thuoc BT x m.
    """

    @staticmethod
    def forward(ctx, x, sel, rest, XT):
        ctx.save_for_backward(sel, rest, XT)
        ctx.m = x.shape[-1]
        flat = x.reshape(-1, x.shape[-1])
        M, N, K = flat.shape[0], sel.numel(), rest.numel()
        out = torch.empty(M, N, device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
        _gather_gemm[grid](flat, XT, sel, rest, out, M, N, K,
                           flat.stride(0), flat.stride(1),
                           XT.stride(0), XT.stride(1),
                           out.stride(0), out.stride(1))
        return out.view(*x.shape[:-1], N)

    @staticmethod
    def backward(ctx, dh):
        sel, rest, XT = ctx.saved_tensors
        dh = dh.reshape(-1, dh.shape[-1])
        dx = torch.zeros(dh.shape[0], ctx.m, device=dh.device, dtype=dh.dtype)
        dx.index_copy_(-1, sel, dh)
        dx.index_copy_(-1, rest, dh @ XT.t())
        return dx, None, None, None


# =========================================================== gan vao model

def _fast_forward_pre(self, x):
    out = self._square(pre_in_proj_fast(x, self._perm, self.k, self._XT))
    return out + self.bias if self.bias is not None else out


def _fast_forward_post(self, x):
    out = post_out_proj_fast(self._square(x), self.sel, self.rest,
                             self._XT, self._n_out)
    return out + self.bias if self.bias is not None else out


def patch(model, verbose=True):
    """Doi forward cua moi RowSpaceLinear sang duong nhanh. Tra ve so lop da doi.

    Chi dung cho basis 'colperm' (structured). Basis svd dung P day dac nen
    khong co gather de bo di.
    """
    from rowspace_peft import RowSpaceLinear
    n = 0
    for mod in model.modules():
        if not isinstance(mod, RowSpaceLinear) or not mod.structured:
            continue
        if hasattr(mod, "_slow_forward_saved"):
            continue
        mod._slow_forward_saved = mod.forward
        # Chuyen vi X mot lan tai day. Ban goc goi .T moi buoc — voi ma tran
        # 4096x10240 thi do la mot view, khong ton gi, nhung .contiguous() cho
        # cuBLAS mot layout tot hon.
        mod.register_buffer("_XT", mod.X.t().contiguous(), persistent=False)
        if mod.orientation == "pre":
            mod.register_buffer("_perm", torch.cat([mod.sel, mod.rest]),
                                persistent=False)
            mod.forward = _fast_forward_pre.__get__(mod)
        else:
            mod._n_out = mod.sel.numel() + mod.rest.numel()
            mod.forward = _fast_forward_post.__get__(mod)
        n += 1
    if verbose:
        print(f"  kernel nhanh: da doi {n} lop RowSpaceLinear", flush=True)
    return n


def unpatch(model):
    n = 0
    for mod in model.modules():
        if hasattr(mod, "_slow_forward_saved"):
            mod.forward = mod._slow_forward_saved
            del mod._slow_forward_saved
            n += 1
    return n
