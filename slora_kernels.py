#!/usr/bin/env python
"""Kernel nhanh cho RowSpaceLinear, theo loi Unsloth.

    import slora_kernels as K
    K.patch(model)                      # doi forward cua moi RowSpaceLinear
    K.unpatch(model)                    # tra lai ban goc

DA DO TREN A10, 2026-09-30. KET LUAN: KHONG NEN DUNG MUC 3.

Ket qua (gate_proj 4096x14336, bf16, BT=4096):

    LoRA                                 8.089 ms   1.00x
    S-LoRA nguyen ban                    9.728 ms   1.20x   <- khoang can bu
    S-LoRA + torch.compile("default")    9.186 ms   1.14x   <- lay lai 1/3
    S-LoRA muc 1 (mot gather + slice)    7.49 vs 6.56 o phep do rieng — CHAM HON
    S-LoRA muc 3 (Triton)              208.069 ms  26.00x   <- thua cuBLAS 32 lan

Kernel Triton DUNG (chung minh o duoi) nhung vo dung. Khuyen nghi: dung
torch.compile, bo muc 1 va muc 3.

--------------------------------------------------------- ba gia thuyet, do het
"Gather la nut that"            -> DUNG.  Gather ton 1.673 ms, dung bang toan bo
                                   chenh lech 1.639 ms giua S-LoRA va LoRA.
"Hai GEMM noi tiep ton them"    -> SAI.   8.024 ms cho hai GEMM noi tiep so voi
                                   8.030 ms cho mot GEMM to. Bang nhau.
"Gather chi so ngau nhien pha   -> SAI.   0.942 ms (ngau nhien) so voi 0.931 ms
 vo coalescing"                    (lien tiep). Khong khac gi.

--------------------------------------------------------- vi sao hai cach deu hong
MUC 1 hong vi gather CA m cot mot lan (1.801 ms) dat hon gather sel roi rest
rieng (0.361 + 0.954 = 1.315 ms), va slice co stride bat cuBLAS lam viec voi
layout xau hon ban sao lien tuc.

MUC 3 hong vi gop gather vao GEMM co nghia la phai TU VIET GEMM, ma GEMM tu
viet thua cuBLAS 32 lan. Gather chi chiem 17% thoi gian; danh doi 17% do de
mat 84% con lai la lo nang. Day la sai lam thiet ke, khong phai loi cai dat:
khong co cach nao vua gop duoc gather vua giu duoc cuBLAS.

--------------------------------------------------------- cho nao CON co the an
Gate va up doc CHUNG mot x (dau ra cua layernorm), k va v cung vay. Neu bat hai
lop cung mot cap dung CHUNG mot phep hoan vi thi mot lan gather phuc vu ca hai,
cat doi chi phi gather. Uoc tinh khoang cach tu 1.20x xuong ~1.10x, cong voi
torch.compile thi gan bang LoRA.

Nhung do la doi PHUONG PHAP chu khong phai doi kernel: pivoted QR se phai chon
mot tap pivot chung cho hai ma tran, tuc khong con toi uu cho tung ma tran. Anh
huong toi chat luong phan ra bao nhieu thi chua do.

--------------------------------------------------------------------- van de goc
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

PRECISION = "tf32"      # doi thanh "ieee" de co fp32 that

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
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     PREC: tl.constexpr = "tf32"):
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
            acc += tl.dot(a, b, input_precision=PREC)

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
        # 'tf32' la mac dinh cua tl.dot; 'ieee' cho fp32 that, cham hon
        # nhung dung de chung minh kernel khong sai logic.
        ctx.save_for_backward(sel, rest, XT)
        ctx.m = x.shape[-1]
        flat = x.reshape(-1, x.shape[-1])
        M, N, K = flat.shape[0], sel.numel(), rest.numel()
        out = torch.empty(M, N, device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
        _gather_gemm[grid](flat, XT, sel, rest, out, M, N, K,
                           flat.stride(0), flat.stride(1),
                           XT.stride(0), XT.stride(1),
                           out.stride(0), out.stride(1),
                           PREC=PRECISION)
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
