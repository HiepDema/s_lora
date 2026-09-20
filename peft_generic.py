#!/usr/bin/env python
"""
Lop truu tuong cho ca GPT-2 (Conv1D) va Qwen2 / Llama (nn.Linear).

Meo chinh: nn.Linear.weight co shape [out, in], con Conv1D.weight la [in, out].
Nen linear.weight.T CHINH LA layout cua Conv1D -> tai dung nguyen phan ra
factorize_conv1d() va module RowSpaceLinear da co, khong viet lai gi.

Qwen2.5-1.5B (28 lop, d=1536, heads=12, kv=2, ffn=8960):
  self_attn.q_proj  1536x1536  VUONG   -> khong phan ra duoc
  self_attn.k_proj   256x1536  a=6.0   -> loi the (1+a)/2 = 3.5x
  self_attn.v_proj   256x1536  a=6.0   -> loi the 3.5x
  self_attn.o_proj  1536x1536  VUONG   -> khong phan ra duoc
  mlp.gate_proj     8960x1536  a=5.83  -> loi the 3.4x
  mlp.up_proj       8960x1536  a=5.83  -> loi the 3.4x
  mlp.down_proj     1536x8960  a=5.83  -> loi the 3.4x
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from rowspace_peft import RowSpaceLinear, factorize_conv1d

# --- tap ma tran dich theo kien truc -----------------------------------------
TARGETS = {
    "gpt2": ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"],
    "qwen2_kv": ["self_attn.k_proj", "self_attn.v_proj"],
    "qwen2_mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "qwen2_all": ["self_attn.k_proj", "self_attn.v_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    # q_proj VUONG (1536x1536) -> chi dung duoc voi --method hybrid hoac lora
    "qwen2_qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    # Llama dung dung ten module nhu Qwen2 -> alias cho de doc, khong phai tap moi
    "llama_kv": ["self_attn.k_proj", "self_attn.v_proj"],
    "llama_mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
}


def get_blocks(model):
    """Tra ve ModuleList cac transformer block, bat ke kien truc."""
    for path in ("transformer.h", "model.layers", "model.decoder.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"khong tim thay block trong {type(model).__name__}")


def resolve(root, path):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _conv1d_style_weight(mod):
    """Tra ve (weight [d_in, d_out], is_linear). Chuan hoa hai kieu lop ve mot."""
    if isinstance(mod, nn.Linear):
        return mod.weight.data.T.contiguous(), True      # [out,in] -> [in,out]
    return mod.weight.data, False                        # Conv1D da la [in,out]


# =========================================================================== LoRA

class LoRALinear(nn.Module):
    """LoRA cho nn.Linear: y = x W^T + b + (alpha/r)(x A) B."""

    def __init__(self, lin, rank, alpha):
        super().__init__()
        self.base = lin
        d_in, d_out = lin.in_features, lin.out_features
        dt, dv = lin.weight.dtype, lin.weight.device
        self.down = nn.Parameter(torch.empty(d_in, rank, dtype=dt, device=dv))
        self.up = nn.Parameter(torch.zeros(rank, d_out, dtype=dt, device=dv))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.down) @ self.up)


class LoRAConv1D(nn.Module):
    """LoRA cho Conv1D cua GPT-2: y = x W + b + (alpha/r)(x A) B."""

    def __init__(self, conv, rank, alpha):
        super().__init__()
        self.base = conv
        d_in, d_out = conv.weight.shape
        dt, dv = conv.weight.dtype, conv.weight.device
        self.down = nn.Parameter(torch.empty(d_in, rank, dtype=dt, device=dv))
        self.up = nn.Parameter(torch.zeros(rank, d_out, dtype=dt, device=dv))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank
        self.nf = conv.nf

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.down) @ self.up)


def apply_lora(model, targets, rank, alpha):
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            wrap = LoRALinear if isinstance(mod, nn.Linear) else LoRAConv1D
            setattr(parent, attr, wrap(mod, rank, alpha))
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv1D)):
            m.down.requires_grad_(True)
            m.up.requires_grad_(True)
    return model


# =========================================================================== VeRA

class VeRAShared(nn.Module):
    """Giu A va B dung chung cho MOI lop.

    Dang ky lam buffer o DAY va chi o day, roi cac lop VeRA tro toi bang
    object.__setattr__ (khong qua nn.Module.__setattr__, neu khong moi lop se
    thanh cha cua no va model.to(dev) se nhan ra 56 ban sao rieng biet).
    persistent=False vi A, B sinh lai duoc tu seed -> khong can vao checkpoint.
    """

    def __init__(self, A, B):
        super().__init__()
        self.register_buffer("A", A, persistent=False)
        self.register_buffer("B", B, persistent=False)


class VeRALayer(nn.Module):
    """VeRA (Kopiczko et al., ICLR 2024): h = x W0 + ((x A) * d) B * b.

    A [d_in, r] va B [r, d_out] ngau nhien, DONG BANG, DUNG CHUNG moi lop;
    chi hai vector d (r) va b (d_out) duoc train.

    Tham so moi lop = r + d_out, KHONG phu thuoc d_in. He qua cho k_proj/v_proj
    cua Qwen (d_in=1536, d_out=256): r=256 moi ton bang S-LoRA r=1. Chay VeRA
    o r nho nhu LoRA la vo nghia.

    b khoi tao 0 -> delta = 0 luc bat dau, dung nhu LoRA.
    """

    def __init__(self, base, shared, d_in, d_out, rank, d_init):
        super().__init__()
        self.base = base
        object.__setattr__(self, "shared", shared)     # xem VeRAShared
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        w = base.weight
        self.vera_d = nn.Parameter(torch.full((rank,), d_init,
                                              dtype=w.dtype, device=w.device))
        self.vera_b = nn.Parameter(torch.zeros(d_out, dtype=w.dtype, device=w.device))
        if not isinstance(base, nn.Linear):
            self.nf = base.nf                          # Conv1D cua GPT-2

    def _AB(self):
        return (self.shared.A[:self.d_in, :self.rank],
                self.shared.B[:self.rank, :self.d_out])

    def forward(self, x):
        A, B = self._AB()
        return self.base(x) + ((((x @ A) * self.vera_d) @ B) * self.vera_b)


def _target_shapes(model, targets):
    """Tra ve list (d_in, d_out) theo layout Conv1D cho moi ma tran dich."""
    out = []
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            w, _ = _conv1d_style_weight(getattr(parent, attr))
            out.append(tuple(w.shape))
    return out


def apply_vera(model, targets, rank, seed=0, d_init=0.1):
    """Gan VeRA. A, B sinh mot lan tu `seed` roi cat lat cho tung lop."""
    shapes = _target_shapes(model, targets)
    max_in = max(s[0] for s in shapes)
    max_out = max(s[1] for s in shapes)
    ref = next(model.parameters())
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Kaiming uniform viet tay thay vi nn.init.kaiming_uniform_(generator=...):
    # kwarg `generator` chi co tu torch 2.4, va _calculate_fan_in_and_fan_out
    # gia dinh layout [out, in] trong khi A cua ta la [in, r]. Tu tinh bound
    # theo dung chieu bi co lai trong phep nhan: 1/sqrt(fan_in) voi a=sqrt(5).
    A = torch.empty(max_in, rank, dtype=ref.dtype).uniform_(
        -1 / math.sqrt(max_in), 1 / math.sqrt(max_in), generator=g)
    B = torch.empty(rank, max_out, dtype=ref.dtype).uniform_(
        -1 / math.sqrt(rank), 1 / math.sqrt(rank), generator=g)
    shared = VeRAShared(A.to(ref.device), B.to(ref.device))
    model.add_module("_vera_shared", shared)           # de model.to(dev) keo theo

    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            w, _ = _conv1d_style_weight(mod)
            setattr(parent, attr,
                    VeRALayer(mod, shared, w.shape[0], w.shape[1], rank, d_init))

    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, VeRALayer):
            m.vera_d.requires_grad_(True)
            m.vera_b.requires_grad_(True)
    return model


@torch.no_grad()
def merge_vera(model, targets):
    """Gop VeRA vao trong so goc -> kien truc tro lai y het model ban dau."""
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            if not isinstance(mod, VeRALayer):
                continue
            A, B = mod._AB()
            d = ((A.double() * mod.vera_d.double()) @ B.double()
                 ) * mod.vera_b.double()               # [d_in, d_out]
            w = mod.base.weight.data
            w += (d.T if isinstance(mod.base, nn.Linear) else d).to(w.dtype)
            setattr(parent, attr, mod.base)
    if hasattr(model, "_vera_shared"):
        del model._vera_shared
    return model


# =========================================================================== rowspace

class RowSpaceLinearWrap(RowSpaceLinear):
    """Nhu RowSpaceLinear nhung nhan nn.Linear va tra ve dung shape [out]."""


def convert_rowspace(model, targets, basis="colperm", rank=0, alpha=None,
                     dtype=torch.float64, verbose=True):
    """Phan ra cac lop dich; dong bang tat ca tru phan train duoc."""
    import time
    report, t0 = [], time.perf_counter()
    blocks = get_blocks(model)
    for i, block in enumerate(blocks):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            w, is_lin = _conv1d_style_weight(mod)
            if w.shape[0] == w.shape[1]:
                raise ValueError(f"h.{i}.{name} vuong {tuple(w.shape)} — bo khoi targets")
            fac, orient, info = factorize_conv1d(w, basis, dtype)
            bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
            layer = RowSpaceLinear(fac, orient, bias, mod.weight.dtype, rank, alpha)
            info.update(layer=f"h.{i}.{name}", target=name, k=layer.k,
                        orient=orient, shape=tuple(w.shape), is_linear=is_lin)
            report.append(info)
            setattr(parent, attr, layer)
        if verbose and (i + 1) % max(1, len(blocks) // 4) == 0:
            print(f"    {i + 1}/{len(blocks)} block  ({time.perf_counter() - t0:.0f}s)", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, RowSpaceLinear):
            if m.rank == 0:
                m.C.requires_grad_(True)
            else:
                m.lora_down.requires_grad_(True)
                m.lora_up.requires_grad_(True)
    return report


def apply_hybrid(model, targets, rank, rank_square, alpha=None, alpha_square=None,
                 basis="colperm", dtype=torch.float64, verbose=True):
    """S-LoRA cho ma tran KHONG VUONG, LoRA thuong cho ma tran VUONG.

    Ma tran vuong (q_proj, o_proj) khong phan ra duoc — [I|X] voi k = n = m thi
    X rong, khong tao ra rang buoc nao. Nen chung phai dung LoRA.

    Hai hang tach roi: `rank` cho phan S-LoRA (k,v), `rank_square` cho phan LoRA
    (q). Diem chinh cua cau hinh nay la S-LoRA doi GIA TUONG DOI giua hai nhom:
    mot bac hang tren (k,v) chi con ton 1024/lop thay vi 3584, trong khi tren q
    van 3072 — nen cung ngan sach thi mua duoc nhieu hang o k,v hon han.
    """
    import time
    report, t0 = [], time.perf_counter()
    blocks = get_blocks(model)
    n_sq = n_fac = 0
    for i, block in enumerate(blocks):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            w, _ = _conv1d_style_weight(mod)
            if w.shape[0] == w.shape[1]:                       # vuong -> LoRA
                wrap = LoRALinear if isinstance(mod, nn.Linear) else LoRAConv1D
                setattr(parent, attr,
                        wrap(mod, rank_square, alpha_square or rank_square))
                n_sq += 1
            else:                                              # khong vuong -> S-LoRA
                fac, orient, info = factorize_conv1d(w, basis, dtype)
                bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
                layer = RowSpaceLinear(fac, orient, bias, mod.weight.dtype, rank, alpha)
                info.update(layer=f"h.{i}.{name}", target=name, k=layer.k,
                            orient=orient, shape=tuple(w.shape))
                report.append(info)
                setattr(parent, attr, layer)
                n_fac += 1
        if verbose and (i + 1) % max(1, len(blocks) // 4) == 0:
            print(f"    {i + 1}/{len(blocks)} block  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    if verbose:
        print(f"    hybrid: {n_fac} ma tran S-LoRA (r={rank}), "
              f"{n_sq} ma tran vuong LoRA (r={rank_square})")

    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, RowSpaceLinear):
            if m.rank == 0:
                m.C.requires_grad_(True)
            else:
                m.lora_down.requires_grad_(True)
                m.lora_up.requires_grad_(True)
        elif isinstance(m, (LoRALinear, LoRAConv1D)):
            m.down.requires_grad_(True)
            m.up.requires_grad_(True)
    return report


@torch.no_grad()
def merge_back(model, targets):
    """Gop ve lop goc -> inference khong ton them chi phi."""
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            if not isinstance(mod, RowSpaceLinear):
                continue
            W = mod.merged_weight()                    # [d_in, d_out]
            lin = nn.Linear(W.shape[0], W.shape[1], bias=mod.bias is not None)
            lin = lin.to(W.device, W.dtype)
            lin.weight.data.copy_(W.T)                 # Linear can [out, in]
            if mod.bias is not None:
                lin.bias.data.copy_(mod.bias.data)
            setattr(parent, attr, lin)
    return model


@torch.no_grad()
def merge_lora(model, targets):
    """Gop LoRA vao trong so goc -> kien truc tro lai y het model ban dau."""
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            if not isinstance(mod, (LoRALinear, LoRAConv1D)):
                continue
            d = mod.scale * (mod.down.data.double() @ mod.up.data.double())   # [in, out]
            w = mod.base.weight.data
            # Linear luu [out, in] nen phai chuyen vi; Conv1D da la [in, out]
            w += (d.T if isinstance(mod.base, nn.Linear) else d).to(w.dtype)
            setattr(parent, attr, mod.base)
    return model


def merge_all(model, targets):
    """Gop bat ke dang nao dang duoc dung."""
    merge_back(model, targets)
    merge_vera(model, targets)
    return merge_lora(model, targets)


# =========================================================================== dem

def param_counts(model, n_orig):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(m.frozen_numel() for m in model.modules() if isinstance(m, RowSpaceLinear))
    tot = sum(p.numel() for p in model.parameters()) + frozen
    return dict(trainable=tr, frozen=frozen, total=tot, n_orig=n_orig)


def predict_params(shapes, rank, L, method):
    """shapes: list (out, in). Tra ve so tham so train duoc du kien.

    VeRA: r + d_out moi ma tran (A, B dong bang va sinh lai duoc tu seed nen
    khong tinh). Voi shapes theo (out, in) thi d_out = o.
    """
    s = 0
    for o, i in shapes:
        if method == "rowspace":
            s += 2 * min(o, i) * rank
        elif method == "vera":
            s += rank + o
        elif method == "hybrid":
            s += 2 * min(o, i) * rank if o != i else rank * (o + i)
        else:
            s += rank * (o + i)
    return s * L


def vera_rank_for(shapes, L, budget):
    """Chon rank VeRA de tong tham so train duoc bam sat `budget`."""
    n = len(shapes) * L
    return max(1, round((budget - sum(o for o, _ in shapes) * L) / n))
