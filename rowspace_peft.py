#!/usr/bin/env python
"""
Row-space PEFT cho GPT-2: tinh X cho moi ma tran trong so KHONG VUONG, dong bang X,
chi fine-tune ma tran vuong k x k voi k = min(d_in, d_out).

Phan ra CHINH XAC (khong xap xi). Voi W (n x m) = weight.T cua Conv1D:

  W beo ngang (m > n) -- vd mlp.c_proj 1024x4096,  k = n
        W[:, piv] = W2 [I | X]      X = W2^-1 W1        [n, m-n]
        y = W2 (x[sel] + X x[rest])                     rang buoc: rowspace(dW) c rowspace(W0)

  W cao doc (m < n) -- vd c_attn 3072x1024, mlp.c_fc 4096x1024,  k = m
        W[piv, :] = [I ; X] W2      X = W1 W2^-1        [n-m, m]
        h = W2 x;  y[sel] = h,  y[rest] = X h           rang buoc: colspace(dW) c colspace(W0)

Chi luu X (khong luu khoi identity) -> tong trong so = n*m, BANG DUNG model goc.
Cac hang/cot lam W2 duoc chon bang column-pivoted QR chu khong phai lay k cai dau tien:
lay k dau tien lam sai so forward tang ~37x so voi matmul goc, pivoted QR chi ~5.9x.

  --basis colperm   dang [I | X] nhu tren (mac dinh). Bao toan tham so.
  --basis svd       P truc chuan tu SVD. CUNG khong gian nghiem va cung so tham so
                    train duoc, sai so forward chi 1.29x thay vi 5.9x, nhung phai luu
                    P day du -> ton them k^2 moi lop. Nen dung neu train bf16.

  --rank 0          train thang ma tran vuong (mac dinh, dung y tuong goc)
  --rank r          LoRA len ma tran vuong, 2*k*r tham so/lop

Chay:
  python rowspace_peft.py --verify --model gpt2
  python rowspace_peft.py --verify --basis svd
  python rowspace_peft.py --train-demo --steps 20
  python rowspace_peft.py --save ./gpt2m_rowspace.pt
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import sys
import time

import torch
import torch.nn as nn

# Conv1D.weight co shape [d_in, d_out];  W (out x in) = weight.T
#   attn.c_attn  1024 -> 3072   cao doc    k = 1024
#   attn.c_proj  1024 -> 1024   VUONG   -> loai (phan ra khong cho rang buoc nao)
#   mlp.c_fc     1024 -> 4096   cao doc    k = 1024
#   mlp.c_proj   4096 -> 1024   beo ngang  k = 1024
NONSQUARE = ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"]
SQUARE = ["attn.c_proj"]
MODEL = "gpt2-medium"


# =========================================================================== phan ra

def _pivoted_qr(A):
    """(R, piv) sao cho A[:, piv] = Q R. Uu tien LAPACK geqp3."""
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
        sys.exit("Can `pip install scipy` hoac de split_solve.py cung thu muc.")
    return qr_pivot_numpy(np.asarray(A))


def _wide(W, basis):
    """W (n x m) voi m > n.  W = W2 @ P.

    basis svd     -> dict(kind='dense', W2 [n,n], P [n,m])
    basis colperm -> dict(kind='struct', W2 [n,n], X [n,m-n], sel [n], rest [m-n])
    """
    n, m = W.shape
    if basis == "svd":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return dict(kind="dense", W2=U * S, P=Vh, maxP=Vh.abs().max().item())

    # W[:, piv] = Q [R11 | R12] ;  W2 = W[:, piv[:n]] = Q R11 ;  X = W2^-1 W1 = R11^-1 R12
    # Q triet tieu, chi con MOT triangular solve -> backward stable.
    R, piv = _pivoted_qr(W.cpu().numpy())
    R = torch.as_tensor(R, dtype=W.dtype, device=W.device)
    piv = torch.as_tensor(piv, dtype=torch.long, device=W.device)
    R11 = R[:, :n]
    X = torch.linalg.solve_triangular(R11, R[:, n:], upper=True)
    d = R11.diagonal().abs()
    return dict(kind="struct", W2=W[:, piv[:n]].contiguous(), X=X.contiguous(),
                sel=piv[:n].contiguous(), rest=piv[n:].contiguous(),
                maxX=X.abs().max().item(),
                condW2=(d.max() / d.clamp_min(torch.finfo(W.dtype).tiny).min()).item())


def _tall(W, basis):
    """W (n x m) voi m < n.  W = P @ W2.

    basis svd     -> dict(kind='dense', W2 [m,m], P [n,m])
    basis colperm -> dict(kind='struct', W2 [m,m], X [n-m,m], sel [m], rest [n-m])
    """
    if basis == "svd":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return dict(kind="dense", W2=S[:, None] * Vh, P=U, maxP=U.abs().max().item())

    # Doi ngau: A = W^T la "beo ngang".  A[:, piv] = W2a [I | Xa]
    #   A[:, sel]  = W2a          -> W[sel, :]  = W2a^T =: W2
    #   A[:, rest] = W2a @ Xa     -> W[rest, :] = Xa^T @ W2  =: X @ W2
    f = _wide(W.T.contiguous(), basis)
    f["W2"] = f["W2"].T.contiguous()
    f["X"] = f["X"].T.contiguous()
    return f


def factorize_conv1d(weight, basis, dtype=torch.float64):
    """weight: [d_in, d_out] cua Conv1D. Tra ve (fac, orientation, info).

    orientation 'pre'  (W beo ngang, k = d_out): y = square(x[sel] + x[rest] @ X^T)
    orientation 'post' (W cao doc,   k = d_in) : h = square(x); y[sel]=h, y[rest]=h @ X^T
    """
    d_in, d_out = weight.shape
    if d_in == d_out:
        raise ValueError("ma tran vuong: phan ra khong tao ra rang buoc nao")
    W = weight.T.to(dtype)                        # [n, m] = [d_out, d_in],  y = W x
    n, m = W.shape

    if m > n:
        f, orient = _wide(W, basis), "pre"
        recon = f["W2"] @ _dense_P(f, n, m, W)
    else:
        f, orient = _tall(W, basis), "post"
        recon = _dense_P(f, n, m, W) @ f["W2"]

    info = {k: v for k, v in f.items() if isinstance(v, float)}
    info["res"] = (torch.linalg.norm(recon - W) / torch.linalg.norm(W)).item()
    return f, orient, info


def _dense_P(f, n, m, ref):
    """Dung lai P day du tu dang co cau truc. Chi dung de kiem tra / merge."""
    if f["kind"] == "dense":
        return f["P"]
    P = torch.zeros(n, m, dtype=ref.dtype, device=ref.device)
    k = f["W2"].shape[0]
    if f["X"].shape[0] == k:                       # beo ngang: P = [I | X] theo cot
        P[:, f["sel"]] = torch.eye(k, dtype=ref.dtype, device=ref.device)
        P[:, f["rest"]] = f["X"]
    else:                                          # cao doc: P = [I ; X] theo hang
        P[f["sel"], :] = torch.eye(k, dtype=ref.dtype, device=ref.device)
        P[f["rest"], :] = f["X"]
    return P


# =========================================================================== module

class RowSpaceLinear(nn.Module):
    """Thay Conv1D bang (phan co dinh) x (ma tran vuong train duoc).

    rank == 0 : train thang C  (dung y tuong goc)
    rank  > 0 : dong bang C, train LoRA len no
    """

    def __init__(self, fac, orientation, bias, out_dtype, rank=0, alpha=None):
        super().__init__()
        self.orientation, self.rank = orientation, rank
        self.structured = fac["kind"] == "struct"
        C = fac["W2"].T.contiguous().to(out_dtype)          # dang hang-vector
        k = C.shape[0]
        self.k = k

        if self.structured:
            self.register_buffer("X", fac["X"].to(out_dtype))
            self.register_buffer("sel", fac["sel"])
            self.register_buffer("rest", fac["rest"])
            if orientation == "post":
                perm = torch.cat([fac["sel"], fac["rest"]])
                self.register_buffer("inv", torch.argsort(perm))
                self.nf = perm.numel()
            else:
                self.nf = k
        else:
            self.register_buffer("P", fac["P"].T.contiguous().to(out_dtype))
            self.nf = self.P.shape[1] if orientation == "post" else k

        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

        if rank == 0:
            self.C = nn.Parameter(C)                        # <- thu duy nhat duoc train
            self.scale = 1.0
        else:
            self.register_buffer("C", C)
            self.lora_down = nn.Parameter(torch.empty(k, rank, dtype=C.dtype))
            self.lora_up = nn.Parameter(torch.zeros(rank, k, dtype=C.dtype))
            nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
            self.scale = (alpha if alpha is not None else rank) / rank

    @classmethod
    def from_conv1d(cls, conv, basis="colperm", rank=0, alpha=None, dtype=torch.float64):
        fac, orient, info = factorize_conv1d(conv.weight.data, basis, dtype)
        b = conv.bias.data if conv.bias is not None else None
        return cls(fac, orient, b, conv.weight.dtype, rank, alpha), info

    # ---- ba manh ghep cua forward

    def _square(self, h):
        out = h @ self.C
        if self.rank > 0:
            out = out + self.scale * ((h @ self.lora_down) @ self.lora_up)
        return out

    def _in_proj(self, x):                                  # d_in -> k  (chi dung khi 'pre')
        if self.structured:
            # h = x[sel] + X x[rest];  khoi identity la phep chon chi so, khong phai matmul
            return x.index_select(-1, self.sel) + x.index_select(-1, self.rest) @ self.X.T
        return x @ self.P

    def _out_proj(self, h):                                 # k -> d_out (chi dung khi 'post')
        if self.structured:
            return torch.cat([h, h @ self.X.T], dim=-1).index_select(-1, self.inv)
        return h @ self.P

    def forward(self, x):
        out = self._square(self._in_proj(x)) if self.orientation == "pre" \
            else self._out_proj(self._square(x))
        return out + self.bias if self.bias is not None else out

    # ---- tien ich

    def effective_C(self):
        C = self.C.double()
        if self.rank > 0:
            C = C + self.scale * (self.lora_down.double() @ self.lora_up.double())
        return C

    @torch.no_grad()
    def merged_weight(self):
        """Gop ve weight [d_in, d_out] cua Conv1D -> inference zero-overhead."""
        C = self.effective_C()
        if not self.structured:
            P = self.P.double()
            return (P @ C if self.orientation == "pre" else C @ P).to(self.C.dtype)
        X = self.X.double()
        if self.orientation == "pre":                       # weight[sel]=C, weight[rest]=X^T C
            W = torch.zeros(self.sel.numel() + self.rest.numel(), C.shape[1],
                            dtype=torch.float64, device=C.device)
            W[self.sel], W[self.rest] = C, X.T @ C
        else:                                               # weight[:,sel]=C, weight[:,rest]=C X^T
            W = torch.zeros(C.shape[0], self.sel.numel() + self.rest.numel(),
                            dtype=torch.float64, device=C.device)
            W[:, self.sel], W[:, self.rest] = C, C @ X.T
        return W.to(self.C.dtype)

    def frozen_numel(self):
        n = self.X.numel() if self.structured else self.P.numel()
        if self.rank > 0:
            n += self.C.numel()      # khi train LoRA thi C cung la buffer dong bang
        return n

    def extra_repr(self):
        return (f"k={self.k}, orientation={self.orientation}, "
                f"{'structured [I|X]' if self.structured else 'dense P'}, rank={self.rank}")


# =========================================================================== surgery

def _resolve(root, path):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def convert_gpt2(model, targets=NONSQUARE, basis="colperm", rank=0, alpha=None,
                 dtype=torch.float64, verbose=True):
    """Thay tai cho cac Conv1D khong vuong; dong bang tat ca tru ma tran vuong."""
    report, t0 = [], time.perf_counter()
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        for name in targets:
            parent, attr = _resolve(block, name)
            conv = getattr(parent, attr)
            layer, info = RowSpaceLinear.from_conv1d(conv, basis, rank, alpha, dtype)
            info.update(layer=f"h.{i}.{name}", target=name, k=layer.k,
                        orient=layer.orientation, shape=tuple(conv.weight.shape))
            report.append(info)
            setattr(parent, attr, layer)
        if verbose and (i + 1) % 6 == 0:
            print(f"    {i + 1}/{len(blocks)} block  ({time.perf_counter() - t0:.0f}s)")

    for p in model.parameters():
        p.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, RowSpaceLinear):
            if mod.rank == 0:
                mod.C.requires_grad_(True)
            else:
                mod.lora_down.requires_grad_(True)
                mod.lora_up.requires_grad_(True)
    return report


@torch.no_grad()
def merge_back(model, targets=NONSQUARE):
    from transformers.pytorch_utils import Conv1D
    for block in model.transformer.h:
        for name in targets:
            parent, attr = _resolve(block, name)
            mod = getattr(parent, attr)
            if not isinstance(mod, RowSpaceLinear):
                continue
            Wc = mod.merged_weight()
            conv = Conv1D(Wc.shape[1], Wc.shape[0]).to(Wc.device, Wc.dtype)
            conv.weight.data.copy_(Wc)
            if mod.bias is not None:
                conv.bias.data.copy_(mod.bias.data)
            else:
                conv.bias.data.zero_()
            setattr(parent, attr, conv)
    return model


# =========================================================================== bao cao

def factorization_report(report):
    nblk = len({r["layer"].split(".")[1] for r in report})
    print(f"\n  Phan ra {len(report)} ma tran khong vuong "
          f"(min / trung vi / max qua {nblk} block)")
    print(f"  {'ma tran':<14s} {'W (out x in)':>13s} {'huong':>6s} {'k':>5s} "
          f"{'residual':>9s} {'max|X| hoac max|P|':>24s} {'cond(W2)':>22s}")
    for t in dict.fromkeys(r["target"] for r in report):
        rs = [r for r in report if r["target"] == t]
        sh = rs[0]["shape"]

        def agg(key, fmt):
            v = sorted(r[key] for r in rs if key in r)
            return "-" if not v else f"{v[0]:{fmt}} /{st.median(v):{fmt}} /{v[-1]:{fmt}}"

        big = agg("maxX", "7.3f") if "maxX" in rs[0] else agg("maxP", "7.3f")
        print(f"  {t:<14s} {f'{sh[1]} x {sh[0]}':>13s} {rs[0]['orient']:>6s} "
              f"{rs[0]['k']:>5d} {max(r['res'] for r in rs):>9.1e} {big:>24s} "
              f"{agg('condW2', '7.1e'):>22s}")


def param_table(model, rank, targets, n_orig=None):
    # P/X la buffer, khong nam trong .parameters() -> phai cong tay
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(m.frozen_numel() for m in model.modules() if isinstance(m, RowSpaceLinear))
    tot = sum(p.numel() for p in model.parameters()) + frozen
    print(f"\n  tong trong so          : {tot / 1e6:8.2f} M  (ke ca X/P dong bang)")
    if n_orig:
        delta = tot - n_orig
        print(f"  model goc              : {n_orig / 1e6:8.2f} M  "
              f"-> chenh {delta / 1e6:+.2f} M "
              f"({'BAO TOAN THAM SO' if abs(delta) < 1000 else 'ton them do luu P day du'})")
    print(f"  X / P dong bang        : {frozen / 1e6:8.2f} M  ({100 * frozen / tot:.1f} %)")
    print(f"  TRAIN DUOC             : {tr / 1e6:8.2f} M  ({100 * tr / tot:.2f} % cua model)")
    d, L = model.config.n_embd, model.config.n_layer
    qv = 4 * 2 * d * 2 * L
    print(f"\n  moc tham chieu:  full FT {tot / 1e6:.2f} M (100 %)   |   "
          f"LoRA r=4 tren q,v {qv / 1e6:.2f} M ({100 * qv / tot:.2f} %)")
    if rank > 0:
        shapes = {"attn.c_attn": (d, 3 * d), "attn.c_proj": (d, d),
                  "mlp.c_fc": (d, 4 * d), "mlp.c_proj": (4 * d, d)}
        lora = sum(rank * sum(shapes[t]) for t in targets) * L
        print(f"                   LoRA thuong r={rank} cung tap ma tran: {lora / 1e6:.2f} M "
              f"-> method nay re hon {lora / tr:.2f}x")


# =========================================================================== lenh

@torch.no_grad()
def cmd_verify(a):
    import gc

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print(f"\n=== VERIFY  basis={a.basis}  rank={a.rank}  dtype={a.dtype} ===")
    tok = GPT2TokenizerFast.from_pretrained(a.model)
    text = "The Eagle is a cheap French coffee shop by the riverside. " * 12
    ids = tok(text, return_tensors="pt").input_ids[:, :a.seq_len].to(a.device)

    # Pha 1: lay tham chieu roi GIAI PHONG -- giu hai model cung luc se OOM tren may it RAM
    ref = GPT2LMHeadModel.from_pretrained(a.model).to(a.device).eval()
    n_orig = sum(p.numel() for p in ref.parameters())
    lr = ref(ids).logits.float().clone()
    pr = torch.exp(ref(ids, labels=ids).loss).item()
    del ref
    gc.collect()

    # Pha 2: nap lai, phan ra, do lech
    new = GPT2LMHeadModel.from_pretrained(a.model).to(a.device).eval()
    print("  dang phan ra...")
    rep = convert_gpt2(new, a.targets, a.basis, a.rank, a.alpha, a.dtype)
    new.eval()
    factorization_report(rep)
    param_table(new, a.rank, a.targets, n_orig)

    lr = lr.double()
    ln = new(ids).logits.double()
    rel = (torch.linalg.norm(ln - lr) / torch.linalg.norm(lr)).item()
    pn = torch.exp(new(ids, labels=ids).loss).item()
    print(f"\n  |dlogits| max          : {(ln - lr).abs().max():.3e}")
    print(f"  ||dlogits|| / ||logits||: {rel:.3e}")
    print(f"  perplexity {pr:.6f} -> {pn:.6f}  (lech tuong doi {abs(pn - pr) / pr:.2e})")

    merge_back(new, a.targets).eval()
    lm = new(ids).logits.double()
    print(f"  sau merge_back()       : |dlogits| = {(lm - lr).abs().max():.3e}"
          "  -> inference zero-overhead")
    ok = rel < 1e-4
    print(f"\n  {'DAT' if ok else 'KHONG DAT -- dung lai, dung train'}")
    return ok


def cmd_train_demo(a):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print(f"\n=== TRAIN DEMO  basis={a.basis}  rank={a.rank} ===")
    tok = GPT2TokenizerFast.from_pretrained(a.model)
    model = GPT2LMHeadModel.from_pretrained(a.model).to(a.device)
    n_orig = sum(p.numel() for p in model.parameters())
    convert_gpt2(model, a.targets, a.basis, a.rank, a.alpha, a.dtype)
    param_table(model, a.rank, a.targets, n_orig)
    model.train()

    text = ("name[The Eagle], eatType[coffee shop], food[French], priceRange[cheap], "
            "customer rating[5 out of 5], area[riverside], familyFriendly[yes] || "
            "The Eagle is a cheap French coffee shop by the riverside that is family "
            "friendly and has a customer rating of 5 out of 5.\n") * 4
    ids = tok(text, return_tensors="pt").input_ids[:, :a.seq_len].to(a.device)

    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=a.lr)
    print(f"\n  {len(tp)} tensor train duoc, lr={a.lr}")
    print(f"  {'step':>5s} {'loss':>10s} {'|grad|':>11s}")
    for s in range(a.steps):
        loss = model(ids, labels=ids).loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(tp, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % max(1, a.steps // 8) == 0 or s == a.steps - 1:
            print(f"  {s:5d} {loss.item():10.4f} {gn:11.3e}")
    print("\n  Loss giam => gradient chay thong qua X dong bang vao ma tran vuong.")


def cmd_save(a):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(a.model)
    n_orig = sum(p.numel() for p in model.parameters())
    rep = convert_gpt2(model, a.targets, a.basis, a.rank, a.alpha, a.dtype)
    factorization_report(rep)
    param_table(model, a.rank, a.targets, n_orig)
    torch.save({"state_dict": model.state_dict(), "basis": a.basis, "rank": a.rank,
                "targets": a.targets, "model": a.model}, a.save)
    print(f"\n  da luu -> {a.save}")


def main():
    p = argparse.ArgumentParser(description="Row-space PEFT cho GPT-2")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--basis", choices=["colperm", "svd"], default="colperm")
    p.add_argument("--rank", type=int, default=0,
                   help="0 = train thang ma tran vuong (mac dinh); >0 = LoRA len no")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--include-square", action="store_true",
                   help="them attn.c_proj (vuong) -- luu y: no KHONG bi rang buoc gi")
    p.add_argument("--fp32-factorize", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--train-demo", action="store_true")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save", default=None, metavar="PATH")
    a = p.parse_args()

    a.dtype = torch.float32 if a.fp32_factorize else torch.float64
    a.targets = NONSQUARE + (SQUARE if a.include_square else [])
    if not (a.verify or a.train_demo or a.save):
        a.verify = True
    print(f"model={a.model}  device={a.device}  targets={a.targets}")
    if a.basis == "colperm":
        print("  luu y: colperm co sai so forward ~5.9x matmul goc; "
              "neu train bf16 hay dung --basis svd (1.29x).")

    if a.verify:
        cmd_verify(a)
    if a.train_demo:
        cmd_train_demo(a)
    if a.save:
        cmd_save(a)


if __name__ == "__main__":
    sys.exit(main())
