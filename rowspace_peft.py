#!/usr/bin/env python
"""Row-space PEFT — GPT-2 verification CLI and backward-compatibility shim.

Core code has moved to the ``slora`` package:
    slora/factorize.py  — pivoted-QR / SVD factorization
    slora/adapters.py   — RowSpaceLinear, LoRA, VeRA modules
    slora/surgery.py    — inject / merge / count

This file re-exports everything the old ``from rowspace_peft import …``
statements need, and keeps the GPT-2-specific CLI (--verify, --train-demo,
--save) that is not part of the generic package.

Usage:
  python rowspace_peft.py --verify --model gpt2
  python rowspace_peft.py --verify --basis svd
  python rowspace_peft.py --train-demo --steps 20
  python rowspace_peft.py --save ./gpt2m_rowspace.pt
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time

import torch
import torch.nn as nn

# ===== Re-exports from slora (backward compat) =============================
from slora.factorize import (                          # noqa: F401
    factorize as factorize_conv1d,
    factorize_cached,
    pivoted_qr as _pivoted_qr,
    reconstruct_P as _dense_P,
    rebuild_from_pivots as _rebuild,
)
from slora.adapters import RowSpaceLinear               # noqa: F401

# ===== GPT-2 specific constants ============================================
NONSQUARE = ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"]
SQUARE    = ["attn.c_proj"]
MODEL     = "gpt2-medium"


# ===== GPT-2 surgery (Conv1D layout) =======================================

def _resolve(root, path):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def convert_gpt2(model, targets=NONSQUARE, basis="colperm", rank=0, alpha=None,
                 dtype=torch.float64, verbose=True):
    """Replace non-square Conv1D layers in GPT-2 with RowSpaceLinear."""
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
    """Merge RowSpaceLinear back to Conv1D (GPT-2 specific)."""
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


# ===== Reporting ============================================================

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


# ===== CLI ==================================================================

@torch.no_grad()
def cmd_verify(a):
    import gc
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print(f"\n=== VERIFY  basis={a.basis}  rank={a.rank}  dtype={a.dtype} ===")
    tok = GPT2TokenizerFast.from_pretrained(a.model)
    text = "The Eagle is a cheap French coffee shop by the riverside. " * 12
    ids = tok(text, return_tensors="pt").input_ids[:, :a.seq_len].to(a.device)

    ref = GPT2LMHeadModel.from_pretrained(a.model).to(a.device).eval()
    n_orig = sum(p.numel() for p in ref.parameters())
    lr = ref(ids).logits.float().clone()
    pr = torch.exp(ref(ids, labels=ids).loss).item()
    del ref
    gc.collect()

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
