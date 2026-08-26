#!/usr/bin/env python
"""
Do toc do train va inference: GPT-2 goc vs LoRA vs rowspace, ca truoc va sau khi merge.

Ly thuyet thi so FLOP cua rowspace BANG DUNG model goc, vi P @ C = W chinh xac va
tong cong viec khong doi:  (n*m - k^2) + k^2 = n*m.  Cai them vao chi la overhead
kernel launch + index_select/cat. Script nay do xem overhead do bao nhieu.

  python bench_speed.py --model gpt2-medium --tf32
"""
from __future__ import annotations

import argparse
import time

import torch

from finetune_e2e import LoRAConv1D, apply_lora
from rowspace_peft import NONSQUARE, convert_gpt2, merge_back


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1000        # ms/lan


@torch.no_grad()
def merge_lora(model):
    """Gop LoRA vao Conv1D goc -> kien truc tro lai y het model ban dau."""
    from rowspace_peft import _resolve
    for block in model.transformer.h:
        for name in NONSQUARE:
            parent, attr = _resolve(block, name)
            mod = getattr(parent, attr)
            if isinstance(mod, LoRAConv1D):
                mod.base.weight.data += mod.scale * (mod.down.data @ mod.up.data)
                setattr(parent, attr, mod.base)
    return model


def build(kind, a):
    from transformers import GPT2LMHeadModel
    # Phai phan ra / gan LoRA TRUOC roi moi .to(device), khong thi tham so moi
    # duoc tao tren CPU trong khi model da o GPU.
    m = GPT2LMHeadModel.from_pretrained(a.model)
    if kind.startswith("lora"):
        apply_lora(m, NONSQUARE, a.lora_rank, a.lora_rank)
        if kind.endswith("merged"):
            merge_lora(m)
    elif kind.startswith("rowspace"):
        convert_gpt2(m, NONSQUARE, a.basis, a.rank, None, torch.float64, verbose=False)
        if kind.endswith("merged"):
            merge_back(m, NONSQUARE)
    return m.to(a.device)


def bench(kind, a):
    m = build(kind, a)
    tok_ids = torch.randint(0, 50000, (a.batch, a.seq), device=a.device)

    # --- 1 buoc train: forward + backward
    # Ban da merge khong con tham so train duoc -> chi do inference cho no.
    # Kien truc cua no y het "original" nen train time cung se bang original.
    tp = [p for p in m.parameters() if p.requires_grad]
    t_train, vram = float("nan"), float("nan")
    if tp:
        m.train()

        def step():
            loss = m(tok_ids, labels=tok_ids).loss
            loss.backward()
            for p in tp:
                p.grad = None

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_train = timed(step, a.iters)
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0

    # --- inference: greedy, sinh a.new_tokens token
    m.eval()
    res = {}
    with torch.no_grad():
        for b in (1, 16):
            ids = torch.randint(0, 50000, (b, 32), device=a.device)
            res[b] = timed(lambda: m.generate(ids, max_new_tokens=a.new_tokens,
                                              do_sample=False, num_beams=1,
                                              pad_token_id=50256), max(3, a.iters // 4), warmup=2)
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return t_train, vram, res[1], res[16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--rank", type=int, default=16, help="hang cua rowspace")
    p.add_argument("--lora-rank", type=int, default=7, help="hang cua LoRA doi chung")
    p.add_argument("--basis", choices=["colperm", "svd"], default="colperm")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"model={a.model} device={a.device} tf32={a.tf32} "
          f"train batch={a.batch}x{a.seq}, sinh {a.new_tokens} token greedy\n")
    kinds = ["original", "lora", "lora-merged", "rowspace", "rowspace-merged"]
    rows = []
    for k in kinds:
        t, v, g1, g16 = bench(k, a)
        rows.append((k, t, v, g1, g16))
        tt = f"{t:7.1f} ms" if t == t else "      --"
        vv = f"{v:5.2f} GiB" if v == v else "   --    "
        print(f"  {k:18s} train {tt}  vram {vv}  "
              f"gen b1 {g1:7.1f} ms  gen b16 {g16:7.1f} ms", flush=True)

    base = rows[0]
    print(f"\n  {'':18s} {'train':>12s} {'gen b1':>12s} {'gen b16':>12s}   (so voi original)")
    for k, t, v, g1, g16 in rows:
        tt = f"{t / base[1]:11.2f}x" if t == t else "         --"
        print(f"  {k:18s} {tt} {g1 / base[3]:11.2f}x {g16 / base[4]:11.2f}x")


if __name__ == "__main__":
    main()
