#!/usr/bin/env python
"""Kiem tra phan ra rowspace tren Qwen2.5-1.5B: chinh xac, dem tham so, merge lai."""
import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft_generic import TARGETS, apply_lora, convert_rowspace, merge_back, param_counts

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
p.add_argument("--target-set", default="qwen2_kv", choices=sorted(TARGETS))
p.add_argument("--rank", type=int, default=8)
p.add_argument("--lora-rank", type=int, default=2)
a = p.parse_args()

T = TARGETS[a.target_set]
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"model={a.model}  targets={T}\n")

tok = AutoTokenizer.from_pretrained(a.model)
txt = "The Eagle is a cheap French coffee shop by the riverside. " * 6
ids = tok(txt, return_tensors="pt").input_ids[:, :128].to(dev)

# --- tham chieu, roi giai phong ngay (1.5B fp32 = 6.2 GB moi ban)
ref = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
n0 = sum(x.numel() for x in ref.parameters())
with torch.no_grad():
    logits_ref = ref(ids).logits.float().clone()
    ppl_ref = torch.exp(ref(ids, labels=ids).loss).item()
del ref
gc.collect()
torch.cuda.empty_cache()
print(f"goc: {n0/1e6:.2f} M tham so, ppl {ppl_ref:.6f}\n")

for meth, rank in [("rowspace", a.rank), ("lora", a.lora_rank)]:
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    if meth == "rowspace":
        rep = convert_rowspace(m, T, "colperm", rank, None, torch.float64, verbose=False)
        res = max(r["res"] for r in rep)
        mx = max(r["maxX"] for r in rep)
        cd = max(r["condW2"] for r in rep)
        sh = rep[0]["shape"]
        print(f"  phan ra {len(rep)} ma tran, W(out x in)={sh[1]}x{sh[0]}, k={rep[0]['k']}, "
              f"huong={rep[0]['orient']}")
        print(f"  residual max {res:.2e} | max|X| {mx:.3f} | cond(W2) max {cd:.2e}")
    else:
        apply_lora(m, T, rank, rank)
    info = param_counts(m, n0)
    m.eval()
    with torch.no_grad():
        ln = m(ids).logits.float()
        ppl = torch.exp(m(ids, labels=ids).loss).item()
    rel = (torch.linalg.norm(ln - logits_ref) / torch.linalg.norm(logits_ref)).item()
    print(f"{meth:9s} r={rank:<3d} train {info['trainable']/1e6:7.3f} M "
          f"({100*info['trainable']/info['total']:.4f} %)  tong {info['total']/1e6:.2f} M  "
          f"||dlogits||/||logits|| {rel:.2e}  ppl {ppl_ref:.6f} -> {ppl:.6f}")

    if meth == "rowspace":
        merge_back(m, T)
        m.eval()
        with torch.no_grad():
            lm = m(ids).logits.float()
        r2 = (torch.linalg.norm(lm - logits_ref) / torch.linalg.norm(logits_ref)).item()
        print(f"  sau merge_back(): ||dlogits||/||logits|| {r2:.2e}  -> inference zero-overhead")
    print()
    del m
    gc.collect()
    torch.cuda.empty_cache()
