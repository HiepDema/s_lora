#!/usr/bin/env python
"""Gop adapter vao trong so goc roi luu ra checkpoint HF chuan.

Dung de chuyen sang vLLM cham diem: sau khi gop, kien truc tro lai y het model
ban dau (merge_all dung lai nn.Linear), nen vLLM nap duoc nhu mot model thuong.

    python export_merged.py --ckpt runs_pissa_r116/hybrid_r116_na_s0_last.pt \\
      --method hybrid --rank 116 --rank-square 116 --target-set all7 \\
      --model mistralai/Mistral-7B-v0.1 --fac-cache fac_cache/mistral7b \\
      --out merged/slora_r116

Phai truyen DUNG --method/--rank/--target-set cua run goc: phan dong bang khong
nam trong checkpoint ma sinh lai tat dinh tu W0, nen lech cau hinh la lech model.
load_trainable_ckpt se bao loi neu so tensor khong khop, khong nap im lang.

Luu bang bf16: vLLM chay bf16, va giu fp32 chi lam file to gap doi (29 GB).
Phep gop chay trong fp64 (merged_weight) roi moi ha xuong, nen khong mat do
chinh xac o buoc gop.
"""
from __future__ import annotations

import argparse
import os

import torch

import finetune_e2e as E
from peft_generic import TARGETS, merge_all


def main():
    p = argparse.ArgumentParser(description="gop adapter -> checkpoint HF")
    p.add_argument("--ckpt", required=True, help="*_last.pt hoac *_best.pt")
    p.add_argument("--out", required=True, help="thu muc luu model da gop")
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--method", default="hybrid",
                   choices=["rowspace", "hybrid", "lora", "vera", "target-ft"])
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--rank-square", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--alpha-square", type=float, default=None)
    p.add_argument("--target-set", default="all7")
    p.add_argument("--basis", default="colperm")
    p.add_argument("--fac-cache", default=None)
    p.add_argument("--factorize-device", default="auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    a = p.parse_args()

    a.targets = TARGETS[a.target_set]
    a.alpha = a.rank if a.alpha is None else a.alpha
    a.rank_square = a.rank if a.rank_square is None else a.rank_square
    a.alpha_square = a.rank_square if a.alpha_square is None else a.alpha_square
    a.dtype = torch.float64          # phan ra luon fp64
    a.model_dtype = "fp32"           # gop trong fp32 roi moi ha xuong khi luu
    a.attn = "auto"
    a.grad_ckpt = False
    a.vera_d_init = 0.1

    print(f"  dung lai model: {a.method} r={a.rank}", flush=True)
    model, _ = E.build_model(a)
    model = model.to(a.device)

    print(f"  nap {a.ckpt}", flush=True)
    E.load_trainable_ckpt(a.ckpt, model, a.device)

    print("  gop adapter vao trong so goc...", flush=True)
    merge_all(model, a.targets)

    out_dt = E.DTYPES[a.save_dtype]
    model = model.to(out_dt)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)

    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)

    mb = sum(os.path.getsize(os.path.join(a.out, f))
             for f in os.listdir(a.out)) / 1e9
    print(f"  da luu {a.out}  ({mb:.1f} GB, {a.save_dtype})")
    print(f"  cham diem: python eval_vllm.py --model {a.out} --eval-math")


if __name__ == "__main__":
    main()
