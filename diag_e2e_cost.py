#!/usr/bin/env python
"""Train hay cham diem moi la cho ton thoi gian cua thi nghiem E2E?

Lan truoc o Mistral/toan, cham chiem 45% wall clock va vLLM cat duoc 44.7x.
E2E khac han: beam search 10 nhanh, chuoi ngan (64 token), nhung 4700 cau.
Phai do truoc khi chon GPU, vi neu cham chi phoi thi GPU to khong giup may.

    python diag_e2e_cost.py Qwen/Qwen2.5-0.5B
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
N_PROMPT = 128                      # mau nho, nhan len sau
N_TEST = 4693                       # so MR duy nhat cua E2E test

tok = AutoTokenizer.from_pretrained(name)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
print(f"{name}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B tham so")

mr = ("name : The Eagle | eatType : coffee shop | food : Japanese | "
      "priceRange : less than 20 | customer rating : low | area : riverside |||")
prompts = [mr] * N_PROMPT
tok.padding_side = "left"


def gen(beams, batch):
    t = time.perf_counter()
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=64, num_beams=beams,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return time.perf_counter() - t


print()
print(f"{'cau hinh':<26}{'128 cau':>10}{'ca 4693':>12}")
print("-" * 50)
for beams, batch in ((10, 16), (10, 64), (5, 64), (1, 64), (1, 256)):
    try:
        s = gen(beams, batch)
        full = s * N_TEST / N_PROMPT
        print(f"  beam {beams:<3} batch {batch:<5}{s:>12.1f}s{full/60:>10.1f} phut")
    except torch.cuda.OutOfMemoryError:
        print(f"  beam {beams:<3} batch {batch:<5}{'OOM':>12}")
        torch.cuda.empty_cache()

print()
print(f"  VRAM dinh: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print()
print("  beam 10 la cau hinh cua paper LoRA, va la cai lan 1.5B truoc da dung.")
print("  Doi beam la doi BLEU, nen khong so duoc voi so cu nua.")
