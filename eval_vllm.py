#!/usr/bin/env python
"""Cham GSM8K + MATH bang vLLM tren model DA GOP adapter.

    python export_merged.py --ckpt ..._last.pt ... --out merged/slora_r116
    python eval_vllm.py --model merged/slora_r116 --eval-math --tag slora_r116

Vi sao tach ra process rieng: model train va vLLM deu muon gan het VRAM, chay
chung mot process thi phai go model train roi moi khoi tao vLLM — de sot bo nho
va hong ca run. Tach ra thi loi cua buoc cham khong keo theo buoc train.

Cham bang DUNG bo cua MetaMath (metamath_eval.py), giong finetune_math.py, nen
so lieu hai duong deu so duoc voi PiSSA/PMSS. Ghi ra cung dinh dang _gen.jsonl
va cung thu tu goc, de phan tich MATH theo chu de van chay duoc.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import metamath_eval as MM
from finetune_math import PROMPT, load_gsm8k, load_math, last_boxed, extract_gold


def run(llm, sp, tests, kind, tag, out_dir):
    """vLLM tu lo continuous batching nen KHONG can tu xep theo do dai."""
    name = "MATH" if kind == "math" else "GSM8K"
    off = MM.score_math if kind == "math" else MM.score_gsm8k
    getg = last_boxed if kind == "math" else extract_gold

    t0 = time.perf_counter()
    outs = llm.generate([PROMPT.format(q=q) for q, _ in tests], sp)
    el = time.perf_counter() - t0

    recs, hit = [], 0
    for (q, gold), o in zip(tests, outs):
        gen = o.outputs[0].text
        # score_gsm8k nhan ca chuoi dap an co "#### X"; score_math nhan
        # ca loi giai con nguyen boxed. Ca hai deu tu trich, truyen thang gold.
        ok = off(gen, gold)
        hit += bool(ok)
        recs.append({"q": q, "gen": gen, "gold": getg(gold),
                     "ok_mm": bool(ok)})

    acc = 100 * hit / max(len(tests), 1)
    os.makedirs(out_dir, exist_ok=True)
    suf = "_math_gen" if kind == "math" else "_gen"
    with open(f"{out_dir}/{tag}{suf}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nomark = sum(1 for r in recs if "The answer is: " not in r["gen"])
    print(f"\n  [{tag}]  {name} accuracy = {acc:.2f}%  ({hit}/{len(tests)}, {el:.0f}s)")
    print(f"           {nomark} cau thieu nhan 'The answer is: ' "
          f"({100*nomark/len(tests):.1f}%) — MetaMath tinh SAI het")
    return dict(acc=acc, hit=hit, n=len(tests), gen_s=round(el, 1),
                benchmark=name, no_marker=nomark)


def main():
    p = argparse.ArgumentParser(description="cham GSM8K/MATH bang vLLM")
    p.add_argument("--model", required=True, help="thu muc model da gop")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--eval-math", action="store_true")
    p.add_argument("--skip-gsm8k", action="store_true")
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-frac", type=float, default=0.90)
    p.add_argument("--max-len", type=int, default=1024,
                   help="prompt 512 + sinh 512; de sat giup vLLM xep nhieu seq hon")
    a = p.parse_args()
    a.tag = a.tag or os.path.basename(a.model.rstrip("/\\"))
    a.out_dir = a.out_dir or f"runs_vllm/{a.tag}"

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_frac,
              max_model_len=a.max_len, enforce_eager=False)
    # temperature 0 = greedy, giong do_sample=False cua HF va giong PiSSA.
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.max_new_tokens)

    res = {"model": a.model, "tag": a.tag}
    if not a.skip_gsm8k:
        t = load_gsm8k()
        if a.limit_eval:
            t = t[:a.limit_eval]
        res["gsm8k"] = run(llm, sp, t, "gsm8k", a.tag, a.out_dir)
    if a.eval_math:
        t = load_math()
        if a.limit_eval:
            t = t[:a.limit_eval]
        res["math"] = run(llm, sp, t, "math", a.tag, a.out_dir)

    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{a.tag}_vllm.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n  da ghi {a.out_dir}/{a.tag}_vllm.json")


if __name__ == "__main__":
    main()
