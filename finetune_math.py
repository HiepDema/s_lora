#!/usr/bin/env python
"""Fine-tune tren MetaMathQA, danh gia tren GSM8K. So sanh S-LoRA / LoRA / hybrid.

Khac E2E o cho chi so la ACCURACY khop dung dap an, khong phai BLEU. Sach hon
nhieu: khong con chuyen val loss va chi so di nguoc chieu nhau nhu da gap ba lan
o E2E, va o 7B thi GSM8K con xa tran.

Tai dung tu finetune_e2e.py: train loop, checkpoint/resume, lich lr, build_model.
Rieng phan du lieu va cham diem la cua rieng file nay.

    python finetune_math.py --method hybrid --rank 4 --rank-square 1 \\
      --model Qwen/Qwen2.5-7B --target-set qwen2_qkv --tf32 \\
      --max-train 100000 --epochs 2 --max-len 512

Luu y: con so cho paper nen cham lai bang bo eval chuan cua MetaMath; bo trich
dap an o day la ban tu implement, giong tinh trang BLEU ben E2E.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request

import torch

import finetune_e2e as E
from peft_generic import TARGETS

MATHQA = ("https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/"
          "MetaMathQA-395K.json")
GSM8K = "hf://datasets/openai/gsm8k/main/test-00000-of-00001.parquet"

# Dung dung prompt cua MetaMath. Doi prompt la doi ket qua, nen giu nguyen de
# so duoc voi so lieu cong bo cua cac paper khac.
PROMPT = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{q}\n\n### Response: Let's think step by step.")


# =========================================================================== du lieu

def load_mathqa(n, seed, only_gsm=False, cache_dir="math_cache"):
    """MetaMathQA 395K. Tai mot lan (~395 MB) roi lay mau."""
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, "MetaMathQA-395K.json")
    if not os.path.exists(p):
        print(f"    tai MetaMathQA (~395 MB) -> {p}...", flush=True)
        urllib.request.urlretrieve(MATHQA, p)
    with open(p, encoding="utf-8") as f:
        rows = json.load(f)
    if only_gsm:
        rows = [r for r in rows if r.get("type", "").startswith("GSM")]
    rng = random.Random(seed)          # rieng biet, khong dung RNG toan cuc cua train
    if n and n < len(rows):
        rows = rng.sample(rows, n)
    return rows


def load_gsm8k():
    """GSM8K test, 1319 bai. Dap an chuan nam sau '#### '."""
    import pandas as pd
    df = pd.read_parquet(GSM8K)
    return [(q, a) for q, a in zip(df["question"], df["answer"])]


def encode_train(rows, tok, max_len):
    """Loss CHI tinh tren phan loi giai, khong tinh tren de bai."""
    out = []
    for r in rows:
        p = tok(PROMPT.format(q=r["query"]), add_special_tokens=False).input_ids
        c = tok(" " + r["response"].strip(),
                add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids, lab = (p + c)[:max_len], ([-100] * len(p) + c)[:max_len]
        if len(lab) > len(p):          # bo vi du bi cat het phan loi giai
            out.append((ids, lab))
    return out


# =========================================================================== cham diem

_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _norm(s):
    """Chuan hoa mot chuoi so ve float. Tra ve None neu khong phai so."""
    s = s.replace(",", "").replace("$", "").rstrip(".").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_pred(text):
    """Lay dap an model sinh ra.

    MetaMath luon ket thuc bang 'The answer is: X' (dung 100% tren mau da kiem),
    nen uu tien mau do. Neu model khong theo dinh dang thi lay SO CUOI CUNG —
    khong phai so dau tien, vi loi giai nhieu buoc chua day so trung gian.
    """
    m = re.search(r"[Tt]he answer is:?\s*(.+?)(?:\n|$)", text)
    if m:
        v = _norm(m.group(1))
        if v is not None:
            return v
        nums = _NUM.findall(m.group(1))     # vd "\\boxed{72}"
        if nums:
            return _norm(nums[-1])
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


def extract_gold(ans):
    """Dap an chuan GSM8K nam sau '#### '."""
    m = re.search(r"####\s*(.+)", ans)
    return _norm(m.group(1)) if m else None


@torch.no_grad()
def evaluate(model, tok, tests, a, tag):
    """Sinh tham lam roi khop dung dap an. Tra ve accuracy."""
    was = model.training
    model.eval()
    tok.padding_side = "left"
    t0, hit, n = time.perf_counter(), 0, 0
    recs = []
    for i in range(0, len(tests), a.eval_batch):
        chunk = tests[i:i + a.eval_batch]
        enc = tok([PROMPT.format(q=q) for q, _ in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=a.eval_max_len,
                  add_special_tokens=False).to(a.device)
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                             do_sample=False, num_beams=1,
                             pad_token_id=tok.eos_token_id)
        for j, (q, gold) in enumerate(chunk):
            gen = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            p, g = extract_pred(gen), extract_gold(gold)
            ok = p is not None and g is not None and abs(p - g) < 1e-4
            hit += ok
            n += 1
            recs.append({"q": q, "gen": gen, "pred": p, "gold": g, "ok": bool(ok)})
        if i % (a.eval_batch * 10) == 0:
            print(f"      {n}/{len(tests)}  acc {100*hit/max(n,1):.2f}%  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    tok.padding_side = "right"
    model.train(was)
    acc = 100 * hit / max(n, 1)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}_gen.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    el = time.perf_counter() - t0
    print(f"\n    [{tag}]  GSM8K accuracy = {acc:.2f}%  ({hit}/{n}, {el:.0f}s)")
    nofmt = sum(1 for r in recs if r["pred"] is None)
    if nofmt:
        print(f"    canh bao: {nofmt} cau khong trich duoc so nao ({100*nofmt/n:.1f}%)")
    return dict(acc=acc, hit=hit, n=n, gen_s=round(el, 1), unparsed=nofmt)


# =========================================================================== main

def main():
    p = argparse.ArgumentParser(description="MetaMathQA -> GSM8K")
    p.add_argument("--method", required=True,
                   choices=["rowspace", "rowspace-full", "lora", "vera", "hybrid",
                            "target-ft", "full", "none"])
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--rank-square", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--alpha-square", type=float, default=None)
    p.add_argument("--basis", choices=["colperm", "svd"], default="colperm")
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--target-set", default="qwen2_qkv", choices=sorted(TARGETS))
    p.add_argument("--vera-d-init", type=float, default=0.1)
    p.add_argument("--match-params", action="store_true")

    p.add_argument("--max-train", type=int, default=100000,
                   help="so vi du lay tu MetaMathQA 395K (cac paper hay dung 100K)")
    p.add_argument("--only-gsm", action="store_true",
                   help="chi lay cac loai GSM_*, bo MATH_* (vi chi eval GSM8K)")
    p.add_argument("--val-frac", type=float, default=0.01,
                   help="ty le tach ra lam validation")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-schedule", choices=["linear", "constant", "cosine"],
                   default="linear")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="0 cho toan hoc: dap an la mot chuoi so xac dinh, lam mem "
                        "nhan chi them nhieu")
    p.add_argument("--max-len", type=int, default=512,
                   help="p95 cua MetaMathQA ~371 token; 512 phu thoai mai")

    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--eval-max-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit-eval", type=int, default=None,
                   help="gioi han so bai GSM8K (mac dinh chay het 1319)")
    p.add_argument("--eval-before", action="store_true", help="do zero-shot truoc")

    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-max", type=int, default=500)
    p.add_argument("--train-eval-max", type=int, default=500)
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--best-epoch", action="store_true")
    p.add_argument("--eval-both", action="store_true")
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--load-ckpt", default=None)
    p.add_argument("--no-save-ckpt", action="store_true")
    p.add_argument("--gen-every", type=int, default=0)
    p.add_argument("--gen-every-max", type=int, default=200)

    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--fp32-factorize", action="store_true")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--bench-batch", type=int, default=8)
    p.add_argument("--bench-seq", type=int, default=512)
    p.add_argument("--bench-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out-dir", default="runs_math")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if a.alpha is None:
        a.alpha = a.rank
    if a.rank_square is None:
        a.rank_square = a.rank
    if a.alpha_square is None:
        a.alpha_square = a.rank_square
    if a.eval_both and a.best_epoch:
        a.best_epoch = False
        print("  --eval-both tat --best-epoch")
    a.dtype = torch.float32 if a.fp32_factorize else torch.float64
    a.targets = a.targets or TARGETS[a.target_set]

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    _sq = f"q{a.rank_square}" if a.rank_square != a.rank else ""
    tag = (f"{a.method}_r{a.rank}{_sq}_"
           f"{a.basis if 'rowspace' in a.method else 'na'}_s{a.seed}")
    a.tag = tag
    print(f"\n=== {tag} ===\nmodel={a.model}  device={a.device}  targets={a.targets}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("  nap MetaMathQA...", flush=True)
    rows = load_mathqa(a.max_train, a.seed, a.only_gsm)
    nval = max(1, int(len(rows) * a.val_frac))
    val_rows, train_rows = rows[:nval], rows[nval:]
    data = encode_train(train_rows, tok, a.max_len)
    val_data = None if a.no_val else encode_train(val_rows, tok, a.max_len)
    tests = load_gsm8k()
    if a.limit_eval:
        tests = tests[:a.limit_eval]
    print(f"  train {len(data)} | val {len(val_data) if val_data else 0} "
          f"| GSM8K test {len(tests)}")

    t0 = time.perf_counter()
    model, info = E.build_model(a)
    print(f"\n  TRAIN DUOC: {info['trainable']/1e6:.3f} M "
          f"({100*info['trainable']/info['total']:.4f} % cua {info['total']/1e6:.2f} M)"
          f"   [{time.perf_counter()-t0:.1f}s]")

    res = dict(tag=tag, config=vars(a) | {"device": str(a.device), "dtype": str(a.dtype)},
               params=info, setup_s=round(time.perf_counter() - t0, 1))
    if a.eval_before:
        res["before"] = evaluate(model, tok, tests, a, tag + "_zeroshot")
    if a.load_ckpt:
        res["loaded"] = E.load_trainable_ckpt(a.load_ckpt, model, a.device)
    elif a.method != "none":
        res["train"] = E.train(model, data, val_data, tok, a)     # pairs=None
        res["final_loss"] = res["train"]["final_loss"]
    res["after"] = evaluate(model, tok, tests, a, tag)

    tr = res.get("train") or {}
    if a.eval_both and tr.get("ckpt_best"):
        be, last = tr.get("best_epoch"), tr.get("epochs_run", 0) - 1
        if be is not None and be != last:
            print(f"\n  === cham lai o epoch tot nhat ({be}) ===")
            E.load_trainable_ckpt(tr["ckpt_best"], model, a.device)
            res["after_best"] = evaluate(model, tok, tests, a, f"{tag}_bestep")
            res["after_best"]["epoch"] = be
            print(f"    accuracy lech {res['after_best']['acc']-res['after']['acc']:+.2f} "
                  f"diem so voi epoch cuoi")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    with open(f"{a.out_dir}/summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(
            tag=tag, method=a.method, rank=a.rank, rank_square=a.rank_square,
            alpha=a.alpha, basis=a.basis if "rowspace" in a.method else "na",
            seed=a.seed, trainable=info["trainable"], model=a.model,
            targets=",".join(a.targets), dataset="metamathqa->gsm8k",
            max_train=a.max_train, only_gsm=a.only_gsm, tf32=bool(a.tf32),
            lr=a.lr, lr_schedule=a.lr_schedule, epochs_planned=a.epochs,
            setup_s=res["setup_s"], **res["after"], **res.get("train", {}),
            **({"after_best": res["after_best"]} if "after_best" in res else {}),
        ), default=str) + "\n")
    print(f"\n  ket qua -> {a.out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
