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
import metamath_eval as MM
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


# --------------------------------------------------------------------- MATH

MATH_SUBS = ["algebra", "counting_and_probability", "geometry",
             "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def load_math():
    """MATH test day du, 5000 bai qua 7 chu de.

    Vi sao khong dung MATH-500: n=500 cho sd ~2.0% o p=0.3, con TE HON GSM8K
    (1319 bai, sd 1.1%). Ca bo 5000 moi xuong 0.65%.
    """
    from datasets import load_dataset
    out = []
    for s in MATH_SUBS:
        d = load_dataset("EleutherAI/hendrycks_math", s, split="test")
        out += [(r["problem"], r["solution"]) for r in d]
    return out


def last_boxed(s):
    """Noi dung \\boxed{...} cuoi cung, ghep cap ngoac dung."""
    i = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def norm_math(s):
    """Chuan hoa bieu thuc LaTeX truoc khi so khop.

    Theo thong le cua cac bo eval MATH (Hendrycks, Minerva): bo don vi va
    \\text, gop \\dfrac/\\tfrac ve \\frac, bo khoang trang va ky hieu trang tri.
    Khong hoan hao — vd khong nhan ra (x+1)^2 = x^2+2x+1 — nhung la chuan chung.
    """
    if s is None:
        return None
    s = s.strip()
    # \text{...} thuong la don vi do ("2\text{ cm}") nen phai XOA. Nhung doi khi
    # chinh no la dap an ("\text{even}") — luc do xoa se thanh rong, giu lai noi
    # dung thay vi xoa.
    stripped = re.sub(r"\\(?:text|mbox|textbf|mathrm)\{[^{}]*\}", "", s).strip()
    s = stripped if stripped else re.sub(
        r"\\(?:text|mbox|textbf|mathrm)\{([^{}]*)\}", r"\1", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""),
                 ("\\;", ""), ("\\:", ""), ("\\ ", ""), ("\\$", ""), ("$", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("^{\\circ}", ""), ("^\\circ", ""), ("\\%", ""), ("%", ""),
                 ("\\cdot", "*"), (" ", ""), (",", "")):
        s = s.replace(a, b)
    s = s.rstrip(".")
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)         # a/b -> \frac{a}{b}
    if m:
        s = "\\frac{%s}{%s}" % (m.group(1), m.group(2))
    if s.startswith("."):
        s = "0" + s
    return s


def _to_float(s):
    """Doi bieu thuc don gian ve so: nguyen, thap phan, hoac \\frac{a}{b}."""
    if s is None:
        return None
    v = _norm(s)
    if v is not None:
        return v
    m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m:
        try:
            den = float(m.group(2))
            return float(m.group(1)) / den if den else None
        except ValueError:
            return None
    return None


def math_equal(pred, gold):
    """Khop chuoi sau chuan hoa, hoac khop so neu ca hai quy duoc ve so."""
    p, g = norm_math(pred), norm_math(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    fp, fg = _to_float(p), _to_float(g)
    return fp is not None and fg is not None and abs(fp - fg) < 1e-6


def extract_pred_math(text):
    """Dap an model sinh ra cho bai MATH: uu tien 'The answer is:', roi \\boxed."""
    m = re.search(r"[Tt]he answer is:?\s*(.+?)(?:\n|$)", text)
    if m:
        v = m.group(1).strip().rstrip(".")
        b = last_boxed(v)
        return b if b is not None else v
    return last_boxed(text)


@torch.no_grad()
def evaluate(model, tok, tests, a, tag, kind="gsm8k"):
    """Sinh tham lam roi khop dung dap an. Tra ve accuracy.

    kind='gsm8k': dap an la so, so sanh bang float.
    kind='math' : dap an la bieu thuc LaTeX, so sanh sau chuan hoa.
    """
    is_math = kind == "math"
    getp = extract_pred_math if is_math else extract_pred
    getg = last_boxed if is_math else extract_gold
    same = math_equal if is_math else (
        lambda p, g: p is not None and g is not None and abs(p - g) < 1e-4)
    # Diem CHINH THUC: bo cham cua MetaMath, dung thu PiSSA/PMSS dung. Diem tu
    # viet giu lai lam doi chieu de biet hai ben lech bao nhieu.
    off = MM.score_math if is_math else MM.score_gsm8k
    was = model.training
    model.eval()
    tok.padding_side = "left"
    t0, hit, n, hit_mm = time.perf_counter(), 0, 0, 0
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
            p, g = getp(gen), getg(gold)
            ok = same(p, g)
            ok_mm = off(gen, g)
            hit += ok
            hit_mm += ok_mm
            n += 1
            recs.append({"q": q, "gen": gen, "pred": p, "gold": g,
                         "ok": bool(ok), "ok_mm": bool(ok_mm)})
        if i % (a.eval_batch * 10) == 0:
            print(f"      {n}/{len(tests)}  MetaMath {100*hit_mm/max(n,1):.2f}%  "
                  f"(tu viet {100*hit/max(n,1):.2f}%)  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    tok.padding_side = "right"
    model.train(was)
    acc = 100 * hit / max(n, 1)
    os.makedirs(a.out_dir, exist_ok=True)
    name = "MATH" if is_math else "GSM8K"
    with open(f"{a.out_dir}/{tag}_gen.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    el = time.perf_counter() - t0
    acc_mm = 100 * hit_mm / max(n, 1)
    print(f"\n    [{tag}]  {name} accuracy = {acc_mm:.2f}%  ({hit_mm}/{n}, {el:.0f}s)"
          f"   <- MetaMath is_equiv, dung de so voi PiSSA/PMSS")
    print(f"    {' ' * len(tag)}    (ham tu viet: {acc:.2f}%, lech {acc - acc_mm:+.2f})")
    nofmt = sum(1 for r in recs if r["pred"] is None)
    if nofmt:
        print(f"    canh bao: {nofmt} cau khong trich duoc so nao ({100*nofmt/n:.1f}%)")
    nomark = sum(1 for r in recs if "The answer is: " not in r["gen"])
    if nomark:
        print(f"    {nomark} cau thieu nhan 'The answer is: ' ({100*nomark/n:.1f}%) "
              f"— MetaMath tinh SAI het, khong co duong lui")
    return dict(acc=acc_mm, hit=hit_mm, n=n, gen_s=round(el, 1), unparsed=nofmt,
                benchmark=name, acc_own=acc, hit_own=hit, no_marker=nomark)


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
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation; batch hieu dung = --batch * --accum")
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
    p.add_argument("--eval-math", action="store_true",
                   help="cham them tren MATH test day du (5000 bai). Nhieu thap hon "
                        "GSM8K (sd 0.65%% so voi 1.10%%) va du dia rong hon nhieu.")
    p.add_argument("--skip-gsm8k", action="store_true",
                   help="bo qua GSM8K, chi cham MATH")
    p.add_argument("--limit-math", type=int, default=None)

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
    p.add_argument("--fac-cache", default=None)
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
    tests = [] if a.skip_gsm8k else load_gsm8k()
    if a.limit_eval:
        tests = tests[:a.limit_eval]
    mtests = []
    if a.eval_math:
        print("  nap MATH test...", flush=True)
        mtests = load_math()
        if a.limit_math:
            mtests = mtests[:a.limit_math]
    print(f"  train {len(data)} | val {len(val_data) if val_data else 0} "
          f"| GSM8K {len(tests)} | MATH {len(mtests)}")

    t0 = time.perf_counter()
    model, info = E.build_model(a)
    print(f"\n  TRAIN DUOC: {info['trainable']/1e6:.3f} M "
          f"({100*info['trainable']/info['total']:.4f} % cua {info['total']/1e6:.2f} M)"
          f"   [{time.perf_counter()-t0:.1f}s]")

    res = dict(tag=tag, config=vars(a) | {"device": str(a.device), "dtype": str(a.dtype)},
               params=info, setup_s=round(time.perf_counter() - t0, 1))
    if a.eval_before:
        if tests:
            res["before"] = evaluate(model, tok, tests, a, tag + "_zeroshot")
        if mtests:
            res["before_math"] = evaluate(model, tok, mtests, a,
                                          tag + "_zeroshot_math", kind="math")
    if a.load_ckpt:
        res["loaded"] = E.load_trainable_ckpt(a.load_ckpt, model, a.device)
    elif a.method != "none":
        res["train"] = E.train(model, data, val_data, tok, a)     # pairs=None
        res["final_loss"] = res["train"]["final_loss"]
    if tests:
        res["after"] = evaluate(model, tok, tests, a, tag)
    if mtests:
        res["after_math"] = evaluate(model, tok, mtests, a, tag + "_math", kind="math")

    tr = res.get("train") or {}
    if a.eval_both and tr.get("ckpt_best"):
        be, last = tr.get("best_epoch"), tr.get("epochs_run", 0) - 1
        if be is not None and be != last:
            print(f"\n  === cham lai o epoch tot nhat ({be}) ===")
            E.load_trainable_ckpt(tr["ckpt_best"], model, a.device)
            if tests:
                res["after_best"] = evaluate(model, tok, tests, a, f"{tag}_bestep")
                res["after_best"]["epoch"] = be
                print(f"    GSM8K lech "
                      f"{res['after_best']['acc']-res['after']['acc']:+.2f} diem")
            if mtests:
                res["after_best_math"] = evaluate(model, tok, mtests, a,
                                                  f"{tag}_bestep_math", kind="math")
                res["after_best_math"]["epoch"] = be

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
            setup_s=res["setup_s"], **res.get("after", {}), **res.get("train", {}),
            **({"after_math": res["after_math"]} if "after_math" in res else {}),
            **({"before": res["before"]} if "before" in res else {}),
            **({"before_math": res["before_math"]} if "before_math" in res else {}),
            **({"after_best": res["after_best"]} if "after_best" in res else {}),
        ), default=str) + "\n")
    print(f"\n  ket qua -> {a.out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
