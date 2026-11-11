#!/usr/bin/env python
"""
Fine-tune + cham DROP (doc hieu, suy luan roi rac tren doan van) theo setup cua
QuanTA (Chen et al., 2024) ma PMSS (phu luc A.1) dung lai:

  train 2000 mau + val 800 mau lay tu tap train DROP, test 1200 mau lay tu tap dev;
  3 epoch; chon checkpoint tot nhat tren val; F1 cua DROP.

Du lieu la ban GOC cua AllenNLP (drop_dataset.zip), khong phai ucinlp/drop tren HF:
ban HF lam phang dap an, tron dap an goc voi validated_answers vao mot list spans,
nen khong con phan biet dap an nhieu span voi cac dap an thay the — F1 se sai.
Cham bang dung thuat toan cua drop_eval.py chinh thuc: chuan hoa so, bo mao tu,
can bang tui tu giua cac span (linear_sum_assignment), lay max tren dap an goc
va moi validated answer.

Dung lai build_model / train / bench_model cua finetune_commonsense.py: base bf16,
adapter fp32 duoi autocast, val loss cuoi moi epoch, --best-epoch khoi phuc epoch
co val loss thap nhat.

  python finetune_drop.py --method lora --rank 32 --alpha 64 --bf16 --fp32-adapters
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import string
import time
import urllib.request
import zipfile

import numpy as np
import torch

import finetune_commonsense as CS
from peft_generic import TARGETS, merge_all

DROP_URL = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip"
SEP = "; "      # noi cac span cua dap an nhieu span; hiem khi nam trong mot span


# =========================================================================== du lieu

def load_drop(cache_dir="drop_cache"):
    """(train, dev): list cau hoi, moi cau giu passage, question, answer, golds."""
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, "drop_dataset.zip")
    if not os.path.exists(p):
        print(f"    tai DROP (~8 MB) -> {p}", flush=True)
        urllib.request.urlretrieve(DROP_URL, p)
    z = zipfile.ZipFile(p)

    def flat(name):
        d = json.load(io.TextIOWrapper(z.open(f"drop_dataset/drop_dataset_{name}.json"),
                                       encoding="utf-8"))
        return [dict(qid=q["query_id"], passage=v["passage"], question=q["question"],
                     answer=q["answer"],
                     golds=[q["answer"]] + q.get("validated_answers", []))
                for v in d.values() for q in v["qa_pairs"]]
    return flat("train"), flat("dev")


def split_drop(train, dev, n_train, n_val, n_test, seed):
    """Train va val roi nhau, cung lay tu tap train; test lay tu tap dev.
    n_train <= 0: moi cau con lai sau val; n_test <= 0: ca tap dev."""
    n_train = len(train) - n_val if n_train <= 0 else n_train
    n_test = len(dev) if n_test <= 0 else n_test
    tv = random.Random(seed).sample(train, n_train + n_val)
    return tv[:n_train], tv[n_train:], random.Random(seed).sample(dev, n_test)


def answer_strings(ans):
    """Nhu answer_json_to_strings cua drop_eval.py: so, cac span, hoac ngay (d m y)."""
    if ans.get("number"):
        return [str(ans["number"])]
    if ans.get("spans"):
        return list(ans["spans"])
    d = ans.get("date") or {}
    return [f"{d.get('day', '')} {d.get('month', '')} {d.get('year', '')}".strip()]


def prompt_of(r):
    return f"Passage: {r['passage']}\nQuestion: {r['question']}\nAnswer:"


def encode(rows, tok, max_len):
    """Prompt (co BOS) + ' dap an' + EOS; loss chi tren dap an."""
    out, cut = [], 0
    for r in rows:
        p = tok(prompt_of(r)).input_ids                      # add_special_tokens -> BOS
        c = tok(" " + SEP.join(answer_strings(r["answer"])),
                add_special_tokens=False).input_ids + [tok.eos_token_id]
        if len(p) + len(c) > max_len:
            cut += 1
            p = p[:1] + p[1 + len(p) + len(c) - max_len:]   # bo dau doan van, giu BOS
        out.append((p + c, [-100] * len(p) + c))
    if cut:
        print(f"    {cut} mau dai hon {max_len} token, da cat dau doan van", flush=True)
    return out


# =========================================================================== F1 cua DROP

PUNCT = set(string.punctuation)


def _is_number(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def _normalize(text):
    out = []
    for tok in re.split(" |-", text):
        tok = tok.lower()
        if not _is_number(tok):
            tok = "".join(ch for ch in tok if ch not in PUNCT)
        if _is_number(tok):
            tok = str(float(tok))
        tok = " ".join(re.sub(r"\b(a|an|the)\b", " ", tok).split())
        if tok.strip():
            out.append(tok)
    return " ".join(out).strip()


def _bags(spans):
    norm = [_normalize(s) for s in spans]
    return norm, [set(n.split()) for n in norm]


def _f1(pred, gold):
    inter = len(gold & pred)
    prec = 1.0 if not pred else inter / len(pred)
    rec = 1.0 if not gold else inter / len(gold)
    return 0.0 if prec == 0 and rec == 0 else 2 * prec * rec / (prec + rec)


def _numbers_match(gold, pred):
    g = {w for w in gold if _is_number(w)}
    return not g or bool(g & {w for w in pred if _is_number(w)})


def drop_metrics(pred_spans, gold_spans):
    """(EM, F1) cua mot du doan voi mot dap an vang, nhu get_metrics() chinh thuc."""
    from scipy.optimize import linear_sum_assignment
    pn, pb = _bags(pred_spans)
    gn, gb = _bags(gold_spans)
    em = float(set(pn) == set(gn) and len(pn) == len(gn))
    sc = np.zeros((len(gb), len(pb)))
    for i, g in enumerate(gb):
        for j, p in enumerate(pb):
            if _numbers_match(g, p):
                sc[i, j] = _f1(p, g)
    rows, cols = linear_sum_assignment(-sc)
    best = np.zeros(max(len(gb), len(pb)))
    for r, c in zip(rows, cols):
        best[r] = max(best[r], sc[r, c])
    return em, round(float(np.mean(best)), 2)


def score(pred_text, golds):
    """Max EM / F1 tren dap an goc va moi validated answer (bo dap an rong, nhu
    evaluate_json() chinh thuc)."""
    spans = [s.strip() for s in pred_text.split(SEP.strip())]
    ms = [drop_metrics(spans, gs) for gs in map(answer_strings, golds) if gs[0].strip()]
    return (max(m[0] for m in ms), max(m[1] for m in ms)) if ms else (0.0, 0.0)


# =========================================================================== cham

@torch.no_grad()
def predict(model, tok, rows, a):
    """Greedy; tu pad trai bang tay de khong phu thuoc padding_side cua tokenizer."""
    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids_all = [tok(prompt_of(r)).input_ids for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(ids_all[i]))
    outs = [None] * len(rows)
    for s in range(0, len(order), a.eval_batch):
        idx = order[s:s + a.eval_batch]
        n = max(len(ids_all[j]) for j in idx)
        ids = torch.full((len(idx), n), pad, dtype=torch.long)
        att = torch.zeros((len(idx), n), dtype=torch.long)
        for k, j in enumerate(idx):
            ids[k, n - len(ids_all[j]):] = torch.tensor(ids_all[j])
            att[k, n - len(ids_all[j]):] = 1
        with CS.amp(a):
            out = model.generate(input_ids=ids.to(a.device), attention_mask=att.to(a.device),
                                 max_new_tokens=a.max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=pad,
                                 eos_token_id=tok.eos_token_id)
        for k, j in enumerate(idx):
            outs[j] = tok.decode(out[k, n:], skip_special_tokens=True).split("\n")[0].strip()
    return outs


def evaluate(model, tok, rows, a, tag):
    t0 = time.perf_counter()
    preds = predict(model, tok, rows, a)
    ems, f1s = zip(*(score(p, r["golds"]) for p, r in zip(preds, rows)))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}_preds.jsonl", "w", encoding="utf-8") as f:
        for p, r, e, s in zip(preds, rows, ems, f1s):
            f.write(json.dumps(dict(qid=r["qid"], question=r["question"], pred=p,
                                    gold=[answer_strings(g) for g in r["golds"]],
                                    em=e, f1=s), ensure_ascii=False) + "\n")
    res = dict(f1=100 * float(np.mean(f1s)), em=100 * float(np.mean(ems)), n=len(rows),
               eval_s=round(time.perf_counter() - t0, 1))
    print(f"\n    [{tag}]  F1 = {res['f1']:.2f}   EM = {res['em']:.2f}   "
          f"({len(rows)} cau, {res['eval_s']:.0f}s)", flush=True)
    return res


# =========================================================================== main

def main():
    p = argparse.ArgumentParser(description="DROP: S-LoRA vs LoRA/DoRA/PiSSA")
    p.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B")
    p.add_argument("--method", required=True,
                   choices=["rowspace", "lora", "hybrid", "dora", "pissa", "none"])
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--rank-square", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--alpha-square", type=float, default=None)
    p.add_argument("--basis", choices=["colperm", "svd"], default="colperm")
    p.add_argument("--slora-norm", choices=["rms"], default=None)
    p.add_argument("--slora-impl", choices=["factored", "lowrank"], default="factored")
    p.add_argument("--target-set", default="qv", choices=sorted(TARGETS))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2,
                   help="PMSS/QuanTA: batch 4 tren 2 GPU = 8")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--n-train", type=int, default=2000, help="0 = moi cau con lai sau val")
    p.add_argument("--n-val", type=int, default=800)
    p.add_argument("--n-test", type=int, default=1200, help="0 = ca tap dev (9536 cau)")
    p.add_argument("--data-seed", type=int, default=42,
                   help="seed lay mau train/val/test, chung cho moi method")
    p.add_argument("--max-len", type=int, default=1024, help="prompt dai nhat ~720 token")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--best-epoch", action="store_true",
                   help="cham test o epoch co val loss thap nhat (PMSS: best checkpoint)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp32-adapters", action="store_true")
    p.add_argument("--fac-cache", default=None)
    p.add_argument("--fp32-factorize", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--bench-batch", type=int, default=8)
    p.add_argument("--bench-seq", type=int, default=128)
    p.add_argument("--bench-iters", type=int, default=20)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--no-save-ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out-dir", default="runs_drop")
    p.add_argument("--data-cache", default="drop_cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    a.dtype = torch.float32 if a.fp32_factorize else torch.float64
    a.targets = TARGETS[a.target_set]
    a.amp = a.bf16 and a.fp32_adapters
    a.alpha = a.rank if a.alpha is None else a.alpha
    # train() cua commonsense doc cac truong nay; DROP khong cham giua chung.
    a.eval_every, a.eval_limit_mid, a.tests = 0, 0, {}
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    if a.method == "hybrid":
        rs = a.rank_square or a.rank
        tag = (f"drop_hybrid_r{a.rank}q{rs}_{a.basis}{'_' + a.slora_norm if a.slora_norm else ''}"
               f"{'_lowrank' if a.slora_impl == 'lowrank' else ''}_s{a.seed}")
    else:
        tag = f"drop_{a.method}_r{a.rank}_na_s{a.seed}"
    a.tag = tag = tag + f"_lr{a.lr:g}"
    print(f"\n=== {tag} ===\nmodel={a.model}  targets={a.targets}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_all, dev_all = load_drop(a.data_cache)
    tr, va, te = split_drop(tr_all, dev_all, a.n_train, a.n_val, a.n_test, a.data_seed)
    data, val_data = encode(tr, tok, a.max_len), encode(va, tok, a.max_len)
    print(f"  train {len(data)} | val {len(val_data)} | test {len(te)}  "
          f"(tu {len(tr_all)} train / {len(dev_all)} dev, seed {a.data_seed})")

    t0 = time.perf_counter()
    model, info = CS.build_model(a)
    setup_s = round(time.perf_counter() - t0, 1)
    print(f"\n  TRAIN DUOC: {info['trainable'] / 1e6:.3f} M   [setup {setup_s}s]")

    res = dict(tag=tag, config={k: v for k, v in vars(a).items() if k != "tests"} |
               {"dtype": str(a.dtype)}, params=info, setup_s=setup_s)
    if a.method != "none":
        res["train"] = CS.train(model, data, tok, a, val_data)
    ep = (res.get("train") or {}).get("restored_epoch")
    print(f"\n  cham test ({'epoch cuoi' if ep is None else f'epoch {ep}'}):")
    res["test"] = evaluate(model, tok, te, a, tag)

    if not a.no_bench:
        res["bench"] = CS.bench_model(model, a)
        if a.method != "none":
            merge_all(model, a.targets)
            res["bench"].update(CS.bench_model(model, a, prefix="merged_"))

    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    with open(f"{a.out_dir}/summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(
            tag=tag, method=a.method, rank=a.rank, alpha=a.alpha,
            rank_square=a.rank_square, alpha_square=a.alpha_square, lr=a.lr,
            slora_norm=a.slora_norm, slora_impl=a.slora_impl, seed=a.seed,
            trainable=info["trainable"], model=a.model, targets=",".join(a.targets),
            n_train=len(data), n_val=len(val_data), n_test=len(te), epochs=a.epochs,
            best_epoch_sel=a.best_epoch, setup_s=setup_s, **res["test"],
            **res.get("train", {}), **res.get("bench", {})), default=str) + "\n")
    print(f"\n  ket qua -> {a.out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
