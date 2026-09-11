#!/usr/bin/env python
"""
Fine-tune + danh gia tren Commonsense reasoning (8 tac vu), song song voi finetune_e2e.py.

Day la benchmark chuan cua mang LoRA-variant: DoRA, PiSSA, PMSS, PiCa deu bao cao bo nay,
nen so lieu dat canh so cua ho duoc.

  train: zwhe99/commonsense_170k  (lay mau --max-train)
  test : 8 tac vu tu LLM-Adapters — BoolQ, PIQA, SIQA, HellaSwag, WinoGrande,
         ARC-e, ARC-c, OBQA. Cham bang ACCURACY (khop chuoi dap an), khong phai BLEU.

Dung chung peft_generic.py voi finetune_e2e.py — phan ra va RowSpaceLinear khong doi gi.
Ten module cua Llama giong Qwen (self_attn.k_proj ...) nen target-set dung lai duoc.

  python finetune_commonsense.py --method rowspace --rank 4
  python finetune_commonsense.py --method lora     --rank 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.request

import torch
import torch.nn as nn

from peft_generic import TARGETS, apply_lora, convert_rowspace, merge_all, param_counts

TRAIN_REPO = "zwhe99/commonsense_170k"
GH = ("https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/dataset/"
      "{task}/test.json")
TASKS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
         "ARC-Easy", "ARC-Challenge", "openbookqa"]
SHORT = {"social_i_qa": "SIQA", "ARC-Easy": "ARC-e", "ARC-Challenge": "ARC-c",
         "openbookqa": "OBQA", "hellaswag": "HellaSwag", "winogrande": "WinoGrande",
         "boolq": "BoolQ", "piqa": "PIQA"}

# Prompt cua LLM-Adapters. Giu NGUYEN VAN de so sanh duoc voi DoRA/PMSS/PiCa.
PROMPT = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Response:\n")


# =========================================================================== du lieu

def load_train(n, seed, cache_dir=None):
    from datasets import load_dataset
    ds = load_dataset(TRAIN_REPO, split="train", cache_dir=cache_dir)
    if n and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def load_tests(cache_dir):
    """Tai 8 tap test tu GitHub, cache lai de khong tai lai moi lan chay."""
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for t in TASKS:
        p = os.path.join(cache_dir, f"{t}.json")
        if not os.path.exists(p):
            with urllib.request.urlopen(GH.format(task=t), timeout=120) as r:
                open(p, "wb").write(r.read())
        out[t] = json.load(open(p, encoding="utf-8"))
    return out


def encode_train(ds, tok, max_len):
    """Loss CHI tinh tren phan response, phan prompt bi mask -100."""
    out = []
    for r in ds:
        instr = r["instruction"] + (("\n" + r["input"]) if r.get("input") else "")
        p = tok(PROMPT.format(instruction=instr), add_special_tokens=False).input_ids
        c = tok(r["output"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p + c)[:max_len]
        lab = ([-100] * len(p) + c)[:max_len]
        if len(lab) > len(p):
            out.append((ids, lab))
    return out


def collate(batch, pad_id):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for j, (i, l) in enumerate(batch):
        ids[j, :len(i)] = torch.tensor(i)
        lab[j, :len(l)] = torch.tensor(l)
        att[j, :len(i)] = 1
    return ids, lab, att


def make_batches(data, batch):
    """Gom theo do dai de gan nhu khong con padding; van giu ngau nhien."""
    idx = list(range(len(data)))
    random.shuffle(idx)
    mega, out = batch * 50, []
    for i in range(0, len(idx), mega):
        chunk = sorted(idx[i:i + mega], key=lambda j: len(data[j][0]))
        out += [chunk[k:k + batch] for k in range(0, len(chunk), batch)]
    random.shuffle(out)
    return out


# =========================================================================== cham diem

def extract(text, task):
    """Rut dap an tu chuoi sinh ra. Theo dung logic cua LLM-Adapters."""
    t = text.strip().lower()
    if task == "boolq":
        opts = ["true", "false"]
    elif task == "piqa":
        opts = ["solution1", "solution2"]
    elif task in ("social_i_qa", "ARC-Easy", "ARC-Challenge", "openbookqa"):
        opts = ["answer1", "answer2", "answer3", "answer4", "answer5"]
    elif task == "hellaswag":
        opts = ["ending1", "ending2", "ending3", "ending4"]
    elif task == "winogrande":
        opts = ["option1", "option2"]
    else:
        opts = []
    # tim lua chon XUAT HIEN SOM NHAT — model hay noi "the correct answer is X"
    hits = [(t.find(o), o) for o in opts if o in t]
    return min(hits)[1] if hits else ""


@torch.no_grad()
def evaluate(model, tok, tests, a):
    model.eval()
    tok.padding_side = "left"
    res, t0 = {}, time.perf_counter()
    for task, rows in tests.items():
        if a.limit_eval:
            rows = rows[:a.limit_eval]
        ok = 0
        for i in range(0, len(rows), a.eval_batch):
            ch = rows[i:i + a.eval_batch]
            prompts = [PROMPT.format(instruction=r["instruction"]) for r in ch]
            # Cat tu BEN TRAI: phan cuoi prompt chua "Answer format: ..." va
            # "### Response:" — cat tu phai se xoa mat chung va model khong biet tra loi gi.
            # 60% prompt HellaSwag dai hon 256 token, nen mac dinh eval_max_len = 640.
            tok.truncation_side = "left"
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=a.eval_max_len, add_special_tokens=False).to(a.device)
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, num_beams=1,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            for j, r in enumerate(ch):
                gen = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
                ok += extract(gen, task) == str(r["answer"]).strip().lower()
        res[task] = 100.0 * ok / max(len(rows), 1)
        print(f"      {SHORT[task]:<11s} {res[task]:6.2f}%  ({len(rows)} cau, "
              f"{time.perf_counter() - t0:.0f}s)", flush=True)
    tok.padding_side = "right"
    res["avg"] = sum(res[t] for t in tests) / len(tests)
    res["eval_s"] = round(time.perf_counter() - t0, 1)
    return res


# =========================================================================== do luong

def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def timeit_cuda(fn, dev, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    _sync(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def bench_model(model, a, prefix=""):
    ids = torch.randint(0, model.config.vocab_size,
                        (a.bench_batch, a.bench_seq), device=a.device)
    ntok, out, was = a.bench_batch * a.bench_seq, {}, model.training
    model.eval()
    with torch.no_grad():
        ms = timeit_cuda(lambda: model(ids), a.device, a.bench_iters)
    out[prefix + "fwd_ms"] = round(ms, 3)
    out[prefix + "fwd_ktok_s"] = round(ntok / ms, 1)
    tp = [p for p in model.parameters() if p.requires_grad]
    if tp:
        model.train()

        def step():
            model(ids, labels=ids).loss.backward()
            for p in tp:
                p.grad = None

        ms = timeit_cuda(step, a.device, max(5, a.bench_iters // 2), warmup=3)
        out[prefix + "fwdbwd_ms"] = round(ms, 3)
    model.train(was)
    return out


# =========================================================================== train

def train(model, data, tok, a):
    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=a.lr, weight_decay=a.weight_decay)
    steps = math.ceil(len(data) / a.batch) * a.epochs
    warm = min(a.warmup, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm
        else max(0.0, (steps - s) / max(steps - warm, 1)))
    lossf = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=a.label_smoothing)

    print(f"\n  train: {len(data)} vi du, {steps} buoc, lr={a.lr}, batch={a.batch}")
    if str(a.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    step, t0, run = 0, time.perf_counter(), None
    for ep in range(a.epochs):
        for bidx in make_batches(data, a.batch):
            ids, lab, att = collate([data[j] for j in bidx], tok.eos_token_id)
            ids, lab, att = ids.to(a.device), lab.to(a.device), att.to(a.device)
            lg = model(input_ids=ids, attention_mask=att).logits
            loss = lossf(lg[:, :-1].reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tp, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            step += 1
            if step % a.log_every == 0:
                el = time.perf_counter() - t0
                print(f"    ep {ep} step {step}/{steps}  loss {run:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {step / el:.2f} it/s  "
                      f"ETA {(steps - step) / max(step / el, 1e-9) / 60:.0f}m", flush=True)
    el = time.perf_counter() - t0
    vram = (torch.cuda.max_memory_allocated() / 2**30
            if str(a.device).startswith("cuda") else 0.0)
    if not a.no_save_ckpt:
        os.makedirs(a.out_dir, exist_ok=True)
        p = f"{a.out_dir}/{a.tag}_last.pt"
        torch.save({"trainable": {n: v.detach().cpu() for n, v in model.named_parameters()
                                  if v.requires_grad},
                    "meta": dict(model=a.model, method=a.method, rank=a.rank, alpha=a.alpha,
                                 basis=a.basis, targets=a.targets, seed=a.seed, which="last")}, p)
        print(f"    luu {p}  ({os.path.getsize(p) / 2**20:.1f} MB)")
    return dict(final_loss=run, train_s=round(el, 1), train_it_s=round(step / el, 3),
                steps_run=step, peak_vram_gb=round(vram, 2))


# =========================================================================== main

def build_model(a):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    n0 = sum(p.numel() for p in m.parameters())
    if a.method == "rowspace":
        print(f"  phan ra {a.targets} (basis={a.basis})...", flush=True)
        convert_rowspace(m, a.targets, a.basis, a.rank, a.alpha, a.dtype, verbose=True)
    elif a.method == "lora":
        apply_lora(m, a.targets, a.rank, a.alpha)
    elif a.method == "none":
        for p in m.parameters():
            p.requires_grad_(False)
    return m.to(a.device), param_counts(m, n0)


def main():
    p = argparse.ArgumentParser(description="Commonsense reasoning: rowspace vs lora")
    p.add_argument("--model", default="unsloth/Llama-3.2-1B")
    p.add_argument("--method", required=True, choices=["rowspace", "lora", "none"])
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--basis", choices=["colperm", "svd"], default="colperm")
    p.add_argument("--target-set", default="qwen2_kv", choices=sorted(TARGETS))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--max-len", type=int, default=256, help="do dai khi TRAIN")
    p.add_argument("--eval-max-len", type=int, default=640,
                   help="do dai khi EVAL — phai du cho HellaSwag (max 410 token)")
    p.add_argument("--load-ckpt", default=None,
                   help="nap lai tham so train duoc va CHI danh gia, khong train")
    p.add_argument("--max-train", type=int, default=40000)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--bench-batch", type=int, default=8)
    p.add_argument("--bench-seq", type=int, default=128)
    p.add_argument("--bench-iters", type=int, default=20)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--no-save-ckpt", action="store_true")
    p.add_argument("--fp32-factorize", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out-dir", default="runs_cs")
    p.add_argument("--test-cache", default="cs_tests")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    a.dtype = torch.float32 if a.fp32_factorize else torch.float64
    a.targets = TARGETS[a.target_set]
    if a.alpha is None:
        a.alpha = a.rank
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    tag = f"cs_{a.method}_r{a.rank}_{a.basis if a.method == 'rowspace' else 'na'}_s{a.seed}"
    a.tag = tag
    print(f"\n=== {tag} ===\nmodel={a.model}  device={a.device}  targets={a.targets}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("  nap du lieu...")
    if a.load_ckpt:
        data = []                       # chi eval, khong can du lieu train
    else:
        ds = load_train(a.max_train, a.seed)
        data = encode_train(ds, tok, a.max_len)
    tests = load_tests(a.test_cache)
    print(f"  train {len(data)} vi du | eval_max_len {a.eval_max_len} | test " +
          ", ".join(f"{SHORT[t]} {len(v)}" for t, v in tests.items()))

    t0 = time.perf_counter()
    model, info = build_model(a)
    setup_s = round(time.perf_counter() - t0, 1)
    tot = info["total"]
    print(f"\n  TRAIN DUOC: {info['trainable'] / 1e6:.3f} M "
          f"({100 * info['trainable'] / tot:.3f} % cua {tot / 1e6:.2f} M)   [setup {setup_s}s]")

    res = dict(tag=tag, config=vars(a) | {"device": str(a.device), "dtype": str(a.dtype)},
               params=info, setup_s=setup_s)
    if a.load_ckpt:
        # Phan ra tat dinh nen dung lai duoc tu W0; chi can nap phan train duoc.
        ck = torch.load(a.load_ckpt, map_location=a.device, weights_only=False)
        cur = {n: p for n, p in model.named_parameters() if p.requires_grad}
        miss = set(cur) ^ set(ck["trainable"])
        assert not miss, f"khong khop tham so: {sorted(miss)[:4]}"
        with torch.no_grad():
            for n, v in ck["trainable"].items():
                cur[n].copy_(v.to(a.device))
        print(f"  nap {a.load_ckpt}  ({len(ck['trainable'])} tensor, meta={ck['meta'].get('which')})")
        res["loaded_ckpt"] = a.load_ckpt
    elif a.method != "none":
        res["train"] = train(model, data, tok, a)
    print("\n  danh gia 8 tac vu (epoch cuoi):")
    res["acc"] = evaluate(model, tok, tests, a)
    print(f"\n    TRUNG BINH: {res['acc']['avg']:.2f}%")

    if not a.no_bench:
        res["bench"] = bench_model(model, a)
        if a.method in ("rowspace", "lora"):
            merge_all(model, a.targets)
            res["bench"].update(bench_model(model, a, prefix="merged_"))
        b = res["bench"]
        print(f"\n  do tre: fwd {b['fwd_ms']:.2f} ms | fwd+bwd {b.get('fwdbwd_ms', 0):.2f} ms"
              f" | sau merge {b.get('merged_fwd_ms', 0):.2f} ms")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    with open(f"{a.out_dir}/summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(
            tag=tag, method=a.method, rank=a.rank, alpha=a.alpha,
            basis=a.basis if a.method == "rowspace" else "na", seed=a.seed,
            trainable=info["trainable"], model=a.model, targets=",".join(a.targets),
            max_train=a.max_train, epochs=a.epochs, setup_s=setup_s,
            **{SHORT[t]: round(res["acc"][t], 2) for t in TASKS},
            avg=round(res["acc"]["avg"], 2), eval_s=res["acc"]["eval_s"],
            **res.get("train", {}), **res.get("bench", {}))) + "\n")
    print(f"\n  ket qua -> {a.out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
