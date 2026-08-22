#!/usr/bin/env python
"""
Fine-tune + danh gia GPT-2 tren E2E NLG Challenge, so sanh:

  --method rowspace   LoRA len cac ma tran VUONG cua phan ra [I|X]   (y tuong cua ban)
  --method rowspace-full   train thang ma tran vuong (khong LoRA)
  --method lora       LoRA thuong len chinh cac Conv1D goc            (doi chung)
  --method full       full fine-tune                                 (tran tren)
  --method none       khong train, chi danh gia zero-shot             (san duoi)

Ca hai nhanh LoRA dung CHUNG mot lop LoRA (cung init, cung scale alpha/r) va CHUNG
mot vong train / sinh cau / cham diem, nen chenh lech chi den tu khac biet phuong phap.

So tham so train duoc (gpt2-medium, d=1024, L=24, 3 ma tran khong vuong):
  rowspace hang r : 3 * 2*d*r * L = 147,456 r
  lora     hang r : (4d + 5d + 5d) * r * L = 344,064 r
  -> cung hang thi rowspace re hon 2.33x. Dung --match-params de LoRA tu chon hang
     tuong duong, hoac quet nhieu hang roi ve duong Pareto (params vs BLEU).

Vi du:
  # thu nhanh 1 epoch tren tap con truoc khi chay that
  python finetune_e2e.py --method rowspace --rank 8 --epochs 1 --max-train 4000

  # chay that, cap doi de so sanh
  python finetune_e2e.py --method rowspace --rank 8 --seed 0
  python finetune_e2e.py --method lora --rank 8 --match-params --seed 0

Chi so BLEU/ROUGE-L o day de theo doi. Con so cho paper phai chay e2e-metrics chinh
thuc tren hai file hyps.txt / refs.txt ma script nay xuat ra.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

from rowspace_peft import NONSQUARE, RowSpaceLinear, convert_gpt2

PARQUET = "hf://datasets/tuetschek/e2e_nlg@refs%2Fconvert%2Fparquet/default"
SEP = " ||| "


# =========================================================================== du lieu

def load_e2e(cache_dir=None):
    from datasets import load_dataset
    files = {s: f"{PARQUET}/{s}/0000.parquet" for s in ("train", "validation", "test")}
    return load_dataset("parquet", data_files=files, cache_dir=cache_dir)


def encode_train(ds, tok, max_len):
    """Moi vi du: '<mr> ||| <ref><eos>', loss CHI tinh tren phan <ref>."""
    out = []
    for mr, ref in zip(ds["meaning_representation"], ds["human_reference"]):
        p = tok(mr + SEP.rstrip(), add_special_tokens=False).input_ids
        c = tok(" " + ref.strip(), add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p + c)[:max_len]
        labels = ([-100] * len(p) + c)[:max_len]
        if len(labels) > len(p):                       # bo vi du bi cat het phan tra loi
            out.append((ids, labels))
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
    """Gom cac vi du dai gan nhau vao cung batch -> gan nhu khong con padding.

    Do dai E2E lech nhau nhieu (p50=58, max=122) nen batch ngau nhien phi 35-55% token
    vao padding. Van giu ngau nhien: xao tron -> sap trong tung mega-batch -> xao tron
    thu tu cac batch.
    """
    idx = list(range(len(data)))
    random.shuffle(idx)
    mega, out = batch * 50, []
    for i in range(0, len(idx), mega):
        chunk = sorted(idx[i:i + mega], key=lambda j: len(data[j][0]))
        out += [chunk[k:k + batch] for k in range(0, len(chunk), batch)]
    random.shuffle(out)
    return out


def group_refs(ds):
    """Gom test set theo MR duy nhat -> [(mr, [ref, ...])]. E2E co toi 45 ref moi MR."""
    g = collections.OrderedDict()
    for mr, ref in zip(ds["meaning_representation"], ds["human_reference"]):
        g.setdefault(mr, []).append(ref.strip())
    return list(g.items())


# =========================================================================== LoRA thuong

class LoRAConv1D(nn.Module):
    """LoRA len Conv1D goc: y = x W + b + (alpha/r) (x A) B.  A kaiming, B = 0."""

    def __init__(self, conv, rank, alpha):
        super().__init__()
        self.base = conv
        d_in, d_out = conv.weight.shape
        dt = conv.weight.dtype
        self.down = nn.Parameter(torch.empty(d_in, rank, dtype=dt))
        self.up = nn.Parameter(torch.zeros(rank, d_out, dtype=dt))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank
        self.nf = conv.nf

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.down) @ self.up)


def apply_lora(model, targets, rank, alpha):
    from rowspace_peft import _resolve
    for block in model.transformer.h:
        for name in targets:
            parent, attr = _resolve(block, name)
            setattr(parent, attr, LoRAConv1D(getattr(parent, attr), rank, alpha))
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRAConv1D):
            m.down.requires_grad_(True)
            m.up.requires_grad_(True)


# =========================================================================== chi so

def toks(s):
    s = s.lower()
    for p in ",.!?;:()[]\"'":
        s = s.replace(p, f" {p} ")
    return s.split()


def corpus_bleu(hyps, refs_list, max_n=4):
    """BLEU-4 corpus, nhieu reference: clip theo max count qua cac ref, BP theo ref gan nhat."""
    num = [0] * max_n
    den = [0] * max_n
    hl = rl = 0
    for hyp, refs in zip(hyps, refs_list):
        h = toks(hyp)
        rs = [toks(r) for r in refs]
        hl += len(h)
        rl += min(rs, key=lambda r: (abs(len(r) - len(h)), len(r))).__len__()
        for n in range(1, max_n + 1):
            hc = collections.Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            mx = collections.Counter()
            for r in rs:
                rc = collections.Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
                for g, c in rc.items():
                    mx[g] = max(mx[g], c)
            num[n - 1] += sum(min(c, mx[g]) for g, c in hc.items())
            den[n - 1] += max(0, len(h) - n + 1)
    if min(num) == 0 or min(den) == 0:
        return 0.0
    logp = sum(math.log(num[i] / den[i]) for i in range(max_n)) / max_n
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    return 100 * bp * math.exp(logp)


def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], (prev + 1 if x == y else max(dp[j], dp[j - 1]))
    return dp[-1]


def rouge_l(hyps, refs_list, beta=1.2):
    tot = 0.0
    for hyp, refs in zip(hyps, refs_list):
        h = toks(hyp)
        best = 0.0
        for r in refs:
            rt = toks(r)
            l = _lcs(h, rt)
            if l == 0:
                continue
            p, rc = l / len(h), l / len(rt)
            best = max(best, (1 + beta**2) * p * rc / (rc + beta**2 * p))
        tot += best
    return 100 * tot / max(len(hyps), 1)


# =========================================================================== train / eval

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def generate(model, tok, pairs, a):
    model.eval()
    tok.padding_side = "left"
    hyps = []
    t0 = time.perf_counter()
    for i in range(0, len(pairs), a.eval_batch):
        chunk = [mr + SEP.rstrip() for mr, _ in pairs[i:i + a.eval_batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                             num_beams=a.beams, length_penalty=a.length_penalty,
                             no_repeat_ngram_size=4, early_stopping=True,
                             pad_token_id=tok.eos_token_id)
        for j in range(out.shape[0]):
            gen = out[j, enc.input_ids.shape[1]:]
            hyps.append(tok.decode(gen, skip_special_tokens=True).strip().replace("\n", " "))
        if i == 0 or (i // a.eval_batch) % 20 == 0:
            done = min(i + a.eval_batch, len(pairs))
            print(f"      sinh {done}/{len(pairs)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    tok.padding_side = "right"
    return hyps


def evaluate(model, tok, pairs, a, tag):
    hyps = generate(model, tok, pairs, a)
    refs = [r for _, r in pairs]
    bleu, rl = corpus_bleu(hyps, refs), rouge_l(hyps, refs)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}_hyps.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    with open(f"{a.out_dir}/{tag}_refs.txt", "w", encoding="utf-8") as f:   # format e2e-metrics
        f.write("\n\n".join("\n".join(r) for r in refs) + "\n")
    print(f"\n    [{tag}]  BLEU-4 = {bleu:.2f}   ROUGE-L = {rl:.2f}   ({len(hyps)} cau)")
    print(f"    vi du: {hyps[0][:110]}")
    return dict(bleu=bleu, rouge_l=rl, n=len(hyps))


def train(model, data, tok, a):
    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=a.lr, weight_decay=a.weight_decay)
    steps = math.ceil(len(data) / a.batch) * a.epochs
    warm = min(a.warmup, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm else max(0.0, (steps - s) / max(steps - warm, 1)))
    lossf = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=a.label_smoothing)

    print(f"\n  train: {len(data)} vi du, {steps} buoc, lr={a.lr}, batch={a.batch}")
    model.train()
    step, t0, run = 0, time.perf_counter(), None
    for ep in range(a.epochs):
        for bidx in make_batches(data, a.batch):
            ids, lab, att = collate([data[j] for j in bidx], tok.eos_token_id)
            ids, lab, att = ids.to(a.device), lab.to(a.device), att.to(a.device)
            logits = model(input_ids=ids, attention_mask=att).logits
            loss = lossf(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1))
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
    return run


# =========================================================================== main

def build_model(a):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(a.model)
    n_orig = sum(p.numel() for p in model.parameters())
    info = dict(n_orig=n_orig)

    if a.method in ("rowspace", "rowspace-full"):
        rank = 0 if a.method == "rowspace-full" else a.rank
        print(f"  phan ra {a.targets} (basis={a.basis})...")
        convert_gpt2(model, a.targets, a.basis, rank, a.alpha, torch.float64, verbose=True)
        info["frozen_X"] = sum(m.frozen_numel() for m in model.modules()
                               if isinstance(m, RowSpaceLinear))
    elif a.method == "lora":
        apply_lora(model, a.targets, a.rank, a.alpha or a.rank)
    elif a.method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif a.method == "none":
        for p in model.parameters():
            p.requires_grad_(False)

    info["trainable"] = count_trainable(model)
    # X thay THE trong so Conv1D cu, khong phai them vao -> cong voi .parameters() hien tai,
    # cong voi n_orig se bi dem trung.
    info["total"] = sum(p.numel() for p in model.parameters()) + info.get("frozen_X", 0)
    return model.to(a.device), info


def main():
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 tren E2E NLG")
    p.add_argument("--method", required=True,
                   choices=["rowspace", "rowspace-full", "lora", "full", "none"])
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=None, help="mac dinh = rank")
    p.add_argument("--basis", choices=["colperm", "svd"], default="colperm")
    p.add_argument("--match-params", action="store_true",
                   help="method=lora: tu chon hang de khop so tham so voi rowspace cung --rank")
    p.add_argument("--targets", nargs="+", default=NONSQUARE)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="mac dinh 2e-4 (PEFT) / 5e-5 (full)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--max-train", type=int, default=None, help="cat bot tap train de thu nhanh")
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--beams", type=int, default=10)
    p.add_argument("--length-penalty", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval-before", action="store_true", help="danh gia truoc khi train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tf32", action="store_true",
                   help="bat TF32 cho matmul (A10): nhanh ~2-3x, mantissa 10 bit. "
                        "Ap dung nhu nhau cho ca hai nhanh nen van cong bang.")
    a = p.parse_args()

    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if a.lr is None:
        a.lr = 5e-5 if a.method == "full" else 2e-4
    if a.method == "lora" and a.match_params:
        d, L = (1024, 24) if "medium" in a.model else (768, 12)
        shp = {"attn.c_attn": (d, 3 * d), "attn.c_proj": (d, d),
               "mlp.c_fc": (d, 4 * d), "mlp.c_proj": (4 * d, d)}
        per_rowspace = sum(2 * min(shp[t]) for t in a.targets)
        per_lora = sum(sum(shp[t]) for t in a.targets)
        a.rank = max(1, round(a.rank * per_rowspace / per_lora))
        print(f"  --match-params: hang cua LoRA doi chung -> {a.rank}")
    if a.alpha is None:
        a.alpha = a.rank

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tag = f"{a.method}_r{a.rank}_{a.basis if 'rowspace' in a.method else 'na'}_s{a.seed}"
    print(f"\n=== {tag} ===\nmodel={a.model}  device={a.device}  targets={a.targets}")

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(a.model)
    tok.pad_token = tok.eos_token

    print("  nap E2E NLG...")
    ds = load_e2e()
    train_rows = ds["train"]
    if a.max_train:
        train_rows = train_rows.select(range(min(a.max_train, len(train_rows))))
    data = encode_train(train_rows, tok, a.max_len)
    pairs = group_refs(ds["test"])
    if a.limit_eval:
        pairs = pairs[:a.limit_eval]
    print(f"  train {len(data)} vi du | test {len(pairs)} MR duy nhat")

    model, info = build_model(a)
    tot = info["total"]
    print(f"\n  TRAIN DUOC: {info['trainable'] / 1e6:.3f} M "
          f"({100 * info['trainable'] / tot:.3f} % cua {tot / 1e6:.2f} M)")

    res = dict(tag=tag, config=vars(a) | {"device": str(a.device)}, params=info)
    if a.eval_before:
        res["before"] = evaluate(model, tok, pairs, a, tag + "_before")
    if a.method != "none":
        res["final_loss"] = train(model, data, tok, a)
    res["after"] = evaluate(model, tok, pairs, a, tag)

    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    with open(f"{a.out_dir}/summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(tag=tag, method=a.method, rank=a.rank, basis=a.basis,
                                seed=a.seed, trainable=info["trainable"],
                                **res["after"])) + "\n")
    print(f"\n  ket qua -> {a.out_dir}/{tag}.json  (va them dong vao summary.jsonl)")
    print(f"  cham diem chinh thuc:  python measure_scores.py "
          f"{a.out_dir}/{tag}_refs.txt {a.out_dir}/{tag}_hyps.txt")


if __name__ == "__main__":
    main()
