#!/usr/bin/env python
"""
Fine-tune + danh gia GPT-2 tren E2E NLG Challenge, so sanh:

  --method rowspace   LoRA len cac ma tran VUONG cua phan ra [I|X]   (y tuong cua ban)
  --method rowspace-full   train thang ma tran vuong (khong LoRA)
  --method lora       LoRA thuong len chinh cac Conv1D goc            (doi chung)
  --method target-ft  train tu do chinh cac ma tran dich                (tran tren dung)
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

from peft_generic import (TARGETS, LoRAConv1D, LoRALinear, apply_hybrid,
                          apply_lora, apply_vera, convert_rowspace, merge_all,
                          param_counts)
from rowspace_peft import NONSQUARE, RowSpaceLinear

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



# =========================================================================== do luong

def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def timeit_cuda(fn, dev, iters=20, warmup=5):
    """ms/lan, lay TRUNG VI de bot nhieu. Bat buoc sync, khong thi so do vo nghia."""
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
    """Do tre forward va forward+backward tren batch CO DINH.

    Tach hoan toan khoi beam search: generate() phu thuoc so buoc decode ma model
    can de dat EOS, nen thoi gian sinh cau KHONG do duoc chi phi cua kien truc.
    """
    ids = torch.randint(0, model.config.vocab_size,
                        (a.bench_batch, a.bench_seq), device=a.device)
    ntok = a.bench_batch * a.bench_seq
    out, was_training = {}, model.training

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
        out[prefix + "fwdbwd_ktok_s"] = round(ntok / ms, 1)

    model.train(was_training)
    return out


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
    return hyps, time.perf_counter() - t0


def evaluate(model, tok, pairs, a, tag):
    hyps, gen_s = generate(model, tok, pairs, a)
    refs = [r for _, r in pairs]
    bleu, rl = corpus_bleu(hyps, refs), rouge_l(hyps, refs)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}_hyps.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    with open(f"{a.out_dir}/{tag}_refs.txt", "w", encoding="utf-8") as f:   # format e2e-metrics
        f.write("\n\n".join("\n".join(r) for r in refs) + "\n")
    print(f"\n    [{tag}]  BLEU-4 = {bleu:.2f}   ROUGE-L = {rl:.2f}   ({len(hyps)} cau)")
    print(f"    vi du: {hyps[0][:110]}")
    ntok = sum(len(h.split()) for h in hyps)
    print(f"    sinh {len(hyps)} cau trong {gen_s:.0f}s "
          f"({len(hyps)/gen_s:.2f} cau/s, {ntok/gen_s:.0f} tu/s)")
    return dict(bleu=bleu, rouge_l=rl, n=len(hyps), gen_s=round(gen_s, 1),
                gen_sent_s=round(len(hyps)/gen_s, 3), gen_word_s=round(ntok/gen_s, 1))


@torch.no_grad()
def eval_loss(model, data, tok, a, max_n=None):
    """NLL trung binh moi token tren tap val.

    KHONG dung label smoothing: smoothing la meo luc train, con day la thuoc do
    tong quat hoa, phai so sanh duoc voi perplexity va voi so trong literature.
    """
    was = model.training
    model.eval()
    lossf = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    idx = sorted(range(len(data)), key=lambda j: len(data[j][0]))   # gom theo do dai
    if max_n:
        idx = idx[:max_n]
    tot = ntok = 0
    for i in range(0, len(idx), a.eval_batch):
        ids, lab, att = collate([data[j] for j in idx[i:i + a.eval_batch]], tok.eos_token_id)
        ids, lab, att = ids.to(a.device), lab.to(a.device), att.to(a.device)
        lg = model(input_ids=ids, attention_mask=att).logits
        tgt = lab[:, 1:].reshape(-1)
        tot += lossf(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt).item()
        ntok += int((tgt != -100).sum())
    model.train(was)
    return tot / max(ntok, 1)


def save_ckpt(path, state, meta):
    """Chi luu tham so train duoc + metadata.

    Phan dong bang (X, sel, rest, C) sinh ra tu W0 bang pivoted QR mot cach TAT DINH,
    nen nap lai chi can chay lai convert_rowspace voi cung config. Nho vay checkpoint
    chi vai MB thay vi 6 GB.
    """
    torch.save({"trainable": {n: v.detach().cpu() for n, v in state.items()},
                "meta": meta}, path)
    return os.path.getsize(path) / 2**20


def trainable_state(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def make_sched(opt, steps, warm, kind):
    """Lich lr. 'linear' giu NGUYEN cong thuc cu de tai lap duoc ket qua da chay.

    Vi sao can 'constant': voi lich linear, lr ve 0 dung o epoch cuoi nen val loss
    luon giam o do — 22/25 run cu deu co best_epoch = epoch cuoi. Do la hieu ung
    cua lich, khong phai bang chung chua hoi tu. Muon do hoi tu that thi lr phai
    khong doi sau warmup, luc do val theo epoch moi so sanh duoc voi nhau.
    """
    if kind == "constant":
        fn = lambda s: min(1.0, s / max(warm, 1))
    elif kind == "cosine":
        fn = lambda s: (s / max(warm, 1) if s < warm else
                        0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    else:
        fn = lambda s: s / max(warm, 1) if s < warm else max(0.0, (steps - s) / max(steps - warm, 1))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def load_trainable_ckpt(path, model, device):
    """Nap tham so train duoc tu *_best.pt / *_last.pt. Tra ve meta cua checkpoint.

    Phan dong bang khong nam trong checkpoint — no sinh lai tat dinh tu W0, nen
    model phai duoc dung voi DUNG --method/--rank/--targets/--basis cua run goc.
    Lech mot tensor nao la bao loi, khong nap im lang.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cur = trainable_state(model)
    miss = set(cur) ^ set(ck["trainable"])
    if miss:
        raise ValueError(f"checkpoint lech {len(miss)} tensor so voi cau hinh hien tai: "
                         f"{sorted(miss)[:3]}... — kiem --method/--rank/--targets")
    with torch.no_grad():
        for n, v in ck["trainable"].items():
            cur[n].copy_(v.to(device))
    m = ck.get("meta", {})
    print(f"  NAP {path}: which={m.get('which','?')} epoch={m.get('epoch','?')} "
          f"val_loss={m.get('val_loss','?')}")
    return m


def save_resume(path, model, opt, sched, step, ep, val_hist, train_hist, best, meta):
    """Checkpoint DAY DU de chay tiep: tham so, optimizer, lich, lich su, va RNG.

    Phai luu RNG vi make_batches() dung random.shuffle — thieu no thi thu tu batch
    sau khi resume se khac, va run bi ngat se khong con trung voi run lien mach.
    Phan dong bang khong luu: no sinh lai tat dinh tu W0 bang pivoted QR.
    """
    torch.save({
        "trainable": {n: v.detach().cpu() for n, v in trainable_state(model).items()},
        "opt": opt.state_dict(),
        "sched_last": sched.last_epoch,
        "step": step,
        "next_epoch": ep + 1,
        "val_hist": val_hist,
        "train_hist": train_hist,
        "best_val": best[0],
        "best_epoch": best[1],
        "best_state": None if best[2] is None else {n: v.cpu() for n, v in best[2].items()},
        "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "meta": meta,
    }, path)
    return os.path.getsize(path) / 2**20


def load_resume(path, model, opt, sched, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = trainable_state(model)
    miss = set(st) ^ set(ck["trainable"])
    if miss:
        raise ValueError(f"checkpoint khong khop cau hinh, lech {len(miss)} tensor: "
                         f"{sorted(miss)[:3]}... — kiem lai --method/--rank/--targets")
    for n, v in ck["trainable"].items():
        st[n].data.copy_(v.to(device))
    opt.load_state_dict(ck["opt"])
    sched.last_epoch = ck["sched_last"]
    random.setstate(ck["rng_python"])
    torch.set_rng_state(ck["rng_torch"])
    if ck.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["rng_cuda"])
    return ck


def train(model, data, val_data, tok, a, pairs=None):
    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=a.lr, weight_decay=a.weight_decay)
    steps = math.ceil(len(data) / a.batch) * a.epochs
    warm = min(a.warmup, steps // 10)
    sched = make_sched(opt, steps, warm, a.lr_schedule)
    lossf = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=a.label_smoothing)

    print(f"\n  train: {len(data)} vi du, {steps} buoc, lr={a.lr} ({a.lr_schedule}), "
          f"batch={a.batch}")
    if str(a.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    step, t0, run = 0, time.perf_counter(), None
    t_start = t0
    val_hist, best, val_s = [], (float("inf"), -1, None), 0.0
    bad, stopped = 0, None            # bad = so lan val khong cai thien lien tiep
    train_hist = []                   # loss tren tap train, do CUNG CACH voi val
    gen_hist = []                     # BLEU/ROUGE theo epoch neu bat --gen-every
    start_ep = 0

    if a.resume:
        ck = load_resume(a.resume, model, opt, sched, a.device)
        step, start_ep = ck["step"], ck["next_epoch"]
        val_hist, train_hist = ck["val_hist"], ck["train_hist"]
        bs = ck["best_state"]
        best = (ck["best_val"], ck["best_epoch"],
                None if bs is None else {n: v.to(a.device) for n, v in bs.items()})
        # get_last_lr() tra ve gia tri da cache luc khoi tao, KHONG cap nhat theo
        # last_epoch vua gan — phai tinh lai tu lambda thi in ra moi dung.
        lr_now = a.lr * sched.lr_lambdas[0](sched.last_epoch)
        print(f"  RESUME tu {a.resume}: tiep tuc o epoch {start_ep}/{a.epochs}, "
              f"buoc {step}/{steps}, lr {lr_now:.2e}, "
              f"best val {best[0]:.4f} (epoch {best[1]})")
        old = ck.get("meta", {})
        if old.get("lr_schedule") and old["lr_schedule"] != a.lr_schedule:
            print(f"  *** CANH BAO: checkpoint dung lich '{old['lr_schedule']}' nhung "
                  f"lan nay la '{a.lr_schedule}' — duong cong se khong lien tuc.")
        if a.lr_schedule == "linear" and old.get("epochs_planned") not in (None, a.epochs):
            print(f"  *** CANH BAO: lich linear phu thuoc tong so epoch. Checkpoint "
                  f"tinh theo {old['epochs_planned']} epoch, lan nay {a.epochs} -> lr "
                  f"se nhay. Dung --lr-schedule constant neu muon chay tiep tu do.")
        if start_ep >= a.epochs:
            print(f"  checkpoint da chay du {a.epochs} epoch — tang --epochs de chay tiep.")

    for ep in range(start_ep, a.epochs):
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

        if val_data and (ep + 1) % a.val_every == 0:
            tv = time.perf_counter()
            vl = eval_loss(model, val_data, tok, a, a.val_max)
            # Cung do tren TAP TRAIN, cung cach (khong smoothing, eval mode) -> khoang
            # cach train/val moi co y nghia. Train loss in ra luc chay CO smoothing nen
            # cao hon val khoang 1.5-2.0 chi vi so hang smoothing, khong phai do fit tot.
            tl = eval_loss(model, data, tok, a, a.train_eval_max) if a.train_eval_max else None
            val_s += time.perf_counter() - tv
            val_hist.append(round(vl, 4))
            if tl is not None:
                train_hist.append(round(tl, 4))
            improved = vl < best[0] - a.min_delta
            if improved:
                best = (vl, ep, {n: v.detach().clone() for n, v in trainable_state(model).items()})
                bad = 0
                mark = "  <- tot nhat"
            else:
                bad += 1
                mark = f"  (khong cai thien {bad}/{a.patience})" if a.patience else ""
            gap = f"  train {tl:.4f}  gap {vl - tl:+.4f}" if tl is not None else ""
            print(f"    [ep {ep}] val loss {vl:.4f}  ppl {math.exp(vl):.3f}{gap}  "
                  f"({time.perf_counter() - tv:.0f}s){mark}", flush=True)

            if a.gen_every and pairs and (ep + 1) % a.gen_every == 0:
                sub = pairs[:a.gen_every_max] if a.gen_every_max else pairs
                ge = evaluate(model, tok, sub, a, f"{a.tag}_ep{ep}")
                gen_hist.append(dict(epoch=ep, bleu=round(ge["bleu"], 2),
                                     rouge_l=round(ge["rouge_l"], 2), n=ge["n"]))
                val_s += ge["gen_s"]           # khong tinh vao thoi gian train
                model.train()                  # generate() da chuyen sang eval mode

            if a.ckpt_every and (ep + 1) % a.ckpt_every == 0:
                os.makedirs(a.out_dir, exist_ok=True)
                rp = f"{a.out_dir}/{a.tag}_resume.pt"
                # Ghi ra file tam roi doi ten: neu bi ngat giua chung thi checkpoint
                # cu van con nguyen, khong bi cut ngang.
                mb = save_resume(rp + ".tmp", model, opt, sched, step, ep, val_hist,
                                 train_hist, best,
                                 dict(model=a.model, method=a.method, rank=a.rank,
                                      alpha=a.alpha, basis=a.basis, targets=a.targets,
                                      seed=a.seed, lr=a.lr, lr_schedule=a.lr_schedule,
                                      epochs_planned=a.epochs))
                os.replace(rp + ".tmp", rp)
                print(f"      resume ckpt -> {rp}  ({mb:.1f} MB)", flush=True)

            if a.patience and bad >= a.patience:
                stopped = ep
                print(f"    EARLY STOPPING: val khong cai thien {bad} lan lien tiep. "
                      f"Tot nhat: epoch {best[1]} (val {best[0]:.4f}).")
                print(f"    luu y: lich lr tuyen tinh chua chay het, lr dung o "
                      f"{sched.get_last_lr()[0]:.2e} thay vi 0.")
                break
    el = time.perf_counter() - t_start - val_s      # tru phan val ra
    vram = (torch.cuda.max_memory_allocated() / 2**30
            if str(a.device).startswith("cuda") else 0.0)

    out = {}
    meta = dict(model=a.model, method=a.method, rank=a.rank, alpha=a.alpha,
                basis=a.basis, targets=a.targets, seed=a.seed, epochs_run=len(val_hist))
    if not a.no_save_ckpt:
        os.makedirs(a.out_dir, exist_ok=True)
        # LAST phai luu TRUOC khi khoi phuc best, khong thi ghi de mat trong so cuoi
        pl = f"{a.out_dir}/{a.tag}_last.pt"
        mb = save_ckpt(pl, trainable_state(model), meta | {"which": "last"})
        out["ckpt_last"] = pl
        print(f"    luu {pl}  ({mb:.1f} MB)")
        if best[2] is not None:
            pb = f"{a.out_dir}/{a.tag}_best.pt"
            save_ckpt(pb, best[2], meta | {"which": "best", "epoch": best[1],
                                           "val_loss": round(best[0], 4)})
            out["ckpt_best"] = pb
            print(f"    luu {pb}  (epoch {best[1]}, val {best[0]:.4f})")

    if val_hist:
        out.update(val_hist=val_hist, val_loss=val_hist[-1],
                   best_val=round(best[0], 4), best_epoch=best[1],
                   val_ppl=round(math.exp(val_hist[-1]), 3),
                   train_eval_hist=train_hist,
                   train_eval_loss=train_hist[-1] if train_hist else None,
                   overfit_gap=round(val_hist[-1] - train_hist[-1], 4) if train_hist else None,
                   val_s=round(val_s, 1), epochs_run=len(val_hist),
                   stopped_epoch=stopped)
        if gen_hist:
            out["gen_hist"] = gen_hist
        if a.best_epoch and best[2] is not None and best[1] != a.epochs - 1:
            with torch.no_grad():
                cur = trainable_state(model)
                for n, v in best[2].items():
                    cur[n].copy_(v)
            print(f"    -> khoi phuc epoch {best[1]} (val {best[0]:.4f}), "
                  f"thay vi epoch cuoi ({val_hist[-1]:.4f})")
            out["restored_epoch"] = best[1]
    return dict(**out, final_loss=run, train_s=round(el, 1), train_it_s=round(step / el, 3),
                train_step_ms=round(el / step * 1000, 2), steps_run=step,
                peak_vram_gb=round(vram, 2))


# =========================================================================== main

def build_model(a):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    n_orig = sum(p.numel() for p in model.parameters())

    if a.method in ("rowspace", "rowspace-full"):
        rank = 0 if a.method == "rowspace-full" else a.rank
        print(f"  phan ra {a.targets} (basis={a.basis})...", flush=True)
        convert_rowspace(model, a.targets, a.basis, rank, a.alpha, a.dtype, verbose=True)
    elif a.method == "lora":
        apply_lora(model, a.targets, a.rank, a.alpha)
    elif a.method == "vera":
        apply_vera(model, a.targets, a.rank, seed=a.seed, d_init=a.vera_d_init)
    elif a.method == "hybrid":
        # S-LoRA cho ma tran khong vuong, LoRA cho ma tran vuong (q_proj, o_proj)
        print(f"  hybrid: S-LoRA r={a.rank} + LoRA r={a.rank_square} tren ma tran vuong",
              flush=True)
        apply_hybrid(model, a.targets, a.rank, a.rank_square, a.alpha,
                     a.alpha_square, a.basis, a.dtype, verbose=True)
    elif a.method == "target-ft":
        # Tran tren DUNG NGHIA: train tu do chinh cac ma tran dich, khong phan ra,
        # khong LoRA. So sanh voi no cho biet rang buoc khong gian con + hang thap
        # lam mat bao nhieu. Full-model FT tra loi cau khac (no doi ca 28 lop).
        from peft_generic import get_blocks, resolve
        for p in model.parameters():
            p.requires_grad_(False)
        for block in get_blocks(model):
            for name in a.targets:
                parent, attr = resolve(block, name)
                for p in getattr(parent, attr).parameters():
                    p.requires_grad_(True)
    elif a.method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif a.method == "none":
        for p in model.parameters():
            p.requires_grad_(False)

    return model.to(a.device), param_counts(model, n_orig)


def main():
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 tren E2E NLG")
    p.add_argument("--method", required=True,
                   choices=["rowspace", "rowspace-full", "lora", "vera", "hybrid",
                            "target-ft",
                            "full", "none"])
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=None, help="mac dinh = rank")
    p.add_argument("--basis", choices=["colperm", "svd"], default="colperm")
    p.add_argument("--rank-square", type=int, default=None,
                   help="method=hybrid: hang LoRA dung cho ma tran VUONG (q_proj, "
                        "o_proj). Mac dinh = --rank. Dat thap hon --rank khi muon "
                        "don ngan sach vao cac ma tran khong vuong.")
    p.add_argument("--alpha-square", type=float, default=None,
                   help="method=hybrid: alpha cho phan LoRA vuong. Mac dinh = rank-square")
    p.add_argument("--vera-d-init", type=float, default=0.1,
                   help="method=vera: gia tri khoi tao vector d (paper dung 0.1)")
    p.add_argument("--match-params", action="store_true",
                   help="method=lora: tu chon hang de khop so tham so voi rowspace cung --rank")
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--target-set", default=None, choices=sorted(TARGETS),
                   help="gpt2 | qwen2_kv | qwen2_mlp | qwen2_all")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="mac dinh 2e-4 (PEFT) / 5e-5 (full)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr-schedule", choices=["linear", "constant", "cosine"],
                   default="linear",
                   help="linear = hanh vi cu (lr ve 0 o epoch cuoi). Dung 'constant' "
                        "khi do HOI TU: lich linear lam val luon giam o epoch cuoi, "
                        "nen khong the phan biet hoi tu that voi hieu ung cua lich.")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="luu checkpoint chay tiep duoc moi N epoch (0 = tat)")
    p.add_argument("--gen-every", type=int, default=0,
                   help="cham BLEU/ROUGE-L moi N epoch trong luc train (0 = tat). "
                        "LUU Y: cham tren tap TEST, nen duong cong nay chi de MO TA. "
                        "Chon epoch van phai dua vao val loss, khong duoc dua vao day.")
    p.add_argument("--gen-every-max", type=int, default=200,
                   help="method=gen-every: so MR toi da moi lan cham giua chung "
                        "(cham het 630 MR moi epoch rat ton). Lan cham CUOI luon day du.")
    p.add_argument("--eval-both", action="store_true",
                   help="cham BLEU/ROUGE o CA epoch cuoi va epoch tot nhat (theo val "
                        "loss). Voi lr khong doi hai cai nay thuong khac nhau. "
                        "Khong dung chung voi --best-epoch.")
    p.add_argument("--load-ckpt", default=None,
                   help="chi cham diem lai mot checkpoint da luu (*_best.pt / *_last.pt), "
                        "KHONG train. Dung de lay BLEU o epoch tot nhat sau khi run da "
                        "ket thuc o epoch cuoi.")
    p.add_argument("--resume", default=None,
                   help="duong dan *_resume.pt de chay tiep. Phai dung y het "
                        "--method/--rank/--targets/--seed cua run goc.")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--max-train", type=int, default=None, help="cat bot tap train de thu nhanh")
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--beams", type=int, default=10)
    p.add_argument("--length-penalty", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval-before", action="store_true", help="danh gia truoc khi train")
    p.add_argument("--val-every", type=int, default=1, help="do val moi N epoch")
    p.add_argument("--val-max", type=int, default=None, help="gioi han so vi du val")
    p.add_argument("--train-eval-max", type=int, default=1000,
                   help="so vi du train de do loss KHONG smoothing (0 = tat)")
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--no-save-ckpt", action="store_true",
                   help="khong luu checkpoint (mac dinh luu ca last va best)")
    p.add_argument("--patience", type=int, default=0,
                   help="dung som neu val khong cai thien N lan lien tiep (0 = tat)")
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="muc cai thien toi thieu de tinh la co tien bo")
    p.add_argument("--best-epoch", action="store_true",
                   help="khoi phuc checkpoint co val loss thap nhat truoc khi eval")
    p.add_argument("--bench-batch", type=int, default=8)
    p.add_argument("--bench-seq", type=int, default=128)
    p.add_argument("--bench-iters", type=int, default=20)
    p.add_argument("--no-bench", action="store_true",
                   help="bo qua phan do do tre kien truc")
    p.add_argument("--fp32-factorize", action="store_true",
                   help="phan ra o fp32 thay vi fp64")
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

    if a.patience and not a.best_epoch:
        a.best_epoch = True          # dung som ma van lay epoch cuoi thi vo nghia
        print("  --patience keo theo --best-epoch")

    if a.eval_both and a.best_epoch:
        # --best-epoch khoi phuc best NGAY TRONG train(), nen lan cham dau se la
        # best chu khong phai last, va --eval-both mat y nghia.
        a.best_epoch = False
        print("  --eval-both tat --best-epoch: cham lan luot epoch cuoi roi epoch tot nhat")
    a.dtype = torch.float32 if a.fp32_factorize else torch.float64
    if a.targets is None:
        key = a.target_set or ("qwen2_kv" if "qwen" in a.model.lower() else "gpt2")
        a.targets = TARGETS[key]
    if a.lr is None:
        # VeRA chi train hai vector d, b nen gradient nho hon han -> paper dung
        # lr cao hon LoRA khoang 50x. Bê nguyen 2e-4 sang thi VeRA gan nhu khong
        # hoc va ta se ket luan sai. Van nen quet lr truoc khi tin ket qua.
        if a.method == "vera":
            a.lr = 1e-2
        else:
            a.lr = 5e-5 if a.method in ("full", "target-ft") else 2e-4
    if a.method == "lora" and a.match_params:
        from transformers import AutoConfig
        from peft_generic import predict_params
        c = AutoConfig.from_pretrained(a.model)
        d = getattr(c, "hidden_size", None) or c.n_embd
        L = getattr(c, "num_hidden_layers", None) or c.n_layer
        ff = getattr(c, "intermediate_size", None) or 4 * d
        kv = getattr(c, "num_key_value_heads", None)
        hd = getattr(c, "head_dim", None) or (d // (getattr(c, "num_attention_heads", None) or c.n_head))
        SH = {"attn.c_attn": (3 * d, d), "attn.c_proj": (d, d),
              "mlp.c_fc": (ff, d), "mlp.c_proj": (d, ff),
              "self_attn.q_proj": (d, d), "self_attn.o_proj": (d, d),
              "self_attn.k_proj": (kv * hd if kv else d, d),
              "self_attn.v_proj": (kv * hd if kv else d, d),
              "mlp.gate_proj": (ff, d), "mlp.up_proj": (ff, d), "mlp.down_proj": (d, ff)}
        shapes = [SH[t] for t in a.targets]
        r_rs = a.rank
        rs = predict_params(shapes, r_rs, L, "rowspace")
        lo = predict_params(shapes, 1, L, "lora")
        a.rank = max(1, round(rs / lo))
        print(f"  --match-params: rowspace r={r_rs} = {rs/1e6:.3f} M "
              f"-> LoRA r={a.rank} = {a.rank*lo/1e6:.3f} M")

    if a.alpha is None:
        a.alpha = a.rank
    if a.rank_square is None:
        a.rank_square = a.rank
    if a.alpha_square is None:
        a.alpha_square = a.rank_square

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    _al = "" if a.alpha == a.rank else f"a{a.alpha:g}"
    tag = f"{a.method}_r{a.rank}{_al}_{a.basis if 'rowspace' in a.method else 'na'}_s{a.seed}"
    a.tag = tag
    print(f"\n=== {tag} ===\nmodel={a.model}  device={a.device}  targets={a.targets}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("  nap E2E NLG...")
    ds = load_e2e()
    train_rows = ds["train"]
    if a.max_train:
        train_rows = train_rows.select(range(min(a.max_train, len(train_rows))))
    data = encode_train(train_rows, tok, a.max_len)
    val_data = None if a.no_val else encode_train(ds["validation"], tok, a.max_len)
    pairs = group_refs(ds["test"])
    if a.limit_eval:
        pairs = pairs[:a.limit_eval]
    print(f"  train {len(data)} vi du | val {len(val_data) if val_data else 0} "
          f"| test {len(pairs)} MR duy nhat")

    t0 = time.perf_counter()
    model, info = build_model(a)
    t_setup = time.perf_counter() - t0
    tot = info["total"]
    print(f"\n  TRAIN DUOC: {info['trainable'] / 1e6:.3f} M "
          f"({100 * info['trainable'] / tot:.3f} % cua {tot / 1e6:.2f} M)"
          f"   [dung model + phan ra: {t_setup:.1f}s]")

    res = dict(tag=tag, config=vars(a) | {"device": str(a.device)}, params=info,
               setup_s=round(t_setup, 1))
    if a.eval_before:
        res["before"] = evaluate(model, tok, pairs, a, tag + "_before")
    if a.load_ckpt:
        # Chi cham diem lai mot checkpoint da luu, KHONG train. Dung de lay
        # BLEU/ROUGE o epoch tot nhat sau khi run da ket thuc o epoch cuoi.
        m = load_trainable_ckpt(a.load_ckpt, model, a.device)
        res["loaded"] = {k: m.get(k) for k in ("which", "epoch", "val_loss")}
        res["loaded"]["path"] = a.load_ckpt
    elif a.method != "none":
        res["train"] = train(model, data, val_data, tok, a, pairs)
        res["final_loss"] = res["train"]["final_loss"]
    res["after"] = evaluate(model, tok, pairs, a, tag)

    # --- cham them o epoch tot nhat -------------------------------------------
    # Voi lr khong doi, epoch cuoi thuong KHONG phai epoch tot nhat (S-LoRA r=2
    # cham day o epoch 5 roi di len). Cham ca hai moi biet duong cong BLEU co
    # bam theo val loss hay khong.
    tr = res.get("train") or {}
    if a.eval_both and tr.get("ckpt_best"):
        be, last_ep = tr.get("best_epoch"), tr.get("epochs_run", 0) - 1
        if be is not None and be != last_ep:
            print(f"\n  === cham lai o epoch tot nhat ({be}) thay vi epoch cuoi "
                  f"({last_ep}) ===")
            load_trainable_ckpt(tr["ckpt_best"], model, a.device)
            res["after_best"] = evaluate(model, tok, pairs, a, f"{tag}_bestep")
            res["after_best"]["epoch"] = be
            d = res["after_best"]["bleu"] - res["after"]["bleu"]
            print(f"    BLEU o epoch tot nhat lech {d:+.2f} so voi epoch cuoi")
        else:
            print(f"\n  epoch tot nhat trung epoch cuoi ({be}) — khong can cham lai")

    # --- do tre kien truc, TACH khoi beam search ---
    # generate() phu thuoc so buoc decode can de dat EOS, nen thoi gian sinh cau
    # khong do duoc chi phi that cua kien truc. Doan nay do tren batch co dinh.
    if not a.no_bench:
        print(f"\n  do tre (batch {a.bench_batch} x seq {a.bench_seq}):")
        res["bench"] = bench_model(model, a)
        if a.method in ("rowspace", "lora"):
            merge_all(model, a.targets)      # eval xong roi nen khong anh huong ket qua
            res["bench"].update(bench_model(model, a, prefix="merged_"))
        b = res["bench"]
        print(f"    forward         {b['fwd_ms']:8.2f} ms  ({b['fwd_ktok_s']:.1f} k token/s)")
        if "fwdbwd_ms" in b:
            print(f"    forward+bwd     {b['fwdbwd_ms']:8.2f} ms  ({b['fwdbwd_ktok_s']:.1f} k token/s)")
        if "merged_fwd_ms" in b:
            ov = 100 * (b["fwd_ms"] / b["merged_fwd_ms"] - 1)
            print(f"    forward da gop  {b['merged_fwd_ms']:8.2f} ms"
                  f"   -> chua gop cham hon {ov:+.1f} %")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    with open(f"{a.out_dir}/summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(tag=tag, method=a.method, rank=a.rank, alpha=a.alpha,
                                basis=a.basis if "rowspace" in a.method else "na",
                                seed=a.seed, trainable=info["trainable"],
                                model=a.model, targets=",".join(a.targets),
                                setup_s=res["setup_s"],
                                **res["after"], **res.get("train", {}),
                                **res.get("bench", {}),
                                # Long vao khoa rieng chu KHONG trai phang: after_best
                                # dung chung ten truong voi after (bleu, rouge_l, ...)
                                # nen trai phang se ghi de mat ket qua epoch cuoi.
                                **({"after_best": res["after_best"]}
                                   if "after_best" in res else {}),
                                **({"loaded": res["loaded"]}
                                   if "loaded" in res else {}),
                                tf32=bool(a.tf32),
                                lr=a.lr, lr_schedule=a.lr_schedule,
                                epochs_planned=a.epochs)) + "\n")
    print(f"\n  ket qua -> {a.out_dir}/{tag}.json  (va them dong vao summary.jsonl)")
    print(f"  cham diem chinh thuc:  python measure_scores.py "
          f"{a.out_dir}/{tag}_refs.txt {a.out_dir}/{tag}_hyps.txt")


if __name__ == "__main__":
    main()
