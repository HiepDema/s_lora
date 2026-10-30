#!/usr/bin/env python
"""VTAB-1k: LoRA vs S-LoRA tren ViT-B/16 (ImageNet-21k goc), 19 tac vu.

Protocol theo NOAH / FacT de so duoc voi so da cong bo:
  train tren train800val200 (1000 anh), cham tren test; anh resize 224x224
  bicubic, chuan hoa ImageNet mean/std, khong augmentation; AdamW, batch 64,
  100 epoch, warmup 10 epoch roi cosine, weight decay 1e-4, drop path 0.1.
  Head (768 -> so lop) train day du o moi method.

  --method lora    LoRA r tren q,k,v (3 adapter rieng tren qkv gop), proj, fc1, fc2
  --method hybrid  S-LoRA r tren qkv gop, fc1, fc2; LoRA r tren proj (vuong)

Du lieu: thu muc theo dinh dang NOAH, moi tac vu mot thu muc con co
train800val200.txt / test.txt ("images/... nhan" moi dong).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from slora.surgery import TARGETS, apply_hybrid, apply_lora, merge_all

# Thu tu va nhom theo bang VTAB cua NOAH/FacT.
GROUPS = {
    "Natural": ["cifar", "caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet",
                "svhn", "sun397"],
    "Specialized": ["patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy"],
    "Structured": ["clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc",
                   "dsprites_ori", "smallnorb_azi", "smallnorb_ele"],
}
TASKS = [t for ts in GROUPS.values() for t in ts]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# =========================================================================== du lieu

def read_list(root, split):
    out = []
    with open(f"{root}/{split}.txt", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                p, l = ln.rsplit(" ", 1)
                out.append((p, int(l)))
    return out


class FileList(torch.utils.data.Dataset):
    def __init__(self, root, items):
        from torchvision import transforms as T
        self.root, self.items = root, items
        self.tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, l = self.items[i]
        return self.tf(Image.open(f"{self.root}/{p}").convert("RGB")), l


def load_split(root, items, workers, batch=256):
    dl = torch.utils.data.DataLoader(FileList(root, items), batch_size=batch,
                                     num_workers=workers, shuffle=False)
    xs, ys = zip(*dl)
    return torch.cat(xs), torch.cat(ys)


# =========================================================================== model

def build_model(a, n_cls):
    import timm
    from timm.models.vision_transformer import _load_weights
    m = timm.create_model("vit_base_patch16_224.orig_in21k", pretrained=False,
                          num_classes=n_cls, drop_path_rate=a.drop_path)
    _load_weights(m, a.ckpt)              # npz goc cua Google; head 21843 lop bi bo qua
    n0 = sum(p.numel() for p in m.parameters())
    tg = TARGETS["vit_all6"]
    if a.method == "lora":
        apply_lora(m, tg, a.rank, a.alpha, split={"attn.qkv": 3})
    elif a.method == "hybrid":
        apply_hybrid(m, tg, a.rank, a.rank, a.alpha, a.alpha, "colperm", torch.float64,
                     verbose=False, cache=a.fac_cache, norm=a.slora_norm,
                     impl=a.slora_impl)
    elif a.method == "linear":
        for p in m.parameters():
            p.requires_grad_(False)
    for p in m.head.parameters():
        p.requires_grad_(True)
    head = sum(p.numel() for p in m.head.parameters())
    adapter = sum(p.numel() for p in m.parameters() if p.requires_grad) - head
    return m.to(a.device), dict(adapter=adapter, head=head, n_orig=n0)


# =========================================================================== train / cham

@torch.no_grad()
def evaluate(model, root, items, a, bs=256):
    """Doc test theo luong: dsprites co 73 728 anh, nap het vao RAM la 44 GB."""
    dl = torch.utils.data.DataLoader(FileList(root, items), batch_size=bs,
                                     num_workers=a.workers, shuffle=False, pin_memory=True)
    model.eval()
    ok = 0
    for xb, yb in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
            ok += (model(xb.to(a.device, non_blocking=True)).argmax(-1).cpu() == yb).sum().item()
    return 100.0 * ok / len(items)


def train(model, x, y, a):
    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=a.lr, weight_decay=a.weight_decay)
    per_ep = math.ceil(len(x) / a.batch)
    steps, warm = per_ep * a.epochs, per_ep * a.warmup_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    x, y = x.to(a.device), y.to(a.device)
    model.train()
    t0, log = time.perf_counter(), []
    for ep in range(a.epochs):
        perm = torch.randperm(len(x), device=a.device)
        tl = tc = 0.0
        for i in range(0, len(x), a.batch):
            idx = perm[i:i + a.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
                out = model(x[idx])
            loss = Fn.cross_entropy(out.float(), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(-1) == y[idx]).sum().item()
        if not math.isfinite(tl):
            raise RuntimeError(f"loss NaN/Inf o epoch {ep}")
        if (ep + 1) % a.log_every == 0 or ep == a.epochs - 1:
            log.append(dict(epoch=ep + 1, loss=tl / len(x), acc=100 * tc / len(x)))
            print(f"      ep {ep + 1:3d}  loss {tl / len(x):.4f}  train acc "
                  f"{100 * tc / len(x):5.1f}%  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    return dict(train_s=round(time.perf_counter() - t0, 1), log=log,
                final_loss=log[-1]["loss"], final_train_acc=log[-1]["acc"])


def bench(model, a, iters=30):
    """Do tre batch 64 x 224: forward, forward+backward, forward sau merge."""
    xb = torch.randn(64, 3, 224, 224, device=a.device)

    def t(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return round(sorted(ts)[len(ts) // 2], 2)

    def fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=a.amp == "bf16"):
            model(xb)

    def fwdbwd():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
            out = model(xb)
        out.float().sum().backward()
        model.zero_grad(set_to_none=True)

    model.eval()
    r = dict(fwd_ms=t(fwd))
    model.train()
    r["fwdbwd_ms"] = t(fwdbwd)
    model.eval()
    merge_all(model, TARGETS["vit_all6"])
    r["merged_fwd_ms"] = t(fwd)
    return r


# =========================================================================== main

def main():
    p = argparse.ArgumentParser(description="VTAB-1k: LoRA vs S-LoRA tren ViT-B/16")
    p.add_argument("--method", required=True, choices=["lora", "hybrid", "linear"])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=None, help="mac dinh = rank (scale 1)")
    p.add_argument("--slora-norm", choices=["rms"], default=None)
    p.add_argument("--slora-impl", choices=["lowrank", "factored"], default="lowrank",
                   help="lowrank: W0 x + delta qua X̄ (sai so nhu LoRA); factored: duong P.C cu")
    p.add_argument("--data", default=os.path.expanduser("~/data/vtab-1k"))
    p.add_argument("--ckpt", default=os.path.expanduser("~/ckpt/ViT-B_16.npz"))
    p.add_argument("--tasks", default="all", help="all | danh sach cach nhau dau phay")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--amp", choices=["none", "bf16"], default="none",
                   help="none = fp32 + TF32 (mac dinh); bf16 = autocast")
    p.add_argument("--max-train", type=int, default=0, help="smoke test: cat bot anh train")
    p.add_argument("--max-test", type=int, default=0, help="smoke test: cat bot anh test")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fac-cache", default="fac_cache/vitb16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--out-dir", default="runs_vtab")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.alpha is None:
        a.alpha = float(a.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tag = (f"vtab_{a.method}_r{a.rank}{'_' + a.slora_norm if a.slora_norm else ''}"
           f"{'_fac' if a.method == 'hybrid' and a.slora_impl == 'factored' else ''}"
           f"_lr{a.lr:g}_s{a.seed}")
    os.makedirs(a.out_dir, exist_ok=True)
    res_path = f"{a.out_dir}/{tag}.jsonl"
    done = set()
    if os.path.exists(res_path):
        done = {json.loads(l)["task"] for l in open(res_path, encoding="utf-8")}
    tasks = TASKS if a.tasks == "all" else a.tasks.split(",")
    print(f"\n=== {tag} ===  {len(tasks)} tac vu, da xong {len(done & set(tasks))}", flush=True)

    benched = any("bench" in json.loads(l) for l in open(res_path, encoding="utf-8")) \
        if os.path.exists(res_path) else False
    for task in tasks:
        if task in done:
            continue
        root = f"{a.data}/{task}"
        tr, te = read_list(root, "train800val200"), read_list(root, "test")
        n_cls = max(l for _, l in tr + te) + 1
        if a.max_train:
            tr = tr[:a.max_train]
        if a.max_test:
            te = te[:a.max_test]
        group = next(g for g, ts in GROUPS.items() if task in ts)
        print(f"\n  [{task}] {group}  {n_cls} lop  train {len(tr)}  test {len(te)}", flush=True)

        t0 = time.perf_counter()
        xtr, ytr = load_split(root, tr, a.workers)
        load_s = time.perf_counter() - t0

        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        t0 = time.perf_counter()
        model, cnt = build_model(a, n_cls)
        setup_s = time.perf_counter() - t0
        torch.cuda.reset_peak_memory_stats()
        tr_info = train(model, xtr, ytr, a)
        peak = torch.cuda.max_memory_allocated() / 2**30
        t0 = time.perf_counter()
        acc = evaluate(model, root, te, a)
        eval_s = time.perf_counter() - t0
        print(f"    => {task}: test {acc:.2f}%  (adapter {cnt['adapter'] / 1e6:.3f}M, "
              f"train {tr_info['train_s']:.0f}s, peak {peak:.1f} GB)", flush=True)

        row = dict(tag=tag, task=task, group=group, n_cls=n_cls, acc=round(acc, 2),
                   method=a.method, rank=a.rank, alpha=a.alpha, slora_norm=a.slora_norm,
                   lr=a.lr, seed=a.seed, amp=a.amp, epochs=a.epochs,
                   n_train=len(tr), n_test=len(te), **cnt, setup_s=round(setup_s, 1),
                   load_s=round(load_s, 1), eval_s=round(eval_s, 1),
                   peak_gb=round(peak, 2), **tr_info)
        if not benched and not a.no_bench:
            row["bench"] = bench(model, a)
            benched = True
            print(f"    do tre b64: {row['bench']}", flush=True)
        with open(res_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        del model, xtr, ytr
        torch.cuda.empty_cache()

    rows = {json.loads(l)["task"]: json.loads(l) for l in open(res_path, encoding="utf-8")}
    print(f"\n  TONG KET {tag}")
    gm = []
    for g, ts in GROUPS.items():
        got = [rows[t]["acc"] for t in ts if t in rows]
        if got:
            gm.append(sum(got) / len(got))
            print(f"    {g:<12} {gm[-1]:6.2f}  ({len(got)}/{len(ts)} tac vu)")
    allv = [r["acc"] for r in rows.values()]
    print(f"    TB 19 tac vu {sum(allv) / len(allv):6.2f}   TB 3 nhom "
          f"{sum(gm) / len(gm):6.2f}")


if __name__ == "__main__":
    main()
