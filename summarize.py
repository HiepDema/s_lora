#!/usr/bin/env python
"""Doc summary.jsonl va in bang so sanh, sap theo so tham so train duoc."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "runs/summary.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

g = defaultdict(list)
for r in rows:                     # gom cac seed cua cung cau hinh
    g[(r.get("model", "?"), r["method"], r["rank"], r.get("alpha", r["rank"]),
       r["basis"], r.get("targets", ""))].append(r)


def ms(v, f, w=6, p=2):
    """mean +/- std cua truong f."""
    x = [r[f] for r in v if r.get(f) is not None]
    if not x:
        return " " * (w + 8)
    m = sum(x) / len(x)
    sd = (sum((y - m) ** 2 for y in x) / (len(x) - 1)) ** .5 if len(x) > 1 else 0.0
    return f"{m:{w}.{p}f} +/-{sd:5.{p}f}"


def one(v, f, w=8, p=2):
    x = [r[f] for r in v if r.get(f) is not None]
    return f"{sum(x)/len(x):{w}.{p}f}" if x else " " * w


for model in sorted({k[0] for k in g}):
    ks = sorted([k for k in g if k[0] == model], key=lambda k: g[k][0]["trainable"])
    print(f"\n{'=' * 92}\n{model}")

    print(f"\nCHAT LUONG\n  {'method':<10s} {'r':>3s} {'a':>3s} {'params':>8s} {'n':>2s} "
          f"{'BLEU-4':>14s} {'ROUGE-L':>14s} {'val loss':>9s} {'val ppl':>8s}")
    print("  " + "-" * 82)
    for k in ks:
        v = g[k]
        print(f"  {k[1]:<10s} {k[2]:>3d} {k[3]:>3g} {v[0]['trainable']/1e6:7.3f}M {len(v):>2d} "
              f"{ms(v,'bleu')} {ms(v,'rouge_l')} {one(v,'val_loss',9,4)} {one(v,'val_ppl',8,3)}")

    if any("train_s" in r for k in ks for r in g[k]):
        print(f"\nTHOI GIAN\n  {'method':<10s} {'r':>3s} {'params':>8s} {'setup_s':>8s} "
              f"{'train_s':>8s} {'val_s':>7s} {'it/s':>7s} {'VRAM_GB':>8s}")
        print("  " + "-" * 74)
        for k in ks:
            v = g[k]
            print(f"  {k[1]:<10s} {k[2]:>3d} {v[0]['trainable']/1e6:7.3f}M "
                  f"{one(v,'setup_s',8,1)} {one(v,'train_s',8,0)} {one(v,'val_s',7,0)} "
                  f"{one(v,'train_it_s',7,2)} {one(v,'peak_vram_gb',8,2)}")

    if any("fwd_ms" in r for k in ks for r in g[k]):
        print(f"\nDO TRE KIEN TRUC  (batch co dinh, tach khoi beam search)\n"
              f"  {'method':<10s} {'r':>3s} {'params':>8s} {'fwd_ms':>8s} {'fwd+bwd':>8s} "
              f"{'gop_ms':>8s} {'gen_s':>7s}")
        print("  " + "-" * 68)
        for k in ks:
            v = g[k]
            print(f"  {k[1]:<10s} {k[2]:>3d} {v[0]['trainable']/1e6:7.3f}M "
                  f"{one(v,'fwd_ms',8,2)} {one(v,'fwdbwd_ms',8,2)} "
                  f"{one(v,'merged_fwd_ms',8,2)} {one(v,'gen_s',7,0)}")

    # duong cong val: dau hieu overfit
    for k in ks:
        h = g[k][0].get("val_hist")
        if h:
            be = g[k][0].get("best_epoch")
            st = g[k][0].get("stopped_epoch")
            trend = " -> ".join(("*" if i == be else "") + f"{x:.4f}"
                                for i, x in enumerate(h))
            tail = f"   (* = tot nhat, epoch {be}" if be is not None else "   ("
            tail += f"; DUNG SOM o epoch {st})" if st is not None else ")"
            print()
            print(f"  val theo epoch [{k[1]} r={k[2]}]: {trend}{tail}")

print("\nGhi chu: 'gop_ms' la forward sau merge — bang chung cho zero-overhead luc deploy.")
print("val loss KHONG co label smoothing, nen so sanh duoc voi perplexity.")
