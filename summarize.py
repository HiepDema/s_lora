#!/usr/bin/env python
"""Doc runs/summary.jsonl va in bang so sanh, sap theo so tham so train duoc."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "runs/summary.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

g = defaultdict(list)
for r in rows:                                    # gom cac seed cua cung cau hinh
    g[(r["method"], r["rank"], r["basis"])].append(r)

print(f"{'method':<14s} {'rank':>4s} {'basis':>8s} {'params':>10s} {'seeds':>5s} "
      f"{'BLEU-4':>16s} {'ROUGE-L':>16s}")
for k in sorted(g, key=lambda k: g[k][0]["trainable"]):
    v = g[k]
    n = len(v)
    def ms(f):
        x = [r[f] for r in v]
        m = sum(x) / n
        sd = (sum((y - m) ** 2 for y in x) / (n - 1)) ** .5 if n > 1 else 0.0
        return f"{m:7.2f} +/-{sd:5.2f}"
    print(f"{k[0]:<14s} {k[1]:>4d} {k[2]:>8s} {v[0]['trainable'] / 1e6:9.3f}M {n:>5d} "
          f"{ms('bleu'):>16s} {ms('rouge_l'):>16s}")
