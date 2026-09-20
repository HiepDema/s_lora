#!/usr/bin/env python
"""Phan tich hoi tu: train loss, val loss, va BLEU/ROUGE theo epoch.

    python plot_convergence.py runs_conv/summary.jsonl
    python plot_convergence.py conv.log chain.log        # doc thang tu log
    python plot_convergence.py runs_conv/*_resume.pt     # ca khi dang chay

LUU Y: BLEU/ROUGE tu --gen-every cham tren tap TEST, khong phai val. Duong cong
do chi de MO TA; chon epoch van phai dua vao val loss.
"""
import json
import re
import sys
from pathlib import Path


def slope(ys):
    """He so goc cua hoi quy tuyen tinh. Am = con giam."""
    n = len(ys)
    if n < 3:
        return None
    mx = (n - 1) / 2
    my = sum(ys) / n
    den = sum((i - mx) ** 2 for i in range(n))
    return sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den if den else None


def from_log(path):
    """Doc conv.log / chain.log: dong [ep N] va dong BLEU-4 cua --gen-every."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    runs = []
    # moi lan "train:" la mot run moi trong cung file
    for block in re.split(r"\n(?=\s*=== )", txt):
        val, tr, gen = [], [], {}
        for m in re.finditer(r"\[ep (\d+)\] val loss ([\d.]+).*?train ([\d.]+)", block):
            val.append(float(m.group(2)))
            tr.append(float(m.group(3)))
        for m in re.finditer(r"_ep(\d+)\]\s+BLEU-4 = ([\d.]+)\s+ROUGE-L = ([\d.]+)", block):
            gen[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        if val:
            tag = re.search(r"=== (\S+) ===", block)
            runs.append((tag.group(1) if tag else Path(path).stem, val, tr, gen))
    return runs


def from_summary(path):
    runs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        gen = {g["epoch"]: (g["bleu"], g["rouge_l"]) for g in d.get("gen_hist", [])}
        runs.append((f"{d['method']} r={d['rank']}", d.get("val_hist", []),
                     d.get("train_eval_hist", []), gen))
    return runs


def from_ckpt(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck.get("meta", {})
    return [(f"{m.get('method','?')} r={m.get('rank','?')}",
             ck.get("val_hist", []), ck.get("train_hist", []), {})]


def report(name, val, tr, gen):
    print(f"\n{'=' * 78}\n{name}   ({len(val)} epoch)")
    print(f"{'ep':>4}{'train':>9}{'val':>9}{'gap':>8}{'BLEU':>8}{'ROUGE':>8}   (test)")
    print("-" * 56)
    for e, v in enumerate(val):
        t = tr[e] if e < len(tr) else None
        b, r = gen.get(e, (None, None))
        mark = " *" if v == min(val) else "  "
        print(f"{e:>4}{(f'{t:.4f}' if t else '-'):>9}{v:>9.4f}"
              f"{(f'{v-t:+.4f}' if t else '-'):>8}"
              f"{(f'{b:.2f}' if b else '-'):>8}{(f'{r:.2f}' if r else '-'):>8}{mark}")

    be = val.index(min(val))
    n = len(val)
    tail = max(3, n // 4)
    sv = slope(val[-tail:])
    print(f"\n  val thap nhat o epoch {be}/{n-1}"
          + ("  <- o 1/4 cuoi, CO THE chua het da" if be >= n - tail
             else "  <- o giua, da qua diem tot nhat"))
    if sv is not None:
        print(f"  do doc val {tail} epoch cuoi: {sv:+.5f}/epoch  "
              + ("van con giam" if sv < -2e-4 else
                 "dang di len (overfit)" if sv > 2e-4 else "PHANG -> da hoi tu"))
    if tr:
        st = slope(tr[-tail:])
        g0 = val[0] - tr[0]
        g1 = val[-1] - tr[-1]
        print(f"  do doc train {tail} epoch cuoi: {st:+.5f}/epoch"
              if st is not None else "")
        print(f"  khoang cach train-val: {g0:+.4f} -> {g1:+.4f} "
              + ("(no rong -> bat dau overfit)" if g1 > g0 + 0.01 else "(on dinh)"))
    if gen:
        eps = sorted(gen)
        bl = [gen[e][0] for e in eps]
        bb = eps[bl.index(max(bl))]
        print(f"  BLEU cao nhat o epoch {bb} ({max(bl):.2f}); val thap nhat o epoch {be}"
              + ("  <- TRUNG nhau" if bb == be else "  <- LECH nhau"))


def main(paths):
    runs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"bo qua (khong thay): {p}")
            continue
        runs += (from_ckpt(p) if p.suffix == ".pt" else
                 from_log(p) if p.suffix == ".log" else from_summary(p))
    if not runs:
        sys.exit("khong doc duoc run nao")
    for r in runs:
        report(*r)
    print(f"\n{'=' * 78}\nBLEU/ROUGE o tren cham tren tap TEST (--gen-every), khong phai val.")
    print("Cac lan cham giua chung dung tap con (--gen-every-max) nen KHONG so duoc")
    print("voi lan cham cuoi tren du 630 MR — chi so duoc theo chieu doc.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs_conv/summary.jsonl"])
