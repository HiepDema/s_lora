#!/usr/bin/env python
"""So sanh duong cong hoi tu cua cac run trong runs_conv/.

    python plot_convergence.py runs_conv/summary.jsonl
    python plot_convergence.py runs_conv/*_resume.pt     # doc duoc ca khi dang chay

Doc duoc checkpoint dang chay nen khong phai doi run ket thuc moi xem duoc.
"""
import json
import sys
from pathlib import Path


def from_summary(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append((f"{d['method']} r={d['rank']}", d.get("val_hist", []),
                    d.get("train_eval_hist", []), d.get("bleu")))
    return out


def from_ckpt(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck.get("meta", {})
    return [(f"{m.get('method','?')} r={m.get('rank','?')}",
             ck.get("val_hist", []), ck.get("train_hist", []), None)]


def main(paths):
    runs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"bo qua (khong thay): {p}")
            continue
        runs += from_ckpt(p) if p.suffix == ".pt" else from_summary(p)
    if not runs:
        sys.exit("khong doc duoc run nao")

    n = max(len(v) for _, v, _, _ in runs)
    print(f"\n{'epoch':>6}" + "".join(f"{name:>22}" for name, _, _, _ in runs))
    print("-" * (6 + 22 * len(runs)))
    for e in range(n):
        row = f"{e:>6}"
        for _, v, t, _ in runs:
            if e < len(v):
                gap = f" ({v[e]-t[e]:+.3f})" if e < len(t) else ""
                row += f"{v[e]:>15.4f}{gap:>7}"
            else:
                row += " " * 22
        print(row)

    print(f"\n{'run':<22}{'best val':>10}{'tai epoch':>11}{'epoch cuoi':>12}"
          f"{'con giam?':>11}{'BLEU':>8}")
    print("-" * 74)
    for name, v, _, bleu in runs:
        if not v:
            continue
        b = min(v)
        be = v.index(b)
        # Neu best roi vao epoch cuoi thi van con du dia -> chua cham day
        still = "CON" if be == len(v) - 1 else f"het tu ep {be}"
        print(f"{name:<22}{b:>10.4f}{be:>11}{len(v)-1:>12}{still:>11}"
              f"{(f'{bleu:.2f}' if bleu else '-'):>8}")

    print("\nDoc the nao: voi lr khong doi, 'best val' roi vao epoch cuoi nghia la")
    print("van con cai thien duoc; roi vao giua nghia la da cham day va bat dau overfit.")
    print("So trong ngoac la val - train (khoang cach tong quat hoa).")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs_conv/summary.jsonl"])
