#!/usr/bin/env python
"""Bieu do day Qwen2.5 tren E2E NLG, bf16. Doc thang tu JSON ket qua.

    python paper/fig_qwen_scale.py

Xuat paper/fig_qwen_scale.png (cho README) va .pdf (cho paper).

Hai panel vi co hai cau hoi khac nhau:
  trai  — BLEU tuyet doi cua ba cau hinh, de thay muc chat luong
  phai  — khoang cach cung ngan sach, de thay no tien ve 0 va doi dau

Truc x dung thang log: day trai 14 lan (0.5B -> 7B), de tuyen tinh thi ba diem
dau don lai mot cho va doan 3B->7B chiem nua hinh.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Palette da validate (dataviz skill, che do light): slot 1/2/3.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

SIZES = [("0.5B", 0.494, 8), ("1.5B", 1.544, 7), ("3B", 3.086, 9), ("7B", 7.616, 8)]
# Day fp32 chi co ba model: 7B chua bao gio chay o fp32.
FP32_GAP = {"0.5B": 1.87, "1.5B": 1.21, "3B": -0.05}


def bleu(d, tag):
    p = os.path.join(ROOT, d, tag + ".json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)["after"]["bleu"]


def main():
    x, s2, sb, lo, gap = [], [], [], [], []
    for name, b, rb in SIZES:
        d = f"results_qwen/bf16/runs_qwen{name}_bf16"
        try:
            a = bleu(d, "rowspace_r2_colperm_s0")
            c = bleu(d, f"rowspace_r{rb}_colperm_s0")
            l = bleu(d, "lora_r2_na_s0")
        except FileNotFoundError as e:
            sys.exit(f"thieu ket qua: {e.filename}")
        x.append(b); s2.append(a); sb.append(c); lo.append(l); gap.append(c - l)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.patch.set_facecolor("#fcfcfb")

    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.set_xscale("log")
        ax.set_xticks([b for _, b, _ in SIZES])
        ax.set_xticklabels([n for n, _, _ in SIZES])
        ax.minorticks_off()
        ax.set_xlabel("Qwen2.5 model size", color=INK2, fontsize=10)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=0)

    # ---- trai: BLEU tuyet doi
    series = [(sb, BLUE, "S-LoRA, budget-matched", "o"),
              (lo, ORANGE, "LoRA $r$=2", "s"),
              (s2, AQUA, "S-LoRA $r$=2", "^")]
    for y, c, lab, mk in series:
        ax1.plot(x, y, color=c, lw=2.0, marker=mk, ms=8, label=lab,
                 zorder=3, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    # Nhan truc tiep o diem cuoi, khong phai moi diem. Ba gia tri o 7B cach
    # nhau 0.09-0.30 BLEU, tuc ~10px tren truc nay, nen phai so le theo chieu
    # doc neu khong se de len nhau.
    for (y, c, lab, _), dy in zip(series, (-14, 1, 12)):
        ax1.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points",
                     xytext=(10, dy), color=c, fontsize=9, fontweight="bold")
    ax1.set_ylabel("BLEU-4", color=INK2, fontsize=10)
    ax1.set_title("E2E NLG, bfloat16", color=INK, fontsize=11, loc="left", pad=28)
    ax1.set_xlim(0.40, 12.0)
    ax1.legend(frameon=False, fontsize=9, ncol=3, labelcolor=INK2,
               loc="lower left", bbox_to_anchor=(0, 1.01), handlelength=1.6,
               columnspacing=1.4, handletextpad=0.5)

    # ---- phai: khoang cach cung ngan sach
    ax2.axhline(0, color=INK2, lw=1.2, ls="-", zorder=2)
    fg = [FP32_GAP.get(n) for n, _, _ in SIZES]
    fx = [b for (n, b, _), g in zip(SIZES, fg) if g is not None]
    fy = [g for g in fg if g is not None]
    ax2.plot(fx, fy, color=ORANGE, lw=2.0, ls="--", marker="s", ms=8,
             label="float32", zorder=3,
             markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    ax2.plot(x, gap, color=BLUE, lw=2.0, marker="o", ms=8, label="bfloat16",
             zorder=4, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    for xi, yi in zip(x, gap):
        ax2.annotate(f"{yi:+.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 11 if yi > 0 else -18), ha="center",
                     color=BLUE, fontsize=9, fontweight="bold")
    ax2.annotate("S-LoRA ahead", (0.03, 0.30), xycoords="axes fraction",
                 color=INK2, fontsize=9)
    ax2.annotate("LoRA ahead", (0.03, 0.14), xycoords="axes fraction",
                 color=INK2, fontsize=9)
    ax2.set_ylabel("BLEU gap at matched budget", color=INK2, fontsize=10)
    ax2.set_title("Gap at matched budget", color=INK, fontsize=11,
                  loc="left", pad=28)
    ax2.set_xlim(0.40, 12.0)
    ax2.set_ylim(-1.6, 5.0)
    ax2.legend(frameon=False, fontsize=9, ncol=2, labelcolor=INK2,
               loc="lower left", bbox_to_anchor=(0, 1.01), handlelength=1.6,
               columnspacing=1.4, handletextpad=0.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"fig_qwen_scale.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None,
                    facecolor=fig.get_facecolor())
        print(f"  da ghi {out}")


if __name__ == "__main__":
    main()
