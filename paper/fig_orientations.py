#!/usr/bin/env python
"""Sinh Figure 1 cua paper: hai truong hop tach khoi cua mot trong so khong vuong.

  python paper/fig_orientations.py            -> paper/fig_orientations.{pdf,png}
  python paper/fig_orientations.py --check    -> in ra cac nhan chong nhau

Ve theo dung cach dan xuat viet tay:  W x = y,  W: n x m,  x: m x 1,  y: n x 1.

  m > n : tach COT   W = [W1 | W2],  W2 vuong n x n,  giai W2 Wbar = W1
          -> W x = W1 x1 + W2 x2 = W2 (Wbar x1 + x2)
  m < n : tach HANG  W = [W1 ; W2],  W2 vuong m x m,  giai Wbar W2 = W1
          -> W x = [W1 x ; W2 x] = [Wbar ; I] W2 x

Ca hai: dong bang Wbar, chi train W2. Trong code Wbar chinh la `X` cua
rowspace_peft.py va W2 la `W2` (tuc `C` trong RowSpaceLinear).

Xanh = W2 (thu duy nhat duoc train), xam = dong bang, net dut = khoi identity.
Ti le cac hinh chu nhat la ti le THAT cua lop duoc dan ten.

Khung ve co dung bang textwidth cua ICLR (5.5in) va fontsize la co chu that
tren giay, nen includegraphics[width=textwidth] khong thu nho chu. Doi figsize
thi phai doi het fontsize theo cung ti le. Sua toa do xong thi chay --check:
no do bbox that cua tung nhan chu khong uoc luong bang mat.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

TRAIN = dict(facecolor="#dbeafe", edgecolor="#1d4ed8", linewidth=1.2)
FROZEN = dict(facecolor="#e8eaed", edgecolor="#6b7280", linewidth=0.9)
IDENT = dict(facecolor="none", edgecolor="#6b7280", linewidth=0.9, linestyle=(0, (2.5, 1.8)))
VEC = dict(facecolor="#f8fafc", edgecolor="#6b7280", linewidth=0.9)
INK, DIM, ACC = "#111827", "#4b5563", "#1d4ed8"

W_IN, XLIM, YLIM = 5.6, 64.0, 52.0            # 1 don vi ~ 6.3pt tren giay
TEXTS = []


def T(x, y, s, fs=8.5, c=INK, ha="left", va="baseline", **kw):
    TEXTS.append((ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, **kw), s))


def box(x, y, w, h, style, label=None, fs=8.5):
    ax.add_patch(Rectangle((x, y), w, h, **style, zorder=2))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)


fig, ax = plt.subplots(figsize=(W_IN, W_IN * YLIM / XLIM))
ax.set_xlim(0, XLIM)
ax.set_ylim(0, YLIM)
ax.set_aspect("equal")
ax.axis("off")

U = 1.8                                       # 1 don vi = k = min(n, m)
VW = 1.8                                      # be rong cot vector

# ======================================================= A.  m > n  (W beo ngang)
T(1.5, 49.5, "A", fs=10.5, c=ACC, fontweight="bold")
T(4.2, 49.5, r"$m > n$:  split the columns,  $W_2$ is $n \times n$", fs=9.5)
T(4.2, 47.3, r"k_proj / v_proj of Qwen2.5-1.5B:  $n = 256$,  $m = 1536$", fs=8, c=DIM)

cy = 39.0
T(13.8, cy + 6.4, "block split", fs=7.5, ha="center", c=DIM, style="italic")
box(3.0, cy - 0.9, 5 * U, U, FROZEN, r"$W_1$", fs=7)
box(3.0 + 5 * U, cy - 0.9, U, U, TRAIN, r"$W_2$", fs=6.5)
T(7.5, cy - 2.6, r"$m - n$", fs=7, ha="center", c=DIM)
T(12.9, cy - 2.6, r"$n$", fs=7, ha="center", c=DIM)
T(2.2, cy, r"$n$", fs=7, ha="right", va="center", rotation=90, c=DIM)

T(15.2, cy, r"$\cdot$", fs=10, ha="center", va="center")
box(16.6, cy - 3.6, VW, 5 * U, VEC, r"$x_1$", fs=7)
box(16.6, cy - 5.4, VW, U, VEC, r"$x_2$", fs=7)
T(19.4, cy + 0.9, r"$m - n$", fs=7, ha="left", va="center", rotation=90, c=DIM)
T(19.4, cy - 4.5, r"$n$", fs=7, ha="left", va="center", rotation=90, c=DIM)

T(22.6, cy, r"$=$", fs=10, ha="center", va="center")
box(24.0, cy - 0.9, VW, U, VEC, r"$y$", fs=7)

T(38.2, cy + 6.4, "factorization", fs=7.5, ha="center", c=DIM, style="italic")
T(28.0, cy, r"$W_0 \;=$", fs=9, va="center")
box(33.0, cy - 0.9, U, U, TRAIN, r"$W_2$", fs=6.5)
T(36.2, cy, r"$\cdot$", fs=10, ha="center", va="center")
box(37.6, cy - 0.9, 5 * U, U, FROZEN, r"$\overline{W}$", fs=7)
box(37.6 + 5 * U, cy - 0.9, U, U, IDENT, r"$I$", fs=6.5)

T(3.0, 31.8, r"solve  $W_2\,\overline{W} = W_1 \;\Rightarrow\; "
             r"\overline{W} = W_2^{-1}W_1$", fs=9)
T(3.0, 28.8, r"$W_0x = W_1x_1 + W_2x_2 = W_2\,(\overline{W}x_1 + x_2)$", fs=9)
T(3.0, 25.8, r"freeze $\overline{W}$, train $W_2$:  $\Delta W = \Delta W_2\,"
             r"[\,\overline{W} \mid I\,] \Rightarrow \mathrm{row}(\Delta W) "
             r"\subseteq \mathrm{row}(W_0)$", fs=8.5, c=ACC)

ax.plot([1.5, 62.5], [24.2, 24.2], color="#d1d5db", lw=0.7)

# ======================================================== B.  m < n  (W cao doc)
T(1.5, 22.6, "B", fs=10.5, c=ACC, fontweight="bold")
T(4.2, 22.6, r"$m < n$:  split the rows,  $W_2$ is $m \times m$", fs=9.5)
T(4.2, 20.4, r"gate_proj / up_proj of Qwen2.5-1.5B:  $n = 8960$,  $m = 1536$",
  fs=8, c=DIM)

cy = 12.4
T(8.7, cy + 6.0, "block split", fs=7.5, ha="center", c=DIM, style="italic")
box(3.0, cy - 3.6, U, 5 * U, FROZEN, r"$W_1$", fs=7)
box(3.0, cy - 5.4, U, U, TRAIN, r"$W_2$", fs=6.5)
T(2.2, cy + 0.9, r"$n - m$", fs=7, ha="right", va="center", rotation=90, c=DIM)
T(2.2, cy - 4.5, r"$m$", fs=7, ha="right", va="center", rotation=90, c=DIM)
T(3.9, cy - 7.1, r"$m$", fs=7, ha="center", c=DIM)

T(6.4, cy, r"$\cdot$", fs=10, ha="center", va="center")
box(7.8, cy - 0.9, VW, U, VEC, r"$x$", fs=7)
T(11.2, cy, r"$=$", fs=10, ha="center", va="center")
box(12.6, cy - 3.6, VW, 5 * U, VEC, r"$y_1$", fs=7)
box(12.6, cy - 5.4, VW, U, VEC, r"$y_2$", fs=7)

T(24.3, cy + 6.0, "factorization", fs=7.5, ha="center", c=DIM, style="italic")
T(17.0, cy, r"$W_0 \;=$", fs=9, va="center")
box(23.0, cy - 3.6, U, 5 * U, FROZEN, r"$\overline{W}$", fs=7)
box(23.0, cy - 5.4, U, U, IDENT, r"$I$", fs=6.5)
T(26.4, cy, r"$\cdot$", fs=10, ha="center", va="center")
box(27.8, cy - 0.9, U, U, TRAIN, r"$W_2$", fs=6.5)

T(33.0, cy + 4.0, r"solve  $\overline{W}\,W_2 = W_1$", fs=9)
T(33.0, cy + 1.2, r"$\Rightarrow\; \overline{W} = W_1W_2^{-1}$", fs=9)
T(33.0, cy - 1.6, r"$W_0x = [\,W_1x \,;\, W_2x\,]$", fs=9)
T(33.0, cy - 4.4, r"$\quad\;\; = [\,\overline{W} \,;\, I\,]\,W_2x$", fs=9)

T(3.0, 2.9, r"freeze $\overline{W}$, train $W_2$:  $\Delta W = "
            r"[\,\overline{W} \,;\, I\,]\,\Delta W_2 \Rightarrow "
            r"\mathrm{col}(\Delta W) \subseteq \mathrm{col}(W_0)$", fs=8.5, c=ACC)

# ==================================================================== legend
for x, style, txt in [
    (3.0, TRAIN, r"$W_2$: trainable"),
    (24.0, FROZEN, r"$W_1$, $\overline{W}$: frozen"),
    (46.0, IDENT, r"$I$: index selection"),
]:
    box(x, 0.4, 2.0, 1.5, style)
    T(x + 2.8, 1.15, txt, fs=7.5, va="center", c=DIM)

# ==================================================================== va cham
if "--check" in sys.argv:
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bbs = [(t.get_window_extent(fig.canvas.get_renderer()).transformed(inv), s)
           for t, s in TEXTS]
    hits = 0
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            a, b = bbs[i][0], bbs[j][0]
            if a.overlaps(b):
                hits += 1
                print(f"OVERLAP  {bbs[i][1][:42]!r:46} x[{a.x0:.1f},{a.x1:.1f}] "
                      f"y[{a.y0:.1f},{a.y1:.1f}]\n         {bbs[j][1][:42]!r:46} "
                      f"x[{b.x0:.1f},{b.x1:.1f}] y[{b.y0:.1f},{b.y1:.1f}]")
    out_of = [(bb, s) for bb, s in bbs if bb.x1 > XLIM - 0.5 or bb.x0 < 0.5]
    for bb, s in out_of:
        print(f"OUT OF FRAME  {s[:42]!r}  x[{bb.x0:.1f},{bb.x1:.1f}]")
    print(f"{hits} overlap(s), {len(out_of)} out of frame, {len(bbs)} labels")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_orientations")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", out + ".pdf", "and", out + ".png")
