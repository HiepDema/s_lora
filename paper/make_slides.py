#!/usr/bin/env python
"""Dung bo slide S-LoRA (.pptx) tu qwen_all_runs.csv + fig_orientations.pdf.

  python paper/make_slides.py [--csv PATH] [--out PATH]

Moi con so trong deck deu tinh lai tu CSV chu khong go tay, tru phan phuong
phap (lay tu paper/iclr_draft.tex). Chay lai sau moi lan them run la deck tu
cap nhat.

Sinh kem paper/fig_pareto.{pdf,png} — dung duoc luon cho hinh Pareto con
\\TODO trong iclr_draft.tex.
"""
import argparse
import csv
import os
import statistics as st
import subprocess
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "_slide_assets")

# --- bang mau, lay tu qwen_report.html de deck va report nhin cung mot he ------
INK = RGBColor(0x12, 0x19, 0x1D)
MUT = RGBColor(0x54, 0x63, 0x6C)
FAINT = RGBColor(0x7D, 0x8B, 0x94)
RULE = RGBColor(0xDD, 0xE4, 0xE8)
RS = RGBColor(0x0D, 0x6F, 0x68)          # S-LoRA (ten trong code: rowspace)
LO = RGBColor(0x9C, 0x55, 0x18)          # LoRA
BAND = RGBColor(0xF5, 0xF7, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RS_HEX, LO_HEX = "#0d6f68", "#9c5518"

SERIF, SANS, MONO = "Georgia", "Segoe UI", "Consolas"
SW, SH = 13.333, 7.5                      # 16:9


# ============================================================== doc du lieu
def load(csv_path):
    """Gom run theo (method, rank) -> mean/sd BLEU, params. Bo run thieu bleu."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["bleu"]:
                continue
            rows.append(r)

    agg = defaultdict(list)
    for r in rows:
        agg[(r["method"], int(r["rank"]))].append(
            (float(r["bleu"]), float(r["rouge_l"]), int(r["trainable"]),
             float(r["val_loss"]) if r["val_loss"] else None,
             float(r["train_eval_loss"]) if r["train_eval_loss"] else None))

    out = {}
    for key, v in agg.items():
        bleu = [b for b, *_ in v]
        rouge = [g for _, g, *_ in v]
        vloss = [x for *_, x, _ in v if x is not None]
        tloss = [x for *_, x in v if x is not None]
        out[key] = dict(
            n=len(v), params=v[0][2] / 1e6,
            bleu=st.mean(bleu), bleu_sd=st.stdev(bleu) if len(bleu) > 1 else 0.0,
            bleus=bleu,
            rouge=st.mean(rouge), rouge_sd=st.stdev(rouge) if len(rouge) > 1 else 0.0,
            vloss=st.mean(vloss) if vloss else None,
            tloss=st.mean(tloss) if tloss else None)
    return out


def timing(csv_path):
    """Trung vi cac cot do thoi gian, gom theo method."""
    cols = ["train_it_s", "peak_vram_gb", "fwdbwd_ms", "merged_fwd_ms", "setup_s"]
    acc = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for c in cols:
                if r[c]:
                    acc[r["method"]][c].append(float(r[c]))
    return {m: {c: st.median(v) for c, v in d.items() if v} for m, d in acc.items()}


# ============================================================== hinh ve
def pareto_chart(D, path):
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    for meth, color, label, mk in [
        ("rowspace", RS_HEX, "S-LoRA  (train $W_2$ vuông 256×256)", "o"),
        ("lora", LO_HEX, "LoRA  (train thẳng trên $W$)", "s"),
    ]:
        pts = sorted((D[k]["params"], D[k]["bleu"], D[k]["bleu_sd"], k[1])
                     for k in D if k[0] == meth)
        xs = [p for p, *_ in pts]
        ys = [b for _, b, *_ in pts]
        es = [e for *_, e, _ in pts]
        ax.errorbar(xs, ys, yerr=es, color=color, marker=mk, ms=6, lw=2,
                    capsize=3, elinewidth=1.3, label=label, zorder=3)
        for x, y, _, r in pts:
            # lech NGANG: thanh sai so la doan doc, lech doc se de chu len no
            ax.annotate(f"r={r}", (x, y), textcoords="offset points",
                        xytext=(11, 12 if meth == "rowspace" else -12),
                        ha="left", va="center", fontsize=8.5, color=color)

    tft = D.get(("target-ft", 0))
    if tft:
        ax.axhline(tft["bleu"], color="#54636c", ls="--", lw=1.2, zorder=1)
        ax.text(0.024, tft["bleu"] + 0.09, f"target-ft: {tft['params']:.1f} M "
                f"tham số tự do → {tft['bleu']:.2f} BLEU",
                fontsize=9, color="#54636c")

    # nhan manh khoang cach ngan sach o cung muc chat luong
    rs2, lo8 = D[("rowspace", 2)], D[("lora", 8)]
    yarr = 66.24                                    # tren moi diem va thanh sai so
    ax.annotate("", xy=(lo8["params"], yarr), xytext=(rs2["params"], yarr),
                arrowprops=dict(arrowstyle="<->", color="#7d8b94", lw=1.1))
    ax.text((rs2["params"] * lo8["params"]) ** 0.5, yarr + 0.07,
            f"LoRA cần {lo8['params'] / rs2['params']:.0f}× tham số "
            f"để vượt",
            ha="center", va="bottom", fontsize=9, color="#54636c")

    ax.set_xscale("log")
    ax.set_xlabel("Tham số train được (triệu, thang log)", fontsize=10)
    ax.set_ylabel("BLEU", fontsize=10)
    ax.set_ylim(61.6, 66.75)
    ax.set_xlim(0.022, 2.6)
    ax.set_xticks([0.03, 0.06, 0.1, 0.2, 0.4, 0.8, 1.6])
    ax.set_xticklabels(["0.03", "0.06", "0.1", "0.2", "0.4", "0.8", "1.6"])
    ax.minorticks_off()
    ax.grid(True, which="both", axis="both", color="#dde4e8", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#dde4e8")
    ax.tick_params(labelsize=9, colors="#54636c")
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax.set_title("Zero-shot (chưa train) = 33.12 BLEU, nằm ngoài khung",
                 fontsize=9, color="#7d8b94", loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=200, facecolor="white")
    fig.savefig(path + ".pdf")
    plt.close(fig)


def render_math(tex, fs=26, color="#12191d", name="eq"):
    """Ket xuat mot cong thuc mathtext thanh PNG trong suot."""
    os.makedirs(TMP, exist_ok=True)
    p = os.path.join(TMP, f"{name}.png")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=fs, color=color)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return p


def png_size_in(path, dpi=300):
    w, h = Image.open(path).size
    return w / dpi, h / dpi


# ============================================================== helper pptx
def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, runs, size=16, font=SANS, color=INK, bold=False,
         align=PP_ALIGN.LEFT, spacing=1.25, space_after=6):
    """runs: str, hoac list cac doan; moi doan la str hoac list (text, **kw)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        for piece in (para if isinstance(para, list) else [para]):
            s, kw = piece if isinstance(piece, tuple) else (piece, {})
            r = p.add_run()
            r.text = s
            r.font.size = Pt(kw.get("size", size))
            r.font.name = kw.get("font", font)
            r.font.bold = kw.get("bold", bold)
            r.font.color.rgb = kw.get("color", color)
    return tb


def head(slide, eyebrow, title, sub=None):
    rect(slide, 0, 0, SW, 0.055, RS)
    text(slide, 0.75, 0.52, 11.5, 0.3, eyebrow, size=10.5, font=MONO, color=RS, bold=True)
    text(slide, 0.75, 0.85, 11.9, 0.8, title, size=30, font=SERIF, color=INK, spacing=1.05)
    y = 1.72
    if sub:
        text(slide, 0.75, y, 11.9, 0.4, sub, size=13.5, color=MUT)
        y += 0.5
    rect(slide, 0.75, y + 0.06, 11.83, 0.012, RULE)
    return y + 0.34


def table(slide, x, y, w, data, widths, size=12.5, head_size=10.5,
          row_h=0.34, colors=None, aligns=None):
    """data[0] la header. colors: dict {row_index: RGBColor} to mau chu cot 0."""
    rows, cols = len(data), len(data[0])
    shp = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * rows))
    tbl = shp.table
    tbl.first_row, tbl.horz_banding = True, False
    total = sum(widths)
    for i, ww in enumerate(widths):
        tbl.columns[i].width = Emu(int(Inches(w) * ww / total))
    for ri, row in enumerate(data):
        tbl.rows[ri].height = Inches(row_h if ri else row_h * 0.92)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if ri else BAND
            cell.margin_left = cell.margin_right = Inches(0.09)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (aligns or [PP_ALIGN.LEFT] * cols)[ci]
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(head_size if ri == 0 else size)
            r.font.name = MONO if (ri == 0 or ci) else SANS
            r.font.bold = ri == 0
            if ri == 0:
                r.font.color.rgb = FAINT
            elif colors and ri in colors:
                r.font.color.rgb = colors[ri]
            else:
                r.font.color.rgb = INK
    return shp


def bullets(slide, x, y, w, items, size=15.5, gap=0.62, dash_color=RS):
    """items: (dam, phan con lai) hoac chuoi."""
    for i, it in enumerate(items):
        text(slide, x, y + i * gap, 0.3, 0.3, "—", size=size, font=MONO, color=dash_color)
        if isinstance(it, tuple):
            text(slide, x + 0.42, y + i * gap, w - 0.42, gap,
                 [[(it[0], dict(bold=True, color=INK)), (it[1], dict(color=MUT))]],
                 size=size, spacing=1.22)
        else:
            text(slide, x + 0.42, y + i * gap, w - 0.42, gap, it, size=size, color=MUT,
                 spacing=1.22)


def picture(slide, path, x, y, w=None, h=None):
    iw, ih = png_size_in(path) if path.endswith(".png") else (None, None)
    if w is None and h is not None and iw:
        w = h * iw / ih
    if h is None and w is not None and iw:
        h = w * ih / iw
    return slide.shapes.add_picture(path, Inches(x), Inches(y),
                                    Inches(w) if w else None, Inches(h) if h else None)


# ============================================================== dung deck
def build(D, TM, out_path, fig_dir):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)

    rs = lambda r: D[("rowspace", r)]
    lo = lambda r: D[("lora", r)]
    tft, zs = D[("target-ft", 0)], D[("none", 8)]
    ratio = lo(8)["params"] / rs(2)["params"]

    # ---------------------------------------------------------------- 1 title
    s = blank(prs)
    rect(s, 0, 0, SW, SH, BAND)
    rect(s, 0, 0, 0.22, SH, RS)
    text(s, 1.3, 2.05, 11, 0.3, "PARAMETER-EFFICIENT FINE-TUNING · E2E NLG",
         size=12, font=MONO, color=RS, bold=True)
    text(s, 1.3, 2.5, 11, 1.1, "S-LoRA", size=62, font=SERIF, color=INK)
    text(s, 1.3, 3.6, 10.4, 1.2,
         "Viết lại lớp tuyến tính thành một nhân tử vuông — "
         "ràng buộc cập nhật vào row space của trọng số gốc",
         size=20, color=MUT, spacing=1.3)
    rect(s, 1.3, 5.05, 3.0, 0.012, RULE)
    text(s, 1.3, 5.32, 11, 0.9,
         [[("Hiep Nguyen", dict(color=INK, bold=True)),
           ("   ·   Qwen2.5-1.5B   ·   k_proj + v_proj   ·   "
            f"{sum(v['n'] for v in D.values())} run độc lập", dict(color=MUT))]],
         size=14)

    # ---------------------------------------------------------------- 2 van de
    s = blank(prs)
    y = head(s, "VẤN ĐỀ", "Adapter trả tiền cho cả hai chiều của ma trận")
    bullets(s, 0.75, y + 0.15, 11.9, [
        ("LoRA cộng thêm ΔW = BA, tốn r(n+m) tham số — ",
         "tỉ lệ với cả hai chiều của lớp mà nó adapt."),
        ("Kiến trúc hiện đại đầy ma trận lệch. ",
         "Grouped-query attention nén k/v: Qwen2.5-1.5B có k_proj 256×1536, "
         "tỉ lệ khung hình a = 6; Falcon-7B lên tới a = 71."),
        ("Trả tiền cho chiều dài là trả cho capacity lớp không có. ",
         "Với a = 6, ba phần tư ngân sách adapter đổ vào chiều 1536."),
        ("Thu nhỏ đối tượng được adapt thì vướng chuyện khác: ",
         "nó phải còn mang thông tin pretrained, nếu không thì DoRA, PiSSA — "
         "những phương pháp khai thác cấu trúc pretrained — không có gì để bám vào."),
    ], gap=0.86)

    # ---------------------------------------------------------------- 3 y tuong
    s = blank(prs)
    y = head(s, "Ý TƯỞNG", "Không cộng thêm — viết lại lớp",
             "Một phép phân rã chính xác, không xấp xỉ, không mất mát.")
    steps = [("1", "Tách khối", "Cắt W thành khối chữ nhật W₁ và khối vuông W₂ "
                                "(k×k, k = min(n,m))."),
             ("2", "Giải", "Khối chữ nhật được xác định hoàn toàn bởi khối vuông: "
                           "giải W₂·W̄ = W₁ ra W̄ = W₂⁻¹W₁."),
             ("3", "Nhân tử", "Lớp trở thành (ma trận vuông) × (phần cố định có "
                              "khối identity). Đóng băng W̄, chỉ train W₂.")]
    for i, (num, t, d) in enumerate(steps):
        x = 0.75 + i * 4.02
        rect(s, x, y + 0.2, 3.62, 0.05, RS)
        text(s, x, y + 0.45, 3.6, 0.5, num, size=32, font=SERIF, color=RS)
        text(s, x, y + 1.12, 3.6, 0.4, t, size=19, font=SERIF, color=INK)
        text(s, x, y + 1.62, 3.6, 1.8, d, size=13.5, color=MUT, spacing=1.3)
    text(s, 0.75, y + 3.35, 11.9, 0.5,
         [[("Kết quả: ", dict(bold=True, color=INK)),
           ("đối tượng được train là một ma trận vuông k×k vẫn mang giá trị "
            "pretrained — nên cả bộ công cụ adapter vẫn dùng được trên nó, "
            "mà rẻ hơn theo hệ số (1+a)/2.", dict(color=MUT))]], size=14.5)

    # ---------------------------------------------------------------- 4 figure
    s = blank(prs)
    y = head(s, "PHƯƠNG PHÁP", "Hai trường hợp: tách cột và tách hàng")
    picture(s, os.path.join(fig_dir, "fig_orientations.png"), 2.55, y + 0.02, h=5.1)

    # ---------------------------------------------------------------- 5 cong thuc
    s = blank(prs)
    y = head(s, "PHƯƠNG PHÁP", "Công thức",
             "W x = y  với  W: n×m,  x: m×1,  y: n×1.")
    for i, (title_, eqs) in enumerate([
        ("m > n   ·   tách cột,  W₂ vuông n×n", [
            r"$W_0=[\,W_1 \mid W_2\,],\quad x=[\,x_1\,;x_2\,]$",
            r"$W_2\,\overline{W}=W_1 \;\Rightarrow\; \overline{W}=W_2^{-1}W_1$",
            r"$W_0x = W_1x_1 + W_2x_2 = W_2\,(\overline{W}x_1+x_2)$",
            r"$\Rightarrow\; \mathrm{row}(\Delta W)\subseteq\mathrm{row}(W_0)$"]),
        ("m < n   ·   tách hàng,  W₂ vuông m×m", [
            r"$W_0=[\,W_1\,;W_2\,],\quad y=[\,y_1\,;y_2\,]$",
            r"$\overline{W}\,W_2=W_1 \;\Rightarrow\; \overline{W}=W_1W_2^{-1}$",
            r"$W_0x = [\,W_1x\,;W_2x\,] = [\,\overline{W}\,;I\,]\,W_2x$",
            r"$\Rightarrow\; \mathrm{col}(\Delta W)\subseteq\mathrm{col}(W_0)$"]),
    ]):
        x = 0.75 + i * 6.3
        rect(s, x, y + 0.1, 5.75, 0.05, RS if i == 0 else FAINT)
        text(s, x, y + 0.32, 5.7, 0.4, title_, size=15, font=SERIF, color=INK)
        for j, eq in enumerate(eqs):
            col = RS_HEX if j == 3 else "#12191d"
            p = render_math(eq, fs=17, color=col, name=f"eq{i}{j}")
            picture(s, p, x, y + 0.92 + j * 0.86, h=min(0.34, png_size_in(p)[1]))
    text(s, 0.75, y + 3.95, 11.9, 0.55,
         [[("Cả hai: ", dict(bold=True, color=INK)),
           ("đóng băng W̄, chỉ train W₂. Ràng buộc là tính chất của cách tham số hoá, "
            "đúng với mọi ΔW₂ — không phụ thuộc optimizer.", dict(color=MUT))]], size=14.5)

    # ---------------------------------------------------------------- 6 bao toan
    s = blank(prs)
    y = head(s, "PHƯƠNG PHÁP", "Bốn thứ không mất gì",
             "Đây là chỗ khác với các phương pháp cắt cụt theo phổ.")
    cards = [("Chính xác", "Không phải xấp xỉ.\nSai số tương đối 6.6×10⁻¹⁵ ở float64."),
             ("Tham số", "k·max(n,m) = n·m.\nĐúng bằng trọng số gốc, từng byte."),
             ("FLOP", "n(m−n) + n² = nm.\nKhối identity là phép chọn chỉ số, "
                      "không phải matmul."),
             ("Inference", f"Gộp lại thành một ma trận sau train.\nĐo được: "
                           f"{TM['rowspace']['merged_fwd_ms']:.0f} ms so với "
                           f"{TM['lora']['merged_fwd_ms']:.0f} ms — trong nhiễu đo.")]
    for i, (t, d) in enumerate(cards):
        x = 0.75 + i * 3.0
        rect(s, x, y + 0.25, 2.72, 2.3, BAND)
        rect(s, x, y + 0.25, 2.72, 0.05, RS)
        text(s, x + 0.22, y + 0.55, 2.3, 0.4, t, size=17, font=SERIF, color=INK)
        text(s, x + 0.22, y + 1.05, 2.34, 1.5, d, size=12.5, color=MUT, spacing=1.28)
    text(s, 0.75, y + 3.05, 11.9, 0.8,
         [[("Đổi lại: ", dict(bold=True, color=INK)),
           ("cập nhật bị giam trong một không gian con k chiều của không gian "
            "max(n,m) chiều. Một ΔW không liên quan gì tới W₀ chỉ giữ được 1/a = "
            "0.167 chuẩn Frobenius bình phương — nên ràng buộc này không hề tầm thường, "
            "và câu hỏi thực nghiệm là nó lọc nhiễu nhiều hơn hay cắt capacity nhiều hơn.",
            dict(color=MUT))]], size=14.5, spacing=1.3)

    # ---------------------------------------------------------------- 7 ban chat
    s = blank(prs)
    y = head(s, "PHÂN TÍCH", "Ràng buộc này thực chất là gì")
    items = [
        ("S-LoRA rank r = LoRA rank r bị hạn chế. ",
         "Đặt adapter lên W₂ rồi khai triển ra: ΔW = (α/r)·B·(A·P). Vẫn là cập nhật "
         "hạng r, nhưng down-projection A·P bị giam trong row space của W₀."),
        ("Nên baseline sắc nhất là LoRA-FA. ",
         "Câu hỏi rút gọn thành: A lấy từ row space của W₀ có hơn A ngẫu nhiên đóng băng "
         "không? Chưa chạy — đây là baseline còn thiếu quan trọng nhất."),
        ("Về gradient, đây là GD có preconditioner cố định. ",
         "ΔW = −η·(∂L/∂W)·(PᵀP), với PᵀP nửa xác định dương, hạng k, ảnh đúng bằng "
         "row(W₀). Cơ sở trực chuẩn cho PᵀP = VVᵀ (phép chiếu thuần); "
         "cơ sở pivoted-QR vừa chiếu vừa co giãn."),
    ]
    bullets(s, 0.75, y + 0.2, 11.9, items, gap=1.12)
    text(s, 0.75, y + 3.65, 11.9, 0.6,
         [[("Chưa đo: ", dict(bold=True, color=LO)),
           ("ρ = ‖ΔW·VVᵀ‖²/‖ΔW‖² trên một lần fine-tune tự do. "
            "ρ ≈ 0.167 nghĩa là ràng buộc vứt đi phần lớn cập nhật hữu ích; "
            "ρ > 0.9 nghĩa là nó có cơ sở.", dict(color=MUT))]], size=14)

    # ---------------------------------------------------------------- 8 chi phi
    s = blank(prs)
    y = head(s, "KẾ TOÁN THAM SỐ", "Rẻ hơn bao nhiêu, và ở đâu thì không rẻ hơn",
             "k = min(n,m),  a = max(n,m)/k.  Với k_proj/v_proj của Qwen: k = 256, a = 6.")
    table(s, 0.75, y + 0.25, 8.4, [
        ["Adapter", "trên W₀", "trên W₂", "tiết kiệm", "a = 6"],
        ["full fine-tuning", "a·k²", "k²", "a", "6.0×"],
        ["LoRA rank r", "r·k(1+a)", "2rk", "(1+a)/2", "3.5×"],
        ["DoRA rank r", "r·k(1+a)+max(n,m)", "2rk+k", "(1+a)/2", "3.5×"],
        ["LoRA-XS rank r", "r²", "r²", "1", "1.0×"],
        ["PMSS rank r", "r²", "r²", "1", "1.0×"],
    ], widths=[3.0, 3.2, 1.8, 1.8, 1.4], size=12,
        colors={5: LO, 6: LO},
        aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4)
    text(s, 9.5, y + 0.3, 3.1, 3.0,
         [[("Nói thẳng chỗ không áp dụng.", dict(bold=True, color=LO, size=15))],
          [("LoRA-XS và PMSS có chi phí độc lập kích thước lớp — thu nhỏ ma trận "
            "được adapt không giúp gì cho chúng. Trích (1+a)/2 để so với hai "
            "phương pháp đó là sai, và reviewer sẽ bắt ngay.", dict(color=MUT))]],
         size=13, spacing=1.3)

    # ---------------------------------------------------------------- 9 setup
    s = blank(prs)
    y = head(s, "THỰC NGHIỆM", "Thiết lập")
    cfg = [
        ("model", "Qwen/Qwen2.5-1.5B — 28 lớp, d=1536, 12 head, 2 KV head, ffn=8960"),
        ("targets", "self_attn.k_proj, self_attn.v_proj — W là 256×1536, a = 6"),
        ("dữ liệu", "E2E NLG — 42,061 train / 4,672 val / 630 MR test"),
        ("train", "5 epoch, batch 8, AdamW lr 2e-4, linear + 500 warmup, "
                  "label smoothing 0.1, weight decay 0.01"),
        ("chọn epoch", "theo val loss, không smoothing"),
        ("sinh câu", "beam 10, length penalty 0.9, no_repeat_ngram 4"),
        ("môi trường", "NVIDIA A10 · torch 2.7.0 · transformers 5.15.1 · fp32 + TF32"),
    ]
    for i, (k, v) in enumerate(cfg):
        yy = y + 0.25 + i * 0.52
        text(s, 0.75, yy, 1.7, 0.35, k, size=12.5, font=MONO, color=RS)
        text(s, 2.6, yy, 9.9, 0.45, v, size=13.5, color=INK)
        rect(s, 0.75, yy + 0.42, 11.83, 0.008, RULE)
    text(s, 0.75, y + 4.15, 11.9, 0.6,
         [[("Tái lập được từng bit. ", dict(bold=True, color=INK)),
           ("Chạy lại cùng seed trên cùng loại GPU cho kết quả giống hệt — "
            "đã kiểm chứng bằng hai lần chạy độc lập, ba máy A10 chung driver "
            "và phiên bản thư viện đã ghim.", dict(color=MUT))]], size=14)

    # ---------------------------------------------------------------- 10 pareto
    s = blank(prs)
    y = head(s, "KẾT QUẢ", "Đường Pareto: chất lượng theo ngân sách")
    picture(s, os.path.join(fig_dir, "fig_pareto.png"), 1.55, y + 0.1, h=4.55)
    text(s, 0.75, y + 4.75, 11.9, 0.5,
         [[(f"S-LoRA chạm {rs(2)['bleu']:.2f} BLEU ở {rs(2)['params']:.3f} M. "
            f"LoRA cần {lo(8)['params']:.3f} M — nhiều hơn {ratio:.0f} lần — "
            f"mới vượt qua ({lo(8)['bleu']:.2f}). ", dict(bold=True, color=INK)),
           ("Hai đường hội tụ quanh 65.7 khi ngân sách đủ lớn.", dict(color=MUT))]],
         size=14.5)

    # ---------------------------------------------------------------- 11 bang rank
    s = blank(prs)
    y = head(s, "KẾT QUẢ", "Chất lượng theo rank", "Trung bình ± độ lệch chuẩn qua các seed.")
    rows = [["method", "r", "params (M)", "n", "BLEU", "ROUGE-L", "val loss"]]
    colors = {}
    for meth, tag, col in [("rowspace", "S-LoRA", RS), ("lora", "LoRA", LO)]:
        for r in (1, 2, 4, 8, 16):
            d = D[(meth, r)]
            rows.append([tag, r, f"{d['params']:.3f}", d["n"],
                         f"{d['bleu']:.2f} ± {d['bleu_sd']:.2f}",
                         f"{d['rouge']:.2f} ± {d['rouge_sd']:.2f}",
                         f"{d['vloss']:.4f}"])
            colors[len(rows) - 1] = col
    table(s, 1.5, y + 0.2, 10.3, rows, widths=[1.5, 0.6, 1.5, 0.6, 2.0, 2.0, 1.5],
          size=11.5, row_h=0.375,
          colors=colors, aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 6)

    # ---------------------------------------------------------------- 12 cung ngan sach
    s = blank(prs)
    y = head(s, "KẾT QUẢ", "So ở cùng ngân sách, không phải cùng rank",
             "Ở cùng rank, LoRA được nhiều hơn 3.5× tham số — so như vậy là so sai.")
    pairs = [(2, 1), (4, 1), (8, 2), (16, 4)]
    rows = [["S-LoRA", "BLEU", "LoRA", "BLEU", "chênh", "ý nghĩa"]]
    for a, b in pairs:
        da, db = rs(a), lo(b)
        d = da["bleu"] - db["bleu"]
        sd = max((da["bleu_sd"] ** 2 + db["bleu_sd"] ** 2) ** 0.5, 1e-9)
        rows.append([f"r={a}  {da['params']:.3f}M", f"{da['bleu']:.2f}",
                     f"r={b}  {db['params']:.3f}M", f"{db['bleu']:.2f}",
                     f"+{d:.2f}", f"{d / sd:.1f}σ"])
    table(s, 1.5, y + 0.3, 10.3, rows, widths=[2.4, 1.3, 2.4, 1.3, 1.3, 1.3],
          size=13, row_h=0.46, aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT] * 3)
    text(s, 1.5, y + 3.0, 10.3, 0.9,
         [[("Lợi thế giảm dần theo ngân sách. ", dict(bold=True, color=INK)),
           ("Từ 4.0σ ở 0.1 M xuống 1.3σ ở 0.4 M, rồi biến mất. Ở rank 1 thì "
            "S-LoRA còn thua: 62.45 so với 62.78 — ít hơn 3.5× tham số nhưng "
            "một hướng duy nhất là quá ít để làm việc. Đây là công cụ cho chế độ "
            "ngân sách thấp, không phải hàng thay thế phổ quát.", dict(color=MUT))]],
         size=14, spacing=1.3)

    # ---------------------------------------------------------------- 13 tran duoi/tren
    s = blank(prs)
    y = head(s, "KẾT QUẢ", "Sàn dưới và trần trên", "Cái gọi là trần trên lại kém hơn.")
    table(s, 1.5, y + 0.3, 10.3, [
        ["cấu hình", "params (M)", "BLEU", "ROUGE-L", "ghi chú"],
        ["zero-shot", "0.000", f"{zs['bleu']:.2f}", f"{zs['rouge']:.2f}",
         "model gốc, chưa train"],
        ["S-LoRA r=2", f"{rs(2)['params']:.3f}", f"{rs(2)['bleu']:.2f}",
         f"{rs(2)['rouge']:.2f}", "điểm vận hành rẻ nhất còn giữ chất lượng"],
        ["target-ft", f"{tft['params']:.3f}", f"{tft['bleu']:.2f}",
         f"{tft['rouge']:.2f}", "train tự do chính k_proj + v_proj"],
    ], widths=[2.0, 1.5, 1.2, 1.4, 4.2], size=13, row_h=0.48,
        colors={2: RS}, aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 3 + [PP_ALIGN.LEFT])
    text(s, 1.5, y + 2.6, 10.3, 1.4,
         [[("target-ft train tự do 22.0 M tham số nhưng chỉ được 64.26 BLEU — "
            "thấp hơn S-LoRA r=2 với 0.057 M, tức ít hơn 384 lần.",
            dict(bold=True, color=INK))],
          [("Nó có val loss tốt nhất bảng (1.1107) nhưng overfit rõ: khoảng cách "
            "train/val nới từ +0.073 lên +0.277 và val chạm đáy ngay epoch 2. "
            "Lưu ý lr của nó là 5e-5 và chưa được tune — đừng bán con số này "
            "mạnh hơn mức nó chịu được.", dict(color=MUT))]], size=14, spacing=1.3)

    # ---------------------------------------------------------------- 14 chi phi thuc te
    s = blank(prs)
    y = head(s, "CHI PHÍ", "Train chậm hơn 2%, inference bằng nhau",
             "Độ trễ đo trên batch cố định 8×128, trung vị 20 lần, có sync CUDA.")
    a, b = TM["rowspace"], TM["lora"]
    table(s, 1.5, y + 0.3, 10.3, [
        ["", "S-LoRA", "LoRA", "chênh"],
        ["phân rã một lần (s)", f"{a['setup_s']:.1f}", f"{b['setup_s']:.1f}",
         "một lần, lúc dựng model"],
        ["train (it/s)", f"{a['train_it_s']:.2f}", f"{b['train_it_s']:.2f}",
         f"−{(1 - a['train_it_s'] / b['train_it_s']) * 100:.0f}%"],
        ["fwd+bwd (ms)", f"{a['fwdbwd_ms']:.0f}", f"{b['fwdbwd_ms']:.0f}",
         f"+{(a['fwdbwd_ms'] / b['fwdbwd_ms'] - 1) * 100:.0f}%"],
        ["VRAM đỉnh (GB)", f"{a['peak_vram_gb']:.2f}", f"{b['peak_vram_gb']:.2f}", "−0.9%"],
        ["fwd sau khi gộp (ms)", f"{a['merged_fwd_ms']:.0f}", f"{b['merged_fwd_ms']:.0f}",
         "trong nhiễu đo"],
    ], widths=[3.2, 1.6, 1.6, 3.9], size=13, row_h=0.46,
        colors={6: RS}, aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT])
    text(s, 1.5, y + 3.35, 10.3, 0.8,
         [[("Sau khi gộp, hai bên bằng nhau. ", dict(bold=True, color=INK)),
           ("Chi phí duy nhất của S-LoRA nằm ở lúc train, cộng một lần phân rã "
            "khoảng 10 giây. Deploy không phải trả gì.", dict(color=MUT))]], size=14.5)

    # ---------------------------------------------------------------- 15 gioi han
    s = blank(prs)
    y = head(s, "TRUNG THỰC", "Những gì còn thiếu")
    bullets(s, 0.75, y + 0.15, 11.9, [
        ("Baseline PMSS — chưa chạy. ",
         "Paper gần nhất: cũng pivoted QR chọn hàng/cột của W₀, cũng ràng buộc vào "
         "không gian con sinh bởi chúng. Khác biệt là họ ép cả hai phía, ta ép một phía. "
         "Không có baseline này thì không qua được vòng novelty."),
        ("Một model, một benchmark. ", "Chỉ E2E NLG trên Qwen2.5-1.5B. Cần DART hoặc WebNLG."),
        ("Một hướng ma trận. ",
         "k_proj/v_proj đều béo ngang. Nhánh cao dọc (gate_proj, up_proj) chưa kiểm chứng."),
        ("Phủ 1.4% model. ",
         "k_proj+v_proj chỉ chiếm 22.0 M trên 1,543.7 M trọng số. Ma trận vuông "
         "(q_proj, o_proj) không phân rã được."),
        ("BLEU tự implement, n=2 ở r=1, 2, 16. ",
         "Số cho paper phải chạy e2e-metrics chính thức; chỉ r=4 và r=8 có đủ 3 seed."),
    ], gap=0.83, dash_color=LO)

    # ---------------------------------------------------------------- 16 ket luan
    s = blank(prs)
    rect(s, 0, 0, SW, SH, BAND)
    rect(s, 0, 0, 0.22, SH, RS)
    text(s, 1.3, 0.85, 11, 0.3, "KẾT LUẬN", size=12, font=MONO, color=RS, bold=True)
    text(s, 1.3, 1.2, 11, 0.7, "Tóm lại", size=36, font=SERIF, color=INK)
    bullets(s, 1.3, 2.2, 10.8, [
        ("Phân rã chính xác, bảo toàn tham số và FLOP, gộp lại được. ",
         "Đối tượng train là ma trận vuông k×k vẫn mang giá trị pretrained."),
        (f"Ở ngân sách thấp, ràng buộc hoạt động như regularizer. "
         f"{rs(2)['bleu']:.2f} BLEU với {rs(2)['params']:.3f} M — ",
         f"vượt cả train tự do 22.0 M ({tft['bleu']:.2f}), và LoRA cần "
         f"{ratio:.0f}× tham số mới theo kịp."),
        ("Lợi thế biến mất ở ngân sách lớn và đảo chiều ở rank 1. ",
         "Trình bày nó đúng như vậy: một công cụ cho chế độ ngân sách thấp."),
    ], gap=1.0)
    rect(s, 1.3, 5.5, 3.0, 0.012, RULE)
    text(s, 1.3, 5.75, 11, 1.2,
         [[("Bước tiếp theo: ", dict(bold=True, color=INK)),
           ("chạy PMSS và LoRA-FA · đo ρ trên target-ft · thêm DART/WebNLG · "
            "kiểm chứng nhánh cao dọc · chốt lại tên (S-LoRA đã bị Sheng et al., "
            "MLSys 2024 dùng cho hệ phục vụ adapter).", dict(color=MUT))]],
         size=14, spacing=1.3)

    prs.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"C:\Users\User\Downloads\ll\_report\qwen_all_runs.csv")
    ap.add_argument("--out", default=os.path.join(HERE, "S-LoRA_slides.pptx"))
    a = ap.parse_args()

    D, TM = load(a.csv), timing(a.csv)
    pareto_chart(D, os.path.join(HERE, "fig_pareto"))
    if not os.path.exists(os.path.join(HERE, "fig_orientations.png")):
        subprocess.run([sys.executable, os.path.join(HERE, "fig_orientations.py")],
                       check=True)
    p = build(D, TM, a.out, HERE)
    n = sum(v["n"] for v in D.values())
    print(f"wrote {p}  ({n} run, {len(D)} cau hinh)")


if __name__ == "__main__":
    main()
