#!/usr/bin/env python
"""Tong ket phep do A/B cua bench_fast.sh.

Tach ra file rieng thay vi heredoc long trong shell: dau ket thuc heredoc ben
trong se dong luon heredoc ben ngoai.
"""
import glob
import json


def load(pat):
    """Doc file ket qua dau tien khop pat -> (dict, duong dan) hoac (None, ly do)."""
    fs = [f for f in glob.glob(pat) if not f.endswith("summary.json")]
    if not fs:
        return None, f"khong thay {pat}"
    try:
        return json.load(open(fs[0], encoding="utf-8")), fs[0]
    except Exception as e:                                  # noqa: BLE001
        return None, f"{fs[0]}: {e}"


def dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    o, po = load("bench_old/*.json")
    n, pn = load("bench_new/hybrid*.json")
    v, pv = load("bench_vllm/*_vllm.json")

    # Bao THIEU gi truoc. Im lang bo qua la cach loi tron mat.
    for tag, d, p in (("A cau hinh cu", o, po), ("B cau hinh moi", n, pn),
                      ("D cham vLLM", v, pv)):
        if d is None:
            print(f"  THIEU {tag}: {p}")

    print()
    print(f"{'':26}{'CU fp32+ckpt':>15}{'MOI bf16':>12}{'ty le':>10}")
    print("-" * 63)
    rows = [("train it/s", dig(o, "train", "train_it_s"),
             dig(n, "train", "train_it_s"), "cao"),
            ("train (giay)", dig(o, "train", "train_s"),
             dig(n, "train", "train_s"), "thap"),
            ("VRAM dinh (GB)", dig(o, "train", "peak_vram_gb"),
             dig(n, "train", "peak_vram_gb"), "thap"),
            ("val loss", dig(o, "train", "val_loss"),
             dig(n, "train", "val_loss"), None)]
    for name, a, b, better in rows:
        if a is None or b is None:
            print(f"  {name:24}{'?':>15}{'?':>12}")
            continue
        tail = ""
        if better:
            r = (b / a) if better == "cao" else (a / b)
            tail = f"{r:>9.2f}x"
        print(f"  {name:24}{a:>15.3f}{b:>12.3f}{tail}")

    print()
    h, w = dig(n, "after", "gen_s"), dig(v, "gsm8k", "gen_s")
    nh, nw = dig(n, "after", "n"), dig(v, "gsm8k", "n")
    if h and w and nh and nw:
        print(f"  cham {nh} bai, HF generate : {h:>7.0f} giay  ({h/nh:.2f} s/bai)")
        print(f"  cham {nw} bai, vLLM        : {w:>7.0f} giay  ({w/nw:.2f} s/bai)")
        print(f"{'':>32}nhanh hon {h/w:.1f}x")
        if nh != nw:
            print(f"  CANH BAO: so bai khac nhau ({nh} vs {nw}), khong so truc tiep duoc")
    else:
        print("  chua do duoc toc do cham (thieu B hoac D)")

    print()
    ah, aw = dig(n, "after", "acc"), dig(v, "gsm8k", "acc")
    if ah is not None and aw is not None:
        d = aw - ah
        # Hai duong chay cung trong so, cung greedy, cung bo cham -> phai trung
        # nhau gan het. Lech lon nghia la duong vLLM sai o dau do, khong phai
        # chuyen "nhieu ngau nhien".
        flag = "OK" if abs(d) < 1.0 else "LECH NHIEU, kiem lai duong vLLM"
        print(f"  accuracy HF {ah:.2f}%   vLLM {aw:.2f}%   lech {d:+.2f}  -> {flag}")
    else:
        print(f"  accuracy HF {ah}   vLLM {aw}")


if __name__ == "__main__":
    main()
