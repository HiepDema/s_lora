#!/usr/bin/env python
"""Kiem ban port metamath_eval.py. Khong can torch — chay duoc tren may local."""
import re
import sys

import metamath_eval as MM

# ban sao ham tu viet trong finetune_math.py, chep ra de khoi phai import torch
_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _norm(s):
    s = s.replace(",", "").replace("$", "").rstrip(".").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_pred(text):
    m = re.search(r"[Tt]he answer is:?\s*(.+?)(?:\n|$)", text)
    if m:
        v = _norm(m.group(1))
        if v is not None:
            return v
        nums = _NUM.findall(m.group(1))
        if nums:
            return _norm(nums[-1])
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


CASES = [
    ("is_equiv 1/2 == frac",      MM.is_equiv("1/2", "\\frac{1}{2}"),           True),
    ("is_equiv 0.5 == frac",      MM.is_equiv("0.5", "\\frac{1}{2}"),           True),
    ("is_equiv 2/4 != frac12",    MM.is_equiv("2/4", "\\frac{1}{2}"),           False),
    ("is_equiv dfrac -> frac",    MM.is_equiv("\\dfrac{3}{4}", "\\frac{3}{4}"), True),
    ("is_equiv bo do",            MM.is_equiv("90^\\circ", "90"),               True),
    ("is_equiv bo don vi",        MM.is_equiv("2\\text{ cm}", "2"),             True),
    ("is_equiv sqrt3 -> sqrt{3}", MM.is_equiv("\\sqrt3", "\\sqrt{3}"),          True),
    ("is_equiv bo 'x = '",        MM.is_equiv("x = 5", "5"),                    True),
    ("is_equiv khoang trang",     MM.is_equiv("\\frac{1}{2} ", "\\frac{1}{2}"), True),
    ("gsm8k co nhan",             MM.extract_gsm8k("...The answer is: 72"),     72),
    ("gsm8k KHONG nhan -> None",  MM.extract_gsm8k("ket qua la 72"),            None),
    ("gsm8k dau phay",            MM.extract_gsm8k("The answer is: 1,234"),     1234),
    ("gsm8k thap phan",           MM.extract_gsm8k("The answer is: 3.50"),      3.5),
    ("score_gsm8k gold ####",     MM.score_gsm8k("The answer is: 72", "x\n#### 72"), True),
    ("score_gsm8k sai so",        MM.score_gsm8k("The answer is: 71", "x\n#### 72"), False),
    ("math trich boxed",          MM.math_gold("vay \\boxed{\\frac{1}{2}}"),    "\\frac{1}{2}"),
    ("math boxed long nhau",      MM.math_gold("\\boxed{\\frac{a}{b}}"),        "\\frac{a}{b}"),
    ("score_math dung",           MM.score_math("The answer is: 1/2",
                                                "\\boxed{\\frac{1}{2}}"),       True),
    ("score_math thieu nhan",     MM.score_math("dap an la 1/2",
                                                "\\boxed{\\frac{1}{2}}"),       False),
    ("score_math gold da boc",    MM.score_math("The answer is: 5", "5"),       True),
]


def main():
    bad = 0
    for name, got, want in CASES:
        ok = got == want
        bad += not ok
        line = f"  {'OK ' if ok else 'SAI'}  {name:<26} -> {got!r}"
        print(line if ok else line + f"   (can {want!r})")

    g = "Sau khi tinh ta duoc 72 qua tao."
    print("\nCho khac biet then chot — cau thieu nhan 'The answer is:', dap an dung 72:")
    print(f"   ham tu viet -> {extract_pred(g)!r:<8} => TINH DUNG  (lui ve so cuoi cung)")
    print(f"   MetaMath    -> {MM.extract_gsm8k(g)!r:<8} => TINH SAI   (khong co duong lui)")
    print(f"\nKET LUAN: {'port dat' if bad == 0 else str(bad) + ' case SAI'}")
    return bad


if __name__ == "__main__":
    sys.exit(main())
