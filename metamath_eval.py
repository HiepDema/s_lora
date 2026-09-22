#!/usr/bin/env python
"""Ban port trung thanh bo cham diem CHINH THUC cua MetaMath.

Vi sao can: PMSS phu luc A.3 ghi ro ho dung implementation cua PiSSA, ma PiSSA
dung util.py cua MetaMath (von ke thua tu repo MATH cua Hendrycks). Muon dat so
cua minh canh 73.31 / 23.12 cua PiSSA thi phai cham bang DUNG ham do. Ham
norm_math/math_equal tu viet trong finetune_math.py giu lai lam doi chieu, de
do xem sai lech bao nhieu — chu khong dung de bao cao.

Nguon: MetaMath eval_gsm8k.py, eval_math.py va util.py.
    https://github.com/meta-math/MetaMath

BA CHO KHAC BIET dang ke so voi ban tu viet, deu theo huong CHAT HON:

 1. Khong co 'The answer is: ' -> SAI, khong co duong lui. Ban tu viet lui ve
    "lay so cuoi cung trong bai", tuc la cham diem cho ca nhung cau model khong
    theo dinh dang. Cai do lam diem CAO hon mot cach khong so sanh duoc.
 2. GSM8K so khop bang float() == float(), khong phai |p-g| < 1e-4.
 3. MATH so khop CHUOI sau chuan hoa, khong quy ve so. Nen "0.5" va
    "\\frac{1}{2}" chi bang nhau nho mot luat dac biet viet tay, con
    "\\frac{2}{4}" thi khong.
"""
from __future__ import annotations

import re


# ===================================================================== util.py

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        if string != "{}/{}".format(a, b):
            return string
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string):
    # "\\text{ " chi xuat hien khi mo ta don vi do — bo tat ca tu do ve sau.
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    # " .5" -> " 0.5", "{.5" -> "{0.5"
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    # bo "k = " o dau neu ve trai ngan
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2):
    """So khop CHUOI sau chuan hoa — dung nguyen ban cua MetaMath."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str1 == str2


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, right_brace_idx, num_left_braces_open = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx:right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    left = "\\boxed "
    if s.startswith(left):
        return s[len(left):]
    return None


def math_gold(solution):
    """Dap an chuan cua MATH: noi dung \\boxed cuoi cung trong loi giai mau."""
    return remove_boxed(last_boxed_only_string(solution))


# ============================================================== bo trich dap an

_MARK = "The answer is: "


def extract_gsm8k(completion):
    """eval_gsm8k.py::extract_answer_number.

    Khong co nhan -> None -> tinh la SAI. Khong lui ve "so cuoi cung".
    """
    text = completion.split(_MARK)
    if len(text) <= 1:
        return None
    m = re.search(r"[\-+]?\d*[\.,]?\d+", text[-1].strip())
    if not m:
        return None
    g = m.group().replace(",", "")
    try:
        return round(float(g), 2) if "." in g else int(g)
    except ValueError:
        return None


def extract_math(completion):
    """eval_math.py::process_results — phan trich dap an."""
    parts = completion.split(_MARK)
    if len(parts) <= 1:
        return None
    ans = parts[-1].split(".\n")[0].strip()
    if ans.endswith("."):
        ans = ans[:-1]
    return ans.strip()


# ================================================================== cham diem

def score_gsm8k(gen, gold):
    """gold: chuoi dap an GSM8K day du (co '#### X') hoac so."""
    p = extract_gsm8k(gen)
    if p is None:
        return False
    if isinstance(gold, str):
        m = re.search(r"####\s*(.+)", gold)
        if not m:
            return False
        gold = m.group(1).strip().replace(",", "")
    try:
        return float(p) == float(gold)
    except (TypeError, ValueError):
        return False


def score_math(gen, gold_solution):
    """gold_solution: loi giai mau cua MATH (con nguyen \\boxed)."""
    p = extract_math(gen)
    if p is None:
        return False
    g = math_gold(gold_solution) if "\\boxed" in str(gold_solution) else gold_solution
    return is_equiv(p, g)
