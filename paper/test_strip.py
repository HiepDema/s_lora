#!/usr/bin/env python
"""Kiem strip_comments tren cac ca de sai, truoc khi cho no dung vao bai that."""
import strip_comments as S

CASES = [
    ("abc % ghi chu",             "abc"),
    ("abc% ghi chu",              "abc%"),
    ("abc.%",                     "abc.%"),
    ("   % ca dong la comment",   ""),
    ("100\\% cua model",          "100\\% cua model"),
    ("\\usepackage{a} % note",    "\\usepackage{a}"),
    ("a \\% b % thuc su la note", "a \\% b"),
    ("\\\\% sau hai gach cheo",   "\\\\%"),
    ("khong co gi",               "khong co gi"),
]

ok = True
for src, exp in CASES:
    got = S.strip(src)[0]
    good = got == exp
    ok &= good
    tag = "OK " if good else "SAI"
    extra = "" if good else f"   mong doi {exp!r}"
    print(f"  {tag} {src!r:32} -> {got!r}{extra}")
print("  => " + ("tat ca dung" if ok else "CO LOI, khong duoc chay tren bai that"))
raise SystemExit(0 if ok else 1)
