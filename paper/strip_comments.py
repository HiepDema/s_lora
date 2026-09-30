#!/usr/bin/env python
"""Xoa comment khoi file .tex ma KHONG doi ket qua bien dich.

    python paper/strip_comments.py paper/iclr2027_conference.tex

Ba truong hop phai phan biet, neu gop chung lam mot la hong bai:

  1. Dong chi co comment            -> xoa ca dong
  2. "abc  % ghi chu"               -> xoa phan comment. An toan: truoc % da co
                                       khoang trang, ma xuong dong trong latex
                                       cung la mot khoang trang.
  3. "abc% ghi chu"  hoac  "abc%"   -> PHAI giu lai mot dau %. Khong co khoang
                                       trang truoc %, nghia la % dang dung de
                                       NOI DONG; bo han se chen them mot
                                       khoang trang khong mong muon.

Va "\\%" la ky tu phan tram that trong van ban, khong phai comment.
"""
import io
import re
import sys

def find_comment(line):
    """Vi tri dau % mo dau comment, hoac -1. Xu ly \\% dung cach."""
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\":
            i += 2                       # bo qua ky tu duoc escape
            continue
        if c == "%":
            return i
        i += 1
    return -1


def strip(text):
    out, n_whole, n_inline, n_join = [], 0, 0, 0
    for line in text.split("\n"):
        if re.match(r"^\s*%", line):
            n_whole += 1
            continue
        i = find_comment(line)
        if i < 0:
            out.append(line)
            continue
        head, tail = line[:i], line[i + 1:]
        if tail.strip() == "" and (i == 0 or not line[i - 1].isspace()):
            out.append(line)             # % noi dong, giu nguyen
            n_join += 1
            continue
        n_inline += 1
        if head and not head[-1].isspace():
            out.append(head + "%")       # giu % de khong chen khoang trang
        else:
            out.append(head.rstrip())
    # gop cac dong trong lien tiep con lai sau khi xoa comment
    res, blank = [], 0
    for line in out:
        if line.strip() == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        res.append(line)
    return "\n".join(res), n_whole, n_inline, n_join


def main():
    if len(sys.argv) != 2:
        sys.exit("dung: python strip_comments.py <file.tex>")
    p = sys.argv[1]
    src = io.open(p, encoding="utf-8", newline="").read()
    new, a, b, c = strip(src)
    io.open(p, "w", encoding="utf-8", newline="").write(new)
    print(f"  {p}")
    print(f"    xoa {a} dong comment nguyen")
    print(f"    xoa {b} comment cuoi dong")
    print(f"    giu {c} dau % noi dong (xoa se doi khoang trang)")
    print(f"    {len(src.splitlines())} -> {len(new.splitlines())} dong")


if __name__ == "__main__":
    main()
