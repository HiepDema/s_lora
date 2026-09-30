#!/usr/bin/env python
"""Kiem paper truoc khi bien dich: citation, goi latex, file phu thuoc.

    python paper/check_paper.py

Bat nhung loi chi lo ra luc chay pdflatex, ma vong lap do cham hon nhieu.
"""
import io
import os
import re
import sys

TEX = "iclr2027_conference.tex"
BIB = "iclr2027_conference.bib"

# Lenh -> goi phai khai bao. Thieu mot dong usepackage la loi bien dich, ma
# thong bao cua latex ("Undefined control sequence") khong noi ro thieu goi nao.
NEEDS = {
    "multirow": "multirow", "toprule": "booktabs", "midrule": "booktabs",
    "bottomrule": "booktabs", "includegraphics": "graphicx",
    "textcolor": "xcolor", "url": "url", "href": "hyperref",
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    bad = 0

    tex = io.open(TEX, encoding="utf-8").read()
    bib = io.open(BIB, encoding="utf-8").read()

    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    used = set()
    for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", tex):
        used |= {k.strip() for k in m.group(1).split(",")}

    miss = sorted(used - keys)
    print(f"  .bib: {len(keys)} muc | .tex trich dan: {len(used)}")
    if miss:
        print(f"  LOI: trich dan khong co trong .bib -> {miss}")
        bad += 1
    else:
        print("  OK: moi trich dan deu co muc trong .bib")
    unused = sorted(keys - used)
    if unused:
        print(f"  canh bao: {len(unused)} muc khong duoc trich dan -> {unused}")

    declared = set()
    for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        declared |= {p.strip() for p in m.group(1).split(",")}
    for cmd, pkg in NEEDS.items():
        if re.search(r"\\" + cmd + r"[\{\[]", tex) and pkg not in declared:
            print(f"  LOI: dung \\{cmd} nhung chua \\usepackage{{{pkg}}}")
            bad += 1
    if not bad:
        print("  OK: moi lenh deu co goi tuong ung")

    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        cand = [f, f + ".pdf", f + ".png"]
        if not any(os.path.exists(c) for c in cand):
            print(f"  LOI: thieu hinh {f}")
            bad += 1
    # Bo qua \input nam trong \IfFileExists — do la nhung file TUY CHON,
    # thieu cung khong sao vi latex tu nhay qua.
    guarded = set(re.findall(r"\\IfFileExists\{([^}]+)\}", tex))
    for f in re.findall(r"\\input\{([^}]+)\}", tex):
        if f in guarded:
            continue
        if not (os.path.exists(f) or os.path.exists(f + ".tex")):
            print(f"  LOI: thieu \\input{{{f}}}")
            bad += 1

    # File style cua ICLR khong nam trong goi ban tai ve, nhung BAT BUOC co.
    for sty, why in (("iclr2027_conference.sty", "\\usepackage"),
                     ("iclr2027_conference.bst", "\\bibliographystyle")):
        if not os.path.exists(sty):
            print(f"  THIEU {sty} ({why}) — tai ca bo template ICLR ve day")
            bad += 1

    n_todo = len(re.findall(r"\\TODO\{", tex))
    if n_todo:
        print(f"  con {n_todo} \\TODO trong bai — phai xu ly truoc khi nop")

    env = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    end = re.findall(r"\\end\{(\w+\*?)\}", tex)
    for e in set(env) | set(end):
        if env.count(e) != end.count(e):
            print(f"  LOI: moi truong '{e}' lech: {env.count(e)} begin, "
                  f"{end.count(e)} end")
            bad += 1

    print("  => " + ("CO LOI, sua truoc khi bien dich" if bad else "san sang bien dich"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
