#!/usr/bin/env python
"""Doi chieu tung muc trong .bib voi DBLP.

    python paper/verify_bib.py

Cac truong volume/pages/booktitle ban dau duoc dien tu tri nho chu khong phai
tu ban ghi goc, nen phai kiem. DBLP la nguon tot nhat cho paper khoa hoc may
tinh: co API JSON mo, khong can khoa, va ghi dung ten hoi nghi.

Script chi BAO CAO, khong tu sua — mot vai truong hop DBLP tra ve ban arXiv
thay vi ban hoi nghi, va quyet dinh lay ban nao la viec cua nguoi viet.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

BIB = "paper/iclr2027_conference.bib"
API = "https://dblp.org/search/publ/api?q={}&format=json&h=5"


def parse_bib(path):
    txt = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            f[fm.group(1).lower()] = " ".join(fm.group(2).split())
        out.append((kind, key, f))
    return out


def clean(s):
    """Bo dau ngoac bao ve chu hoa va chuan hoa de so khop."""
    s = re.sub(r"[{}\\]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def dblp(title):
    q = urllib.parse.quote(clean(title)[:120])
    try:
        with urllib.request.urlopen(API.format(q), timeout=25) as r:
            d = json.load(r)
    except Exception as e:                                   # noqa: BLE001
        return None, f"loi mang: {type(e).__name__}"
    hits = d.get("result", {}).get("hits", {}).get("hit", [])
    if not hits:
        return None, "DBLP khong tim thay"
    want = clean(title)
    best = None
    for h in hits:
        info = h.get("info", {})
        if clean(info.get("title", "")) == want:
            # Uu tien ban hoi nghi/tap chi hon ban arXiv neu ca hai cung khop
            if best is None or info.get("venue") not in ("CoRR", "arXiv"):
                best = info
    return (best or hits[0].get("info", {})), None


def main():
    entries = parse_bib(BIB)
    print(f"Doi chieu {len(entries)} muc voi DBLP\n")
    issues = 0
    for kind, key, f in entries:
        title = f.get("title", "")
        info, err = dblp(title)
        time.sleep(0.6)                 # lich su voi API cong cong
        if err:
            print(f"  [?] {key:<24} {err}")
            issues += 1
            continue

        probs = []
        y_bib, y_db = f.get("year", ""), str(info.get("year", ""))
        if y_bib and y_db and y_bib != y_db:
            probs.append(f"year {y_bib} -> DBLP {y_db}")

        venue = info.get("venue", "")
        venue = " ".join(venue) if isinstance(venue, list) else venue
        bt = f.get("booktitle", "") or f.get("journal", "")
        if venue and bt:
            vk, bk = clean(venue), clean(bt)
            # ten viet tat cua DBLP vs ten day du trong bib: coi la khop neu
            # chu viet tat xuat hien trong ten day du, hoac nguoc lai
            if vk not in bk and bk not in vk and not any(
                    w in bk for w in vk.split() if len(w) > 3):
                probs.append(f"venue '{bt}' -> DBLP '{venue}'")

        na_bib = len(re.split(r"\s+and\s+", f.get("author", ""))) if f.get("author") else 0
        au = info.get("authors", {}).get("author", [])
        na_db = len(au) if isinstance(au, list) else 1
        if na_bib and na_db and abs(na_bib - na_db) > 0 and "others" not in f.get("author", ""):
            probs.append(f"so tac gia {na_bib} -> DBLP {na_db}")

        if probs:
            print(f"  [!] {key:<24} {'; '.join(probs)}")
            issues += 1
        else:
            print(f"  [ok] {key:<24} {y_db}  {venue}")

    print(f"\n  {issues}/{len(entries)} muc can xem lai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
