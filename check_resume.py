#!/usr/bin/env python
"""Smoke test cho save_resume / load_resume. Chay TRUOC khi dung that.

    python check_resume.py

Kiem 5 thu: buoc/epoch, lich su val, trang thai optimizer (moment cua AdamW),
lich lr, va RNG cua Python — thieu cai cuoi thi thu tu batch sau resume se khac
va run bi ngat khong con trung voi run lien mach.
"""
import os
import random
import tempfile

import torch
import torch.nn as nn

import finetune_e2e as F

torch.manual_seed(0)
m = nn.Linear(4, 3)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
sched = F.make_sched(opt, steps=100, warm=10, kind="constant")

for _ in range(25):                       # chay 25 buoc
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    sched.step()
    opt.zero_grad()

w_at_save = m.weight.detach().clone()
mom_at_save = opt.state_dict()["state"][0]["exp_avg"].clone()
p = os.path.join(tempfile.mkdtemp(), "r.pt")
F.save_resume(p, m, opt, sched, step=25, ep=4, val_hist=[1.2, 1.1], train_hist=[1.0],
              best=(1.1, 1, {n: v.detach().clone()
                             for n, v in F.trainable_state(m).items()}),
              meta={"method": "test", "lr_schedule": "constant", "epochs_planned": 20})
rng_next = random.random()                # gia tri RNG ke tiep, phai khoi phuc dung

for _ in range(5):                        # lam nhieu moi thu
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    sched.step()
    opt.zero_grad()
[random.random() for _ in range(10)]

ck = F.load_resume(p, m, opt, sched, "cpu")

ok = []
ok.append(("step", ck["step"] == 25, f"{ck['step']} (can 25)"))
ok.append(("next_epoch", ck["next_epoch"] == 5, f"{ck['next_epoch']} (can 5)"))
ok.append(("val_hist", ck["val_hist"] == [1.2, 1.1], str(ck["val_hist"])))
ok.append(("trong so", torch.equal(m.weight, w_at_save), "khoi phuc dung"))
ok.append(("AdamW moment",
           torch.allclose(opt.state_dict()["state"][0]["exp_avg"], mom_at_save),
           "khoi phuc dung"))
ok.append(("sched.last_epoch", sched.last_epoch == 25, f"{sched.last_epoch} (can 25)"))
lr = 1e-3 * sched.lr_lambdas[0](sched.last_epoch)
ok.append(("lr", abs(lr - 1e-3) < 1e-12, f"{lr:.3e} (constant sau warmup -> 1e-3)"))
ok.append(("RNG python", abs(random.random() - rng_next) < 1e-15, "trung gia tri ke tiep"))

print()
for name, passed, detail in ok:
    print(f"  {'OK  ' if passed else '*SAI'}  {name:<18} {detail}")
bad = sum(not p for _, p, _ in ok)
print(f"\n{'TAT CA DAT' if bad == 0 else f'{bad} MUC SAI'}\n")

# Khop sai cau hinh thi phai bao loi, khong duoc nap im lang
m2 = nn.Linear(4, 5)
try:
    F.load_resume(p, m2, torch.optim.AdamW(m2.parameters()),
                  F.make_sched(torch.optim.AdamW(m2.parameters()), 100, 10, "constant"),
                  "cpu")
    print("  *SAI  nap duoc checkpoint sai shape — dang le phai bao loi")
except Exception as e:
    print(f"  OK    checkpoint sai cau hinh bi tu choi: {type(e).__name__}")
