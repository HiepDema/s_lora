#!/bin/bash
# S-LoRA tren Mistral-7B theo DUNG cau hinh PiSSA (Meng et al., NeurIPS 2024).
#
#   bash run_pissa.sh sweep   # 2 diem lr, 30K mau, xep theo val loss
#   bash run_pissa.sh main    # run chinh: r=64, 100K, 1 epoch
#
# ------------------------------------------------------------------ cau hinh
# PiSSA muc 5 ghi ro: AdamW, batch 128, lr 2e-5, cosine, warmup 0.03, khong
# weight decay, lora_alpha LUON bang lora_r, dropout 0, gan vao MOI lop tuyen
# tinh, Float32 cho ca base model lan adapter, 100K mau, DUNG 1 epoch, loss chi
# tinh tren phan response.
#
# Ba thu minh da khop san, khong phai sua:
#   - float32: finetune_e2e.py nap model o torch.float32 san.
#   - alpha=r: --alpha mac dinh bang rank, nen he so scale = 1 nhu PiSSA.
#   - loss tren response: encode_train() che de bai bang -100.
#
# Viec nay cung go duoc mau thuan trong PMSS: Bang 10 ghi "Epochs 3" nhung phu
# luc A.3 ghi "only one epoch" va dan PiSSA. PiSSA muc 5.1 noi thang 1 epoch,
# nen A.3 dung va Bang 10 sai. Dung 1 epoch.
#
# ------------------------------------------------------------------ chon rank
# Khop SO THAM SO voi PiSSA de so sanh truc tiep. LoRA va PiSSA o muc 168M cua
# PiSSA/PMSS chinh la r=64: 2.621.440 x 64 = 167.77M. S-LoRA ton 1.441.792 moi
# don vi rank, nen r=116 cho 167.25M — lech 0.31%, khop tot hon r=117 (168.69M).
#
# Y nghia: CUNG mot ngan sach 168M, PiSSA va LoRA mua duoc rank 64, S-LoRA mua
# duoc rank 116 — gap 1.81 lan. Chenh lech den tu ty le canh: nam trong bay ma
# tran cua Mistral-7B khong vuong, va S-LoRA tra 2kr thay vi r(n+m).
#
#   moi tang, moi don vi rank    LoRA      S-LoRA
#     q_proj  4096x4096 (vuong)  8192      8192   (vuong -> di duong LoRA)
#     o_proj  4096x4096 (vuong)  8192      8192
#     k_proj  4096x1024          5120      2048   2kr voi k=1024
#     v_proj  4096x1024          5120      2048
#     gate    4096x14336        18432      8192   2kr voi k=4096
#     up      4096x14336        18432      8192
#     down    14336x4096        18432      8192
#                               -----     -----
#                               81920     45056
#
# ------------------------------------------------------------------ doi chieu
# PiSSA Bang 2, Mistral-7B, trung binh 3 run:
#   Full FT  7242M  GSM8K 69.91+-0.25  MATH 18.64+-0.35
#   LoRA(g)   168M        69.50+-0.42       20.08+-0.20
#   LoRA(k)   168M        69.40+-0.25       19.99+-0.44
#   PiSSA     168M        73.31+-0.23       23.12+-0.52
#
# PMSS Bang 4 chay lai cung model, cung du lieu:
#   Full FT         67.02  18.60      LoRA     168M  67.70  19.68
#   PiSSA    168M   72.86  21.54      CURLoRA 87.5M  72.40  20.40
#   PMSS    87.5M   73.31  21.34
#
# CANH BAO khi doc ket qua: hai phong thi nghiem chay CUNG method tren CUNG
# du lieu ma lech 1.80 diem GSM8K (LoRA 69.50 vs 67.70) va 1.86 diem MATH
# (PiSSA 23.12 vs 21.54). Mot run don cua minh phai vuot duoc muc nhieu do thi
# moi co y nghia. Chenh lech duoi ~2 diem GSM8K khong ket luan duoc gi.
set -u

N="${1:?dung: bash run_pissa.sh sweep|main}"
R="${R:-116}"
LR="${LR:-2e-5}"

# batch 128 = 8 x accum 16 vi VRAM; fp32 nen phai grad checkpointing.
COMMON="--model mistralai/Mistral-7B-v0.1 --target-set all7 --method hybrid
        --tf32 --grad-ckpt --seed 0 --max-len 512 --batch 8 --accum 16
        --lr-schedule cosine --warmup-ratio 0.03 --weight-decay 0
        --fac-cache fac_cache/mistral7b"

if [ "$N" = "sweep" ]; then
  # 2e-5 la so cua PiSSA va chuyen sang duoc TRUC TIEP vi alpha/r = 1 o ca hai
  # ben — khac han 1e-3 cua PMSS, von tune cho scale alpha/max{c,r} = 2 cua ho.
  # Van quet vi S-LoRA hoc delta tren THUA SO VUONG chu khong tren W, nen do
  # lon gradient khong nhat thiet giong PiSSA.
  for L in 2e-5 2e-4; do
    echo "=== lr=$L ==="
    python -u finetune_math.py --rank "$R" --rank-square "$R" $COMMON \
      --lr "$L" --epochs 1 --max-train 30000 --limit-eval 300 \
      --no-bench --no-save-ckpt --out-dir "pissa_sweep_$L"
  done
  echo SWEEP_DONE
  exit 0
fi

echo "==> S-LoRA r=$R (167.25M, khop PiSSA 167.77M), lr=$LR, 100K, 1 epoch"
python -u finetune_math.py --rank "$R" --rank-square "$R" $COMMON \
  --lr "$LR" --epochs 1 --max-train 100000 --eval-math \
  --out-dir runs_pissa

echo "PISSA_DONE"
