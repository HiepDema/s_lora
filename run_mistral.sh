#!/bin/bash
# S-LoRA tren Mistral-7B, MetaMathQA -> GSM8K + MATH.
# Tai lap thiet lap cua PMSS de KHONG phai chay lai baseline nao.
#
#   bash run_mistral.sh sweep   # quet lr truoc (bat buoc, xem ghi chu duoi)
#   bash run_mistral.sh 1       # 87.95M — khop PMSS 87.5M
#   bash run_mistral.sh 2       # 23.07M — thap hon 3.8x
#
# So cua PMSS de doi chieu (Bang 4, Mistral-7B):
#   Full FT  7242M  GSM8K 67.02  MATH 18.60
#   LoRA      168M        67.70       19.68
#   PiSSA     168M        72.86       21.54
#   CURLoRA  87.5M        72.40       20.40
#   PMSS     87.5M        73.31       21.34
#
# Sieu tham so theo Bang 10 cot Mistral-7B: batch 128, cosine, warmup 0.03,
# weight decay 0, ca 7 lop tuyen tinh. Batch 128 = 8 x accum 16 vi VRAM.
#
# HAI CHO LECH SO VOI PMSS, deu phai ghi vao paper:
#  1. lr cua ho la 1e-3 nhung do la lr DA TUNE CHO PMSS. S-LoRA dung co che
#     scale khac (alpha/r = 1, con ho la alpha/max{c,r} = 2). Dung thang 1e-3
#     la lap lai dung sai lam alpha=32 hoi truoc. Vi vay phai quet lr.
#  2. Phu luc A.3 viet "only one epoch" nhung Bang 10 ghi Epochs 3 — paper tu
#     mau thuan. Mac dinh o day la 3 theo Bang 10; doi EP=1 neu muon theo A.3.
set -u

N="${1:?dung: bash run_mistral.sh sweep|1|2}"
EP="${EP:-3}"
LR="${LR:-1e-3}"
COMMON="--model mistralai/Mistral-7B-v0.1 --target-set all7 --method hybrid
        --tf32 --grad-ckpt --seed 0 --max-len 512 --batch 8 --accum 16
        --lr-schedule cosine --warmup-ratio 0.03 --weight-decay 0
        --max-train 100000 --out-dir runs_mistral
        --fac-cache fac_cache/mistral7b"

if [ "$N" = "sweep" ]; then
  # Batch hieu dung 128 nen 8000 mau chi cho 62 buoc optimizer — qua it de
  # phan biet lr. 30000 mau cho ~234 buoc, ~27 phut moi diem.
  # Xep hang theo VAL LOSS chu khong theo accuracy: 300 bai GSM8K co sd ~2.6%,
  # qua nhieu de so ba lr voi nhau.
  for L in 2e-4 5e-4 1e-3; do
    echo "=== lr=$L ==="
    python -u finetune_math.py --rank 61 --rank-square 61 $COMMON \
      --lr "$L" --epochs 1 --max-train 30000 --limit-eval 300 \
      --no-bench --no-save-ckpt --out-dir "sweep_$L"
  done
  echo SWEEP_DONE
  exit 0
fi

case "$N" in
  1) R=61; TAG="87.95M — khop PMSS" ;;
  2) R=16; TAG="23.07M — thap hon 3.8x" ;;
  *) echo "N phai la sweep, 1 hoac 2"; exit 1 ;;
esac

echo "==> run $N: S-LoRA hybrid r=$R rq=$R  ($TAG), lr=$LR, $EP epoch"
python -u finetune_math.py --rank "$R" --rank-square "$R" $COMMON \
  --lr "$LR" --epochs "$EP" --eval-both --eval-math --ckpt-every 1

echo "MISTRAL_DONE_$N"
