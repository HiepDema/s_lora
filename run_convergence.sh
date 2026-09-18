#!/bin/bash
# Do HOI TU cua S-LoRA va LoRA tren Qwen2.5-1.5B / E2E NLG.
#
#   bash run_convergence.sh rowspace      # GPU 1
#   bash run_convergence.sh lora          # GPU 2
#
# Vi sao --lr-schedule constant: voi lich linear (mac dinh) lr ve 0 dung o epoch
# cuoi, nen val loss luon giam o do — 22/25 run cu deu co best_epoch = epoch cuoi.
# Do la hieu ung cua lich, khong phai bang chung chua hoi tu. Voi lr khong doi sau
# warmup, val theo tung epoch moi so sanh duoc voi nhau va duong cong moi co nghia.
#
# TU DONG CHAY TIEP: neu da co *_resume.pt thi script noi tiep tu do. Ngat giua
# chung (mat mang, tat may) khong mat qua mot epoch.
set -u

# KHONG dat { } trong thong bao cua ${1:?...}: dau } se dong som phep khai trien
# va phan con lai bi noi vao gia tri (METHOD thanh "rowspace}").
METHOD="${1:?dung: bash run_convergence.sh rowspace hoac lora}"
RANK="${2:-2}"
EPOCHS="${3:-20}"
SEED="${4:-0}"
OUT="runs_conv"

case "$METHOD" in
  rowspace) TAG="rowspace_r${RANK}_colperm_s${SEED}" ;;
  lora)     TAG="lora_r${RANK}_na_s${SEED}" ;;
  *) echo "method phai la rowspace hoac lora"; exit 1 ;;
esac

CK="$OUT/${TAG}_resume.pt"
RESUME=""
if [ -f "$CK" ]; then
  RESUME="--resume $CK"
  echo "==> tim thay $CK, se chay tiep tu do"
else
  echo "==> chua co checkpoint, chay tu dau"
fi

echo "==> $METHOD r=$RANK, $EPOCHS epoch, lr khong doi, seed $SEED"
python -u finetune_e2e.py \
  --method "$METHOD" --rank "$RANK" --seed "$SEED" \
  --model Qwen/Qwen2.5-1.5B --target-set qwen2_kv \
  --epochs "$EPOCHS" --lr-schedule constant --tf32 \
  --ckpt-every 1 --out-dir "$OUT" \
  $RESUME

echo "CONV_DONE_${METHOD}"
