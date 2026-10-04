#!/bin/bash
# S-LoRA vs LoRA tren E2E NLG, quet kich thuoc Qwen2.5.
#
#   bash run_qwen_scale.sh 0.5B             # fp32, de so voi day cu
#   DT=bf16 bash run_qwen_scale.sh 1.5B     # bf16, nhanh ~2x, nua VRAM
#
# ------------------------------------------------------------------ thiet ke
# Ba run moi model cho hai phep so sanh:
#
#   S-LoRA r=2  vs  LoRA r=2   -> CUNG RANK,      S-LoRA dung ~4x it tham so
#   S-LoRA r=RB vs  LoRA r=2   -> CUNG NGAN SACH, S-LoRA duoc ~4x rank
#
# RB khac nhau theo model vi ty le tiet kiem la (1+a)/2 va a khac nhau. Ca bon
# deu khop ngan sach cua LoRA r=2 den 0.0%:
#
#   model  a   ty le   LoRA r2    RB   S-LoRA rRB
#   0.5B   7   4.00x   0.098M      8   0.098M
#   1.5B   6   3.50x   0.201M      7   0.201M
#   3B     8   4.50x   0.332M      9   0.332M
#   7B     7   4.00x   0.459M      8   0.459M
#
# Day fp32 cu chay 1.5B voi r=8 chu khong phai r=7, tuc S-LoRA duoc du 14%
# ngan sach va con so +1.21 o do hoi co loi cho no. Day bf16 dung r=7, khop dung.
#
# --------------------------------------------------------------- ve do chinh xac
# DO DUOC tren Qwen2.5-0.5B, giu nguyen moi thu tru dtype:
#
#            fp32 BLEU   bf16 BLEU   phat     thoi gian       VRAM
#   S-LoRA     64.19       63.03    -1.15   61.1 -> 32.1 ph   9.24 -> 4.64 GB
#   LoRA       62.31       60.22    -2.10   60.5 -> 28.3 ph
#
# bf16 nhanh ~2x va ton nua VRAM, nhung KHONG trung lap giua hai phuong phap:
# no phat LoRA nang gan gap doi, lam loi the S-LoRA phong tu +1.87 len +2.82.
# Vi vay KHONG duoc tron fp32 va bf16 trong cung mot bang — moi day phai chay
# het o mot dtype.
#
# Gia thuyet cu cua toi nguoc lai: bf16 hai RIENG S-LoRA, vi chi no phai dung
# lai W qua phan ra (sai so 9.4e-03 so voi ~1e-6 o fp32). Phep do bac bo dieu
# do. Sai so tai dung lon that, nhung ton that chat luong lai roi vao LoRA
# nhieu hon — co che chua ro.
#
# PiSSA muc 5 dung Float32 cho ca base model lan adapter. LoRA (Hu et al.) va
# VeRA (Kopiczko et al.) KHONG neu dtype o bat ky dau; da tim toan van ca hai.
#
# ------------------------------------------------------------------ doi chieu
# Qwen2.5-1.5B fp32 (day cu):
#   S-LoRA r2  0.057M  BLEU 65.22 (sd 0.22, n=2)
#   S-LoRA r8  0.229M       65.45 (sd 0.38, n=3)
#   LoRA   r2  0.201M       64.24 (sd 0.50, n=2)
#
# CANH BAO: voi 1 seed, do lech cua hieu hai lan rut la sqrt(0.22^2+0.50^2) =
# 0.55 BLEU. Chenh tren ~1.5 doc duoc, duoi ~1 thi khong.
set -u
PY="${PY:-$HOME/venv/bin/python}"
DT="${DT:-fp32}"

N="${1:?dung: bash run_qwen_scale.sh 0.5B|1.5B|3B|7B}"
case "$N" in
  0.5B) MODEL=Qwen/Qwen2.5-0.5B; RB=8 ;;
  1.5B) MODEL=Qwen/Qwen2.5-1.5B; RB=7 ;;
  3B)   MODEL=Qwen/Qwen2.5-3B;   RB=9 ;;
  7B)   MODEL=Qwen/Qwen2.5-7B;   RB=8 ;;
  *)    echo "N phai la 0.5B, 1.5B, 3B hoac 7B"; exit 1 ;;
esac

OUT="runs_qwen${N}_${DT}"
COMMON="--model $MODEL --target-set qwen2_kv --epochs 5 --best-epoch --seed 0
        --attn sdpa --eval-batch 32 --model-dtype $DT --out-dir $OUT"

run () {
  echo
  echo "=================================================================="
  echo "==> $N  $DT  $1 r=$2"
  echo "=================================================================="
  "$PY" -u finetune_e2e.py --method "$1" --rank "$2" $COMMON
  echo "RUN_DONE_${N}_${DT}_$1_r$2"
}

run rowspace 2        # S-LoRA, cung rank voi LoRA r=2
run rowspace "$RB"    # S-LoRA, cung ngan sach voi LoRA r=2
run lora 2            # doi chung

echo "ALL_DONE_${N}_${DT}"
