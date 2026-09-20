#!/bin/bash
# Mot run cho moi GPU. Dung:  bash run_hybrid.sh {1|2|3|4}
#
# Bon cau hinh phu ba phep so sanh tren tap q,k,v (28 lop, Qwen2.5-1.5B):
#
#   1  hybrid r_kv=4 r_q=1   0.201M  ┐ hang o q dang gia bao nhieu
#   2  hybrid r_kv=4 r_q=2   0.287M  ┘
#   3  LoRA   r_kv=2 r_q=1   0.287M  <- KHOP CHINH XAC voi (2): cung ngan sach,
#                                       cung tap ma tran, khac moi cach phan bo
#   4  LoRA   r_kv=4 r_q=1   0.487M  <- cung HANG voi (1): hybrid re hon 2.43x
#
# Mac 0.201M khong co nghiem nguyen nao cho LoRA nen khong ghep cap duoc.
set -u

N="${1:?dung: bash run_hybrid.sh 1|2|3|4}"
COMMON="--model Qwen/Qwen2.5-1.5B --target-set qwen2_qkv --tf32 --seed 0
        --epochs 5 --best-epoch --eval-both --out-dir runs_hybrid"

case "$N" in
  1) M=hybrid; RKV=4; RQ=1; EXP=200704 ;;
  2) M=hybrid; RKV=4; RQ=2; EXP=286720 ;;
  3) M=lora;   RKV=2; RQ=1; EXP=286720 ;;
  4) M=lora;   RKV=4; RQ=1; EXP=487424 ;;
  *) echo "N phai la 1..4"; exit 1 ;;
esac

echo "==> run $N: $M  r_kv=$RKV  r_q=$RQ  (du kien $EXP tham so = $(echo "scale=3; $EXP/1000000" | bc)M)"
python -u finetune_e2e.py --method "$M" --rank "$RKV" --rank-square "$RQ" $COMMON

echo "HYBRID_DONE_$N"
