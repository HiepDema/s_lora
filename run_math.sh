#!/bin/bash
# MetaMathQA -> GSM8K tren Qwen2.5-7B. Mot run moi GPU:
#
#   bash run_math.sh 1     # hybrid r_kv=128 r_q=32   13.763M
#   bash run_math.sh 2     # LoRA   r_kv=32  r_q=32   13.763M
#
# Hai cau hinh co DUNG 13,762,560 tham so — khop tuyet doi, khong xap xi — nen
# khac nhau duy nhat o cach chia ngan sach giua ma tran vuong (q) va khong vuong
# (k, v). Hybrid mua duoc hang gap 4 lan tren k,v o cung chi phi.
#
# Muc nay la chuan cua literature (LoRA r=32) = 0.181% tham so model. Luu y:
# tren Qwen2.5-1.5B loi the cua S-LoRA chi song duoi 0.016% va bien mat tren do;
# quy sang 7B la ~1.2M. 13.76M cao gap 11 lan nguong ay, nen ket qua HOA nhau la
# kha nang cao — do la thong tin, khong phai that bai.
#
# --grad-ckpt BAT BUOC: seq 512 batch 8 fp32 can ~91 GB neu khong bat, vuot ca
# H100-80. Bat vao con 47 GB.
set -u

N="${1:?dung: bash run_math.sh 1 hoac 2}"
COMMON="--model Qwen/Qwen2.5-7B --target-set qwen2_qkv --tf32 --grad-ckpt
        --seed 0 --max-train 100000 --epochs 2 --max-len 512 --batch 8
        --eval-both --ckpt-every 1 --out-dir runs_math"

case "$N" in
  1) M=hybrid; RKV=128; RQ=32; EXTRA="--eval-before" ;;   # do luon zero-shot
  2) M=lora;   RKV=32;  RQ=32; EXTRA="" ;;
  *) echo "N phai la 1 hoac 2"; exit 1 ;;
esac

echo "==> run $N: $M  r_kv=$RKV  r_q=$RQ  (du kien 13,762,560 tham so)"
python -u finetune_math.py --method "$M" --rank "$RKV" --rank-square "$RQ" \
  $COMMON $EXTRA

echo "MATH_DONE_$N"
