#!/bin/bash
# Chay noi tiep tren MOT GPU:
#   1. doi run 1 (hybrid) dang chay ket thuc
#   2. cham MATH cho checkpoint cua run 1   (run 1 chay truoc khi co --eval-math)
#   3. chay run 2 (LoRA) voi CA GSM8K lan MATH
#
# Dat chay khi run 1 DANG chay: script tu doi.
set -u

COMMON="--model Qwen/Qwen2.5-7B --target-set qwen2_qkv --tf32 --grad-ckpt
        --seed 0 --max-len 512 --batch 8 --out-dir runs_math"
T1=hybrid_r128q32_na_s0

echo "==> doi run 1 ket thuc..."
# Loc theo finetune_math.py chu KHONG theo ten script nay, neu khong pgrep se
# khop chinh no va doi mai mai.
while pgrep -f "finetune_math.py" > /dev/null; do sleep 60; done
echo "==> run 1 xong"

CK="runs_math/${T1}_last.pt"
if [ -f "$CK" ]; then
  echo "==> cham MATH cho run 1 (tu $CK)"
  python -u finetune_math.py --method hybrid --rank 128 --rank-square 32 \
    $COMMON --load-ckpt "$CK" --eval-math --skip-gsm8k --no-bench
else
  echo "==> KHONG thay $CK — bo qua buoc cham MATH cho run 1"
  ls -la runs_math/ 2>/dev/null | tail -5
fi

echo "==> chay run 2 (LoRA r_kv=32 r_q=32), cham ca hai tap"
python -u finetune_math.py --method lora --rank 32 --rank-square 32 \
  $COMMON --max-train 100000 --epochs 2 --eval-both --ckpt-every 1 --eval-math

echo "MATH_CHAIN_DONE"
