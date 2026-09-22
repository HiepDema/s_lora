#!/bin/bash
# Doi run 1 (S-LoRA r=116) xong -> chay run 2 (LoRA r=64) tren dung pipeline do.
#
#   cd ~/slora && setsid nohup bash chain2.sh > chain2.log 2>&1 < /dev/null &
#
# Vi sao doi run 2 tu S-LoRA r=64 sang LoRA r=64: baseline dang dung la so
# TRICH DAN tu paper khac, ma PiSSA va PMSS bao cao lech nhau 1.80 diem GSM8K
# tren cung method cung du lieu. Tu chay LoRA tren chinh duong ong nay thi phep
# so sanh khong con phu thuoc vao chuyen do nua.
set -u
cd ~/slora

# Loc theo 'finetune_math.py', TUYET DOI khong theo ten script nay: shell ssh
# chua chuoi do trong dong lenh va se tu khop, gay doi mai mai.
echo "[$(date -u +%H:%M:%S)] doi run 1 (S-LoRA r=116) xong..."
while ps -eo args --no-headers | grep -q "[f]inetune_math.py.*runs_pissa_r116"; do
  sleep 60
done

if [ ! -s runs_pissa_r116/*.json ] 2>/dev/null && ! grep -q "MATH accuracy" chain.log 2>/dev/null; then
  echo "CANH BAO: khong thay ket qua cua run 1. Van chay run 2, nguoi xem log sau."
fi

echo "[$(date -u +%H:%M:%S)] run 1 xong, bat dau LoRA r=64"
LR=2e-5 bash run_pissa.sh lora
echo "CHAIN2_DONE"
