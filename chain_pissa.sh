#!/bin/bash
# Doi quet lr xong -> chon lr theo VAL LOSS -> chay ca hai run chinh noi tiep.
# Chay duoi setsid nohup nen song sot qua viec dong SSH.
#
#   cd ~/slora && setsid nohup bash chain_pissa.sh > chain.log 2>&1 < /dev/null &
#
# Vi sao xep hang theo val loss chu khong theo accuracy: diem quet chi cham 300
# bai GSM8K, sd ~2.6% — qua nhieu de phan biet hai muc lr.
set -u
cd ~/slora

# --- 1. doi quet xong -----------------------------------------------------
# Loc theo 'finetune_math.py' chu TUYET DOI khong theo ten script nay: shell
# ssh cua chinh minh chua chuoi do trong dong lenh va se tu khop, gay doi mai
# mai. Da dinh bay nay mot lan roi.
echo "[$(date -u +%H:%M:%S)] doi quet lr xong..."
while ps -eo args --no-headers | grep -q "[f]inetune_math.py.*pissa_sweep"; do
  sleep 60
done

if ! grep -q SWEEP_DONE pissa_sweep.log 2>/dev/null; then
  echo "LOI: quet khong ket thuc binh thuong, dung lai de nguoi xem log."
  exit 1
fi

# --- 2. chon lr -----------------------------------------------------------
BEST=$(tr '\r' '\n' < pissa_sweep.log | awk '
  /^=== lr=/       { lr = $0; sub(/^=== lr=/, "", lr); sub(/ ===$/, "", lr) }
  /\[ep [0-9]+\] val loss/ { for (i = 1; i <= NF; i++) if ($i == "loss") v = $(i+1)
                             if (best == "" || v + 0 < best + 0) { best = v; bl = lr } }
  END             { print bl }')

echo "[$(date -u +%H:%M:%S)] val loss tung muc lr:"
tr '\r' '\n' < pissa_sweep.log | grep -E "^=== lr=|\[ep [0-9]+\] val loss"
echo "[$(date -u +%H:%M:%S)] chon lr=$BEST"

if [ -z "$BEST" ]; then
  echo "LOI: khong doc duoc val loss nao tu log. Dung lai."
  exit 1
fi

# --- 3. hai run chinh -----------------------------------------------------
LR="$BEST" bash run_pissa.sh both
echo "CHAIN_DONE"
