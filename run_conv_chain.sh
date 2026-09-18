#!/bin/bash
# Chay noi tiep tren MOT box:  cho r=2 xong -> cham lai o best epoch -> chay r=4.
#
#   bash run_conv_chain.sh rowspace     # box 1
#   bash run_conv_chain.sh lora         # box 2
#
# Dat chay khi r=2 DANG chay: script se doi tien trinh do ket thuc roi lam tiep.
set -u

METHOD="${1:?dung: bash run_conv_chain.sh rowspace hoac lora}"
EPOCHS="${2:-20}"
SEED="${3:-0}"
OUT="runs_conv"
COMMON="--model Qwen/Qwen2.5-1.5B --target-set qwen2_kv --tf32 --seed $SEED"

tag_of() {   # $1 = method, $2 = rank
  case "$1" in
    rowspace) echo "rowspace_r${2}_colperm_s${SEED}" ;;
    lora)     echo "lora_r${2}_na_s${SEED}" ;;
  esac
}

# --- 1. doi run r=2 hien tai ket thuc -----------------------------------------
# Loc theo "finetune_e2e" chu KHONG theo ten script nay, neu khong pgrep se khop
# chinh no va vong lap doi mai mai.
echo "==> doi run r=2 hien tai ket thuc..."
while pgrep -f "finetune_e2e.py" > /dev/null; do sleep 60; done
echo "==> r=2 da xong"

# --- 2. cham lai r=2 o epoch tot nhat -----------------------------------------
T2=$(tag_of "$METHOD" 2)
BEST="$OUT/${T2}_best.pt"
if [ -f "$BEST" ]; then
  echo "==> cham lai $T2 o epoch tot nhat"
  python -u finetune_e2e.py --method "$METHOD" --rank 2 $COMMON \
    --load-ckpt "$BEST" --out-dir "${OUT}_bestep" --no-bench
else
  echo "==> khong thay $BEST, bo qua buoc cham lai"
fi

# --- 3. chay r=4 voi logic moi ------------------------------------------------
# --gen-every 3 : BLEU/ROUGE moi 3 epoch (mo ta duong cong, KHONG dung de chon)
# --eval-both   : cham ca epoch cuoi lan epoch tot nhat khi ket thuc
T4=$(tag_of "$METHOD" 4)
CK4="$OUT/${T4}_resume.pt"
R4=""
[ -f "$CK4" ] && R4="--resume $CK4"
echo "==> chay r=4 ($METHOD), $EPOCHS epoch"
python -u finetune_e2e.py --method "$METHOD" --rank 4 $COMMON \
  --epochs "$EPOCHS" --lr-schedule constant \
  --ckpt-every 1 --gen-every 3 --gen-every-max 200 --eval-both \
  --out-dir "$OUT" $R4

echo "CHAIN_DONE_${METHOD}"
