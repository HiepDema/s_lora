#!/bin/bash
# DROP tren LLaMA-3.1-8B, setup cua QuanTA/PMSS (q_proj, v_proj). Moi lan mot run:
#
#   bash run_drop5.sh lora           # LoRA  r=32                 13,631,488
#   bash run_drop5.sh dora           # DoRA  r=32 (+ magnitude)   13,795,328
#   bash run_drop5.sh pissa          # PiSSA r=32                 13,631,488
#   bash run_drop5.sh slora          # S-LoRA cung rank r=32      10,485,760 (77%)
#   bash run_drop5.sh slora_budget   # S-LoRA r=80 tren v         13,631,488 (khop dung)
#
# EXTRA: co them, dat sau nen ghi de cau hinh (smoke test dung). SUF: hau to log/thu muc.
#
# Theo PMSS phu luc A.1 + bang 7: 2000 train / 800 val (tu tap train), 1200 test (tu
# tap dev), 3 epoch, batch 4 x 2, lr giam tuyen tinh, khong weight decay, chon epoch
# tot nhat tren val, F1 cua DROP. Giong cac dot truoc: mot lr chung 1e-4, alpha = 2r,
# base bf16 + adapter fp32.
#
# Luu y: q_proj cua LLaMA-3.1-8B vuong (4096x4096) nen S-LoRA di duong LoRA o do;
# S-LoRA chi khac LoRA o v_proj (4096 -> 1024, a = 4).
set -u
cd "$(dirname "$0")"

RUN="${1:?dung: bash run_drop5.sh lora|dora|pissa|slora|slora_budget}"
PY="${PY:-$HOME/venv/bin/python}"
M=unsloth/Meta-Llama-3.1-8B

COMMON="--model $M --target-set qv --epochs 3 --batch 4 --grad-accum 2 --lr 1e-4
        --weight-decay 0 --warmup 0 --n-train 2000 --n-val 800 --n-test 1200
        --data-seed 42 --best-epoch --bf16 --fp32-adapters --seed 0 --log-every 25
        --fac-cache fac_cache/llama31"
SL="--method hybrid --rank-square 32 --alpha-square 64 --slora-norm rms --slora-impl lowrank"

case "$RUN" in
  lora)         CFG="--method lora  --rank 32 --alpha 64" ;;
  dora)         CFG="--method dora  --rank 32 --alpha 64" ;;
  pissa)        CFG="--method pissa --rank 32 --alpha 64" ;;
  slora)        CFG="$SL --rank 32 --alpha 64" ;;
  slora_budget) CFG="$SL --rank 80 --alpha 160" ;;
  *)            echo "phai la lora, dora, pissa, slora hoac slora_budget"; exit 1 ;;
esac

EXTRA="${EXTRA:-}"
SUF="${SUF:-}"
OUT="runs_drop_llama31${SUF}"
mkdir -p "$OUT" logs
LOG="logs/drop_llama31_${RUN}${SUF}.log"
echo "### $RUN$SUF bat dau $(date -u '+%F %T') UTC" | tee "$LOG"
"$PY" -u finetune_drop.py $COMMON $CFG $EXTRA --out-dir "$OUT" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] && grep -qi "out of memory" "$LOG"; then
  echo "### OOM -> chay lai voi gradient checkpointing" | tee -a "$LOG"
  "$PY" -u finetune_drop.py $COMMON $CFG $EXTRA --grad-ckpt --out-dir "$OUT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
fi
echo "### $RUN$SUF ket thuc rc=$rc $(date -u '+%F %T') UTC" | tee -a "$LOG"
exit "$rc"
