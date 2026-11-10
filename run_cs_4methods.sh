#!/bin/bash
# Commonsense 170k -> 8 tac vu: S-LoRA, LoRA, DoRA, PiSSA, cau hinh MoC/DoRA
# (q,k,v,up,down). Moi box mot run:
#
#   bash run_cs_4methods.sh lora           # LoRA  r=32               lr 1e-4
#   bash run_cs_4methods.sh dora           # DoRA  r=32 (+ magnitude) lr 1e-4
#   bash run_cs_4methods.sh pissa          # PiSSA r=32               lr 1e-4
#   bash run_cs_4methods.sh slora          # S-LoRA CUNG RANK r=32 + rms, lr 1e-4
#   bash run_cs_4methods.sh slora_budget   # S-LoRA CUNG NGAN SACH voi LoRA r=32
#   bash run_cs_4methods.sh lora_3e-4      # LoRA dung cau hinh LoRA* cua DoRA/MoC
#
#   MODEL=llama3 bash run_cs_4methods.sh ...    # LLaMA3-8B thay vi Qwen2.5-7B
#
# ------------------------------------------------------------------ cau hinh
# Giong run_cs_moc.sh: commonsense_170k day du, 3 epoch, batch 16 x 2, max_len 256,
# weight decay 0.01, warmup 100, beam 4 tren toan bo 8 tap test, alpha = 2r.
#
# Khac lan truoc o MOT cho, ap dung cho ca bon: base bf16 nhung adapter fp32,
# forward duoi autocast bf16 (--fp32-adapters, nhu PEFT). Lan truoc adapter bf16:
# buoc Adam nho hon nua ulp cua phan lon phan tu nen bi lam tron mat — LoRA lr 3e-4
# sap, va tren DART PiSSA dung yen. Do tren CPU: o lr 1e-5, adapter bf16 chi 0.8%
# phan tu cua A nhuc nhich, adapter fp32 la 100%.
#
# Mot lr chung 1e-4 cho ca bon: la lr cua DoRA tren LLaMA3-8B va cua run S-LoRA
# 77.4 truoc.
#
# S-LoRA: --slora-impl lowrank — phan dong bang chay qua W0 nguyen ban nhu LoRA,
# nen sai so lam tron bf16 va toc do nhu LoRA (duong phan ra lech ~1e-2 o bf16).
# q_proj vuong nen di duong LoRA r=32 nhu ba method kia.
#
# ------------------------------------------------------------------ ngan sach
#                LoRA/PiSSA r=32   DoRA r=32     S-LoRA r=32 (cung rank)   S-LoRA cung ngan sach
#   Qwen2.5-7B    54,132,736       54,892,544    21,102,592 (39%)          r=104  54,132,736
#   LLaMA3-8B     56,623,104       57,409,536    29,360,128 (52%)          r=73   56,229,888
set -u
cd "$(dirname "$0")"

RUN="${1:?dung: bash run_cs_4methods.sh lora|dora|pissa|slora|slora_budget|lora_3e-4}"
PY="${PY:-$HOME/venv/bin/python}"
case "${MODEL:-qwen7b}" in
  qwen7b) M=Qwen/Qwen2.5-7B;    KEY=qwen7b; RB=104 ;;
  llama3) M=unsloth/llama-3-8b; KEY=llama3; RB=73 ;;
  *)      echo "MODEL phai la qwen7b hoac llama3"; exit 1 ;;
esac

COMMON="--model $M --target-set qkvud --batch 16 --grad-accum 2 --epochs 3
        --max-len 256 --weight-decay 0.01 --warmup 100 --eval-beams 4 --eval-batch 16
        --bf16 --fp32-adapters --seed 0 --log-every 200 --max-train 0
        --eval-every 2000 --eval-limit-mid 200 --fac-cache fac_cache/$KEY"
#        ^ cham giua chung moi 2000 buoc (200 cau/tac vu, greedy) de bat som model sup
SL="--method hybrid --rank-square 32 --alpha-square 64 --slora-norm rms
    --slora-impl lowrank --lr 1e-4"

case "$RUN" in
  lora)         CFG="--method lora  --rank 32 --alpha 64 --lr 1e-4" ;;
  dora)         CFG="--method dora  --rank 32 --alpha 64 --lr 1e-4" ;;
  pissa)        CFG="--method pissa --rank 32 --alpha 64 --lr 1e-4" ;;
  slora)        CFG="$SL --rank 32 --alpha 64" ;;
  slora_budget) CFG="$SL --rank $RB --alpha $((2 * RB))" ;;
  lora_3e-4)    CFG="--method lora  --rank 32 --alpha 64 --lr 3e-4" ;;
  *)            echo "phai la lora, dora, pissa, slora, slora_budget hoac lora_3e-4"; exit 1 ;;
esac

# EXTRA: co them, dat SAU nen ghi de cau hinh tren (argparse lay lan cuoi).
# SUF: hau to cho thu muc/log, de smoke test khong lan vao run that.
EXTRA="${EXTRA:-}"
SUF="${SUF:-}"
OUT="runs_cs4_${KEY}${SUF}"
mkdir -p "$OUT" logs
LOG="logs/cs4_${KEY}_${RUN}${SUF}.log"
echo "### $KEY $RUN$SUF bat dau $(date -u '+%F %T') UTC" | tee "$LOG"
"$PY" -u finetune_commonsense.py $COMMON $CFG $EXTRA --out-dir "$OUT" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] && grep -qi "out of memory" "$LOG"; then
  echo "### OOM -> chay lai voi gradient checkpointing" | tee -a "$LOG"
  "$PY" -u finetune_commonsense.py $COMMON $CFG $EXTRA --grad-ckpt --out-dir "$OUT" 2>&1 \
    | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
fi
echo "### $KEY $RUN$SUF ket thuc rc=$rc $(date -u '+%F %T') UTC" | tee -a "$LOG"
exit "$rc"
