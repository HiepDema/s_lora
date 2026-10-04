#!/bin/bash
# Mot box A10: kiem lai o 3B dang ngo, roi chay ca ba run 7B. Tat ca bf16.
#
#   cd ~/slora && setsid nohup bash chain_3b_7b.sh > chain37.log 2>&1 < /dev/null &
#
# ---------------------------------------------------------------- vi sao run dau
# O 3B LoRA r=2 bf16 cho BLEU 61.16, tut 4.22 so voi fp32 — nhung val loss chi
# xau di 0.026. Cac hang khac hai chi so di cung muc (0.5B: val +0.042, BLEU
# -2.09). Da loai kha nang sinh hong: do dai 133.3 so voi 132.7 ky tu, khong
# cau rong, khong lap.
#
# O do dang ganh TOAN BO su dao chieu giua hai bang: fp32 cho -0.05 o 3B con
# bf16 cho +3.93. Chay lai voi seed 1 de biet no that hay la nhieu.
#
# ------------------------------------------------------------------- ve 7B
# RB=8 vi a=7, khop DUNG ngan sach cua LoRA r=2 (ca hai 0.459M).
# VRAM ngoai suy ~20.3 GB tu ba diem do duoc (4.64 / 6.59 / 10.33 GB), vua A10
# 24GB voi bien ~3.7 GB. Bien do mong nen bat expandable_segments cho chac.
set -u
cd ~/slora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$HOME/venv/bin/python"

echo "===================================================================="
echo "==> 3B LoRA r=2 bf16 SEED 1 — kiem lai o dang ngo"
echo "===================================================================="
"$PY" -u finetune_e2e.py --method lora --rank 2 \
  --model Qwen/Qwen2.5-3B --target-set qwen2_kv --epochs 5 --best-epoch --seed 1 \
  --attn sdpa --eval-batch 32 --model-dtype bf16 --out-dir runs_qwen3B_bf16_s1
echo "RUN_DONE_3B_seed1"

DT=bf16 bash run_qwen_scale.sh 7B

echo CHAIN_37_DONE
