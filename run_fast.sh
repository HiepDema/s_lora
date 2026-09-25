#!/bin/bash
# S-LoRA vs LoRA tren Mistral-7B — duong nhanh.
#
#   bash run_fast.sh 116 hybrid    # S-LoRA r=116 (167.25M)
#   bash run_fast.sh  64 lora      # LoRA r=64   (167.77M)
#   bash run_fast.sh sweep         # quet ngan sach: 5 muc x 2 phuong phap
#
# Khac run_pissa.sh o BA cho, tat ca deu ap dung DONG DEU cho ca hai nhanh:
#
#  1. bf16 thay vi fp32. PiSSA quy dinh fp32 cho LoRA/PiSSA, nhung chinh Bang 7
#     cua ho cho thay Mistral-7B chay bf16 TOT HON fp32 (GSM8K 73.09 vs 65.88),
#     va ho viet thang "the experiments did not prove which precision is better".
#     Mot khi ca hai nhanh cung bf16 thi phep so sanh van sach; cai mat di la
#     doi chieu truc tiep voi con so cong bo cua ho.
#
#  2. Bo gradient checkpointing. Peak VRAM do duoc o run fp32 chi 34/80 GB, ma
#     grad ckpt dang tra ~30% toc do de doi VRAM minh khong thieu. Bo no trong
#     fp32 thi sat mep (fp32 weights da an 29 GB); bf16 con 14.5 GB nen thoai mai.
#
#  3. Cham diem bang vLLM tren model da gop, thay cho model.generate() cua HF.
#     Day cung la thu MetaMath/PiSSA dung. Cham chiem 45-49% wall clock o run cu
#     (1.93h tren tong 4.3h), nen day la cho an nhieu nhat.
#
# Neu can mot diem doi chieu truc tiep voi PiSSA thi chay lai dung run_pissa.sh:
# fp32, grad ckpt, cham bang HF. Duong nhanh nay de quet, khong de bao cao tuyet doi.
set -u

M="${1:?dung: bash run_fast.sh <rank> <hybrid|lora>  |  bash run_fast.sh sweep}"
LR="${LR:-2e-5}"
NTRAIN="${NTRAIN:-100000}"

COMMON="--model mistralai/Mistral-7B-v0.1 --target-set all7
        --model-dtype bf16 --attn sdpa --tf32
        --seed 0 --max-len 512 --batch 16 --accum 8
        --lr-schedule cosine --warmup-ratio 0.03 --weight-decay 0
        --fac-cache fac_cache/mistral7b"
#        ^ batch 16 x accum 8 = 128, van dung batch hieu dung cua PiSSA.
#          bf16 du cho cho batch 16; neu OOM thi ha ve 8 va accum 16.

one () {
  local r="$1" meth="$2" tag="${meth}_r${r}"
  local rk="--rank $r"
  [ "$meth" = hybrid ] && rk="--rank $r --rank-square $r"

  echo "=================================================================="
  echo "==> $meth r=$r  lr=$LR  ${NTRAIN} mau  1 epoch  seed 0"
  echo "=================================================================="

  python -u finetune_math.py --method "$meth" $rk $COMMON \
    --lr "$LR" --epochs 1 --max-train "$NTRAIN" \
    --skip-gsm8k --no-bench --out-dir "fast_$tag" || return 1
  #  ^ --skip-gsm8k: bo hoan phan cham cua HF, de vLLM lam o buoc sau.

  python -u export_merged.py --ckpt "fast_$tag/${meth}"*_last.pt \
    --method "$meth" $rk --target-set all7 \
    --model mistralai/Mistral-7B-v0.1 --fac-cache fac_cache/mistral7b \
    --out "merged/$tag" || return 1

  python -u eval_vllm.py --model "merged/$tag" --tag "$tag" \
    --eval-math --out-dir "fast_$tag" || return 1

  # 14 GB moi model da gop; xoa ngay khong thi quet 10 run la day dia.
  rm -rf "merged/$tag"
  echo "RUN_DONE_$tag"
}

if [ "$M" = "sweep" ]; then
  # Gia thuyet can kiem: rang buoc row(dW) thuoc row(W0) co dat them khi ngan
  # sach lon khong? Neu khoang cach S-LoRA/LoRA thu hep o ngan sach thap thi
  # rang buoc chi dat khi cap nhat phai lon; neu no khong doi thi nguyen nhan
  # nam cho khac va phai tim tiep.
  #
  # Moi cap la CUNG ngan sach: S-LoRA r = LoRA r x 1.8177 (81920/45056).
  for pair in "15 27" "29 52" "58 105" "64 116"; do
    set -- $pair
    one "$2" hybrid
    one "$1" lora
  done
  echo SWEEP_DONE
  exit 0
fi

one "$M" "${2:?thieu phuong phap: hybrid hoac lora}"
echo ALL_DONE
