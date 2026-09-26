#!/bin/bash
# Do code MOI nhanh hon code CU bao nhieu — ca hai tren CUNG box, cung du lieu.
#
# Khong so voi so cu tren box SXM5: do la H100 PCIe khac card, se lan lon giua
# code va phan cung. Bai hoc tu lan truoc — minh tung bao VeRA cham gap doi
# S-LoRA, hoa ra toan bo chenh lech den tu TF32 bat/tat chu khong tu method.
#
# Ca hai cau hinh dung batch hieu dung 128 (8x16 va 16x8) nen CUNG so buoc
# optimizer tren cung du lieu -> it/s so truc tiep duoc.
set -u
cd ~/slora
export PATH="$HOME/.local/bin:$PATH"

N="${N:-7760}"        # 60 buoc optimizer: 7760 - 1% val = 7682, /128 = 60
EV="${EV:-500}"       # so bai GSM8K de do toc do cham

BASE="--method hybrid --rank 116 --rank-square 116
      --model mistralai/Mistral-7B-v0.1 --target-set all7
      --fac-cache fac_cache/mistral7b --seed 0 --max-len 512
      --lr-schedule cosine --warmup-ratio 0.03 --weight-decay 0 --lr 2e-5
      --epochs 1 --max-train $N --log-every 10 --no-bench"

echo "################################################################"
echo "# A: cau hinh CU — fp32, grad-ckpt, batch 8 x accum 16"
echo "################################################################"
python -u finetune_math.py $BASE --tf32 --grad-ckpt --batch 8 --accum 16 \
  --skip-gsm8k --no-save-ckpt --out-dir bench_old
echo "DONE_A"

echo
echo "################################################################"
echo "# B: cau hinh MOI — bf16, sdpa, batch 16 x accum 8, KHONG grad-ckpt"
echo "################################################################"
python -u finetune_math.py $BASE --model-dtype bf16 --attn sdpa --tf32 \
  --batch 16 --accum 8 --limit-eval "$EV" --out-dir bench_new
echo "DONE_B"

echo
echo "################################################################"
echo "# C: gop adapter -> checkpoint HF"
echo "################################################################"
python -u export_merged.py --ckpt bench_new/hybrid_r116_na_s0_last.pt \
  --method hybrid --rank 116 --rank-square 116 --target-set all7 \
  --model mistralai/Mistral-7B-v0.1 --fac-cache fac_cache/mistral7b \
  --out merged/bench
echo "DONE_C"

echo
echo "################################################################"
echo "# D: cham CUNG $EV bai do bang vLLM"
echo "################################################################"
python -u eval_vllm.py --model merged/bench --tag bench \
  --limit-eval "$EV" --out-dir bench_vllm
echo "DONE_D"

echo
echo "################################################################"
echo "# TONG KET"
echo "################################################################"
python -u bench_summary.py
echo BENCH_DONE
