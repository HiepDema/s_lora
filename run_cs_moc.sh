#!/bin/bash
# Lan 2: S-LoRA da sua (--slora-norm rms) + LoRA tham chieu, commonsense 8 tac vu.
#
#   setsid nohup bash cs_auto2.sh > cs_auto2.log 2>&1 </dev/null &
#
# Lan 1 hong vi P = [I | X] khuech dai delta 17-244x (up_proj te nhat) -> model
# sup ve mot dap an. rms chia scale moi tang cho ||P||_F/sqrt(k).
# Run ngan 1299 buoc da len 75.8% (greedy, 500 cau/tac vu).
#
# Thu tu: LLaMA3 S-LoRA -> LLaMA3 LoRA -> Mistral S-LoRA. Mot run hong thi van
# chay run sau. Cham giua chung moi 2000 buoc (200 cau/tac vu, greedy) de bat som.
#
# LoRA tham chieu: r=32 a=64 lr 3e-4 = cau hinh LoRA* cua DoRA/MoC (Table 1),
# de biet pipeline nay cho LoRA bao nhieu so voi 80.8 trong bai.
set -uo pipefail
PY=~/venv/bin/python
export PATH="$HOME/venv/bin:$PATH"
M_LLAMA=unsloth/llama-3-8b
M_MISTRAL=mistralai/Mistral-7B-Instruct-v0.3
cd ~/s_lora
echo "### bat dau $(date '+%F %T')"

COMMON="--target-set qkvud --batch 16 --grad-accum 2 --epochs 3 --max-len 256
        --weight-decay 0.01 --warmup 100 --eval-beams 4 --eval-batch 16 --bf16
        --seed 0 --log-every 200 --max-train 0 --eval-every 2000 --eval-limit-mid 200"
SLORA="--method hybrid --slora-norm rms --rank 73 --rank-square 32 --alpha 146
       --alpha-square 64 --lr 1e-4"
LORA="--method lora --rank 32 --alpha 64 --lr 3e-4"

run () {   # $1=model  $2=ten run  $3=fac cache  $4..=cau hinh method
  local m="$1" name="$2" fc="$3"; shift 3
  local d="runs_cs2_$name"
  mkdir -p "$d"
  echo "### [run] $name  bat dau $(date '+%F %T')"
  $PY finetune_commonsense.py --model "$m" $COMMON "$@" \
      --fac-cache "fac_cache/$fc" --out-dir "$d" >> "$d/run.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] && grep -qi "out of memory" "$d/run.log"; then
    echo "    OOM -> chay lai voi gradient checkpointing"
    $PY finetune_commonsense.py --model "$m" $COMMON "$@" --grad-ckpt \
        --fac-cache "fac_cache/$fc" --out-dir "$d" >> "$d/run.log" 2>&1
    rc=$?
  fi
  echo "### [run] $name ket thuc rc=$rc  $(date '+%F %T')"
  grep -E "TRUNG BINH|>> step" "$d/run.log" | tail -9
  tar czf ~/cs2_$name.tar.gz --exclude='*.pt' "$d"
}

run "$M_LLAMA"   llama3_slora llama3  $SLORA
run "$M_LLAMA"   llama3_lora  llama3  $LORA
run "$M_MISTRAL" mistral_slora mistral $SLORA

tar czf ~/cs2_all.tar.gz --exclude='*.pt' runs_cs2_* 2>/dev/null
touch ~/CS2_DONE
echo "### TAT CA XONG $(date '+%F %T')"
