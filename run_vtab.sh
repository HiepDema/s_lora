#!/bin/bash
# VTAB-1k, ViT-B/16 in21k goc, 19 tac vu, 3 run song song tren 1 A100:
#   1. LoRA r=8 tren q,k,v,o,fc1,fc2            1.33M tham so
#   2. S-LoRA r=8  (qkv gop, fc1, fc2) + LoRA o  0.59M  cung rank
#   3. S-LoRA r=18 (qkv gop, fc1, fc2) + LoRA o  1.33M  cung ngan sach
# lr 1e-3 chung, AdamW, batch 64, 100 epoch, warmup 10, wd 1e-4, drop path 0.1,
# fp32 + TF32. S-LoRA: --slora-norm rms, impl lowrank (W0 x + delta qua X).
cd ~/s_lora
PY=~/venv/bin/python
echo "### bat dau $(date "+%F %T")"
mkdir -p runs_vtab
$PY finetune_vtab.py --method lora   --rank 8  --out-dir runs_vtab > runs_vtab/lora_r8.log 2>&1 &
p1=$!
$PY finetune_vtab.py --method hybrid --rank 8  --slora-norm rms --out-dir runs_vtab > runs_vtab/slora_r8.log 2>&1 &
p2=$!
$PY finetune_vtab.py --method hybrid --rank 18 --slora-norm rms --out-dir runs_vtab > runs_vtab/slora_r18.log 2>&1 &
p3=$!
wait $p1; r1=$?; wait $p2; r2=$?; wait $p3; r3=$?
echo "### ket thuc rc=$r1/$r2/$r3 $(date "+%F %T")"
tar czf ~/vtab_all.tar.gz runs_vtab
touch ~/VTAB_DONE
