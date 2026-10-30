#!/bin/bash
# Doi 3 run chinh xong, roi:
#   1. S-LoRA r=4 (0.29M) va LoRA r=4 (0.66M) tren 19 tac vu, chay song song,
#      cung cau hinh voi 3 run truoc.
#   2. Do lai do tre cua ca 5 cau hinh khi GPU trong (so do trong luc cac run
#      chay song song bi lan nhau, khong dung duoc).
until [ -f ~/VTAB_DONE ]; do sleep 60; done
cd ~/s_lora
PY=~/venv/bin/python
echo "### r4 bat dau $(date "+%F %T")"
$PY finetune_vtab.py --method hybrid --rank 4 --slora-norm rms --out-dir runs_vtab > runs_vtab/slora_r4.log 2>&1 &
p1=$!
$PY finetune_vtab.py --method lora --rank 4 --out-dir runs_vtab > runs_vtab/lora_r4.log 2>&1 &
p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
echo "### r4 ket thuc rc=$r1/$r2 $(date "+%F %T")"
mkdir -p bench_vtab
for cfg in "--method lora --rank 4" "--method lora --rank 8" "--method hybrid --rank 4 --slora-norm rms" "--method hybrid --rank 8 --slora-norm rms" "--method hybrid --rank 18 --slora-norm rms"; do
  $PY finetune_vtab.py $cfg --tasks dtd --epochs 1 --warmup-epochs 1 --max-test 64 --out-dir bench_vtab >> bench_vtab/bench.log 2>&1
done
grep -E "===|do tre" bench_vtab/bench.log
echo "### bench xong $(date "+%F %T")"
tar czf ~/vtab_all.tar.gz runs_vtab bench_vtab
touch ~/VTAB4_DONE
