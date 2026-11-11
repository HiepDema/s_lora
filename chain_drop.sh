#!/bin/bash
# Chay 5 run DROP noi tiep tren MOT GPU: smoke test ca nam truoc, roi run that.
#
#   cd ~/s_lora && setsid nohup bash chain_drop.sh > ~/chain_drop.log 2>&1 < /dev/null &
#
# Hang doi ~/drop_queue.txt, moi dong "smoke:<run>" hoac "full:<run>"; ghi them dong
# trong luc chay la xep them viec. Smoke hong -> dung chuoi (~/DROP_SMOKE_FAILED).
# Run that hong -> ~/DROP_FAILED_<run> roi lam muc tiep theo.
set -u
cd ~/s_lora
Q=~/drop_queue.txt
RUNS="lora pissa slora slora_budget dora"
if [ ! -s "$Q" ]; then
  for r in $RUNS; do echo "smoke:$r"; done > "$Q"
  for r in $RUNS; do echo "full:$r"; done >> "$Q"
fi

# 64 mau x 2 epoch: di qua train, val cuoi epoch, khoi phuc epoch tot nhat, luu
# checkpoint, sinh + cham F1, do toc do va gop adapter.
SMOKE="--n-train 64 --n-val 16 --n-test 32 --epochs 2 --log-every 2"

smoke_ok () {      # loss phai huu han va giam tu lan log dau den lan log cuoi
  local log="logs/drop_llama31_$1_smoke.log" a b
  a=$(sed -n 's/.*step [0-9]*\/[0-9]*  loss \([^ ]*\).*/\1/p' "$log" | head -n 1)
  b=$(sed -n 's/.*step [0-9]*\/[0-9]*  loss \([^ ]*\).*/\1/p' "$log" | tail -n 1)
  echo "    loss $a -> $b"
  awk -v a="$a" -v b="$b" 'BEGIN { exit !(a != "" && b == b + 0 && b + 0 < a + 0) }'
}

while true; do
  item=$(head -n 1 "$Q")
  [ -z "$item" ] && break
  sed -i 1d "$Q"
  kind=${item%%:*}; run=${item#*:}
  echo "### [$item] bat dau $(date -u '+%F %T') UTC"
  if [ "$kind" = smoke ]; then
    EXTRA="$SMOKE" SUF=_smoke bash run_drop5.sh "$run" > /dev/null 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] || ! smoke_ok "$run"; then
      echo "### [$item] HONG rc=$rc — dung chuoi"
      tail -n 25 "logs/drop_llama31_${run}_smoke.log"
      touch ~/DROP_SMOKE_FAILED
      exit 1
    fi
    tail -n 1 runs_drop_llama31_smoke/summary.jsonl | python3 -c "
import json, sys; d = json.loads(sys.stdin.read())
print(f\"    ok: {d['trainable']/1e6:.2f}M tham so, F1 {d['f1']:.1f}, val {d.get('val_hist')}, \"
      f\"VRAM {d['peak_vram_gb']} GB, merged fwd {d.get('merged_fwd_ms')} ms\")"
  else
    # FULL_EXTRA / FULL_SUF: cau hinh rieng cho run that, vd ca bo DROP:
    #   FULL_EXTRA="--n-train 0 --n-test 0 --log-every 200" FULL_SUF=_full
    EXTRA="${FULL_EXTRA:-}" SUF="${FULL_SUF:-}" bash run_drop5.sh "$run" > /dev/null 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || touch ~/DROP_FAILED_"$run"
    grep -E "F1 = " "logs/drop_llama31_${run}${FULL_SUF:-}.log" | tail -n 1
  fi
  echo "### [$item] ket thuc rc=$rc $(date -u '+%F %T') UTC"
done
touch ~/DROP_CHAIN_DONE
echo "### CHUOI XONG $(date -u '+%F %T') UTC"
