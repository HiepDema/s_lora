#!/bin/bash
# Chay 4 method commonsense noi tiep tren MOT GPU: smoke test ca bon truoc, roi
# moi den run that.
#
#   cd ~/s_lora && setsid nohup bash chain_cs4.sh > ~/chain_cs4.log 2>&1 < /dev/null &
#
# Hang doi la ~/cs4_queue.txt, moi dong "smoke:<run>" hoac "full:<run>". Chuoi doc
# dong dau moi khi bat dau mot muc, nen ghi them dong vao file trong luc dang chay
# la xep them viec (dung de chay lai mot run sau khi sua loi).
#
# Smoke test hong -> dung ca chuoi va ghi ~/CS4_SMOKE_FAILED: khong dot 9 gio vao
# code loi. Run that hong -> ghi ~/CS4_FAILED_<run> roi lam muc tiep theo.
set -u
cd ~/s_lora
Q=~/cs4_queue.txt
[ -s "$Q" ] || printf '%s\n' smoke:lora smoke:pissa smoke:slora smoke:dora \
                             full:lora full:pissa full:slora full:dora > "$Q"

# 1000 mau x 2 epoch = ~64 buoc: du di qua train, cham giua chung, cham cuoi
# epoch, luu checkpoint, cham cuoi, do toc do va gop adapter.
SMOKE="--max-train 1000 --epochs 2 --limit-eval 20 --eval-every 20
       --eval-limit-mid 10 --log-every 5"

smoke_ok () {      # loss phai huu han va giam tu lan log dau den lan log cuoi
  local log="logs/cs4_qwen7b_$1_smoke.log" a b
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
    EXTRA="$SMOKE" SUF=_smoke bash run_cs_4methods.sh "$run" > /dev/null 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] || ! smoke_ok "$run"; then
      echo "### [$item] HONG rc=$rc — dung chuoi"
      tail -n 25 "logs/cs4_qwen7b_${run}_smoke.log"
      touch ~/CS4_SMOKE_FAILED
      exit 1
    fi
    tail -n 1 runs_cs4_qwen7b_smoke/summary.jsonl | python3 -c "
import json, sys; d = json.loads(sys.stdin.read())
print(f\"    ok: {d['trainable']/1e6:.2f}M tham so, {d['train_it_s']} it/s, \"
      f\"VRAM {d['peak_vram_gb']} GB, merged fwd {d.get('merged_fwd_ms')} ms\")"
  else
    bash run_cs_4methods.sh "$run" > /dev/null 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || touch ~/CS4_FAILED_"$run"
    grep -E "TRUNG BINH" "logs/cs4_qwen7b_${run}.log" | tail -n 1
  fi
  echo "### [$item] ket thuc rc=$rc $(date -u '+%F %T') UTC"
done
touch ~/CS4_CHAIN_DONE
echo "### CHUOI XONG $(date -u '+%F %T') UTC"
