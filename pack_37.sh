#!/bin/bash
# Doi chuoi 3B+7B xong roi dong goi. Chay TREN BOX nen may ca nhan tat van duoc.
#
#   cd ~/slora && setsid nohup bash pack_37.sh > pack37.log 2>&1 < /dev/null &
#
# Doi CHAIN_37_DONE chu khong phai ALL_DONE: ALL_DONE xuat hien sau rieng phan
# 7B, con RUN_DONE_3B_seed1 thi truoc do — cho nham se dong goi thieu.
set -u
cd ~/slora
LOG=chain37.log
OUT=~/qwen_37_bf16.tar.gz

echo "[$(date -u +%H:%M:%S)] doi CHAIN_37_DONE..."
while ! grep -q CHAIN_37_DONE "$LOG" 2>/dev/null; do
  # Loc theo 'finetune_e2e.py', khong bao gio theo ten script nay: shell cua
  # ssh chua chuoi do trong dong lenh va se tu khop, gay doi mai mai.
  if ! pgrep -f "[f]inetune_e2e.py" > /dev/null; then
    sleep 150
    if ! pgrep -f "[f]inetune_e2e.py" > /dev/null \
       && ! grep -q CHAIN_37_DONE "$LOG" 2>/dev/null; then
      echo "[$(date -u +%H:%M:%S)] khong con tien trinh ma cung chua xong —"
      echo "  run da chet giua chung. Van dong goi nhung gi co."
      break
    fi
  fi
  sleep 60
done

echo "[$(date -u +%H:%M:%S)] dong goi..."
tar czf "$OUT" "$LOG" \
  $(find runs_qwen3B_bf16_s1 runs_qwen7B_bf16 -maxdepth 1 -type f \
      \( -name '*.json' -o -name '*.jsonl' -o -name '*_hyps.txt' \
         -o -name '*_refs.txt' \) 2>/dev/null) 2>/dev/null

echo "[$(date -u +%H:%M:%S)] xong: $OUT ($(du -h "$OUT" | cut -f1))"
tar tzf "$OUT" | sed 's/^/    /'
echo PACK_37_DONE
