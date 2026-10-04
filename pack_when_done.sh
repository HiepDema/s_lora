#!/bin/bash
# Doi mot run xong roi tu dong dong goi ket qua lai mot file.
#
#   cd ~/slora && setsid nohup bash pack_when_done.sh qwen3b.log runs_qwen3B \
#       ~/qwen3b_results.tar.gz > pack.log 2>&1 < /dev/null &
#
# Vi sao chay o day chu khong o may ca nhan: vong lap tai ve o may ca nhan se
# chet khi may ngu. Chay tren box thi may ca nhan tat bao lau cung duoc, luc
# quay lai chi can keo mot file.
#
# Khong gom *.pt (checkpoint vai tram MB moi cai). Neu can thi keo rieng.
set -u
LOG="${1:?thieu file log}"
DIR="${2:?thieu thu muc ket qua}"
OUT="${3:?thieu duong dan file .tar.gz}"

echo "[$(date -u +%H:%M:%S)] doi $LOG co ALL_DONE..."

# Loc theo 'finetune_e2e.py', TUYET DOI khong theo ten script nay: shell cua
# ssh chua chuoi do trong dong lenh va se tu khop, gay doi mai mai.
while ! grep -q "ALL_DONE" "$LOG" 2>/dev/null; do
  if ! pgrep -f "[f]inetune_e2e.py" > /dev/null; then
    sleep 120                      # co the dang giua hai run, cho mot nhip
    if ! pgrep -f "[f]inetune_e2e.py" > /dev/null && ! grep -q "ALL_DONE" "$LOG" 2>/dev/null; then
      echo "[$(date -u +%H:%M:%S)] KHONG con tien trinh nao ma cung chua ALL_DONE —"
      echo "  run da chet giua chung. Van dong goi nhung gi co."
      break
    fi
  fi
  sleep 60
done

echo "[$(date -u +%H:%M:%S)] dong goi..."
tar czf "$OUT" "$LOG" \
    $(find "$DIR" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.jsonl' -o -name '*_hyps.txt' \
           -o -name '*_refs.txt' \) 2>/dev/null) 2>/dev/null

echo "[$(date -u +%H:%M:%S)] xong: $OUT  ($(du -h "$OUT" | cut -f1))"
tar tzf "$OUT" | sed 's/^/    /'
echo PACK_DONE
