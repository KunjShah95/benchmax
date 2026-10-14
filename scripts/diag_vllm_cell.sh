#!/usr/bin/env bash
# Show the tail of a specific backend log, e.g. diag_vllm_cell.sh qwen3-0.6b 8bit
PAT="${1:-qwen3-0.6b}_${2:-8bit}"
LOG=$(ls -t /mnt/c/benchmaxing/results/logs/vllm_${PAT}_*.log 2>/dev/null | head -1)
echo "log: $LOG"
[ -z "$LOG" ] && { echo "(not found)"; exit 1; }
echo
echo "=== terminal lines ==="
grep -E 'Error|error|RuntimeError|OSError|ValueError|ImportError|not found|cannot|Cannot' "$LOG" \
  | grep -vE '^\s*$' | tail -12 | cut -c1-210
echo
echo "=== last 12 raw lines ==="
tail -12 "$LOG" | cut -c1-200