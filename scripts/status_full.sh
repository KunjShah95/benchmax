#!/usr/bin/env bash
echo "=== record counts ==="
for k in perf_run quality_run cell_summary startup_failure run_error quality_error; do
  n=$(grep -c "\"kind\": \"$k\"" /mnt/c/benchmaxing/results/raw/runs.jsonl 2>/dev/null || echo 0)
  printf '  %-18s %s\n' "$k" "$n"
done
echo
echo "=== cells completed ==="
grep '\[cell\] done' ~/benchmax-logs/slice.log | sort -u | wc -l
echo
echo "=== last log lines ==="
tail -12 ~/benchmax-logs/slice.log | cut -c1-110
echo
echo "=== supervisor output ==="
tail -8 ~/benchmax-logs/slice-1.log 2>/dev/null | cut -c1-100
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo
echo "=== live procs ==="
ps -eo pid,etime,args | grep -E 'benchmax.cli|llama-server|vllm serve' | grep -v grep | cut -c1-95 | head -4