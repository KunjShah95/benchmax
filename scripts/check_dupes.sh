#!/usr/bin/env bash
echo "=== all benchmax / server processes ==="
ps -eo pid,ppid,etime,pcpu,args --sort=start_time | grep -E 'stage5_run|benchmax.cli|vllm|llama-server' | grep -v grep
echo
echo "=== count of distinct run launchers ==="
pgrep -f stage5_run | wc -l
echo
echo "=== GPU compute apps ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo
echo "=== GPU utilization sample (3x) ==="
for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader; sleep 1; done