#!/usr/bin/env bash
# Hard-stop every benchmark process and verify the GPU is idle.
# pkill patterns must also match the bare `python -m benchmax.cli ... run`
# command, which is how the orchestrator actually appears in the process table.
set -uo pipefail

for pat in stage5_run.sh stage4_download.sh benchmax.cli llama-server "vllm serve" \
           "VLLM::EngineCore" sglang.launch_server; do
  pkill -9 -f "$pat" 2>/dev/null && echo "  killed: $pat"
done

# orphaned vLLM engine workers survive the parent
pkill -9 -f "VLLM" 2>/dev/null
sleep 3

echo
echo "=== survivors (should be empty) ==="
ps -eo pid,etime,args | grep -E 'benchmax|llama-server|vllm|sglang|EngineCore' | grep -v grep || echo "  none"
echo
echo "=== GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "  util: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"