#!/usr/bin/env bash
# Root-cause the newest vLLM startup failure (post CUDA_HOME fix).
LOG=$(ls -t /mnt/c/benchmaxing/results/logs/vllm_*.log 2>/dev/null | head -1)
echo "log: $LOG"
echo
echo "=== terminal error lines ==="
grep -nE '^\(EngineCore.*(Error|error)|RuntimeError|ValueError|ImportError|ModuleNotFound|AssertionError|torch\.utils|Triton|nvcc|CUDA_HOME|cuda_home' "$LOG" \
  | tail -20 | cut -c1-210
echo
echo "=== last 15 lines ==="
tail -15 "$LOG" | cut -c1-200