#!/usr/bin/env bash
# Root-cause the newest vLLM startup failure.
LOG=$(ls -t /mnt/c/benchmaxing/results/logs/vllm_*.log 2>/dev/null | head -1)
echo "log: $LOG"
echo
echo "=== first ERROR / exception context ==="
grep -nE 'ERROR|Error|Exception|not supported|No module|ImportError|Assertion' "$LOG" \
  | grep -vE 'core_client|engine/utils|async_llm|v1/engine/core|entry.py' \
  | head -18 | cut -c1-200
echo
echo "=== quantization / kernel selection lines ==="
grep -inE 'quant|gptq|awq|fp8|marlin|exllama|compressed.tensors|backend' "$LOG" \
  | head -20 | cut -c1-200