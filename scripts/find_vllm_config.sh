#!/usr/bin/env bash
# Find a vLLM configuration that starts in this environment.
# FlashInfer's bundled CCCL headers clash with the CUDA 13.4 toolkit, so its
# JIT-compiled sampling kernel fails. Try progressively more conservative
# backend selections and report which one reaches a healthy server.
set -uo pipefail
V=~/benchmax-env/venv-vllm/bin
LOGDIR=~/benchmax-logs; mkdir -p "$LOGDIR"
MODEL="$HOME/benchmax-env/models/Qwen__Qwen3-0.6B"
SP="$HOME/benchmax-env/venv-vllm/lib/python3.12/site-packages"
CU="$SP/nvidia/cu13"

export CUDA_HOME="$CU" CUDA_PATH="$CU"
export PATH="$V:$CU/bin:$PATH"
export LD_LIBRARY_PATH="$CU/lib:$CU/lib64:${LD_LIBRARY_PATH:-}"

try() {
  local name="$1"; shift
  local port="$1"; shift
  local log="$LOGDIR/vllmcfg-$name.log"
  "$V/vllm" serve "$MODEL" --host 127.0.0.1 --port "$port" \
    --max-model-len 4096 --gpu-memory-utilization 0.80 \
    --seed 1234 "$@" > "$log" 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok; then
      echo "  $name: HEALTHY after ${i}s"
      kill $pid 2>/dev/null; wait $pid 2>/dev/null
      return 0
    fi
    if ! kill -0 $pid 2>/dev/null; then
      local err=$(grep -oE 'Ninja build failed|CUDA compiler and CUDA toolkit[^"]*|No module named[^ ]*|Engine core initialization failed|OSError:[^\n]*' "$log" | head -1)
      echo "  $name: DIED after ${i}s  [${err:0:80}]"
      return 1
    fi
    sleep 1
  done
  echo "  $name: TIMEOUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  return 1
}

echo "=== baseline (vLLM defaults) ==="
try default 8931 || true

echo "=== no-enable-flashinfer-autotune ==="
try noauto 8932 --no-enable-flashinfer-autotune || true

echo "=== TRITON_ATTN ==="
try triton 8933 --attention-backend TRITON_ATTN || true

echo "=== TRITON_ATTN + no-autotune ==="
try tritonnoauto 8934 --attention-backend TRITON_ATTN --no-enable-flashinfer-autotune || true

echo "=== FLASH_ATTN ==="
try flashattn 8935 --attention-backend FLASH_ATTN || true