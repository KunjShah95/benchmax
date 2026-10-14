#!/usr/bin/env bash
# FlashInfer's bundled CCCL refuses to compile when the nvcc minor version
# (13.4) disagrees with the CUDA runtime headers torch ships (13.0). The header
# exposes CCCL_DISABLE_CTK_COMPATIBILITY_CHECK as the documented escape hatch;
# nvcc honours NVCC_APPEND_FLAGS for every compilation it drives, so inject it
# there rather than patching site-packages.
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
  local log="$LOGDIR/vllmfix-$name.log"
  env "$@" "$V/vllm" serve "$MODEL" --host 127.0.0.1 --port "$port" \
    --max-model-len 4096 --gpu-memory-utilization 0.80 --seed 1234 \
    > "$log" 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok; then
      echo "  $name: HEALTHY after ${i}s"
      kill $pid 2>/dev/null; wait $pid 2>/dev/null
      return 0
    fi
    if ! kill -0 $pid 2>/dev/null; then
      local err=$(grep -oE 'Ninja build failed|incompatible[^"]*|CUDA error[^ ]*|no kernel image[^ ]*' "$log" | head -1)
      echo "  $name: DIED after ${i}s  [${err:0:70}]"
      return 1
    fi
    sleep 1
  done
  echo "  $name: TIMEOUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
}

echo "=== NVCC_APPEND_FLAGS escape hatch ==="
try appendflag 8941 NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK && exit 0

echo "=== also with CXXFLAGS ==="
try cxxflags 8942 NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK CXXFLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK && exit 0

echo "=== FLASHINFER_CUDA_FLAGS variant ==="
try fiflags 8943 FLASHINFER_CUDA_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK && exit 0

echo "no configuration worked"