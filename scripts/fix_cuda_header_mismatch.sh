#!/usr/bin/env bash
# Align the CUDA runtime headers with the nvcc that vLLM uses.
#
# vLLM's wheel stack installed nvcc 13.4.92 alongside nvidia-cuda-runtime
# 13.0.96. FlashInfer's bundled CCCL hard-errors unless the compiler's minor
# version equals the toolkit headers' minor version, so every JIT'd kernel
# failed with "CUDA compiler and CUDA toolkit headers are incompatible".
# Upgrading the runtime to 13.4.x makes CUDART_VERSION agree with nvcc.
set -euo pipefail
V=~/benchmax-env/venv-vllm/bin
source $V/activate

echo "=== before ==="
pip list 2>/dev/null | grep -E 'nvidia-cuda-(runtime|nvcc|crt)'

echo
echo "=== upgrading CUDA runtime headers to match nvcc 13.4 ==="
pip install --quiet "nvidia-cuda-runtime==13.4.92"

echo
echo "=== after ==="
pip list 2>/dev/null | grep -E 'nvidia-cuda-(runtime|nvcc|crt)'

echo
echo "=== verify header version ==="
SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
grep -m1 -E '#define CUDART_VERSION' "$SP/nvidia/cuda_runtime/include/cuda_runtime_api.h" 2>/dev/null || \
  grep -rm1 -E '#define CUDART_VERSION' "$SP/nvidia/cuda_runtime/include/" 2>/dev/null | head -1