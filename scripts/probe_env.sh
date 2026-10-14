#!/usr/bin/env bash
# Probe the WSL2 environment for benchmarking prerequisites.
set -uo pipefail

echo "=== OS ==="
cat /etc/os-release | grep -E '^(NAME|VERSION)=' 
echo "kernel: $(uname -r)"

echo
echo "=== TOOLS ==="
for t in nvcc gcc g++ make cmake git curl wget python3 jq nvidia-smi; do
  p=$(command -v "$t" 2>/dev/null || echo "MISSING")
  printf '%-12s %s\n' "$t" "$p"
done

echo
echo "=== CUDA ==="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -2
else
  echo "nvcc not installed (vLLM ships its own wheels; only needed to BUILD llama.cpp with CUDA)"
fi

echo
echo "=== GPU PASSTHROUGH ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1

echo
echo "=== PYTHON ==="
python3 --version
python3 -m pip --version 2>&1
python3 -c "import venv; print('venv: ok')" 2>&1

echo
echo "=== MEMORY / CPU / DISK ==="
free -m | head -2
echo "cores: $(nproc)"
df -h / /mnt/c | tail -3

echo
echo "=== PRE-INSTALLED ML PACKAGES ==="
python3 -m pip list 2>/dev/null | grep -Ei 'torch|vllm|transformers|sglang|huggingface|accelerate' || echo "(none relevant)"