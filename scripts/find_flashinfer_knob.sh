#!/usr/bin/env bash
# Find how to turn off FlashInfer in this vLLM build.
SP="$HOME/benchmax-env/venv-vllm/lib/python3.12/site-packages"

echo "=== env vars mentioning flashinfer ==="
grep -rnoE 'VLLM_[A-Z0-9_]*FLASHINFER[A-Z0-9_]*' "$SP/vllm/envs.py" 2>/dev/null | sort -u | head -20

echo
echo "=== all VLLM_USE_* sampler-related ==="
grep -rnoE 'VLLM_USE_[A-Z0-9_]+' "$SP/vllm/envs.py" 2>/dev/null | sort -u | head -30

echo
echo "=== CLI flags mentioning flashinfer/sampler ==="
V=~/benchmax-env/venv-vllm/bin
$V/vllm serve --help=all 2>/dev/null | grep -iE 'flashinfer|sampler|attention.backend' | head -20

echo
echo "=== flashinfer bundled cccl version marker ==="
find "$SP/flashinfer/data/cccl" -name '*.h' -path '*cuda_toolkit*' 2>/dev/null | head -2
grep -rnoE 'CUDA_VERSION[^\n]{0,40}' "$SP/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h" 2>/dev/null | head -5