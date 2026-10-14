#!/usr/bin/env bash
# Query which versions of the pip CUDA wheels actually exist for py3.12 linux.
ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv/bin/activate"
for pkg in nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
           nvidia-cuda-nvrtc-cu12 nvidia-cuda-cccl-cu12 nvidia-cuda-cupti-cu12; do
  echo "### $pkg"
  python -m pip index versions "$pkg" 2>&1 | head -3
done