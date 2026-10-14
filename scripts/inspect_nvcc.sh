#!/usr/bin/env bash
ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv/bin/activate"
SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "=== cuda_nvcc tree ==="
find "$SP/nvidia/cuda_nvcc" -maxdepth 3 | sed "s|$SP/||"
echo
echo "=== bin listing (raw) ==="
ls -la "$SP/nvidia/cuda_nvcc/bin/" 2>&1
echo
echo "=== RECORD from wheel metadata ==="
cat "$SP"/nvidia_cuda_nvcc_cu12-*.dist-info/RECORD 2>/dev/null | cut -d, -f1 | head -40