#!/usr/bin/env bash
# Inspect where the pip CUDA wheels actually put things.
ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv/bin/activate"
SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "site-packages: $SP"
echo
echo "=== nvidia/* package dirs ==="
ls -1 "$SP/nvidia/" 2>/dev/null || echo "(no nvidia dir)"
echo
for d in "$SP"/nvidia/*/; do
  echo "--- $(basename "$d") ---"
  find "$d" -maxdepth 2 -type d | sed "s|$SP/||" | head -12
done
echo
echo "=== where is nvcc? ==="
find "$SP/nvidia" -name 'nvcc*' 2>/dev/null | head
echo
echo "=== cuda_runtime contents ==="
find "$SP/nvidia/cuda_runtime" -maxdepth 2 2>/dev/null | sed "s|$SP/||" | head -20
echo
echo "=== pip show ==="
python -m pip list 2>/dev/null | grep -i nvidia