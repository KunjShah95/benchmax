#!/usr/bin/env bash
# FlashInfer's bundled CCCL headers clash with the CUDA 13.4 toolkit, breaking
# the JIT sampling kernel. Try newer FlashInfer builds; report what is available.
V=~/benchmax-env/venv-vllm/bin
source $V/activate

echo "=== current ==="
pip list 2>/dev/null | grep -i flashinfer
echo
echo "=== available versions ==="
pip index versions flashinfer-python 2>&1 | head -3
echo
echo "=== flashinfer bundled cccl provenance ==="
SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
sed -n '25,45p' "$SP/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h" 2>/dev/null
echo
echo "=== does the CUDA 13.4 toolkit also ship cuda/std? ==="
CU="$SP/nvidia/cu13"
ls "$CU/include/cuda/std/__cccl/cuda_toolkit.h" 2>/dev/null && echo "  toolkit HAS cuda/std (both on include path -> conflict)" || echo "  toolkit has no cuda/std"