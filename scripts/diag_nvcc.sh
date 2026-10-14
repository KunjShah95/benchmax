#!/usr/bin/env bash
# Which nvcc can vLLM use? Its venv ships cu13 wheels, which may include a real
# nvcc binary (unlike the 12.8 wheel, which shipped only ptxas).
V=~/benchmax-env/venv-vllm
SP=$(~/benchmax-env/venv-vllm/bin/python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")

echo "=== nvcc binaries in venv ==="
find "$SP/nvidia" -name 'nvcc' -type f 2>/dev/null | head
echo
echo "=== ptxas in venv ==="
find "$SP/nvidia" -name 'ptxas' -type f 2>/dev/null | head -3
echo
echo "=== cuda_nvcc tree ==="
ls -1 "$SP/nvidia/cuda_nvcc/bin/" 2>/dev/null
echo
echo "=== torch's cuda home expectations ==="
~/benchmax-env/venv-vllm/bin/python - <<'PY'
import os, glob
sp = __import__('sysconfig').get_paths()['purelib']
cands = sorted(glob.glob(os.path.join(sp, 'nvidia', 'cuda_nvcc')))
print("candidate CUDA_HOME:", cands)
for c in cands:
    nv = os.path.join(c, 'bin', 'nvcc')
    print(f"  {c}/bin/nvcc exists: {os.path.exists(nv)}")
    inc = os.path.join(c, 'include', 'cuda_runtime.h')
    print(f"  include/cuda_runtime.h exists: {os.path.exists(inc)}")
PY
echo
echo "=== rootless toolkit ==="
ls -1 ~/benchmax-env/cuda-12.8/bin/ | head -8
echo
echo "=== torch version / cuda ==="
~/benchmax-env/venv-vllm/bin/python -c "import torch;print(torch.__version__, torch.version.cuda)"