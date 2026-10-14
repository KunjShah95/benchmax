#!/usr/bin/env bash
# Stage 2b: provision a pip-based CUDA toolkit (nvcc + headers + cublas) and
# assemble it into a single CUDA_HOME that llama.cpp's build can consume.
# Rationale: apt/nvidia driver installs need sudo and can break the WSL GPU
# passthrough. The pip nvidia-* wheels ship the same toolkit pieces.
set -euo pipefail

ENV_ROOT="$HOME/benchmax-env"
VENV="$ENV_ROOT/venv"
source "$VENV/bin/activate"

CUDA_VER="12.8"

echo "=== installing CUDA toolkit pieces (cu${CUDA_VER}) ==="
# NOTE: there is no separate `nvidia-cuda-cudart-cu12` wheel; cudart ships
# inside nvidia-cuda-runtime-cu12. Asking for it makes pip fail outright.
python -m pip install --quiet \
  "nvidia-cuda-nvcc-cu12==${CUDA_VER}.93" \
  "nvidia-cuda-runtime-cu12==${CUDA_VER}.90" \
  "nvidia-cublas-cu12==${CUDA_VER}.4.1" \
  "nvidia-cuda-nvrtc-cu12==${CUDA_VER}.93" \
  "nvidia-cuda-cccl-cu12==${CUDA_VER}.90" \
  "nvidia-cuda-cupti-cu12==${CUDA_VER}.90"

SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "site-packages: $SP"

CUDA_HOME="$ENV_ROOT/cuda-12.8"
rm -rf "$CUDA_HOME"
mkdir -p "$CUDA_HOME"/{bin,include,lib64}

echo "=== assembling CUDA_HOME at $CUDA_HOME ==="
# headers from every nvidia wheel that ships any
for d in "$SP"/nvidia/*/include; do
  [ -d "$d" ] || continue
  for f in "$d"/*; do
    b=$(basename "$f")
    [ -e "$CUDA_HOME/include/$b" ] || ln -sf "$f" "$CUDA_HOME/include/$b"
  done
done

# binaries (nvcc, ptxas, nvlink, fatbinary, cicc...)
for f in "$SP"/nvidia/cuda_nvcc/bin/*; do
  [ -f "$f" ] && ln -sf "$f" "$CUDA_HOME/bin/$(basename "$f")"
done

# shared libraries -> lib64
for d in "$SP"/nvidia/*/lib; do
  [ -d "$d" ] || continue
  for f in "$d"/*.so*; do
    [ -f "$f" ] && ln -sf "$f" "$CUDA_HOME/lib64/$(basename "$f")"
  done
done

export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

echo
echo "=== verify nvcc ==="
"$CUDA_HOME/bin/nvcc" --version | tail -3

echo
echo "=== header spot-check (needed by ggml-cuda) ==="
for h in cuda_runtime.h cublas_v2.h cuda_fp16.h thrust/version.h cub/cub.cuh; do
  if [ -e "$CUDA_HOME/include/$h" ]; then echo "  OK      $h"; else echo "  MISSING $h"; fi
done

echo
echo "=== nvcc compiles a trivial kernel? ==="
cat > "$ENV_ROOT/t.cu" <<'CU'
#include <cstdio>
__global__ void k(float*o){o[threadIdx.x]=threadIdx.x;}
int main(){float*d;cudaMalloc(&d,4*4);k<<<1,4>>>(d);
  float h[4];cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
  printf("%g %g %g %g\n",h[0],h[1],h[2],h[3]);return 0;}
CU
"$CUDA_HOME/bin/nvcc" -arch=sm_89 -o "$ENV_ROOT/t" "$ENV_ROOT/t.cu" 2>&1 | tail -5
if [ -x "$ENV_ROOT/t" ]; then
  echo -n "  kernel result: "; "$ENV_ROOT/t"
else
  echo "  NVCC COMPILE FAILED"
fi
rm -f "$ENV_ROOT/t" "$ENV_ROOT/t.cu"

echo
echo "STAGE2B OK"