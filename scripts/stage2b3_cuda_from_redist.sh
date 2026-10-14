#!/usr/bin/env bash
# Stage 2b3: build a complete, rootless CUDA_HOME from NVIDIA's official
# redistributable component tarballs. The pip wheels ship only fragments
# (e.g. ptxas without nvcc), so we go to the source tarballs instead.
set -euo pipefail

ENV_ROOT="$HOME/benchmax-env"
CUDA_HOME="$ENV_ROOT/cuda-12.8"
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
cd "$ENV_ROOT"

COMPONENTS=(
  "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz"
  "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz"
  "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz"
  "libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.4.1-archive.tar.xz"
)

rm -rf "$CUDA_HOME" redist_tmp
mkdir -p redist_tmp "$CUDA_HOME"/{bin,include,lib64,nvvm}

echo "=== downloading (parallel) ==="
for rel in "${COMPONENTS[@]}"; do
  fn=$(basename "$rel")
  echo "  -> $fn"
  curl -fsSL -o "redist_tmp/$fn" "$BASE/$rel" &
done
wait
ls -la redist_tmp/

echo
echo "=== extracting and merging into CUDA_HOME ==="
for tarball in redist_tmp/*.tar.xz; do
  tmpdir=$(mktemp -d)
  tar -xf "$tarball" -C "$tmpdir"
  root=$(find "$tmpdir" -maxdepth 1 -mindepth 1 -type d | head -1)
  echo "  merged $(basename "$root")"
  # bin/
  [ -d "$root/bin" ] && cp -a "$root/bin/." "$CUDA_HOME/bin/"
  # include/ (merge, never clobber wholesale)
  [ -d "$root/include" ] && cp -a "$root/include/." "$CUDA_HOME/include/"
  # nvvm/
  [ -d "$root/nvvm" ] && cp -a "$root/nvvm/." "$CUDA_HOME/nvvm/"
  # lib/ -> lib64/
  if [ -d "$root/lib" ]; then
    cp -a "$root/lib/." "$CUDA_HOME/lib64/"
  fi
  rm -rf "$tmpdir"
done

# CUDA convention: tools look in lib64, but some link scripts expect lib
ln -sfn "$CUDA_HOME/lib64" "$CUDA_HOME/lib"

# cublas ships versioned .so plus a .so symlink; ensure the unversioned one exists
cd "$CUDA_HOME/lib64"
for base in libcublas libcublasLt libcudart; do
  if [ ! -e "$base.so" ] && ls "${base}.so."* >/dev/null 2>&1; then
    latest=$(ls -1 "${base}.so."* | sort -V | tail -1)
    ln -sfn "$(basename "$latest")" "$base.so"
    echo "  linked $base.so -> $(basename "$latest")"
  fi
done
cd "$ENV_ROOT"

echo
echo "=== nvcc check ==="
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -3

echo
echo "=== compiler binaries present ==="
for b in nvcc cicc cudafe++ ptxas nvlink fatbinary bin2c; do
  [ -x "$CUDA_HOME/bin/$b" ] && echo "  OK      $b" || echo "  MISSING $b"
done

echo
echo "=== header spot-check ==="
for h in cuda_runtime.h cublas_v2.h cuda_bf16.h cublasLt.h thrust/version.h cub/cub.cuh; do
  [ -e "$CUDA_HOME/include/$h" ] && echo "  OK      $h" || echo "  MISSING $h"
done

echo
echo "=== nvcc actually compiles + runs a kernel on sm_89 ==="
cat > "$ENV_ROOT/t.cu" <<'CU'
#include <cstdio>
__global__ void k(float*o){o[threadIdx.x]=threadIdx.x*2.0f;}
int main(){
  float*d=nullptr; cudaMalloc(&d,4*sizeof(float));
  k<<<1,4>>>(d); cudaDeviceSynchronize();
  float h[4]; cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
  printf("kernel output: %g %g %g %g\n",h[0],h[1],h[2],h[3]);
  return cudaGetLastError()!=cudaSuccess;
}
CU
nvcc -arch=sm_89 -o "$ENV_ROOT/t" "$ENV_ROOT/t.cu"
"$ENV_ROOT/t"
rm -f "$ENV_ROOT/t" "$ENV_ROOT/t.cu"

rm -rf "$ENV_ROOT/redist_tmp"
echo
echo "STAGE2B3 OK"