#!/usr/bin/env bash
# Rebuild llama.cpp. Configure step is cached in build-llamacpp, so this goes
# straight to compilation. Full output is preserved for diagnosis.
set -uo pipefail

ENV_ROOT="$HOME/benchmax-env"
CUDA_HOME="$ENV_ROOT/cuda-12.8"
BUILD="$ENV_ROOT/build-llamacpp"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source "$ENV_ROOT/venv/bin/activate"

echo "=== configuring (cached) ==="
cmake -S "$ENV_ROOT/src/llama.cpp" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON > /tmp/cmake_cfg.log 2>&1
echo "configure exit=$?"
tail -5 /tmp/cmake_cfg.log

echo
echo "=== compiling (-j 8; leaving headroom so the WSL host stays responsive) ==="
cmake --build "$BUILD" --target llama-server llama-cli -j 8 2>&1 | tee /tmp/llamacpp_build.log | grep -E 'error|Error|FAILED|warning: unused' | head -30
BUILD_RC=${PIPESTATUS[0]}
echo "build exit=$BUILD_RC"
echo "--- last lines ---"
tail -8 /tmp/llamacpp_build.log

echo
echo "=== artifacts ==="
ls -la "$BUILD/bin/" 2>/dev/null | grep -E 'llama-server|llama-cli' || echo "NO BINARIES"
if [ -x "$BUILD/bin/llama-server" ]; then
  "$BUILD/bin/llama-server" --version 2>&1 | head -3
  echo
  echo "STAGE2C OK"
fi