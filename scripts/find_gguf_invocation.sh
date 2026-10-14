#!/usr/bin/env bash
# Find the working invocation form for vLLM's GGUF loader.
#
# vLLM documents loading a GGUF as "<repo>/<file.gguf>" (a hub-style path), not
# as a bare local filesystem path. Passing a local .gguf makes transformers
# try to parse it as JSON config.json and fail with a misleading OSError.
set -uo pipefail
V=~/benchmax-env/venv-vllm/bin
LOGDIR=~/benchmax-logs; mkdir -p "$LOGDIR"
SP="$HOME/benchmax-env/venv-vllm/lib/python3.12/site-packages"
CU="$SP/nvidia/cu13"
export CUDA_HOME="$CU" CUDA_PATH="$CU"
export PATH="$V:$CU/bin:$PATH"
export LD_LIBRARY_PATH="$CU/lib:$CU/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CU/lib:$CU/lib64:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"

try() {
  local name="$1"; local target="$2"; local port="$3"
  local log="$LOGDIR/gguf-$name.log"
  timeout 420 "$V/vllm" serve "$target" --host 127.0.0.1 --port "$port" \
    --quantization gguf --max-model-len 4096 \
    --gpu-memory-utilization 0.80 --seed 1234 > "$log" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok; then
      echo "  $name: HEALTHY after ${i}s"
      curl -s -m 30 "http://127.0.0.1:$port/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$target\",\"prompt\":\"Hello\",\"max_tokens\":8,\"temperature\":0}" \
        | head -c 220; echo
      kill $pid 2>/dev/null; wait $pid 2>/dev/null
      return 0
    fi
    if ! kill -0 $pid 2>/dev/null; then
      local e=$(grep -oE 'OSError: .{0,70}|error: .{0,60}|not a valid JSON|NotImplementedError: .{0,60}|Unsupported .{0,50}' "$log" | head -1)
      echo "  $name: FAILED [${e:0:90}]"
      return 1
    fi
    sleep 1
  done
  echo "  $name: TIMEOUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
}

echo "=== A: local path (current approach) ==="
try localpath "$HOME/benchmax-env/models/unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf" 8961 || true

echo "=== B: hub-style repo path ==="
try hubpath "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf" 8962 || true