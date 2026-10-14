#!/usr/bin/env bash
# Validate the full vLLM environment exactly as the backend sets it up.
set -uo pipefail
V=~/benchmax-env/venv-vllm/bin
LOGDIR=~/benchmax-logs; mkdir -p "$LOGDIR"
MODEL="$HOME/benchmax-env/models/Qwen__Qwen3-0.6B"
SP="$HOME/benchmax-env/venv-vllm/lib/python3.12/site-packages"
CU="$SP/nvidia/cu13"

export CUDA_HOME="$CU" CUDA_PATH="$CU"
export PATH="$V:$CU/bin:$PATH"
export LD_LIBRARY_PATH="$CU/lib:$CU/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CU/lib:$CU/lib64:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"

PORT=8951
LOG="$LOGDIR/vllmfix-libpath.log"
"$V/vllm" serve "$MODEL" --host 127.0.0.1 --port $PORT \
  --max-model-len 4096 --gpu-memory-utilization 0.80 --seed 1234 > "$LOG" 2>&1 &
PID=$!

for i in $(seq 1 600); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
    echo "HEALTHY after ${i}s"
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "DIED after ${i}s"
    grep -oE 'cannot find -l[a-zA-Z0-9_.]+|error: .{0,90}|Ninja build failed' "$LOG" | sort -u | head -8
    exit 1
  fi
  sleep 1
done

if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok; then
  echo
  echo "=== quick request ==="
  source ~/benchmax-env/venv-bench/bin/activate
  cd /mnt/c/benchmaxing
  PYTHONPATH=/mnt/c/benchmaxing python - <<PY
from benchmax.core import OpenAICompatClient
c = OpenAICompatClient("http://127.0.0.1:$PORT/v1", model="$MODEL")
r = c.complete_streaming(prompt="The capital of France is", max_tokens=24,
                         run_id="t", workload="t")
print("  err:", r.error, "tok:", r.completion_tokens, "text_len:", r.text_len)
PY
fi

kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "DONE"