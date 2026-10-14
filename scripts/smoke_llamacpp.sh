#!/usr/bin/env bash
# Smoke test: start llama-server on a real model and drive it with the harness
# client. Catches protocol or CUDA problems before committing to the full matrix.
set -uo pipefail
ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv-bench/bin/activate"
cd /mnt/c/benchmaxing

# persistent log dir: /tmp is cleared on WSL restart, and a missing redirect
# target silently prevents the server from launching at all
LOGDIR="$HOME/benchmax-logs"
mkdir -p "$LOGDIR"
SRVLOG="$LOGDIR/smoke_server.log"

MODEL="${1:-$ENV_ROOT/models/unsloth__Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf}"
BIN="$ENV_ROOT/build-llamacpp/bin/llama-server"
PORT=8888

# CUDA runtime libs are not on WSL's default search path
export LD_LIBRARY_PATH="$ENV_ROOT/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

echo "model: $MODEL"
ls -la "$MODEL" || { echo "MODEL MISSING"; exit 1; }
echo "ldd check:"; ldd "$BIN" 2>&1 | grep -E 'not found' && echo "  !! unresolved deps above" || echo "  all deps resolved"

echo
echo "=== starting llama-server ==="
setsid nohup "$BIN" -m "$MODEL" --host 127.0.0.1 --port $PORT \
  -ngl 999 -c 8192 -np 8 --seed 1234 --no-webui --cont-batching \
  > "$SRVLOG" 2>&1 &
SRV=$!
echo "pid=$SRV  log=$SRVLOG"

for i in $(seq 1 180); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok; then
    echo "healthy after ${i}s"; break
  fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "SERVER DIED after ${i}s"; tail -20 "$SRVLOG"; exit 1
  fi
  sleep 1
done

echo
echo "=== server log highlights ==="
grep -E 'CUDA|offloaded|model size|n_layer|CUDA0|graph|slot|BLAS' "$SRVLOG" | head -14

echo
echo "=== driving with harness client ==="
python - <<PY
from benchmax.core import OpenAICompatClient, run_concurrent, summarize
from benchmax.workloads import build_prompts
import json, requests

r = requests.get("http://127.0.0.1:$PORT/health", timeout=5)
print("health:", r.status_code, r.text.strip())

c = OpenAICompatClient("http://127.0.0.1:$PORT/v1", model="$MODEL")
res = run_concurrent(c, build_prompts("w1_single_short"), max_tokens=64,
                     run_id="smoke", workload="w1_single_short")
for x in res:
    print(f"  err={x.error} ttft={x.ttft_s and round(x.ttft_s*1000,1)}ms "
          f"itl_p50={x.itl_p50_s and round(x.itl_p50_s*1000,2)}ms "
          f"tok={x.completion_tokens} ptok={x.prompt_tokens} text={x.text_len}")
s = summarize(res, 0.001)
print("  summary:", json.dumps({k:v for k,v in s.items() if k in
      ("n_ok","ttft_ms_mean","tpot_ms_mean","output_throughput_tok_s")}, default=str))
PY

echo
echo "=== logprobs/echo support (perplexity path) ==="
python - <<PY
from benchmax.core import OpenAICompatClient
c = OpenAICompatClient("http://127.0.0.1:$PORT/v1", model="$MODEL")
try:
    got = c.complete_scored("The quick brown fox jumps over the lazy dog. " * 8,
                            max_tokens=1, run_id="smoke", workload="q")
    lp = got["raw"]["choices"][0].get("logprobs") or {}
    tl = lp.get("token_logprobs")
    print("echo+logprobs:", "OK" if tl else "UNSUPPORTED",
          f"({len(tl) if tl else 0} token logprobs)")
except Exception as e:
    print("echo+logprobs: FAILED", type(e).__name__, e)
PY

echo
echo "=== stopping ==="
kill -TERM -$(ps -o pgid= $SRV 2>/dev/null | tr -d ' ') 2>/dev/null || kill $SRV 2>/dev/null
sleep 2
pkill -f "llama-server.*$PORT" 2>/dev/null
echo "SMOKE DONE"