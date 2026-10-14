#!/usr/bin/env bash
# Smoke test for the vLLM arm, including the quality path (echo + prompt
# logprobs) which llama.cpp cannot serve.
set -uo pipefail
ENV_ROOT="$HOME/benchmax-env"
LOGDIR="$HOME/benchmax-logs"; mkdir -p "$LOGDIR"
cd /mnt/c/benchmaxing

MODEL="${1:-$ENV_ROOT/models/Qwen__Qwen3-1.7B}"
[ -d "$MODEL" ] || { echo "MODEL MISSING: $MODEL"; exit 1; }

VENV="$ENV_ROOT/venv-vllm/bin"
export PATH="$VENV:$PATH"
PORT=8890

echo "model: $MODEL"
setsid nohup "$VENV/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port $PORT \
  --max-model-len 8192 --gpu-memory-utilization 0.80 \
  --max-num-seqs 128 --dtype auto --seed 1234 \
  --enforce-eager \
  > "$LOGDIR/smoke_vllm.log" 2>&1 &
echo "launched, waiting for health..."
for i in $(seq 1 300); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && { echo "healthy after ${i}s"; break; }
  if ! pgrep -f "vllm serve" >/dev/null; then echo "VLLM DIED after ${i}s"; tail -30 "$LOGDIR/smoke_vllm.log"; exit 1; fi
  sleep 1
done

echo
echo "=== engine info ==="
grep -E 'quantization|dtype|KV cache|GPU KV|blocks|Capturing|model loading took|Graph capturing|memory' "$LOGDIR/smoke_vllm.log" | tail -12

echo
echo "=== driving with harness client ==="
source "$ENV_ROOT/venv-bench/bin/activate"
python - <<PY
from benchmax.core import OpenAICompatClient, run_concurrent, summarize
from benchmax.workloads import build_prompts
import time, json
c = OpenAICompatClient("http://127.0.0.1:$PORT/v1", model="$MODEL")
for wl in ("w1_single_short", "w4_concurrent_32"):
    t0=time.perf_counter()
    res = run_concurrent(c, build_prompts(wl), max_tokens=128, run_id="smoke", workload=wl)
    wall=time.perf_counter()-t0
    s = summarize(res, wall)
    print(f"  {wl}: n_ok={s['n_ok']} ttft_mean={s['ttft_ms_mean']:.1f}ms "
          f"tpot_mean={s['tpot_ms_mean']:.1f}ms out_tok/s={s['output_throughput_tok_s']:.1f} "
          f"goodput={s['goodput_output_tok_s']}")
    for x in res[:2]:
        print(f"     err={x.error} ptok={x.prompt_tokens} tok={x.completion_tokens} text={x.text_len}")

print()
print("=== quality path: echo + prompt logprobs ===")
got = c.complete_scored("The capital of France is Paris. Water freezes at zero degrees Celsius."*4,
                        max_tokens=1, run_id="smoke", workload="q")
lp = (got["raw"]["choices"][0].get("logprobs") or {})
tl = lp.get("token_logprobs") or []
print("  prompt logprobs:", len(tl), "entries" if tl else "UNSUPPORTED")
if tl:
    import math, statistics
    vals=[v for v in tl if v is not None]
    print(f"  -> perplexity from echo: {math.exp(-statistics.mean(vals)):.4f} over {len(vals)} tokens")

print()
print("=== quality path: top-logprobs for MC ===")
ch = got["raw"]["choices"][0]
lp2 = ch.get("logprobs") or {}
tops = lp2.get("top_logprobs") or []
print("  top_logprobs frames:", len(tops))
if tops: print("  first frame:", list(tops[0].items())[:6])
PY

echo
echo "=== stopping ==="
pkill -f "vllm serve" 2>/dev/null; sleep 3
pkill -9 -f "vllm serve" 2>/dev/null
echo "SMOKE_VLLM DONE"