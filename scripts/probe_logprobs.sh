#!/usr/bin/env bash
# Find a working way to get prompt logprobs from llama-server for the
# perplexity axis. Tries several request shapes and reports which the server
# accepts.
set -uo pipefail
ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv-bench/bin/activate"
cd /mnt/c/benchmaxing
LOGDIR="$HOME/benchmax-logs"; mkdir -p "$LOGDIR"
export LD_LIBRARY_PATH="$ENV_ROOT/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

MODEL="$ENV_ROOT/models/unsloth__Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf"
PORT=8889
setsid nohup "$ENV_ROOT/build-llamacpp/bin/llama-server" -m "$MODEL" \
  --host 127.0.0.1 --port $PORT -ngl 999 -c 8192 -np 4 --seed 1234 --no-webui \
  > "$LOGDIR/logprob_probe.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && break
  sleep 1
done
echo "server up"

python - <<'PY'
import requests, json
BASE="http://127.0.0.1:8889/v1/completions"
MODEL="/home/jksha/benchmax-env/models/unsloth__Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf"
P="The capital of France is Paris. The capital of Germany is Berlin. Water freezes at zero degrees Celsius at sea level."

variants = [
  ("echo+logprobs=0",  {"echo":True,  "logprobs":0, "max_tokens":1}),
  ("echo+logprobs=1",  {"echo":True,  "logprobs":1, "max_tokens":1}),
  ("echo only",       {"echo":True,  "max_tokens":1}),
  ("logprobs=0 only", {"logprobs":0, "max_tokens":1}),
  ("logprobs=5 only", {"logprobs":5, "max_tokens":8}),
  ("echo+n_probs",    {"echo":True,  "logprobs":5, "n_probs":5, "max_tokens":1}),
  ("echo+max0",       {"echo":True,  "logprobs":0, "max_tokens":0}),
  ("echo+top_n",      {"echo":True,  "logprobs":5, "top_n":5, "max_tokens":1}),
]
for name, extra in variants:
    payload = {"model":MODEL, "prompt":P, "temperature":0.0, "seed":1234, **extra}
    try:
        r = requests.post(BASE, json=payload, timeout=60)
        if r.status_code != 200:
            print(f"  {name:20} HTTP {r.status_code}: {r.text[:150]}")
            continue
        o = r.json(); ch = o["choices"][0]
        lp = ch.get("logprobs") or {}
        tl = lp.get("token_logprobs") or []
        toks = lp.get("tokens") or []
        print(f"  {name:20} OK  echo_text_len={len(ch.get('text',''))} "
              f"tokens={len(toks)} token_logprobs={len(tl)} "
              f"top_logprobs={len(lp.get('top_logprobs') or [])}")
    except Exception as e:
        print(f"  {name:20} ERR {type(e).__name__}: {e}")

# streaming echo (in case non-streaming is the problem)
print("\n  --- streaming variants ---")
for name, extra in [("stream echo+lp0", {"echo":True,"logprobs":0,"max_tokens":1}),
                    ("stream echo+lp5", {"echo":True,"logprobs":5,"max_tokens":1})]:
    payload = {"model":MODEL,"prompt":P,"temperature":0.0,"seed":1234,"stream":True, **extra}
    try:
        r = requests.post(BASE, json=payload, stream=True, timeout=60)
        got_lp=False; frames=0
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:") and line[5:].strip()!="[DONE]":
                frames+=1
                try:
                    o=json.loads(line[5:].strip())
                    lp=(o.get("choices") or [{}])[0].get("logprobs")
                    if lp and lp.get("token_logprobs"): got_lp=True
                except Exception: pass
        print(f"  {name:20} HTTP {r.status_code} frames={frames} prompt_logprobs={got_lp}")
    except Exception as e:
        print(f"  {name:20} ERR {type(e).__name__}: {e}")
PY

echo
echo "=== server-side errors ==="
grep -iE 'error|exception|500|invalid' "$LOGDIR/logprob_probe.log" | tail -12

kill $SRV 2>/dev/null; sleep 1; pkill -f "llama-server.*8889" 2>/dev/null
echo "PROBE DONE"