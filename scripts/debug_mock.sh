#!/usr/bin/env bash
# Debug the mock endpoint's SSE framing directly.
source ~/benchmax-env/venv-bench/bin/activate
cd /mnt/c/benchmaxing
python - <<'PY'
import threading, time, requests, json
from benchmax import mock_cloud

mock_cloud.STATE.ttft_s = 0.05
mock_cloud.STATE.itl_s = 0.01
threading.Thread(target=mock_cloud.serve, args=(8998,), daemon=True).start()
time.sleep(1.0)

base = "http://127.0.0.1:8998/v1"
print("health:", requests.get("http://127.0.0.1:8998/health", timeout=5).status_code)

payload = {"model":"mock-model","prompt":"hello world foo bar","max_tokens":5,
           "stream": True, "temperature":0.0, "seed":1234}
r = requests.post(base+"/completions", json=payload, stream=True, timeout=30)
print("status:", r.status_code)
print("headers:", dict(r.headers))
raw = b""
for line in r.iter_lines(decode_unicode=False):
    print("LINE:", repr(line))
    raw += (line or b"") + b"\n"
print("---- non-streaming ----")
payload2 = dict(payload); payload2["stream"]=False; payload2["echo"]=True; payload2["logprobs"]=0
r2 = requests.post(base+"/completions", json=payload2, timeout=30)
print("status:", r2.status_code)
obj = r2.json()
print("keys:", list(obj.keys()))
ch = obj["choices"][0]
print("choice keys:", list(ch.keys()))
lp = ch.get("logprobs")
print("logprobs type:", type(lp))
if isinstance(lp, dict):
    print("  logprobs keys:", list(lp.keys()))
    print("  n token_logprobs:", len(lp.get("token_logprobs") or []))
PY