#!/usr/bin/env python
"""Regression test for --resume accounting.

This bug cost a full matrix run: `--resume` compared a set of unique workload
NAMES against len(workloads) * repeats, a threshold that can never be met, so
every restart re-ran the same cells and vLLM cells were never reached.
"""
import json
import os
import tempfile

from benchmax.runner import BenchmarkRunner
from benchmax.workloads import WORKLOADS

tmp = tempfile.mkdtemp()
raw = os.path.join(tmp, "raw")
os.makedirs(raw, exist_ok=True)
path = os.path.join(raw, "runs.jsonl")

REPEATS = 3
WLS = list(WORKLOADS.keys())
EXPECTED = len(WLS) * REPEATS

r = BenchmarkRunner(env_root="x", results_dir=tmp, models_dir=tmp,
                    backends=["llamacpp"], repeats=REPEATS, workloads=WLS)

def emit(cell_id, n):
    with open(path, "a") as f:
        for i in range(n):
            wl = WLS[i % len(WLS)]
            f.write(json.dumps({
                "kind": "perf_run",
                "workload": wl, "replicate": i % REPEATS,
                "cell": {"backend": "llamacpp", "model_key": cell_id,
                         "quant_level": "4bit"},
                "summary": {"n_ok": 1},
            }) + "\n")

emit("full-cell", EXPECTED)
emit("partial-cell", EXPECTED - 3)
emit("empty-cell", 0)

done = r._completed_keys()
print(f"expected threshold: {EXPECTED}")
print(f"marked complete:   {sorted(x[1] for x in done)}")
ok = done == {("llamacpp", "full-cell", "4bit")}
print("RESULT:", "PASS" if ok else "FAIL")

# a cell whose runs all failed must NOT count as complete
with open(path, "a") as f:
    for i in range(EXPECTED):
        f.write(json.dumps({
            "kind": "perf_run", "workload": WLS[i % len(WLS)],
            "replicate": i % REPEATS,
            "cell": {"backend": "llamacpp", "model_key": "failed-cell",
                     "quant_level": "4bit"},
            "summary": {"n_ok": 0},
        }) + "\n")
done2 = r._completed_keys()
ok2 = not any(x[1] == "failed-cell" for x in done2)
print("RESULT (failed runs excluded):", "PASS" if ok2 else "FAIL")
raise SystemExit(0 if (ok and ok2) else 1)