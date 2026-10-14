#!/usr/bin/env python
"""Final matrix status: which cells measured, which are skipped, which are thin."""
import json
from collections import defaultdict

from benchmax import config as C

cells = C.build_matrix(C.MODELS, ["llamacpp", "vllm", "vllm_gguf"])

runs = defaultdict(set)
qual = defaultdict(int)
summary = {}
with open("results/raw/runs.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        c = r.get("cell") or {}
        k = (c.get("backend"), c.get("model_key"), c.get("quant_level"))
        if r.get("kind") == "perf_run":
            if (r.get("summary") or {}).get("n_ok"):
                runs[k].add((r.get("workload"), r.get("replicate")))
        elif r.get("kind") == "quality_run":
            qual[k] += 1
        elif r.get("kind") == "cell_summary":
            summary[k] = r

expected = 4 * 3  # workloads * replicates

print(f"{'backend':10} {'model':12} {'q':6} {'runs':>5} {'qual':>4} "
      f"{'startup_s':>9}  status")
print("-" * 78)
complete = partial = skipped = 0
for c in cells:
    k = (c.backend, c.model_key, c.quant_level)
    n = len(runs.get(k, ()))
    q = qual.get(k, 0)
    st = summary.get(k, {}).get("startup_s")
    if c.status == "skipped":
        status = "SKIPPED (verified)"
        skipped += 1
    elif n >= expected:
        status = "complete"
        complete += 1
    elif n:
        status = f"PARTIAL ({n}/{expected})"
        partial += 1
    else:
        status = "NO DATA"
        partial += 1
    print(f"{c.backend:10} {c.model_key:12} {c.quant_level:6} {n:5} {q:4} "
          f"{(f'{st:.1f}' if st else '-'):>9}  {status}")

print(f"\ncomplete={complete}  partial={partial}  skipped={skipped}  "
      f"total={len(cells)}")