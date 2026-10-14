#!/usr/bin/env python
"""Show the per-request errors recorded for a failing workload."""
import json
import sys

path = "results/raw/runs.jsonl"
want = sys.argv[1] if len(sys.argv) > 1 else "w2_single_prefill"

seen = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("kind") != "perf_run" or r.get("workload") != want:
            continue
        seen += 1
        c = r["cell"]
        print(f"--- {c['backend']} {c['model_key']} {c['quant_level']} "
              f"rep{r['replicate']} ---")
        for req in r.get("requests", [])[:3]:
            print(f"   status={req.get('http_status')} "
                  f"finish={req.get('finish_reason')} "
                  f"ptok={req.get('prompt_tokens')} err={req.get('error')}")
        if seen >= 2:
            break