#!/usr/bin/env python
"""Show quality_run error details."""
import json

for line in open("results/raw/runs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") != "quality_error":
        continue
    c = r["cell"]
    print(f"{c['backend']:10} {c['model_key']:12} {c['quant_level']:6} "
          f"{str(r.get('error'))[:130]}")

print("\n=== quality_run status detail (non-ok) ===")
for line in open("results/raw/runs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") != "quality_run":
        continue
    c, q = r["cell"], r["quality"]
    p = q.get("perplexity") or {}
    m = q.get("mc_accuracy") or {}
    if p.get("status") != "ok" or m.get("status") != "ok":
        print(f"{c['backend']:10} {c['model_key']:12} {c['quant_level']:6} "
              f"ppl={p.get('status')} {str(p.get('errors'))[:70]} | "
              f"mc={m.get('status')} {str(m.get('errors') or m.get('reason'))[:60]}")