#!/usr/bin/env python
"""Perplexity / MC table across quantization levels."""
import json

rows = []
for line in open("results/raw/runs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") != "quality_run":
        continue
    c, q = r["cell"], r["quality"]
    p = q.get("perplexity") or {}
    m = q.get("mc_accuracy") or {}
    rows.append((c["model_key"], c["backend"], c["quant_level"],
                 c["quant_scheme"], p.get("ppl"), p.get("ppl_stderr"),
                 p.get("context_tokens") or p.get("total_tokens_scored"),
                 p.get("method", "server echo"),
                 m.get("accuracy"), m.get("n_scored")))

print(f"{'model':14} {'backend':10} {'q':6} {'scheme':22} {'ppl':>9} "
      f"{'+/-':>7} {'ctx/tok':>8} {'mc':>6} {'method':>10}")
print("-" * 104)
for x in sorted(rows):
    ppl = f"{x[4]:.4f}" if isinstance(x[4], (int, float)) else "-"
    se = f"{x[5]:.3f}" if isinstance(x[5], (int, float)) else "-"
    mc = f"{x[8]:.3f}" if isinstance(x[8], (int, float)) else "-"
    meth = "offline" if "offline" in str(x[7]) else "echo"
    print(f"{x[0]:14} {x[1]:10} {x[2]:6} {str(x[3])[:22]:22} {ppl:>9} {se:>7} "
          f"{str(x[6]):>8} {mc:>6} {meth:>10}")