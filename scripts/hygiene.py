#!/usr/bin/env python
"""Find duplicated records and likely GPU contention.

Two hygiene checks that must pass before any number is published:
  1. duplicate (backend, model, quant, workload, replicate) records
  2. single-stream runs reporting very high mean GPU utilisation, which a lone
     small model cannot produce and which indicates another process was on the
     card (an orphaned server from a killed run).
"""
import json
from collections import Counter, defaultdict

records = defaultdict(list)
for line in open("results/raw/runs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") != "perf_run":
        continue
    if not (r.get("summary") or {}).get("n_ok"):
        continue
    c = r["cell"]
    key = (c["backend"], c["model_key"], c["quant_level"],
           r["workload"], r["replicate"])
    records[key].append(r)

dups = {k: v for k, v in records.items() if len(v) > 1}
print(f"successful perf records: {sum(len(v) for v in records.values())}")
print(f"unique (cell,workload,replicate) keys: {len(records)}")
print(f"keys with more than one successful record: {len(dups)}")
cell_dups = Counter((k[0], k[1], k[2]) for k in dups)
for cell, n in cell_dups.most_common():
    print(f"   {cell}  -> {n} duplicated keys")
print()

# GPU utilisation at batch 1
print("=== mean GPU utilisation, single-request workload (w1) ===")
print(f"{'backend':10} {'model':12} {'q':6} {'util%':>7} {'tpot':>7}  flag")
util_rows = defaultdict(list)
for key, rs in records.items():
    if key[3] != "w1_single_short":
        continue
    for r in rs:
        m = r.get("memory") or {}
        if m.get("gpu_util_mean_pct") is None:
            continue
        util_rows[key[:3]].append(
            (m["gpu_util_mean_pct"], (r["summary"].get("tpot_ms_mean") or 0)))
flagged = []
for k, vals in sorted(util_rows.items()):
    u = sum(v[0] for v in vals) / len(vals)
    t = sum(v[1] for v in vals) / len(vals)
    flag = "  <-- HIGH for batch-1" if u > 85 else ""
    if flag:
        flagged.append((k, u, t))
    print(f"{k[0]:10} {k[1]:12} {k[2]:6} {u:7.1f} {t:7.1f}{flag}")
print()
print(f"flagged cells: {len(flagged)}")
for k, u, t in flagged:
    print("   ", k, f"util={u:.1f}% tpot={t:.1f}ms")