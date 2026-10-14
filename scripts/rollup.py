#!/usr/bin/env python
"""Compact per-config rollup from raw records (median across replicates)."""
import json
import statistics
import sys
from collections import defaultdict

path = "results/raw/runs.jsonl"
rows = defaultdict(list)
rows_vram = defaultdict(list)
meta = {}

with open(path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("kind") != "perf_run":
            continue
        c, s = r["cell"], r["summary"]
        if s.get("n_ok", 0) == 0:
            continue
        k = (c["backend"], c["model_key"], c["quant_level"], r["workload"])
        rows[k].append(s)
        meta[k] = c
        # memory stats live on the record, not inside summary
        vram_peak = (r.get("memory") or {}).get("gpu_peak_mib")
        if vram_peak:
            rows_vram[k].append(vram_peak)

def med(vals):
    return statistics.median(vals) if vals else None

workload = sys.argv[1] if len(sys.argv) > 1 else "w4_concurrent_32"
hdr = (f"{'backend':10} {'model':12} {'q':6} {'scheme':20} "
       f"{'ttft_p50':>9} {'ttft_p95':>9} {'tpot':>7} {'itl95':>7} "
       f"{'tok/s':>8} {'goodput':>8} {'vram':>7}")
print(f"--- {workload} (median across replicates) ---")
print(hdr)
print("-" * len(hdr))
for k in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
    if k[3] != workload:
        continue
    ss = rows[k]
    c = meta[k]
    vram = rows_vram[k]
    print(f"{k[0]:10} {k[1]:12} {k[2]:6} {str(c.get('quant_scheme'))[:20]:20} "
          f"{med([s['ttft_ms_mean'] for s in ss]) or 0:9.1f} "
          f"{med([s['ttft_ms_p95'] for s in ss]) or 0:9.1f} "
          f"{med([s['tpot_ms_mean'] for s in ss]) or 0:7.1f} "
          f"{med([s['itl_ms_p95'] for s in ss]) or 0:7.1f} "
          f"{med([s['output_throughput_tok_s'] for s in ss]) or 0:8.1f} "
          f"{med([s['goodput_output_tok_s'] for s in ss]) or 0:8.1f} "
          f"{max(vram) if vram else 0:7.0f}")