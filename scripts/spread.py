#!/usr/bin/env python
"""Per-replicate spread for suspicious cells, to spot contamination."""
import json
import sys

want_model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
want_wl = sys.argv[2] if len(sys.argv) > 2 else "w1_single_short"

print(f"{'backend':10} {'q':6} {'rep':>3} {'ttft':>8} {'tpot':>7} {'tok/s':>8} "
      f"{'vram':>7} {'util%':>6} {'n_err':>5}")
print("-" * 76)
for line in open("results/raw/runs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") != "perf_run" or r.get("workload") != want_wl:
        continue
    c = r["cell"]
    if c["model_key"] != want_model:
        continue
    s, m = r["summary"], r.get("memory") or {}
    print(f"{c['backend']:10} {c['quant_level']:6} {r['replicate']:3} "
          f"{(s.get('ttft_ms_mean') or 0):8.1f} {(s.get('tpot_ms_mean') or 0):7.1f} "
          f"{(s.get('output_throughput_tok_s') or 0):8.1f} "
          f"{(m.get('gpu_peak_mib') or 0):7} {(m.get('gpu_util_mean_pct') or 0):6} "
          f"{s.get('n_errored', 0):5}")