#!/usr/bin/env python
"""Quick console view of raw benchmark results (used during a run)."""
import json
import sys
import os

path = sys.argv[1] if len(sys.argv) > 1 else "results/raw/runs.jsonl"

perf, qual, other = [], [], []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        k = r.get("kind")
        if k == "perf_run":
            perf.append(r)
        elif k == "quality_run":
            qual.append(r)
        else:
            other.append(r)

print(f"{len(perf)} perf runs, {len(qual)} quality runs, {len(other)} other records\n")

hdr = (f"{'backend':10} {'model':22} {'q':6} {'workload':18} "
       f"{'ttft':>8} {'tpot':>8} {'itl95':>8} {'tok/s':>9} {'goodput':>8} {'vram':>7}")
print(hdr)
print("-" * len(hdr))
for r in perf:
    s, c, m = r["summary"], r["cell"], r.get("memory") or {}
    if s.get("n_ok", 0) == 0:
        print(f"{c['backend']:10} {c['model_key']:22} {c['quant_level']:6} "
              f"{r['workload']:18}  ERROR: all requests failed")
        continue

    def g(k, f="{:.1f}"):
        v = s.get(k)
        return f.format(v) if isinstance(v, (int, float)) else "-"

    print(f"{c['backend']:10} {c['model_key']:22} {c['quant_level']:6} "
          f"{r['workload']:18} {g('ttft_ms_mean'):>8} {g('tpot_ms_mean'):>8} "
          f"{g('itl_ms_p95'):>8} {g('output_throughput_tok_s'):>9} "
          f"{g('goodput_output_tok_s'):>8} "
          f"{m.get('gpu_peak_mib') if m.get('gpu_peak_mib') is not None else '-':>7}")

print("\n=== quality ===")
for r in qual:
    c, q = r["cell"], r["quality"]
    p, mc = q.get("perplexity") or {}, q.get("mc_accuracy") or {}
    ppl = p.get("ppl")
    print(f"{c['backend']:10} {c['model_key']:22} {c['quant_level']:6} "
          f"ppl={ppl if ppl is None else round(ppl, 4)}"
          f"{' +/- ' + str(round(p['ppl_stderr'], 4)) if p.get('ppl_stderr') else ''}"
          f"  [{p.get('method', 'server echo')[:34]}]"
          f"  mc={mc.get('accuracy') if mc.get('accuracy') is not None else mc.get('status')}")

print("\n=== startup / lifecycle ===")
for r in other:
    if r.get("kind") != "cell_summary":
        continue
    c, m = r["cell"], r.get("memory_overall") or {}
    print(f"{c['backend']:10} {c['model_key']:22} {c['quant_level']:6} "
          f"startup={r.get('startup_s', 0):7.1f}s  "
          f"vram_peak={m.get('gpu_peak_mib')}  rss_peak={m.get('host_rss_peak_mib')}")

fails = [r for r in other if r.get("kind") in ("startup_failure", "run_error", "quality_error")]
if fails:
    print("\n=== failures ===")
    for r in fails:
        c = r.get("cell") or {}
        print(f"{r.get('kind'):18} {c.get('backend')} {c.get('model_key')} "
              f"{c.get('quant_level')}: {str(r.get('error'))[:150]}")