#!/usr/bin/env python
"""List cells that failed to start, with the root cause."""
import json

for line in open("results/raw/runs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") != "startup_failure":
        continue
    c = r["cell"]
    err = str(r.get("startup_error", "")) or "(no message)"
    first = err.splitlines()[0][:90] if err.splitlines() else "(empty)"
    print(f"{c['backend']:10} {c['model_key']:12} {c['quant_level']:6} "
          f"{str(c.get('quant_scheme'))[:24]:24} {first}")
    # last meaningful line of the log tail
    tail = [ln for ln in str(r.get("log_tail", "")).splitlines()
            if "Error" in ln or "error" in ln or "not found" in ln]
    for ln in tail[-2:]:
        print(f"{'':58}-> {ln.strip()[:110]}")