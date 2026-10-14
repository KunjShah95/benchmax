#!/usr/bin/env python
"""Import every module and confirm the CLI parses. Cheap structural check that
catches syntax errors and bad imports before a multi-hour run."""
import importlib
import sys

mods = ["benchmax", "benchmax.core", "benchmax.backends", "benchmax.config",
        "benchmax.quality", "benchmax.models", "benchmax.analysis",
        "benchmax.report", "benchmax.cli", "benchmax.cloud",
        "benchmax.mock_cloud", "benchmax.workloads"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK      {m}")
    except Exception as e:
        print(f"  FAIL    {m}: {type(e).__name__}: {e}")
        bad.append(m)

# sanity-check the matrix builds
from benchmax import config as C
cells = C.build_matrix(C.MODELS, ["llamacpp", "vllm", "vllm_gguf"])
print(f"\nmatrix: {len(cells)} cells "
      f"({sum(1 for c in cells if c.status != 'skipped')} runnable)")
print("VERDICT:", "FAIL " + ",".join(bad) if bad else "all modules OK")
sys.exit(1 if bad else 0)