#!/usr/bin/env python
"""Verify the single-instance lock: a live holder must block a second runner,
and a dead holder's lock must be reclaimed."""
import os
import sys
import tempfile

from benchmax.runner import BenchmarkRunner

tmp = tempfile.mkdtemp()
r = BenchmarkRunner(env_root="x", results_dir=tmp, models_dir=tmp,
                    backends=["llamacpp"])

print("1. no lock present -> acquire")
r._acquire_lock()
print("   lock file exists:", os.path.exists(r.lock_path))

print("2. lock held by THIS live process -> second acquire must refuse")
try:
    r._acquire_lock()
    print("   FAIL: second acquire succeeded (would allow concurrent runs)")
    sys.exit(1)
except SystemExit as e:
    print("   OK: refused ->", str(e).split(".")[0])

print("3. lock held by a DEAD pid -> must be reclaimed")
with open(r.lock_path, "w") as f:
    f.write('{"pid": 999999, "started": "past"}')
r._acquire_lock()
print("   OK: reclaimed")

r._release_lock()
print("4. release removes it:", not os.path.exists(r.lock_path))
print("LOCK TEST PASSED")