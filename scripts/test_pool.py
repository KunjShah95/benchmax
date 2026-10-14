#!/usr/bin/env python
"""Verify the client is not the bottleneck under concurrency.

Fires N simultaneous requests against an idle endpoint and reports how long
each thread spent *before* the server saw it. Any spread here is our own
overhead and would otherwise be misreported as server latency.
"""
import statistics
import sys
import threading
import time

from benchmax.core import OpenAICompatClient, run_concurrent

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
PORT = sys.argv[2] if len(sys.argv) > 2 else "8899"

c = OpenAICompatClient(f"http://127.0.0.1:{PORT}/v1", model="mock-model")

results = run_concurrent(c, ["warmup text " * 20] * N, max_tokens=8,
                         run_id="pooltest", workload="pooltest")

ok = [r for r in results if r.error is None and r.ttft_s is not None]
ttfts = sorted(r.ttft_s * 1000 for r in ok)
print(f"concurrent={N}  ok={len(ok)}/{N}")
if ttfts:
    print(f"  TTFT ms  min={ttfts[0]:.1f}  p50={ttfts[len(ttfts)//2]:.1f}  "
          f"p95={ttfts[int(0.95*len(ttfts))]:.1f}  max={ttfts[-1]:.1f}")
    print(f"  spread (max/min) = {ttfts[-1]/max(ttfts[0],1e-9):.1f}x")

# Absolute check: against an endpoint whose injected TTFT is 100ms, every
# measurement should land near 100ms regardless of concurrency.
bad = [t for t in ttfts if t > 400]
print(f"  requests exceeding 400ms (4x the injected 100ms): {len(bad)}")
print("  VERDICT:", "client is not bottlenecking" if not bad
      else "CLIENT BOTTLENECK -- numbers would be invalid")