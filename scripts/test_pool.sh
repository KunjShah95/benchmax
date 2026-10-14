#!/usr/bin/env bash
# Run the client-pool test against the mock endpoint (known 100ms injected TTFT).
set -uo pipefail
source ~/benchmax-env/venv-bench/bin/activate
cd /mnt/c/benchmaxing
export PYTHONPATH=/mnt/c/benchmaxing

for N in 1 8 32 64; do
  python - <<PY &
from benchmax import mock_cloud
mock_cloud.STATE.ttft_s = 0.100
mock_cloud.STATE.itl_s = 0.010
mock_cloud.STATE.batching = False
mock_cloud.serve(8901)
PY
  SRV=$!
  sleep 1.5
  echo "--- N=$N ---"
  python scripts/test_pool.py "$N" 8901
  kill $SRV 2>/dev/null
  wait $SRV 2>/dev/null
  sleep 0.5
done