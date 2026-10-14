#!/usr/bin/env bash
# Re-measure one model's cells in isolation, then compare with the originals.
#
# llamacpp/qwen3-1.7b reported ~8x slower per-token than llamacpp/qwen3-0.6b,
# which is not physically plausible for a 2.8x larger model. Either the card was
# contended at the time or it is a real (if strange) property. This re-runs the
# cell on a quiet machine so the two possibilities can be told apart.
set -uo pipefail
MODEL="${1:-qwen3-1.7b}"
BACKEND="${2:-llamacpp}"

source ~/benchmax-env/venv-bench/bin/activate
cd /mnt/c/benchmaxing

echo "=== pre-check: GPU must be idle and free of other servers ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "util: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
ps -eo pid,args | grep -E 'llama-server|vllm serve' | grep -v grep | head -3 || true

echo
echo "=== dropping prior records for these cells so the re-run is unambiguous ==="
PYTHONPATH=/mnt/c/benchmaxing python - "$MODEL" "$BACKEND" <<'PY'
import json, sys
model, backend = sys.argv[1], sys.argv[2]
src = "results/raw/runs.jsonl"
arch = "results/raw/runs.jsonl.recheck-archived.jsonl"
kept, dropped = [], []
with open(src, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        c = r.get("cell") or {}
        if (c.get("model_key") == model and c.get("backend") == backend
                and r.get("kind") in ("perf_run", "quality_run", "cell_summary")):
            dropped.append(r)
        else:
            kept.append(line)
with open(src, "w", encoding="utf-8") as f:
    for l in kept:
        f.write(l + "\n")
with open(arch, "a", encoding="utf-8") as f:
    for r in dropped:
        f.write(json.dumps(r) + "\n")
print(f"archived {len(dropped)} prior records for {backend}/{model}")
PY

echo
echo "=== re-measuring ${BACKEND} / ${MODEL} ==="
PYTHONPATH=/mnt/c/benchmaxing python -m benchmax.cli \
  --results results --reports reports run \
  --backends "$BACKEND" --models "$MODEL" --repeats 3 --warmup 3

echo
echo "=== comparison: old vs new (w1_single_short) ==="
PYTHONPATH=/mnt/c/benchmaxing python scripts/spread.py "$MODEL" w1_single_short
echo
echo "NOTE: prior records archived to results/raw/runs.jsonl.recheck-archived.jsonl"