#!/usr/bin/env bash
# Execute the benchmark matrix.
# Usage: stage5_run.sh <backends...> -- [--models m1 m2] [--repeats N] [--quants ...]
set -uo pipefail

ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv-bench/bin/activate"
cd /mnt/c/benchmaxing

# Persistent log: stdout may be lost when the WSL VM restarts mid-run, and the
# supervisor needs to read progress from a file that survives that.
LOGDIR="$HOME/benchmax-logs"
mkdir -p "$LOGDIR"
echo "=== stage5 started $(date -Is) ===" >> "$LOGDIR/slice.log"

BACKENDS=()
MODELS=()
QUANTS=()
REPEATS=3
WARMUP=3
WORKLOADS=()
NO_QUALITY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) shift; while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
    --quants) shift; while [[ $# -gt 0 && "$1" != --* ]]; do QUANTS+=("$1"); shift; done ;;
    --workloads) shift; while [[ $# -gt 0 && "$1" != --* ]]; do WORKLOADS+=("$1"); shift; done ;;
    --repeats) shift; REPEATS="$1"; shift ;;
    --warmup) shift; WARMUP="$1"; shift ;;
    --no-quality) NO_QUALITY=1; shift ;;
    --) shift; break ;;
    *) BACKENDS+=("$1"); shift ;;
  esac
done

ARGS=(run --repeats "$REPEATS" --warmup "$WARMUP" --resume)
[[ ${#BACKENDS[@]} -gt 0 ]] && ARGS+=(--backends "${BACKENDS[@]}")
[[ ${#MODELS[@]}   -gt 0 ]] && ARGS+=(--models "${MODELS[@]}")
[[ ${#QUANTS[@]}   -gt 0 ]] && ARGS+=(--quants "${QUANTS[@]}")
[[ ${#WORKLOADS[@]} -gt 0 ]] && ARGS+=(--workloads "${WORKLOADS[@]}")
[[ $NO_QUALITY -eq 1 ]] && ARGS+=(--no-quality)

echo "executing: python -m benchmax.cli run ${ARGS[*]}"
python -m benchmax.cli --results results --reports reports "${ARGS[@]}" 2>&1 \
  | tee -a "$LOGDIR/slice.log"

echo
echo "=== raw record counts ===" | tee -a "$LOGDIR/slice.log"
for kind in perf_run quality_run cell_summary startup_failure run_error quality_error; do
  n=$(grep -c "\"kind\": \"$kind\"" results/raw/runs.jsonl 2>/dev/null || echo 0)
  echo "  $kind: $n" | tee -a "$LOGDIR/slice.log"
done

echo "STAGE5 DONE" | tee -a "$LOGDIR/slice.log"