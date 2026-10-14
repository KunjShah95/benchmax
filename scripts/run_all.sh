#!/usr/bin/env bash
# One-shot pipeline: selftest -> preflight -> download -> run -> report.
#
# Ordering matters: all downloading and installation happens BEFORE any
# measurement. A pip install or a model download running concurrently would
# steal CPU and disk IO from the server under test and silently inflate every
# latency number.
set -uo pipefail

PROJ="/mnt/c/benchmaxing"
ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv-bench/bin/activate"
cd "$PROJ"

BACKENDS="${BACKENDS:-llamacpp vllm}"
REPEATS="${REPEATS:-3}"
MODELS="${MODELS:-}"

echo "=============================================="
echo " benchmax pipeline"
echo " backends : $BACKENDS"
echo " repeats  : $REPEATS"
echo " models   : ${MODELS:-<all>}"
echo "=============================================="

echo
echo "### 0. measurement-core selftest"
python -m benchmax.cli --results results selftest --port 8991 || {
  echo "SELFTEST FAILED - refusing to benchmark with a broken harness"; exit 1; }

echo
echo "### 1. matrix preview"
python -m benchmax.cli --results results list --backends $BACKENDS ${MODELS:+--models $MODELS}

echo
echo "### 2. artifact preflight"
python -m benchmax.cli --results results preflight --backends $BACKENDS ${MODELS:+--models $MODELS}

echo
echo "### 3. download artifacts (network-bound, nothing measured yet)"
python -m benchmax.cli --results results download --backends $BACKENDS ${MODELS:+--models $MODELS}

echo
echo "### 4. run benchmark"
python -m benchmax.cli --results results --reports reports run \
  --backends $BACKENDS \
  --repeats "$REPEATS" \
  --resume \
  ${MODELS:+--models $MODELS}

echo
echo "### 5. report"
python -m benchmax.cli --results results --reports reports report

echo
echo "### 6. data appendix + PDFs"
PYTHONPATH=/mnt/c/benchmaxing python scripts/build_appendix.py
bash scripts/stage6_pdf.sh

echo
echo "### 7. cloud arm status"
python -m benchmax.cli --results results cloud status

echo
echo "PIPELINE DONE"