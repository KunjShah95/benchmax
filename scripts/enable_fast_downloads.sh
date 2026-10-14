#!/usr/bin/env bash
# Stop any in-flight download job and enable fast HF transfers.
# hf_transfer does parallel chunked range requests; the plain resolver was
# pulling at well under 1 MB/s, which is hours for a ~55 GiB matrix.
set -uo pipefail

pkill -f stage4_download.sh 2>/dev/null
pkill -f "benchmax.cli" 2>/dev/null
sleep 1

ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv-bench/bin/activate"
pip install --quiet hf_transfer
python -c 'import hf_transfer; print("hf_transfer installed OK")'

echo
echo "note: set HF_HUB_ENABLE_HF_TRANSFER=1 (or pass enable_hf_transfer=True)"
echo "      to huggingface_hub for the remaining downloads."