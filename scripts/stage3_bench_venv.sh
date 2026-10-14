#!/usr/bin/env bash
# Stage 3: dedicated venv for the benchmax harness itself.
# Kept separate from venv-vllm so that installing analysis libraries cannot
# perturb the environment being measured.
set -euo pipefail

ENV_ROOT="$HOME/benchmax-env"
VENV="$ENV_ROOT/venv-bench"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --without-pip "$VENV"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$ENV_ROOT/get-pip.py"
  "$VENV/bin/python" "$ENV_ROOT/get-pip.py" --quiet
  rm -f "$ENV_ROOT/get-pip.py"
fi

source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel
python -m pip install --quiet \
  requests pandas matplotlib psutil huggingface_hub

echo "=== harness venv ready ==="
python - <<'PY'
import requests, pandas, matplotlib, psutil, huggingface_hub
print("requests     ", requests.__version__)
print("pandas       ", pandas.__version__)
print("matplotlib   ", matplotlib.__version__)
print("psutil       ", psutil.__version__)
print("hf_hub       ", huggingface_hub.__version__)
PY

echo
echo "STAGE3 OK"