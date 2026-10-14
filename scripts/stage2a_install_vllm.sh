#!/usr/bin/env bash
# Stage 2a: install vLLM into its own venv (isolated so concurrent llama.cpp
# builds don't fight over pip). vLLM does not support native Windows, so this
# lives in WSL2.
set -euo pipefail

ENV_ROOT="$HOME/benchmax-env"
VENV="$ENV_ROOT/venv-vllm"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --without-pip "$VENV"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$ENV_ROOT/get-pip.py"
  "$VENV/bin/python" "$ENV_ROOT/get-pip.py" --quiet
  rm -f "$ENV_ROOT/get-pip.py"
fi

source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel

echo "=== installing vLLM (this pulls torch, CUDA libs, transformers) ==="
python -m pip install vllm

echo
echo "=== versions ==="
python - <<'PY'
import torch, vllm, transformers
print("torch      ", torch.__version__)
print("cuda avail ", torch.cuda.is_available())
print("cuda ver   ", torch.version.cuda)
if torch.cuda.is_available():
    print("device     ", torch.cuda.get_device_name(0))
    print("capability ", torch.cuda.get_device_capability(0))
print("vllm       ", vllm.__version__)
print("transformers", transformers.__version__)
PY

echo
echo "STAGE2A OK"