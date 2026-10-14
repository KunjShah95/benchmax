#!/usr/bin/env bash
# Stage 1 bootstrap v2: no-sudo pip bootstrap, then build tooling.
set -euo pipefail

ENV_ROOT="$HOME/benchmax-env"
VENV="$ENV_ROOT/venv"

mkdir -p "$ENV_ROOT"
cd "$ENV_ROOT"

echo "=== fetching get-pip.py ==="
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
echo "downloaded: $(wc -c < get-pip.py) bytes"

echo "=== creating venv without pip ==="
rm -rf "$VENV"
python3 -m venv --without-pip "$VENV"

echo "=== bootstrapping pip into venv ==="
"$VENV/bin/python" get-pip.py --quiet
rm -f get-pip.py

source "$VENV/bin/activate"
echo "pip: $(python -m pip --version)"

echo "=== build tooling ==="
python -m pip install --quiet --upgrade setuptools wheel
python -m pip install --quiet cmake ninja

echo
echo "=== versions ==="
cmake --version | head -1
ninja --version
python --version

echo
echo "=== ext4 disk speed (why venv lives here) ==="
dd if=/dev/zero of="$ENV_ROOT/ddtest" bs=1M count=512 conv=fdatasync 2>&1 | tail -1
rm -f "$ENV_ROOT/ddtest"

echo
echo "STAGE1 OK"