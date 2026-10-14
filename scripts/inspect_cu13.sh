#!/usr/bin/env bash
SP="$HOME/benchmax-env/venv-vllm/lib/python3.12/site-packages"
CU="$SP/nvidia/cu13"
echo "=== cu13 exists? ==="; [ -d "$CU" ] && echo yes || echo no
echo
echo "=== top level ==="; ls -1 "$CU" 2>/dev/null
echo
echo "=== bin ==="; ls -1 "$CU/bin" 2>/dev/null | head -20
echo
echo "=== nvvm ==="; ls -1 "$CU/nvvm" 2>/dev/null
echo
echo "=== include (count + sample) ==="
ls "$CU/include" 2>/dev/null | wc -l; ls "$CU/include" 2>/dev/null | head -6
echo
echo "=== lib (count + sample) ==="
ls "$CU/lib" 2>/dev/null | wc -l; ls "$CU/lib" 2>/dev/null | head -6
echo
echo "=== nvcc runs? ==="
"$CU/bin/nvcc" --version 2>&1 | tail -2