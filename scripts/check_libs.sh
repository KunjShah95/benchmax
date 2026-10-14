#!/usr/bin/env bash
SP="$HOME/benchmax-env/venv-vllm/lib/python3.12/site-packages"
CU="$SP/nvidia/cu13"

echo "=== cudart in cu13/lib ==="
ls -la "$CU/lib/" | grep -E 'cudart' || echo "  none"
echo
echo "=== libcuda in WSL driver dir ==="
ls -la /usr/lib/wsl/lib/ | grep -E 'libcuda\.' | head -6
echo
echo "=== does ld find them with the current LIBRARY_PATH? ==="
export LIBRARY_PATH="$CU/lib:$CU/lib64:/usr/lib/wsl/lib"
echo 'int main(){return 0;}' > /tmp/t.c
c++ /tmp/t.c -o /tmp/t -lcudart -lcuda 2>&1 | head -5 && echo "  link OK" || echo "  link FAILED (above)"
rm -f /tmp/t.c /tmp/t