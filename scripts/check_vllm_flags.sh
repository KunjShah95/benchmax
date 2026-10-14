#!/usr/bin/env bash
# Check which vllm serve flags exist in the installed version.
source ~/benchmax-env/venv-vllm/bin/activate
HELP=$(vllm serve --help 2>/dev/null)
for f in --enforce-eager --max-num-seqs --gpu-memory-utilization --seed \
         --quantization --max-model-len --dtype --served-model-name \
         --disable-log-requests --disable-log-stats --max-num-batched-tokens; do
  if grep -q -- "$f" <<< "$HELP"; then echo "OK      $f"; else echo "MISSING $f"; fi
done
echo
echo "vllm version: $(python -c 'import vllm;print(vllm.__version__)')"