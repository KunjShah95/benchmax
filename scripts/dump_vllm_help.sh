#!/usr/bin/env bash
source ~/benchmax-env/venv-vllm/bin/activate
vllm serve --help > /tmp/vh.txt 2>&1
echo "rc=$? bytes=$(wc -c < /tmp/vh.txt)"
echo "--- first 30 lines ---"
head -30 /tmp/vh.txt
echo "--- grep key flags ---"
for f in enforce-eager max-num-seqs gpu-memory-utilization seed quantization max-model-len dtype; do
  printf '%-28s ' "$f"
  grep -c -- "$f" /tmp/vh.txt
done