#!/usr/bin/env bash
source ~/benchmax-env/venv-vllm/bin/activate
vllm serve --help=all > /tmp/vhall.txt 2>&1
echo "bytes=$(wc -c < /tmp/vhall.txt)"
echo "--- grep flags we want ---"
for f in enforce-eager max-num-seqs gpu-memory-utilization seed quantization \
         max-model-len dtype max-num-batched-tokens served-model-name \
         max-logprobs logprobs-mode; do
  n=$(grep -c -- "--$f" /tmp/vhall.txt)
  printf '%-30s %s\n' "--$f" "$n"
done
echo
echo "--- lines mentioning our flags ---"
grep -nE '^\s*--(enforce-eager|max-num-seqs|gpu-memory-utilization|seed|quantization|max-model-len|dtype|max-num-batched-tokens|max-logprobs)' /tmp/vhall.txt | head -20