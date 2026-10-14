#!/usr/bin/env bash
# Build llama-perplexity, which computes wikitext perplexity directly from a
# GGUF file.
#
# Why this exists: llama-server's OpenAI-compatible layer rejects echo=true and
# returns empty logprobs, so server-side perplexity is impossible for the
# llama.cpp arm. Measuring the same artifact offline is the honest alternative.
set -uo pipefail
ENV_ROOT="$HOME/benchmax-env"
export CUDA_HOME="$ENV_ROOT/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source "$ENV_ROOT/venv/bin/activate"

# fetch wikitext-2 once into the shared corpus cache
CORPUS_DIR=/mnt/c/benchmaxing/results/raw
mkdir -p "$CORPUS_DIR"
WIKI="$CORPUS_DIR/wikitext2-valid.txt"
if [ ! -s "$WIKI" ]; then
  echo "=== fetching wikitext-2 validation split ==="
  curl -fsSL -o "$WIKI" \
    https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt
fi
echo "corpus: $WIKI ($(wc -c < "$WIKI") bytes)"

echo
echo "=== building llama-perplexity ==="
cmake --build "$ENV_ROOT/build-llamacpp" --target llama-perplexity -j 8 2>&1 | tail -6

BIN="$ENV_ROOT/build-llamacpp/bin/llama-perplexity"
if [ -x "$BIN" ]; then
  echo
  echo "=== sanity run on a tiny slice ==="
  head -c 200000 "$WIKI" > /tmp/wiki_small.txt
  LD_LIBRARY_PATH="$CUDA_HOME/lib64" "$BIN" \
    -m "$ENV_ROOT/models/unsloth__Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf" \
    -f /tmp/wiki_small.txt -c 2048 -ngl 999 2>&1 | tail -12
  echo "PERPLEXITY TOOL OK"
else
  echo "BUILD FAILED - no $BIN"
  exit 1
fi