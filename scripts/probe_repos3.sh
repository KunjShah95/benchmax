#!/usr/bin/env bash
source ~/benchmax-env/venv-bench/bin/activate
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
cands = [
  "unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-3B-Instruct",
  "meta-llama/Llama-3.2-1B-Instruct",
  # more 4-bit candidates for the small Qwen3 models
  "cpatonn/Qwen3-0.6B-AWQ-INT4", "wakataka/Qwen3-0.6B-AWQ",
  "wakataka/Qwen3-1.7B-AWQ", "ill893/Qwen3-0.6B-AWQ",
  "castorini/Qwen3-0.6B-AWQ", "FireworkAI/Qwen3-0.6B-AWQ",
  "cloudflare/Qwen3-0.6B-AWQ", "btbtyler09/Qwen3-0.6B-AWQ-INT4",
  "Valdemar112/Qwen3-0.6B-AWQ-INT4", "Valdemar112/Qwen3-1.7B-AWQ-INT4",
  "adamlinj/Qwen3-1.7B-AWQ", "stelterlab/Qwen3-1.7B-AWQ",
  "peculiar-thinking/Qwen3-0.6B-AWQ",
  "Qwen/Qwen3-0.6B-GPTQ-Int4", "Qwen/Qwen3-1.7B-GPTQ-Int4",
  "Qwen/Qwen3-4B-GPTQ-Int8", "Qwen/Qwen3-4B-GPTQ-Int4",
]
for repo in cands:
    try:
        files = api.list_repo_files(repo)
        st = sorted(f for f in files if f.endswith(".safetensors"))
        print(f"  OK   {repo:50} st={len(st)} {st[:3]}")
    except Exception as e:
        print(f"  --   {repo:50} {type(e).__name__}")
PY