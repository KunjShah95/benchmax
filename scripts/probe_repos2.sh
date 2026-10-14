#!/usr/bin/env bash
# Wider probe for real repo names (exact-name matching, no search API since
# list_models' signature differs across hub versions).
source ~/benchmax-env/venv-bench/bin/activate
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()

cands = [
  # --- Llama 3.2 GGUF ---
  "bartowski/Llama-3.2-1B-Instruct-GGUF",
  "bartowski/Llama-3.2-3B-Instruct-GGUF",
  "unsloth/Llama-3.2-1B-GGUF",
  "unsloth/Llama-3.2-3B-GGUF",
  "unsloth/Llama-3.2-1B-Instruct-GGUF",
  "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
  "lmstudio-community/Llama-3.2-3B-Instruct-GGUF",
  "mradermacher/Llama-3.2-1B-Instruct-GGUF",
  "mradermacher/Llama-3.2-3B-Instruct-GGUF",
  # --- Qwen3 AWQ / GPTQ / FP8 ---
  "Qwen/Qwen3-4B-AWQ",
  "Qwen/Qwen3-4B-FP8",
  "Qwen/Qwen3-0.6B-FP8",
  "Qwen/Qwen3-1.7B-FP8",
  "Qwen/Qwen3-0.6B-GPTQ-Int8",
  "Qwen/Qwen3-1.7B-GPTQ-Int8",
  "cpatonn/Qwen3-4B-AWQ-INT4",
  "cpatonn/Qwen3-8B-AWQ-INT4",
  "soliotech/Qwen3-0.6B-AWQ",
  "soliotech/Qwen3-1.7B-AWQ",
  "MaxQuis/Qwen3-0.6B-AWQ",
  "MaxQuis/Qwen3-1.7B-AWQ",
  "TechxGenus/Meta-Llama-3.2-1B-Instruct-AWQ",
  "TechxGenus/Meta-Llama-3.2-3B-Instruct-AWQ",
  "mradermacher/Meta-Llama-3.2-1B-Instruct-AWQ-INT4",
  "neuralmagic/Meta-Llama-3.2-1B-Instruct-FP8",
  "neuralmagic/Meta-Llama-3.2-3B-Instruct-FP8",
  "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic",
  "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic",
  "neuralmagic/Qwen3-4B-FP8-dynamic",
]
for repo in cands:
    try:
        files = api.list_repo_files(repo)
        st = sorted(f for f in files if f.endswith(".safetensors"))
        gg = sorted(f for f in files if f.endswith(".gguf"))
        quant = "?"
        try:
            import json
            from huggingface_hub import hf_hub_download
        except Exception:
            pass
        print(f"  OK   {repo}")
        if st: print(f"         safetensors: {st[:4]}")
        if gg: print(f"         gguf: {[g for g in gg if any(k in g.upper() for k in ('Q4_K_M','Q8_0','F16'))][:6]}")
    except Exception as e:
        print(f"  --   {repo:55} {type(e).__name__}")
PY