#!/usr/bin/env bash
# Probe candidate HF repos to find which 4-bit checkpoints actually exist.
source ~/benchmax-env/venv-bench/bin/activate
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()

print("=== SEARCH: qwen3 awq ===")
for q in ["Qwen3 AWQ", "Qwen3 GPTQ-Int4", "Llama-3.2 AWQ", "Llama-3.2 GPTQ"]:
    try:
        res = list(api.list_models(search=q, limit=12, sort="downloads", direction=-1))
        print(f"\n-- search: {q!r}")
        for m in res:
            print(f"   {m.id:65} dl={getattr(m,'downloads',0)}")
    except Exception as e:
        print(f"  search {q!r} failed: {type(e).__name__}")

print("\n=== EXACT repo probes ===")
cands = [
    "Qwen/Qwen3-0.6B-AWQ", "Qwen/Qwen3-1.7B-AWQ", "Qwen/Qwen3-4B-AWQ",
    "Qwen/Qwen3-0.6B-FP8", "Qwen/Qwen3-1.7B-FP8",
    "cpatonn/Qwen3-0.6B-AWQ-INT4", "cpatonn/Qwen3-1.7B-AWQ-INT4",
    "hugging-quants/Meta-Llama-3.2-1B-Instruct-AWQ-INT4",
    "hugging-quants/Meta-Llama-3.2-3B-Instruct-AWQ-INT4",
    "unsloth/Qwen3-0.6B-AWQ", "unsloth/Qwen3-1.7B-AWQ",
    "unsloth/Llama-3.2-1B-Instruct-AWQ", "unsloth/Llama-3.2-3B-Instruct-AWQ",
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
    "unsloth/Llama-3.2-1B", "unsloth/Llama-3.2-3B",
    "unsloth/Qwen3-0.6B-GGUF", "unsloth/Qwen3-1.7B-GGUF", "unsloth/Qwen3-4B-GGUF",
    "bartowski/Llama-3.2-1B-GGUF", "bartowski/Llama-3.2-3B-GGUF",
]
for repo in cands:
    try:
        files = api.list_repo_files(repo)
        st = [f for f in files if f.endswith(".safetensors")]
        gg = [f for f in files if f.endswith(".gguf")]
        cfg = "config.json" in files
        print(f"  OK   {repo:58} cfg={cfg} st={len(st)} gguf={len(gg)}")
    except Exception as e:
        print(f"  --   {repo:58} {type(e).__name__}")
PY