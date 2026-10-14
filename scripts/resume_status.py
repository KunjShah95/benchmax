#!/usr/bin/env python
"""Show which cells the resume logic considers complete."""
from benchmax.runner import BenchmarkRunner

r = BenchmarkRunner(env_root="x", results_dir="results", models_dir="m",
                    backends=["llamacpp", "vllm", "vllm_gguf"],
                    repeats=3)
done = sorted(r._completed_keys())
print(f"cells with complete data: {len(done)}")
for d in done:
    print("  ", d)