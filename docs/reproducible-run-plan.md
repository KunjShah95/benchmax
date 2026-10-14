# Reproducible run plan

Goal: obtain one small, trustworthy vLLM vs llama.cpp comparison before expanding the matrix.

## 1. Environment setup (WSL2 Ubuntu)

Run once on a quiet machine:

```bash
cd /mnt/c/benchmaxing
bash scripts/stage1_bootstrap.sh
bash scripts/stage2b3_cuda_from_redist.sh
bash scripts/stage2a_install_vllm.sh
bash scripts/stage2c_build_llamacpp.sh
bash scripts/stage3_bench_venv.sh
```

Do not benchmark while the desktop/GPU is under other load.

## 2. Sanity checks

```bash
python -m benchmax.cli selftest
python -m benchmax.cli preflight
```

Both must pass. Fix any import/model resolution issue before measuring.

## 3. First controlled matrix

Use only one model, one workload at a time, and one repeat:

```bash
python -m benchmax.cli run \
  --backends llamacpp vllm \
  --models qwen3-0.6b \
  --quants 4bit 8bit \
  --workloads w1_single_short w4_concurrent_32 \
  --repeats 1
```

Then inspect:

```bash
python -c "import json; print(sum(1 for _ in open('results/raw/runs.jsonl')))"
python -m benchmax.cli report
```

## 4. Trust checks before scaling

- No physically implausible TTFT/TPOT outliers.
- `n_failed` and skipped cells are present in the raw data.
- `token_usage_available` / `usage_coverage` are not silently treated as valid totals when usage is absent.
- Generation-eval records exist and include `dataset_version` and `dataset_hash`.
- GPU utilization at batch size 1 is consistent with an idle, uncontended run.

## 5. Scale only after the small run is clean

- Increase models/quants/workloads one dimension at a time.
- Keep `n=3` for the final comparison.
- Archive, do not delete, superseded/contracted records.

## 6. Deliverable

- Fresh `results/raw/runs.jsonl`, environment, versions.
- Regenerated `reports/`.
- Technical write-up explaining TTFT vs throughput, quantization cost, and backend choice by workload.
```

Commit and push.<|end|><|start|><|start|>assistant to=shell<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
cd C:\benchmaxing; git add docs/reproducible-run-plan.md; git commit -m "docs: add reproducible run plan for trustworthy first backend comparison"; git push origin feat/work-on-open-issues