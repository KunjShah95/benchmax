#!/usr/bin/env bash
# Validate benchmax.analysis against the synthetic dataset, including a fresh
# generation of that dataset (the earlier one predates the analysis module).
set -uo pipefail
source ~/benchmax-env/venv-bench/bin/activate
cd /mnt/c/benchmaxing

bash /mnt/c/benchmaxing/scripts/test_report_pipeline.sh >/dev/null 2>&1
echo "synthetic dataset regenerated"

python -m benchmax.cli --results /tmp/benchmax-reporttest \
  --reports /tmp/benchmax-reporttest/reports report 2>&1 | tail -20

echo
echo "=== derived tables ==="
ls -1 /tmp/benchmax-reporttest/reports/tables/ | sed 's/^/  /'

echo
echo "=== frontier sample ==="
python - <<'PY'
import pandas as pd
d="/tmp/benchmax-reporttest/reports/tables"
f=pd.read_csv(f"{d}/frontier_by_workload.csv")
print(f[f.workload=="w4_concurrent_32"][["backend","model","quant_level","ttft_ms","out_tok_s","on_frontier"]].to_string(index=False))
print()
q=pd.read_csv(f"{d}/quantization_cost.csv")
print("quantization_cost rows:", len(q), "| axes:", sorted(q.axis.unique()))
print(q[q.axis=="perplexity"].head(4).to_string(index=False))
print()
m=pd.read_csv(f"{d}/memory_efficiency.csv")
print(m[["backend","model_key","quant_level","out_tok_s","gpu_peak_mib","tok_s_per_mib","vram_mib_per_1b_params"]].head(5).to_string(index=False))
print()
s=pd.read_csv(f"{d}/same_weights_backend_comparison.csv", header=[0,1], index_col=[0,1])
print("same_weights pivot shape:", s.shape)
PY