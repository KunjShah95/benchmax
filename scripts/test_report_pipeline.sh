#!/usr/bin/env bash
# End-to-end test of the reporting layer using synthetic run records shaped
# exactly like the real ones. Catches schema drift between runner.py and
# report.py without needing the GPU, so reporting can be trusted on arrival of
# real data.
set -uo pipefail
ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv-bench/bin/activate"
cd /mnt/c/benchmaxing

TESTDIR=/tmp/benchmax-reporttest
rm -rf "$TESTDIR"; mkdir -p "$TESTDIR/raw"

python - <<'PY'
import json, math, os, random
random.seed(7)
d="/tmp/benchmax-reporttest/raw/runs.jsonl"
recs=[]
backends=["llamacpp","vllm","vllm_gguf"]
models=[("qwen3-1.7b","Qwen3-1.7B","Qwen3",1.7),
        ("llama32-1b","Llama-3.2-1B-Instruct","Llama-3.2",1.0)]
quants=["4bit","8bit","16bit"]
wls=["w1_single_short","w2_single_prefill","w3_concurrent_8","w4_concurrent_32"]

recs.append({"kind":"session","environment":{"cpu":"i7","gpu":"RTX 4060","ram_gib":10.5},
             "versions":{"vllm":"0.31.0"},"corpus_source":"test",
             "client_overhead":{"synthetic_frame_p50_ms":0.002},
             "workloads":{},"slo":{"ttft_ms":1000,"tpot_ms":100},"repeats":3})

# base throughput/latency characteristics per backend so the report has signal
prof={"llamacpp":(70,0.9),"vllm":(45,1.6),"vllm_gguf":(180,0.7)}
for b in backends:
    for (mk,ml,fam,pb) in models:
        for q in quants:
            ttft0,mult=prof[b]
            for wl in wls:
                for rep in range(3):
                    n={"w1_single_short":1,"w2_single_prefill":1,
                       "w3_concurrent_8":8,"w4_concurrent_32":32}[wl]
                    ttft=ttft0*mult*random.uniform(0.9,1.1)
                    tpot=10*mult*random.uniform(0.9,1.1)
                    ptok=50*n
                    ctok=128*n
                    wall=ctok/(60*mult)*random.uniform(0.95,1.05)
                    recs.append({"kind":"perf_run",
                        "run_id":f"{b}|{mk}|{q}|{wl}|r{rep}",
                        "cell":{"backend":b,"model_key":mk,"model_label":ml,"family":fam,
                                "params_b":pb,"quant_level":q,"quant_scheme":"TEST"},
                        "workload":wl,"replicate":rep,
                        "workload_spec":{"n_requests":n},
                        "summary":{"n_ok":n,"n_errored":0,
                            "ttft_ms_mean":ttft,"ttft_ms_p95":ttft*1.2,
                            "tpot_ms_mean":tpot,"tpot_ms_p95":tpot*1.2,
                            "itl_ms_p50":tpot,"itl_ms_p95":tpot*2,"itl_ms_p99":tpot*3,
                            "output_throughput_tok_s":ctok/wall,
                            "prefill_throughput_tok_s":ptok/wall,
                            "goodput_output_tok_s":ctok/wall*0.8,
                            "group_wall_s":wall,
                            "total_output_tokens":ctok,"total_prompt_tokens":ptok},
                        "memory":{"gpu_peak_mib":1200+pb*300,"gpu_mean_mib":1100,
                                  "host_rss_peak_mib":900.0,"gpu_util_mean_pct":70},
                        "requests":[]})
            base=13.0 if q=="16bit" else {"4bit":13.5,"8bit":13.2,"int8":13.1}[q]
            recs.append({"kind":"quality_run",
                "run_id":f"{b}|{mk}|{q}|quality",
                "cell":{"backend":b,"model_key":mk,"model_label":ml,"family":fam,
                        "params_b":pb,"quant_level":q,"quant_scheme":"TEST"},
                "corpus_source":"test","corpus_chars":1000,
                "quality":{"perplexity":{"status":"ok","ppl":base*random.uniform(0.99,1.01),
                                          "ppl_stderr":0.2,"method":"test","context_tokens":2048},
                           "mc_accuracy":{"status":"ok","accuracy":random.uniform(0.4,0.8),
                                          "n_scored":60,"wilson_lo":0.3,"wilson_hi":0.9}}})
recs.append({"kind":"cell_summary",
             "cell":{"backend":"vllm","model_key":"qwen3-1.7b","model_label":"Qwen3-1.7B",
                     "quant_level":"4bit","quant_scheme":"TEST"},
             "startup_s":42.5,"memory_overall":{"gpu_peak_mib":1900,"host_rss_peak_mib":6000}})

with open(d,"w") as f:
    for r in recs: f.write(json.dumps(r)+"\n")
print(f"wrote {len(recs)} synthetic records")
PY

echo
echo "=== running report ==="
python -m benchmax.cli --results "$TESTDIR" --reports "$TESTDIR/reports" report

echo
echo "=== outputs ==="
find "$TESTDIR/reports" -type f | sed "s|$TESTDIR/reports/||" | sort

echo
echo "=== sample table ==="
head -6 "$TESTDIR/reports/tables/perf_rollup_config.csv" 2>/dev/null || echo "  (missing)"