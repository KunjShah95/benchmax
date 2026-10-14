import json
import os

import pandas as pd

from benchmax.report import write_all, load_raw


def _make_raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "kind": "perf_run",
        "run_id": "r0",
        "workload": "w1_single_short",
        "replicate": 0,
        "cell": {
            "backend": "llamacpp",
            "model_key": "qwen3_0_6b",
            "model_label": "Qwen3-0.6B",
            "family": "qwen",
            "params_b": 0.6,
            "quant_level": "f16",
            "quant_scheme": "GGUF F16",
            "device": "cuda",
        },
        "summary": {
            "n_requests": 1,
            "n_ok": 1,
            "total_prompt_tokens": 10,
            "total_output_tokens": 5,
            "ttft_ms_mean": 50.0,
            "ttft_ms_p95": 50.0,
            "ttft_ms_p99": 50.0,
            "tpot_ms_mean": 5.0,
            "tpot_ms_p95": 5.0,
            "itl_ms_p50": 4.0,
            "itl_ms_p95": 4.5,
            "itl_ms_p99": 4.8,
            "output_throughput_tok_s": 100.0,
            "prefill_throughput_tok_s": 200.0,
            "goodput_output_tok_s": 100.0,
            "group_wall_s": 0.05,
        },
        "memory": {
            "gpu_peak_mib": 1024.0,
            "gpu_mean_mib": 900.0,
            "host_rss_peak_mib": 512.0,
            "gpu_util_mean_pct": 50.0,
        },
    }
    with open(raw_dir / "runs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    # environment + versions passthrough (reproducibility metadata)
    with open(raw_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump({"gpu": "mock", "python": "3.13"}, f)
    with open(raw_dir / "versions.json", "w", encoding="utf-8") as f:
        json.dump({"llamacpp": {"commit": "abc"}}, f)
    return tmp_path


def test_report_generation_checks(tmp_path):
    results = _make_raw(tmp_path)
    reports = tmp_path / "reports"
    out = write_all(str(results), str(reports))
    assert out["n_perf_runs"] == 1
    assert os.path.exists(reports / "tables" / "all_perf_runs.csv")
    assert os.path.exists(reports / "tables" / "perf_by_workload_backend_model.csv")
    # environment/versions passthrough present
    assert "environment" in out
    assert "versions" in out
    perf, qual, other = load_raw(str(results))
    assert len(perf) == 1
    assert qual.empty


def test_report_handles_empty_perf_without_crashing(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # empty runs file
    (raw_dir / "runs.jsonl").write_text("", encoding="utf-8")
    reports = tmp_path / "reports"
    out = write_all(str(tmp_path), str(reports))
    assert out["n_perf_runs"] == 0
    assert isinstance(out.get("figures", []), list)
