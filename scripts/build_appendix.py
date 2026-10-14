#!/usr/bin/env python
"""Build a data appendix PDF from the generated CSV tables.

The report itself explains the numbers; this carries them. A reader should be
able to take the appendix alone and check every claim without re-running
anything.
"""
from __future__ import annotations

import argparse
import base64
import os

import pandas as pd


def _f(v, nd=2):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, nd: int = 2, max_rows: int = 400) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    d = df.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and float(v) == float(v):
                cells.append(_f(v, 1 if abs(v) >= 1000 else nd))
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


COLUMN_LABELS = {
    "ttft_ms_mean_mean": "ttft_ms", "ttft_ms_mean_std": "ttft_sd",
    "ttft_ms_p95_mean": "ttft_p95",
    "tpot_ms_mean_mean": "tpot_ms", "tpot_ms_mean_std": "tpot_sd",
    "tpot_ms_p95_mean": "tpot_p95",
    "itl_ms_p50_mean": "itl_p50", "itl_ms_p99_mean": "itl_p99",
    "out_tok_s_mean": "tok_s", "out_tok_s_std": "tok_s_sd",
    "prefill_tok_s_mean": "prefill_tok_s",
    "goodput_tok_s_mean": "goodput",
    "gpu_peak_mib_max": "vram_mib", "gpu_peak_mib_mean": "vram_mean_mib",
    "host_rss_peak_mib_max": "rss_mib",
    "quant_scheme": "scheme", "model_key": "model",
    "ppl_context_tokens": "ppl_ctx",
    "corpus_chars_used": "ppl_corpus_chars",
}


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Rename report.py's machine column names for human reading."""
    return df.rename(columns={k: v for k, v in COLUMN_LABELS.items()
                              if k in df.columns})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="reports/pdf/DATA_APPENDIX.md")
    a = ap.parse_args()

    import json
    T = a.tables
    out = [
        "# Data appendix — benchmax",
        "",
        "Every table below is generated from `results/raw/runs.jsonl` by "
        "`python -m benchmax.cli report`. Figures are in `reports/figures/`. "
        "n = 3 replicates per (backend, model, quantization, workload) unless "
        "noted.",
        "",
        "---",
        "",
    ]

    # environment + versions
    for name, label in [("environment.json", "Environment"),
                        ("versions.json", "Backend versions")]:
        p = os.path.join(a.raw, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            out.append(f"## {label}")
            out.append("")
            out.append("| key | value |")
            out.append("|---|---|")
            for k, v in data.items():
                out.append(f"| `{k}` | {str(v).replace('|', '/')} |")
            out.append("")

    def safe(name):
        p = os.path.join(T, name)
        if os.path.exists(p):
            return pd.read_csv(p)
        return None

    # matrix status
    mp = os.path.join(a.raw, "matrix.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            matrix = json.load(f)["matrix"]
        out += ["## Matrix (every cell and why it was or was not run)", "",
                "| backend | model | quant | scheme | status | reason |", "|---|---|---|---|---|---|"]
        for c in matrix:
            reason = (c.get("skip_reason") or "").replace("|", "/")[:220]
            out.append(f"| {c['backend']} | {c['model_key']} | "
                       f"{c['quant_level']} | {c.get('quant_scheme')} | "
                       f"{c.get('status')} | {reason} |")
        out.append("")

    df = safe("perf_rollup_config.csv")
    if df is not None:
        # report.py's groupby.agg produces "<metric>_<stat>", so the mean of
        # ttft_ms_mean is literally the column `ttft_ms_mean_mean`. Using the
        # short name here silently drops the column, so map explicitly.
        rename = {
            "ttft_ms_mean_mean": "ttft_ms", "ttft_ms_mean_std": "ttft_sd",
            "tpot_ms_mean_mean": "tpot_ms", "tpot_ms_mean_std": "tpot_sd",
            "itl_ms_p99_mean": "itl_p99",
            "out_tok_s_mean": "tok_s", "out_tok_s_std": "tok_s_sd",
            "goodput_tok_s_mean": "goodput",
            "gpu_peak_mib_max": "vram_mib",
            "host_rss_peak_mib_max": "rss_mib",
            "quant_scheme": "scheme",
        }
        keep = ["backend", "model", "quant_level", "quant_scheme",
                "ttft_ms_mean_mean", "ttft_ms_mean_std",
                "tpot_ms_mean_mean", "itl_ms_p99_mean",
                "out_tok_s_mean", "goodput_tok_s_mean", "gpu_peak_mib_max",
                "n_reps"]
        keep = [c for c in keep if c in df.columns]
        d2 = df[keep].rename(columns=rename)
        out += ["## Performance rollup (all workloads combined)", "",
                "Latency in ms, throughput in tok/s, memory in MiB. "
                "Averaged across all four workloads, so use the per-workload "
                "table for concurrency-specific claims.", "",
                md_table(d2)]

        df = safe("perf_by_workload_backend_quant.csv")
    if df is not None:
        out += ["## Performance by workload, backend and quantization", "",
                "This is the table to cite for any concurrency-specific claim.",
                "", md_table(pretty(df))]

    df = safe("frontier_by_workload.csv")
    if df is not None:
        out += ["## TTFT / throughput frontier", "",
                "`on_frontier = True` means no other configuration beats this "
                "one on both throughput and TTFT.", "", md_table(pretty(df))]

    df = safe("quantization_cost.csv")
    if df is not None:
        ppl = df[df.axis == "perplexity"]
        if not ppl.empty:
            out += ["## Quantization cost — perplexity", "",
                    "Normalised per model against that model's own unquantized "
                    "run. Cross-backend perplexity is NOT comparable "
                    "(different context windows and tools).", "",
                    md_table(ppl)]
        thr = df[df.axis == "throughput"]
        if not thr.empty:
            out += ["## Quantization cost — throughput", "",
                    "Normalised per (backend, model) against the same "
                    "configuration's 16-bit reference.", "",
                    md_table(thr)]

    df = safe("lifecycle.csv")
    if df is not None:
        out += ["## Lifecycle — time-to-ready and peak memory", "",
                md_table(df)]

    df = safe("memory_efficiency.csv")
    if df is not None:
        out += ["## Memory efficiency", "",
                "`tok_s_per_mib` = output throughput per MiB of peak VRAM. "
                "`vram_mib_per_1b_params` is the footprint scaling factor.",
                "", md_table(df)]

    df = safe("quality_runs.csv")
    if df is not None:
        keep = [c for c in ["backend", "model", "quant_level", "quant_scheme",
                            "ppl", "ppl_stderr", "ppl_context_tokens",
                            "ppl_method", "mc_acc", "mc_n", "mc_wilson_lo",
                            "mc_wilson_hi"] if c in df.columns]
        out += ["## Quality", "",
                "`ppl_method` distinguishes llama.cpp's offline perplexity tool "
                "from vLLM's server-side echo path — they are different "
                "measurements and must not be compared directly.", "",
                md_table(df[keep])]

    df = safe("capability_gaps.csv")
    if df is not None:
        out += ["## Capability gaps", "",
                "Recorded rather than hidden: these are properties of the "
                "backends, not failures of the harness.", "",
                md_table(df)]

    note = os.path.join(T, "same_weights_unavailable.json")
    if os.path.exists(note):
        with open(note, encoding="utf-8") as f:
            data = json.load(f)
        out += ["## Why there is no same-weights comparison", "",
                "This was the intended anchor of the whole benchmark and it "
                "could not be produced.", "",
                f"**Reason.** {data['reason']}", "",
                "Verified by:", ""]
        for v in data["verified_by"]:
            out.append(f"- {v}")
        out.append("")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1024:.0f} KiB)")


if __name__ == "__main__":
    main()