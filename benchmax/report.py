"""Turn raw run records into tables, figures and machine-readable summaries.

Nothing is computed twice: this reads `runs.jsonl` (the append-only raw log) and
re-derives every aggregate from scratch. If a number in the report is wrong,
re-running this module with a corrected aggregation reproduces the truth without
re-running a single inference.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_raw(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    perf_rows: list[dict] = []
    qual_rows: list[dict] = []
    other: list[dict] = []

    path = os.path.join(results_dir, "raw", "runs.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no raw results at {path}")

    # (cell, workload, replicate) -> row, so a cell measured twice (e.g. a run
    # slice that was interrupted and retried) contributes one observation
    # rather than silently averaging two different sessions together.
    perf_by_key: dict[tuple, dict] = {}
    qual_by_key: dict[tuple, dict] = {}
    superseded = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = rec.get("kind")
            if kind == "perf_run":
                row = _perf_row(rec)
                if row:
                    key = (row["backend"], row["model_key"], row["quant_level"],
                           row["workload"], row["rep"])
                    if key in perf_by_key:
                        superseded += 1
                    perf_by_key[key] = row          # last wins
            elif kind == "quality_run":
                row = _qual_row(rec)
                if row:
                    key = (row["backend"], row["model_key"], row["quant_level"])
                    qual_by_key[key] = row
            else:
                other.append(rec)

    perf_rows = list(perf_by_key.values())
    qual_rows = list(qual_by_key.values())
    if superseded:
        print(f"[report] collapsed {superseded} duplicate perf records "
              f"(same cell/workload/replicate measured more than once)")
    return pd.DataFrame(perf_rows), pd.DataFrame(qual_rows), other


def _perf_row(rec: dict) -> dict | None:
    s = rec.get("summary") or {}
    c = rec.get("cell") or {}
    m = rec.get("memory") or {}
    if not s or s.get("n_ok", 0) == 0:
        return None
    return {
        "run_id": rec.get("run_id"),
        "backend": c.get("backend"),
        "model_key": c.get("model_key"),
        "model": c.get("model_label"),
        "family": c.get("family"),
        "params_b": c.get("params_b"),
        "quant_level": c.get("quant_level"),
        "quant_scheme": c.get("quant_scheme"),
        "device": c.get("device"),
        "workload": rec.get("workload"),
        "rep": rec.get("replicate"),
        "n_req": s.get("n_ok"),
        "prompt_tokens_total": s.get("total_prompt_tokens"),
        "output_tokens_total": s.get("total_output_tokens"),
        "ttft_ms_mean": s.get("ttft_ms_mean"),
        "ttft_ms_p95": s.get("ttft_ms_p95"),
        "ttft_ms_p99": s.get("ttft_ms_p99"),
        "tpot_ms_mean": s.get("tpot_ms_mean"),
        "tpot_ms_p95": s.get("tpot_ms_p95"),
        "itl_ms_p50": s.get("itl_ms_p50"),
        "itl_ms_p95": s.get("itl_ms_p95"),
        "itl_ms_p99": s.get("itl_ms_p99"),
        "out_tok_s": s.get("output_throughput_tok_s"),
        "prefill_tok_s": s.get("prefill_throughput_tok_s"),
        "goodput_tok_s": s.get("goodput_output_tok_s"),
        "wall_s": s.get("group_wall_s"),
        "gpu_peak_mib": m.get("gpu_peak_mib"),
        "gpu_mean_mib": m.get("gpu_mean_mib"),
        "host_rss_peak_mib": m.get("host_rss_peak_mib"),
        "gpu_util_mean_pct": m.get("gpu_util_mean_pct"),
    }


def _qual_row(rec: dict) -> dict | None:
    q = rec.get("quality") or {}
    ppl = q.get("perplexity") or {}
    mc = q.get("mc_accuracy") or {}
    c = rec.get("cell") or {}
    return {
        "backend": c.get("backend"),
        "model_key": c.get("model_key"),
        "model": c.get("model_label"),
        "family": c.get("family"),
        "quant_level": c.get("quant_level"),
        "quant_scheme": c.get("quant_scheme"),
        "ppl": ppl.get("ppl"),
        "ppl_stderr": ppl.get("ppl_stderr"),
        "ppl_method": ppl.get("method", "server echo logprobs"),
        "ppl_context_tokens": ppl.get("context_tokens"),
        "ppl_tokens": ppl.get("total_tokens_scored"),
        "ppl_chunks": ppl.get("n_chunks_scored"),
        "ppl_status": ppl.get("status"),
        "ppl_prompt_eval_tok_s": ppl.get("prompt_eval_tok_s"),
        "mc_acc": mc.get("accuracy"),
        "mc_n": mc.get("n_scored"),
        "mc_wilson_lo": mc.get("wilson_lo"),
        "mc_wilson_hi": mc.get("wilson_hi"),
        "mc_status": mc.get("status"),
        "mc_reason": mc.get("reason"),
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

AGG = {
    "ttft_ms_mean": ["mean", "std"],
    "ttft_ms_p95": ["mean"],
    "tpot_ms_mean": ["mean", "std"],
    "tpot_ms_p95": ["mean"],
    "itl_ms_p50": ["mean"],
    "itl_ms_p99": ["mean"],
    "out_tok_s": ["mean", "std"],
    "prefill_tok_s": ["mean", "std"],
    "goodput_tok_s": ["mean", "std"],
    "gpu_peak_mib": ["max"],
    "host_rss_peak_mib": ["max"],
}


def aggregate_perf(perf: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Group and aggregate.

    Column names are rebuilt as `<metric>_<stat>` rather than the pandas default
    `<metric>_<stat>` doubled up (e.g. `ttft_ms_mean_mean`), which is noisy when
    the metric name already carries a qualifier.
    """
    if perf.empty:
        return perf
    g = perf.groupby(group_cols, dropna=False).agg(AGG)
    g.columns = [f"{col}_{stat}" for col, stat in g.columns]
    n = perf.groupby(group_cols, dropna=False).size().rename("n_reps")
    return g.join(n).reset_index()


def startup_table(other: list[dict]) -> pd.DataFrame:
    rows = []
    for r in other:
        if r.get("kind") != "cell_summary":
            continue
        c = r.get("cell") or {}
        rows.append({
            "backend": c.get("backend"),
            "model": c.get("model_label"),
            "quant_level": c.get("quant_level"),
            "quant_scheme": c.get("quant_scheme"),
            "startup_s": r.get("startup_s"),
            "gpu_peak_mib": (r.get("memory_overall") or {}).get("gpu_peak_mib"),
            "host_rss_peak_mib": (r.get("memory_overall") or {}).get("host_rss_peak_mib"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

BACKEND_STYLE = {
    "llamacpp": {"color": "#d1495b", "marker": "o"},
    "vllm": {"color": "#2e86ab", "marker": "s"},
    "vllm_gguf": {"color": "#8a9a5b", "marker": "^"},
    "ollama": {"color": "#8e6c88", "marker": "D"},
}


def _style(backend: str) -> dict:
    return BACKEND_STYLE.get(backend, {"color": "gray", "marker": "o"})


def make_figures(perf: pd.DataFrame, qual: pd.DataFrame, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    made: list[str] = []

    if not perf.empty:
        # --- 1. TTFT vs throughput frontier, per workload ------------------
        # Coloured by BACKEND (not per configuration): the point of the plot is
        # the backend split, and one legend entry per config buried it.
        for wl, sub in perf.groupby("workload"):
            agg = sub.groupby(["backend", "model", "quant_level"]).agg(
                ttft_ms=("ttft_ms_mean", "mean"),
                out_tok_s=("out_tok_s", "mean"),
            ).reset_index()
            fig, ax = plt.subplots(figsize=(9, 6))
            for backend, g in agg.groupby("backend"):
                st = _style(backend)
                ax.scatter(g["out_tok_s"], g["ttft_ms"],
                           s=95, alpha=0.85, label=backend,
                           color=st["color"], marker=st["marker"],
                           edgecolors="white", linewidths=0.6)
                for _, r in g.iterrows():
                    ax.annotate(f"{r['model']}\n{r['quant_level']}",
                                (r["out_tok_s"], r["ttft_ms"]),
                                fontsize=6, alpha=0.75,
                                xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel("aggregate output throughput (tok/s)")
            ax.set_ylabel("mean TTFT (ms)")
            ax.set_title(f"TTFT / throughput frontier — {wl}\n"
                         f"(lower-right is better on both axes)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10, title="backend")
            fig.tight_layout()
            p = os.path.join(out_dir, f"frontier_{wl}.png")
            fig.savefig(p, dpi=140)
            plt.close(fig)
            made.append(p)

        # --- 2. latency breakdown by backend -------------------------------
        sub = perf[perf["workload"] == "w1_single_short"]
        if not sub.empty:
            g = sub.groupby("backend").agg(
                ttft=("ttft_ms_mean", "mean"),
                tpot=("tpot_ms_mean", "mean"),
            )
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            x = range(len(g))
            ax.bar([i - 0.2 for i in x], g["ttft"], width=0.4, label="TTFT (ms)",
                   color="#d1495b")
            ax.bar([i + 0.2 for i in x], g["tpot"], width=0.4, label="TPOT (ms)",
                   color="#2e86ab")
            ax.set_xticks(list(x))
            ax.set_xticklabels(g.index, rotation=10)
            ax.set_ylabel("milliseconds (lower is better)")
            ax.set_title("Single-stream latency (w1_single_short)\n"
                         "llama.cpp leads per-token at batch size 1")
            ax.legend()
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            p = os.path.join(out_dir, "latency_by_backend.png")
            fig.savefig(p, dpi=140)
            plt.close(fig)
            made.append(p)

        # --- 3. quantization effect on throughput --------------------------
        sub = perf[perf["workload"] == "w4_concurrent_32"]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for backend, g in sub.groupby("backend"):
                gg = g.groupby(["model", "quant_level"])["out_tok_s"].mean().reset_index()
                for model, gm in gg.groupby("model"):
                    ax.plot(gm["quant_level"], gm["out_tok_s"], marker="o",
                            label=f"{backend} {model}")
            ax.set_ylabel("output throughput (tok/s)")
            ax.set_xlabel("quantization level")
            ax.set_title("Throughput vs quantization (w4_concurrent_32)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            p = os.path.join(out_dir, "quant_vs_throughput.png")
            fig.savefig(p, dpi=140)
            plt.close(fig)
            made.append(p)

        # --- 4. memory: peak VRAM by model ---------------------------------
        # Minimum footprint per (model, backend), i.e. the smallest
        # quantization that runs. Peak-across-quantizations muddles the point,
        # because llama.cpp's max is dominated by its own 16-bit variant while
        # vLLM's is flat either way.
        g = perf.groupby(["model", "backend"])["gpu_peak_mib"].min().reset_index()
        if not g.empty:
            models = sorted(g["model"].unique())
            backends = sorted(g["backend"].unique())
            fig, ax = plt.subplots(figsize=(9, 5))
            w = 0.8 / max(len(backends), 1)
            for i, backend in enumerate(backends):
                vals = [g[(g.model == m) & (g.backend == backend)]["gpu_peak_mib"]
                        .min() if len(g[(g.model == m) & (g.backend == backend)]) else 0
                        for m in models]
                st = _style(backend)
                ax.bar([j + i * w for j in range(len(models))], vals,
                       width=w, label=backend, color=st["color"])
                for j, v in enumerate(vals):
                    if v:
                        ax.text(j + i * w, v + 40, f"{v/1024:.1f}G",
                                ha="center", fontsize=6.5)
            ax.set_xticks([j + w * (len(backends) - 1) / 2 for j in range(len(models))])
            ax.set_xticklabels(models, rotation=12, fontsize=8)
            ax.set_ylabel("peak GPU memory (MiB)")
            ax.set_title("Smallest running footprint per model and backend\n"
                         "vLLM reserves ~80% of the card regardless of model size; "
                         "llama.cpp scales with the model")
            ax.legend()
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            p = os.path.join(out_dir, "memory_by_config.png")
            fig.savefig(p, dpi=140)
            plt.close(fig)
            made.append(p)

        # --- 4b. goodput: throughput vs SLO-satisfying throughput -----------
        g = perf.groupby(["backend", "workload"]).agg(
            throughput=("out_tok_s", "mean"),
            goodput=("goodput_tok_s", "mean"),
        ).reset_index()
        if not g.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            wls = sorted(g["workload"].unique())
            for i, backend in enumerate(sorted(g["backend"].unique())):
                st = _style(backend)
                vals = [g[(g.backend == backend) & (g.workload == w)]
                       ["goodput"].mean() or 0 for w in wls]
                ax.bar([j + i * w for j in range(len(wls))], vals,
                       width=w, label=f"{backend} goodput",
                       color=st["color"])
            ax.set_xticks([j + w * (len(sorted(g["backend"].unique())) - 1) / 2
                           for j in range(len(wls))])
            ax.set_xticklabels(wls, rotation=10, fontsize=8)
            ax.set_ylabel("tokens/s meeting TTFT<=1s AND TPOT<=100ms")
            ax.set_title("Goodput — throughput that actually meets the latency SLO\n"
                         "raw throughput alone would overstate llama.cpp badly")
            ax.legend()
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            p = os.path.join(out_dir, "goodput_by_backend.png")
            fig.savefig(p, dpi=140)
            plt.close(fig)
            made.append(p)

    # --- 5. quality under quantization -----------------------------------
    if not qual.empty and qual["ppl"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        for (model), g in qual.groupby("model"):
            gg = g.dropna(subset=["ppl"]).sort_values("quant_level")
            if not gg.empty:
                ax.plot(gg["quant_level"], gg["ppl"], marker="o",
                        label=f"{model} (PPL)")
        ax.set_ylabel("perplexity (lower = closer to unquantized)")
        ax.set_xlabel("quantization level")
        ax.set_title("Perplexity under quantization")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(out_dir, "quality_perplexity.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(p)

    return made


# --------------------------------------------------------------------------
# markdown helpers
# --------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = "{:.1f}", max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    d = df.head(max_rows) if max_rows else df

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if isinstance(v, float):
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            return floatfmt.format(v)
        return str(v)

    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_all(results_dir: str, reports_dir: str) -> dict[str, Any]:
    os.makedirs(reports_dir, exist_ok=True)
    perf, qual, other = load_raw(results_dir)

    summary_dir = os.path.join(reports_dir, "tables")
    fig_dir = os.path.join(reports_dir, "figures")
    os.makedirs(summary_dir, exist_ok=True)

    outputs: dict[str, Any] = {}

    if not perf.empty:
        for cols, name in [
            (["workload", "backend", "model", "quant_level"],
             "perf_by_workload_backend_model"),
            (["workload", "backend", "quant_level"],
             "perf_by_workload_backend_quant"),
            (["backend", "model", "quant_level"],
             "perf_rollup_config"),
        ]:
            sub = perf[perf["workload"].isin(["w1_single_short", "w4_concurrent_32"])] \
                if "workload" in cols and len(cols) == 4 else perf
            agg = aggregate_perf(perf if len(cols) <= 3 else sub, cols)
            agg.to_csv(os.path.join(summary_dir, f"{name}.csv"), index=False)
            outputs[name] = agg

        perf.to_csv(os.path.join(summary_dir, "all_perf_runs.csv"), index=False)

    if not qual.empty:
        qual.to_csv(os.path.join(summary_dir, "quality_runs.csv"), index=False)
        outputs["quality"] = qual

    starts = startup_table(other)
    if not starts.empty:
        starts.to_csv(os.path.join(summary_dir, "startup_and_memory.csv"), index=False)
        outputs["startup"] = starts

    # environment / versions passthrough
    for key, fname in [("environment", "environment.json"),
                       ("versions", "versions.json")]:
        src = os.path.join(results_dir, "raw", fname)
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                outputs[key] = json.load(f)

    outputs["figures"] = make_figures(perf, qual, fig_dir)
    outputs["n_perf_runs"] = len(perf)
    outputs["n_quality_runs"] = len(qual)
    return outputs


if __name__ == "__main__":
    import sys
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    rep = sys.argv[2] if len(sys.argv) > 2 else "reports"
    out = write_all(res, rep)
    print(f"perf runs: {out.get('n_perf_runs')}")
    print(f"quality runs: {out.get('n_quality_runs')}")
    print(f"figures: {len(out.get('figures', []))}")