"""Derived trade-off analysis.

The raw tables answer "what was each number?". This module answers the
question the comparison actually exists for: *given a constraint, which
backend wins, and what does choosing it cost?*

Everything here is derived from `results/raw/runs.jsonl` via report.load_raw,
so nothing depends on state accumulated during the run.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from .report import load_raw


# --------------------------------------------------------------------------
# 1. the latency/throughput frontier
# --------------------------------------------------------------------------

def pareto_frontier(df: pd.DataFrame, x: str = "out_tok_s",
                    y: str = "ttft_ms_mean") -> pd.DataFrame:
    """Configs that are not simultaneously beaten on throughput and TTFT.

    A point is on the frontier if no other config has both >= throughput and
    <= TTFT with at least one strict. This is the right lens for backend
    selection because it surfaces the actual operating choices rather than a
    single averaged "winner".
    """
    if df.empty:
        return df
    d = df.dropna(subset=[x, y]).sort_values([x, y])
    keep: list[int] = []
    best_y = float("inf")
    for idx, row in d.iterrows():
        if row[y] < best_y:
            keep.append(idx)
            best_y = row[y]
    return d.loc[keep]


def frontier_by_workload(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return perf
    rows = []
    for wl, sub in perf.groupby("workload"):
        agg = sub.groupby(["backend", "model", "quant_level"]).agg(
            ttft_ms=("ttft_ms_mean", "mean"),
            out_tok_s=("out_tok_s", "mean"),
            tpot_ms=("tpot_ms_mean", "mean"),
            goodput_tok_s=("goodput_tok_s", "mean"),
        ).reset_index()
        agg["on_frontier"] = False
        if not agg.empty:
            # columns were renamed above, so the default axis names no longer
            # apply here
            front = pareto_frontier(agg, x="out_tok_s", y="ttft_ms")
            agg.loc[front.index, "on_frontier"] = True
        agg.insert(0, "workload", wl)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# 2. same weights, different backend (the controlled comparison)
# --------------------------------------------------------------------------

def format_vs_backend_anchor(perf: pd.DataFrame,
                             anchor: str = "llamacpp") -> pd.DataFrame:
    """Backends serving IDENTICAL weights at an IDENTICAL quantization.

    This was the intended anchor of the comparison: holding the model artifact
    fixed while varying only the serving backend. It cannot be produced here,
    because vLLM 0.31 has no GGUF support at all (verified: no `gguf` method
    in QUANTIZATION_METHODS, no quantization.gguf module, no GGUFConfig). Since
    llama.cpp's quantization surface is GGUF and vLLM's is
    safetensors+AWQ/FP8/GPTQ, there is no configuration in which both serve the
    same weights.

    The function therefore returns an empty frame and the report explains the
    gap rather than publishing an empty pivot as if it were a measurement.
    """
    return pd.DataFrame()


def same_weights_unavailable_note() -> dict[str, str]:
    return {
        "status": "unavailable",
        "reason": (
            "vLLM 0.31 does not support GGUF. llama.cpp consumes GGUF "
            "(Q4_K_M/Q8_0/BF16); vLLM consumes safetensors with AWQ, FP8, GPTQ "
            "or compressed-tensors. The two quantization surfaces are disjoint, "
            "so 'same weights, different backend' cannot be measured by "
            "construction. Any llama.cpp-vs-vLLM throughput delta therefore "
            "confounds backend with quantization scheme and model artifact."
        ),
        "verified_by": [
            "no 'gguf' key in vllm QUANTIZATION_METHODS",
            "ModuleNotFoundError: vllm.model_executor.layers.quantization.gguf",
            "ImportError: cannot import GGUFConfig from vllm.transformers_utils.config",
            "serving a local .gguf path -> OSError 'not a valid JSON file'",
            "serving 'repo/file.gguf' -> OSError malformed repo id",
        ],
    }


def ratio_table(perf: pd.DataFrame, baseline: str = "llamacpp",
                value: str = "out_tok_s") -> pd.DataFrame:
    """Throughput of each backend relative to the baseline, same workload+model.

    >1 means faster than the baseline. Reported with the 16-bit reference of the
    same model where available, so the reader can see whether a backend's
    advantage is a constant factor or disappears at higher precision.
    """
    if perf.empty:
        return perf
    agg = perf.groupby(["workload", "model", "quant_level", "backend"])[value].mean().reset_index()
    base = agg[agg["backend"] == baseline][["workload", "model", "quant_level", value]]
    base = base.rename(columns={value: "baseline"})
    m = agg.merge(base, on=["workload", "model", "quant_level"], how="left")
    m["ratio_vs_" + baseline] = m[value] / m["baseline"]
    return m


# --------------------------------------------------------------------------
# 3. quantization cost
# --------------------------------------------------------------------------

def quantization_cost(perf: pd.DataFrame, qual: pd.DataFrame) -> pd.DataFrame:
    """Cost of quantization relative to each model's own 16-bit reference.

    Perplexity is normalized per model, because raw perplexity is not comparable
    across tokenizers. Throughput is normalized per (model, backend), because
    the interesting question is "how much did quantizing cost *this*
    configuration", not which config is fastest in the room.
    """
    rows = []

    if not qual.empty:
        q = qual.dropna(subset=["ppl"]).copy()
        if not q.empty:
            for (bk, mk), g in q.groupby(["backend", "model_key"]):
                ref = g[g["quant_level"] == "16bit"]
                if ref.empty:
                    continue
                base_ppl = float(ref["ppl"].iloc[0])
                for _, r in g.iterrows():
                    rows.append({
                        "axis": "perplexity",
                        "backend": bk, "model": r["model"],
                        "quant_level": r["quant_level"],
                        "quant_scheme": r.get("quant_scheme"),
                        "value": r["ppl"], "reference": base_ppl,
                        "ratio": float(r["ppl"]) / base_ppl if base_ppl else None,
                    })

    if not perf.empty:
        p = perf.groupby(["workload", "backend", "model_key", "quant_level",
                          "quant_scheme"])["out_tok_s"].mean().reset_index()
        for (wl, bk, mk), g in p.groupby(["workload", "backend", "model_key"]):
            ref = g[g["quant_level"] == "16bit"]
            if ref.empty:
                continue
            base = float(ref["out_tok_s"].iloc[0])
            for _, r in g.iterrows():
                rows.append({
                    "axis": "throughput",
                    "workload": wl, "backend": bk, "model": r["model_key"],
                    "quant_level": r["quant_level"],
                    "quant_scheme": r.get("quant_scheme"),
                    "value": r["out_tok_s"], "reference": base,
                    "ratio": float(r["out_tok_s"]) / base if base else None,
                })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. memory efficiency
# --------------------------------------------------------------------------

def memory_efficiency(perf: pd.DataFrame) -> pd.DataFrame:
    """Throughput per MiB of peak VRAM, and VRAM per billion parameters.

    The second ratio is the one that determines whether a given model fits at
    all on a given card, which is often a harder constraint than speed.
    """
    if perf.empty:
        return perf
    d = perf.dropna(subset=["gpu_peak_mib", "out_tok_s"]).copy()
    if d.empty:
        return d
    agg = d.groupby(["workload", "backend", "model_key", "model", "params_b",
                     "quant_level", "quant_scheme"]).agg(
        out_tok_s=("out_tok_s", "mean"),
        gpu_peak_mib=("gpu_peak_mib", "max"),
        host_rss_peak_mib=("host_rss_peak_mib", "max"),
    ).reset_index()
    agg["tok_s_per_mib"] = agg["out_tok_s"] / agg["gpu_peak_mib"]
    agg["vram_mib_per_1b_params"] = agg["gpu_peak_mib"] / agg["params_b"]
    return agg.sort_values(["workload", "tok_s_per_mib"], ascending=[True, False])


# --------------------------------------------------------------------------
# 5. lifecycle / startup cost
# --------------------------------------------------------------------------

def lifecycle(other: list[dict]) -> pd.DataFrame:
    rows = []
    for r in other:
        if r.get("kind") != "cell_summary":
            continue
        c = r.get("cell") or {}
        m = r.get("memory_overall") or {}
        rows.append({
            "backend": c.get("backend"),
            "model": c.get("model_label"),
            "quant_level": c.get("quant_level"),
            "quant_scheme": c.get("quant_scheme"),
            "startup_s": r.get("startup_s"),
            "gpu_peak_mib": m.get("gpu_peak_mib"),
            "host_rss_peak_mib": m.get("host_rss_peak_mib"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. capability gaps (recorded rather than hidden)
# --------------------------------------------------------------------------

def capability_gaps(results_dir: str) -> pd.DataFrame:
    """Startup failures, unsupported quality paths, and skipped cells."""
    rows = []
    path = os.path.join(results_dir, "raw", "runs.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    import json
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
            c = rec.get("cell") or {}
            if kind == "startup_failure":
                rows.append({"category": "startup_failure",
                             "backend": c.get("backend"), "model": c.get("model_key"),
                             "quant_level": c.get("quant_level"),
                             "detail": rec.get("error") or rec.get("startup_error")})
            elif kind == "quality_run":
                q = rec.get("quality") or {}
                ppl = q.get("perplexity") or {}
                mc = q.get("mc_accuracy") or {}
                if ppl.get("status") != "ok":
                    rows.append({"category": "perplexity_unavailable",
                                 "backend": c.get("backend"), "model": c.get("model_key"),
                                 "quant_level": c.get("quant_level"),
                                 "detail": ppl.get("status")})
                if mc.get("status") not in ("ok",):
                    rows.append({"category": "mc_unavailable",
                                 "backend": c.get("backend"), "model": c.get("model_key"),
                                 "quant_level": c.get("quant_level"),
                                 "detail": mc.get("reason") or mc.get("status")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_all(results_dir: str, reports_dir: str) -> dict[str, Any]:
    perf, qual, other = load_raw(results_dir)
    tdir = os.path.join(reports_dir, "tables")
    os.makedirs(tdir, exist_ok=True)

    out: dict[str, Any] = {}

    for name, df in [
        ("frontier_by_workload", frontier_by_workload(perf)),
        ("quantization_cost", quantization_cost(perf, qual)),
        ("memory_efficiency", memory_efficiency(perf)),
        ("lifecycle", lifecycle(other)),
        ("capability_gaps", capability_gaps(results_dir)),
    ]:
        if df is not None and not getattr(df, "empty", True):
            df.to_csv(os.path.join(tdir, f"{name}.csv"), index=False)
            out[name] = df

    piv = format_vs_backend_anchor(perf)
    if piv is not None and not getattr(piv, "empty", True):
        piv.to_csv(os.path.join(tdir, "same_weights_backend_comparison.csv"))
        out["same_weights"] = piv

    note = same_weights_unavailable_note()
    with open(os.path.join(tdir, "same_weights_unavailable.json"), "w") as f:
        json.dump(note, f, indent=2)
    out["same_weights_note"] = note

    for anchor in ("llamacpp", "vllm"):
        r = ratio_table(perf, baseline=anchor)
        if r is not None and not r.empty:
            r.to_csv(os.path.join(tdir, f"ratio_vs_{anchor}.csv"), index=False)
            out[f"ratio_{anchor}"] = r

    return out