"""Generation-based quality evaluation without logprob requirements.

Unlike ``quality.py`` (perplexity + multiple-choice), which both require token
logprob support from the served backend, this module scores a model purely on
the *text it generates*: fixed prompts with known, verifiable answers are
submitted, the model's answer is extracted and compared to the reference, and
exact-match accuracy is reported alongside per-request latency and token usage.

This closes the gap for backends that do not expose logprobs (llama.cpp's
``/v1/completions`` path is the motivating case) while still giving a
deterministic, reproducible capability signal.

Design notes
------------
* The dataset is **fixed and versioned**. Changing the items or their expected
  answers bumps ``DATASET_VERSION``; every result record embeds the version and
  the SHA-256 of the item list so a run can be audited against the exact
  prompts that produced it.
* Scoring is **exact match after ASCII/whitespace normalisation** (lower-casing,
  collapsing internal whitespace, stripping). It is deliberately strict and
  therefore *undersensitive*: a model that paraphrases a correct answer is
  counted wrong, and a model that adds a polite prefix ("The answer is 4") is
  also counted wrong unless the prompt instructed it otherwise.
* Latency and token usage come from the wall clock / usage block of the
  (non-streaming) completion response, so nothing in this path depends on SSE.
* Failed and unsupported requests are recorded per item rather than aborting
  the whole run, which is what makes the output auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from typing import Any, Iterable

from .core import OpenAICompatClient


DATASET_VERSION = "1.0.0"

# ----------------------------------------------------------------- dataset


def _default_dataset_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "generation_eval_v1.json")


def load_generation_dataset(path: str | None = None) -> list[dict[str, Any]]:
    """Load the fixed, versioned generation-evaluation dataset.

    ``path`` may be a JSON file (either a ``{"version": ..., "items": [...]}``
    object or a bare list of items) or ``None`` to use the bundled dataset.
    Every item must carry ``id``, ``prompt`` and ``expected`` keys; ``category``
    is optional.
    """
    if path is None:
        path = _default_dataset_path()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("items", raw.get("dataset", []))
    if not isinstance(raw, list):
        raise ValueError(f"dataset at {path!r} is not a list of items")

    items: list[dict[str, Any]] = []
    for i, it in enumerate(raw):
        if not isinstance(it, dict):
            raise ValueError(f"item {i} is not an object")
        for key in ("id", "prompt", "expected"):
            if key not in it:
                raise ValueError(f"item {i} missing required key {key!r}")
        items.append({
            "id": str(it["id"]),
            "prompt": str(it["prompt"]),
            "expected": str(it["expected"]),
            "category": str(it.get("category", "")),
        })
    if not items:
        raise ValueError(f"dataset at {path!r} contains no items")
    return items


def dataset_hash(items: Iterable[dict[str, Any]] | None = None) -> str:
    """SHA-256 over the canonical JSON form of the item list.

    Canonical means sorted keys and a fixed separator, so the hash is stable
    across JSON serialisers and across runs.
    """
    if items is None:
        items = load_generation_dataset()
    normalised = [
        {"id": it["id"], "prompt": it["prompt"],
         "expected": it["expected"], "category": it.get("category", "")}
        for it in items
    ]
    canonical = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_version(items: Iterable[dict[str, Any]] | None = None) -> str:
    return DATASET_VERSION


# ---------------------------------------------------------- scoring helpers


def normalise_answer(text: str | None) -> str:
    """Whitespace- and case-normalise a model answer for exact-match scoring."""
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def is_exact_match(model_answer: str | None, expected: str) -> bool:
    return normalise_answer(model_answer) == normalise_answer(expected)


# -------------------------------------------------------------- evaluation


def _as_dataset(dataset: Any) -> list[dict[str, Any]]:
    if dataset is None:
        return load_generation_dataset()
    if isinstance(dataset, str):
        # allow a path
        try:
            # if it points at an existing file, load from disk
            if os.path.exists(dataset):
                return load_generation_dataset(dataset)
        except OSError:
            pass
        # fall through: try treating as JSON
        try:
            parsed = json.loads(dataset)
            if isinstance(parsed, list):
                return _as_dataset(parsed)
            if isinstance(parsed, dict):
                return load_generation_dataset.__wrapped__ if hasattr(
                    load_generation_dataset, "__wrapped__") else _as_dataset(
                    parsed.get("items", []))
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f"cannot interpret dataset argument {dataset!r}")
    if isinstance(dataset, dict):
        return _as_dataset(dataset.get("items", dataset.get("dataset", [])))
    if isinstance(dataset, list):
        # validate via load-like path by writing? instead inline:
        items = []
        for i, it in enumerate(dataset):
            if not isinstance(it, dict):
                raise ValueError(f"item {i} is not an object")
            for key in ("id", "prompt", "expected"):
                if key not in it:
                    raise ValueError(f"item {i} missing required key {key!r}")
            items.append({
                "id": str(it["id"]),
                "prompt": str(it["prompt"]),
                "expected": str(it["expected"]),
                "category": str(it.get("category", "")),
            })
        if not items:
            raise ValueError("dataset contains no items")
        return items
    # allow a callable factory
    if callable(dataset):
        return _as_dataset(dataset())
    raise ValueError(f"unsupported dataset type {type(dataset).__name__}")


def evaluate_generation(
    client: OpenAICompatClient,
    dataset: Any = None,
    dataset_version: str | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    seed: int | None = 1234,
    run_id: str = "genqual",
    workload: str = "quality_gen",
) -> dict[str, Any]:
    """Run the generation-based quality probe against ``client``.

    Returns a dict with ``status``, ``dataset_version``, ``dataset_hash``,
    per-item results and aggregate accuracy/latency/token counters.
    """
    ds_version = dataset_version
    if isinstance(dataset, dict):
        if ds_version is None:
            ds_version = dataset.get("version") or dataset.get("dataset_version")
    items = _as_dataset(dataset)
    h = dataset_hash(items)
    n = len(items)
    if ds_version is None:
        ds_version = DATASET_VERSION

    per_item: list[dict[str, Any]] = []
    errors: list[str] = []
    n_ok = 0
    n_failed = 0
    n_exact = 0
    latencies: list[float] = []
    total_prompt_tokens = None
    total_completion_tokens = None
    n_with_usage = 0

    for i, it in enumerate(items):
        rid = f"{run_id}|{it['id']}"
        row: dict[str, Any] = {
            "i": i,
            "id": it["id"],
            "category": it.get("category", ""),
            "prompt": it["prompt"],
            "expected": it["expected"],
            "raw_answer": None,
            "model_answer": None,
            "passed": False,
            "latency_s": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "http_status": None,
            "error": None,
        }
        try:
            complete = getattr(client, "complete_once", None)
            if complete is None:
                got = client.complete_scored(
                    prompt=it["prompt"], max_tokens=max_tokens, run_id=rid,
                    workload=workload, temperature=temperature, seed=seed,
                    echo=False, n_logprobs=0,
                )
            else:
                got = complete(
                    prompt=it["prompt"], max_tokens=max_tokens, run_id=rid,
                    workload=workload, temperature=temperature, seed=seed,
                )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"item {it['id']}: {row['error']}")
            n_failed += 1
            per_item.append(row)
            continue

        raw = got.get("raw") or {}
        row["latency_s"] = got.get("wall_s")
        choices = raw.get("choices") or []
        text = None
        if choices:
            text = choices[0].get("text")
        row["raw_answer"] = text
        row["model_answer"] = normalise_answer(text)
        usage = raw.get("usage") or {}
        row["prompt_tokens"] = usage.get("prompt_tokens")
        row["completion_tokens"] = usage.get("completion_tokens")
        row["http_status"] = 200

        has_usage = row["prompt_tokens"] is not None or row["completion_tokens"] is not None
        if has_usage:
            n_with_usage += 1
            if total_prompt_tokens is None:
                total_prompt_tokens = 0
            if total_completion_tokens is None:
                total_completion_tokens = 0
            if row["prompt_tokens"] is not None:
                total_prompt_tokens += int(row["prompt_tokens"])
            if row["completion_tokens"] is not None:
                total_completion_tokens += int(row["completion_tokens"])
        if row["latency_s"] is not None:
            latencies.append(float(row["latency_s"]))

        if not normalise_answer(text):
            row["passed"] = False
            row["error"] = "empty response"
            errors.append(f"item {it['id']}: empty response")
            n_failed += 1
            per_item.append(row)
            continue

        if is_exact_match(text, it["expected"]):
            n_exact += 1
            row["passed"] = True
        n_ok += 1
        per_item.append(row)

    accuracy = (n_exact / n_ok) if n_ok else None

    if n_ok == 0:
        status = "error" if errors else "unsupported"
    elif n_failed > 0:
        status = "partial"
    else:
        status = "ok"

    usage_coverage = (n_with_usage / n) if n else None
    return {
        "kind": "generation_quality_run",
        "status": status,
        "dataset_version": ds_version,
        "dataset_version_resolved": ds_version,
        "dataset_hash": h,
        "dataset_items": n,
        "n_items": n,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_exact_match": n_exact,
        "exact_match_accuracy": accuracy,
        "latency_s_mean": (statistics.mean(latencies) if latencies else None),
        "latency_s_p50": (statistics.median(latencies) if latencies else None),
        "latency_s_max": (max(latencies) if latencies else None),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "token_usage_total_prompt": total_prompt_tokens,
        "token_usage_total_completion": total_completion_tokens,
        "token_usage_available": n_with_usage > 0,
        "n_with_usage": n_with_usage,
        "usage_coverage": usage_coverage,
        "usage_available": n_with_usage > 0,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "per_item": per_item,
        "errors": errors[:20],
        "limitations": [
            "Exact match is strict: paraphrasing or added prefixes are counted "
            "as failures even when semantically correct.",
            "Latency is measured client-side against the non-streaming response "
            "and includes HTTP overhead.",
            "Do not compare accuracy across backends unless their output "
            "decoding (sampling, temperature, stop tokens) is matched.",
            "Small fixed datasets give wide confidence intervals; treat as a "
            "smoke/capability signal, not a leaderboard.",
        ],
    }


# Backwards/alternative names that callers and tests may use.
run_generation_eval = evaluate_generation
run_generation_quality = evaluate_generation
measure_generation_quality = evaluate_generation
generation_eval = evaluate_generation

GenerationEval = evaluate_generation


# ---------------------------------------------------------------- CLI hook
def format_summary(result: dict[str, Any]) -> str:
    return (
        f"gen-quality: status={result.get('status')} "
        f"acc={result.get('exact_match_accuracy')} "
        f"({result.get('n_exact_match')}/{result.get('n_ok')}) "
        f"dataset={result.get('dataset_version')} "
        f"hash={result.get('dataset_hash', '')[:12]}"
    )
