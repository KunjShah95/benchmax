"""Deterministic workload generation.

Every prompt is generated from a seeded RNG over a fixed vocabulary, so the
same workload is byte-identical across backends, models and machines. The only
thing that varies between runs is the thing we are trying to measure.

Prompt lengths are specified in *words* because the client cannot count tokens
portably before the server's tokenizer is known. The actual token count is read
back from the server response (`prompt_tokens`) and recorded, so cross-tokenizer
comparisons stay honest.
"""

from __future__ import annotations

import random
from typing import Any


# A fixed vocabulary with a realistic Zipf-ish shape. Drawn from a small set of
# topical words rather than uniform noise, so the tokenizer's behaviour (and
# therefore prefill cost) is representative of natural text.
VOCAB = [
    "system", "model", "server", "token", "memory", "latency", "throughput",
    "quantization", "inference", "runtime", "kernel", "attention", "context",
    "sequence", "batch", "cache", "compute", "network", "storage", "dataset",
    "training", "evaluation", "baseline", "architecture", "framework", "library",
    "interface", "protocol", "request", "response", "scheduler", "pipeline",
    "parallel", "distribution", "optimization", "compression", "accuracy",
    "benchmark", "measurement", "overhead", "latency-bound", "throughput-bound",
    "reproducible", "deterministic", "controlled", "experiment", "hypothesis",
    "configuration", "environment", "resource", "constraint", "tradeoff",
    "hardware", "accelerator", "processor", "allocator", "profiler", "trace",
    "telemetry", "observability", "diagnostic", "anomaly", "regression",
]

CONNECTIVES = [
    "the", "a", "of", "and", "to", "in", "is", "that", "for", "with", "as",
    "by", "on", "at", "from", "it", "this", "that", "which", "be", "are",
    "not", "but", "than", "then", "when", "where", "while", "into", "over",
]


def _zipf_weights(n: int) -> list[float]:
    """1/rank weighting gives a natural-looking token distribution."""
    return [1.0 / (r + 1) for r in range(n)]


def make_body(target_words: int, seed: int) -> str:
    rng = random.Random(seed)
    vw = _zipf_weights(len(VOCAB))
    cw = _zipf_weights(len(CONNECTIVES))
    out: list[str] = []
    for _ in range(target_words):
        if rng.random() < 0.35:
            out.append(rng.choices(CONNECTIVES, weights=cw)[0])
        else:
            out.append(rng.choices(VOCAB, weights=vw)[0])
    return " ".join(out)


# A small set of realistic instruction stems. We deliberately use the
# completions endpoint, so these read as prefixes rather than chat turns.
STEMS = [
    "Summarize the following system design discussion.\n\n",
    "Explain the trade-offs between batched and unbatched inference.\n\n",
    "Given a serving workload, recommend a backend and justify it.\n\n",
    "List the operational risks of running this configuration in production.\n\n",
    "Rewrite the following paragraph in plain language.\n\n",
    "Analyze the performance characteristics described below.\n\n",
    "Continue the following technical note.\n\n",
]


def make_prompt(target_words: int, seed: int, stem: int | None = None) -> str:
    stem = seed % len(STEMS) if stem is None else stem
    return STEMS[stem % len(STEMS)] + make_body(target_words, seed)


# --------------------------------------------------------------------------
# workload definitions
# --------------------------------------------------------------------------

# Each workload is a point in the (input length) x (concurrency) plane. The set
# is chosen so that the three regimes every backend argues about are covered:
#   - prefill-dominated, no batching   -> raw single-stream speed
#   - decode-dominated, no batching    -> per-token latency floor
#   - heavily batched                  -> aggregate throughput ceiling
WORKLOADS: dict[str, dict[str, Any]] = {
    # --- single request, short prompt: the "interactive chat" case ---------
    "w1_single_short": {
        "n_requests": 1,
        "prompt_words": 40,
        "max_tokens": 128,
        "weight": 1.30,   # real-world arrival weight, used for goodput
        "description": "1 request, ~40-word prompt, 128 output tokens",
    },
    # --- single request, long prompt: prefill-dominated --------------------
    # Sized so prompt + max_tokens fits comfortably inside llama.cpp's
    # per-slot context (ctx / -np). At ~1000 words this is roughly 1350 tokens,
    # which leaves real headroom under a 2048-token slot while still putting
    # enough work in prefill to dominate TTFT.
    "w2_single_prefill": {
        "n_requests": 1,
        "prompt_words": 1000,
        "max_tokens": 64,
        "weight": 1.30,
        "description": "1 request, ~1000-word prompt (~1.3k tokens), 64 output tokens",
    },
    # --- moderate concurrency: light continuous batching --------------------
    "w3_concurrent_8": {
        "n_requests": 8,
        "prompt_words": 100,
        "max_tokens": 128,
        "weight": 1.20,
        "description": "8 concurrent requests, ~100-word prompts, 128 output tokens",
    },
    # --- high concurrency: throughput ceiling ------------------------------
    "w4_concurrent_32": {
        "n_requests": 32,
        "prompt_words": 150,
        "max_tokens": 128,
        "weight": 1.00,
        "description": "32 concurrent requests, ~150-word prompts, 128 output tokens",
    },
}


def build_prompts(workload: str, seed: int = 1234) -> list[str]:
    w = WORKLOADS[workload]
    return [
        make_prompt(w["prompt_words"], seed + i * 7919, stem=i)
        for i in range(w["n_requests"])
    ]


# Service-level objective used to compute goodput. Chosen to match what
# interactive products actually target rather than an arbitrary threshold.
TTFT_SLO_S = 1.0        # first token within 1s
TPOT_SLO_S = 0.100      # inter-token within 100ms