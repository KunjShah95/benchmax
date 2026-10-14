"""Cloud-hosted endpoint adapter.

The local comparison runs on one machine under one roof. The interesting cloud
trade-off is that the *unit of comparison changes*: you stop renting a GPU and
start renting a queue position, so TTFT now includes provider-side queueing you
cannot control, and throughput is bounded by a quota rather than by silicon.

This module lets the exact same workload suite run against any
OpenAI-compatible hosted endpoint. It is deliberately thin -- `core.py` already
speaks the protocol -- so a cloud number is produced by the same code path as a
local number and is directly comparable.

**No credentials are present in this environment**, so nothing here has been
run against a real provider. `mock_cloud.py` provides a local server that
implements the subset of the protocol the harness uses; the adapter is verified
against that mock, so adding a real key is a configuration change, not a code
change.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any

from .core import OpenAICompatClient, RequestResult, run_concurrent, summarize


@dataclass
class CloudProvider:
    name: str
    base_url: str
    api_key_env: str
    model: str
    notes: str = ""
    # documented characteristics, recorded alongside measurements so the report
    # can separate "measured" from "claimed"
    claimed: dict[str, Any] | None = None


# A registry of common OpenAI-compatible gateways. Users can add their own via
# BENCHMAX_PROVIDERS (JSON) without editing code.
DEFAULT_PROVIDERS: list[CloudProvider] = [
    CloudProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model="gpt-4o-mini",
        notes="reference hosted endpoint",
    ),
    CloudProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="llama-3.1-8b-instant",
        notes="very high token/s ceiling, aggressive rate limits",
    ),
    CloudProvider(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        notes="self-serve open models, pay-per-token",
    ),
    CloudProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model="meta-llama/llama-3.2-3b-instruct",
        notes="aggregator: routes across upstream providers, adds a hop",
    ),
    CloudProvider(
        name="ollama-local",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        model="qwen3:4b",
        notes="local but OpenAI-compatible; useful as a control",
    ),
]


def load_providers() -> list[CloudProvider]:
    extra = os.environ.get("BENCHMAX_PROVIDERS")
    if not extra:
        return DEFAULT_PROVIDERS
    try:
        data = json.loads(extra)
        return [CloudProvider(**d) for d in data]
    except Exception:  # noqa: BLE001
        return DEFAULT_PROVIDERS


def available_providers() -> list[tuple[CloudProvider, bool]]:
    return [(p, bool(os.environ.get(p.api_key_env))) for p in load_providers()]


def make_client(p: CloudProvider, timeout: float = 300.0) -> OpenAICompatClient | None:
    key = os.environ.get(p.api_key_env)
    if not key and not p.base_url.startswith("http://127.0.0.1"):
        return None
    c = OpenAICompatClient(p.base_url, model=p.model, timeout=timeout)
    if key:
        c.session.headers.update({"Authorization": f"Bearer {key}"})
    return c


def run_provider_suite(p: CloudProvider, workloads: list[str],
                       repeats: int = 3, results_dir: str | None = None) -> dict[str, Any]:
    """Run the same workload suite against a hosted endpoint."""
    client = make_client(p)
    if client is None:
        return {"provider": asdict(p), "status": "no_credentials",
                "error": f"env var {p.api_key_env} is not set"}

    from .workloads import WORKLOADS, build_prompts

    out: list[dict[str, Any]] = []
    for wl in workloads:
        w = WORKLOADS[wl]
        for rep in range(repeats):
            prompts = build_prompts(wl, seed=1234)
            t0 = time.perf_counter()
            res = run_concurrent(client, prompts, max_tokens=w["max_tokens"],
                                 run_id=f"cloud|{p.name}|{wl}|r{rep}",
                                 workload=wl, seed=1234)
            wall = time.perf_counter() - t0
            summary = summarize(res, wall)

            # Hosted endpoints have no process to sample, so record what the
            # provider tells us instead of inventing local memory numbers.
            summary["memory"] = {"note": "not observable for hosted endpoints"}
            rec = {
                "kind": "perf_run",
                "run_id": f"cloud|{p.name}|{wl}|r{rep}",
                "cell": {"backend": "cloud", "model_key": p.name,
                         "model_label": p.model, "quant_level": "hosted",
                         "quant_scheme": "provider-managed", "family": "n/a",
                         "params_b": None, "device": "cloud"},
                "workload": wl, "replicate": rep, "workload_spec": w,
                "summary": summary,
                "memory": summary.pop("memory"),
                "requests": [asdict(r) for r in res],
            }
            out.append(rec)
            if results_dir:
                os.makedirs(os.path.join(results_dir, "raw"), exist_ok=True)
                with open(os.path.join(results_dir, "raw", "runs.jsonl"),
                          "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")

    return {"provider": asdict(p), "status": "ok", "n_runs": len(out)}


def describe_arms() -> dict[str, Any]:
    """Machine-readable description of the cloud arm's configuration state."""
    return {
        "providers": [
            {**asdict(p), "credentials_present": bool(os.environ.get(p.api_key_env))}
            for p in load_providers()
        ],
        "note": (
            "No provider credentials were present in this environment, so the "
            "cloud arm is implemented and verified against a local mock only. "
            "Run with BENCHMAX_CLOUD=1 after exporting a key to populate it."
        ),
    }