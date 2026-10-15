"""Request-level metrics: TTFT, inter-token latency, throughput.

All timing is done client-side with a monotonic clock on arrival of each SSE
chunk. This is deliberately the *external* view of latency -- it includes HTTP
overhead, JSON parsing and event-loop scheduling. For sub-100ms deltas those
overheads matter, so we measure the floor (see `calibrate_overhead`) and record
it alongside results rather than pretending it is zero.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import requests


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibrate_overhead(url: str, samples: int = 200) -> dict[str, float]:
    """Measure the client's own SSE-frame overhead.

    We time N synthetic chunk parses to estimate the fixed per-frame cost added
    by our client. Reported so downstream readers can judge how much of ITL at
    the low end is measurement artifact.
    """
    frames = []
    for _ in range(samples):
        t0 = time.perf_counter()
        chunk = {"choices": [{"text": "x", "index": 0}]}
        _ = chunk["choices"][0]["text"]
        frames.append(time.perf_counter() - t0)
    frames.sort()
    return {
        "synthetic_frame_p50_ms": frames[len(frames) // 2] * 1000.0,
        "synthetic_frame_max_ms": frames[-1] * 1000.0,
    }


# --------------------------------------------------------------------------
# result records
# --------------------------------------------------------------------------

@dataclass
class RequestResult:
    """One completion request, fully decomposed."""

    run_id: str
    workload: str
    request_id: str
    prompt_chars: int
    target_max_tokens: int

    # timing (seconds unless noted)
    start_offset_s: float = 0.0     # offset from concurrent-group start
    ttft_s: float | None = None
    total_s: float = 0.0
    e2e_s: float = 0.0

    # tokens
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    # derived
    inter_token_latencies_s: list[float] = field(default_factory=list)
    tpot_s: float | None = None     # (t_last - t_first) / (n_tok - 1)
    decode_tok_per_s: float | None = None
    prefill_tok_per_s: float | None = None
    itl_p50_s: float | None = None
    itl_p95_s: float | None = None
    itl_p99_s: float | None = None
    itl_std_s: float | None = None
    itl_max_s: float | None = None

    # integrity
    text_len: int = 0
    finish_reason: str | None = None
    error: str | None = None
    http_status: int | None = None

    def finalize(self) -> "RequestResult":
        """Compute derived stats from raw inter-token latencies."""
        itl = self.inter_token_latencies_s
        n = self.completion_tokens if self.completion_tokens is not None else 0
        pt = self.prompt_tokens or 0

        if len(itl) >= 2:
            self.itl_p50_s = statistics.median(itl)
            s = sorted(itl)
            self.itl_p95_s = s[min(len(s) - 1, int(0.95 * len(s)))]
            self.itl_p99_s = s[min(len(s) - 1, int(0.99 * len(s)))]
            self.itl_std_s = statistics.pstdev(itl) if len(itl) > 1 else 0.0
            self.itl_max_s = s[-1]
            self.tpot_s = sum(itl) / len(itl)

        if self.e2e_s and n:
            # end-to-end output throughput, excluding nothing (this is what a
            # user waiting on the request experiences)
            self.decode_tok_per_s = n / self.e2e_s
        if self.ttft_s and pt:
            # tokens processed per second before first token arrived
            self.prefill_tok_per_s = pt / self.ttft_s

        return self


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------

class OpenAICompatClient:
    """Minimal SSE streaming client for OpenAI-compatible /v1/completions.

    We use the *completions* endpoint rather than /v1/chat/completions on
    purpose: chat endpoints apply each model's chat template, and Qwen3-style
    hybrid-thinking models inject a reasoning block that changes token counts
    per model. Completions keeps the workload byte-identical across every model
    and backend, which is the entire point of a controlled comparison.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 600.0,
                 extra_body: dict | None = None):
        # Normalize: providers are configured with a /v1 suffix (that's the
        # OpenAI convention), but the request paths below already carry it.
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.session = requests.Session()

        # urllib3's default pool is 10 connections. With 32 concurrent
        # requests that makes 22 threads queue on the CLIENT before their
        # request is even sent -- and since TTFT is measured from the moment we
        # call post(), that client-side queueing was being reported as server
        # latency (it showed up as TTFT climbing to several seconds while the
        # server was idle). Size the pool above the largest workload.
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=128, pool_maxsize=128, max_retries=0,
            pool_block=False)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def complete_streaming(
        self,
        prompt: str,
        max_tokens: int,
        run_id: str,
        workload: str,
        temperature: float = 0.0,
        seed: int | None = 1234,
        group_start: float | None = None,
        request_id: str | None = None,
        stream_usage: bool = True,
    ) -> RequestResult:
        rid = request_id or uuid.uuid4().hex[:12]
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if seed is not None:
            payload["seed"] = seed
        if stream_usage:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.extra_body)

        res = RequestResult(
            run_id=run_id,
            workload=workload,
            request_id=rid,
            prompt_chars=len(prompt),
            target_max_tokens=max_tokens,
        )

        t0 = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                stream=True,
                timeout=self.timeout,
                headers={"Accept": "text/event-stream"},
            )
            res.http_status = resp.status_code
            resp.raise_for_status()

            arrival: list[float] = []
            text_parts: list[str] = []

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if not raw.startswith("data:"):
                    continue
                body = raw[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()

                usage = obj.get("usage")
                if usage:
                    res.prompt_tokens = usage.get("prompt_tokens")
                    res.completion_tokens = usage.get("completion_tokens")

                for ch in obj.get("choices", []):
                    delta = ch.get("text")
                    if delta:
                        # only count frames that actually carry a token
                        arrival.append(now)
                        text_parts.append(delta)
                    fr = ch.get("finish_reason")
                    if fr:
                        res.finish_reason = fr

            res.e2e_s = time.perf_counter() - t0

            if len(arrival) >= 1:
                res.ttft_s = arrival[0] - t0
                res.inter_token_latencies_s = [
                    b - a for a, b in zip(arrival, arrival[1:])
                ]
            if res.completion_tokens is None:
                # fall back to counting text frames if the server omits usage
                res.completion_tokens = len(arrival)

            res.text_len = len("".join(text_parts))
            if group_start is not None:
                res.start_offset_s = arrival[0] - group_start if arrival else res.e2e_s

        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            res.error = f"{type(exc).__name__}: {exc}"
            res.e2e_s = time.perf_counter() - t0

        return res.finalize()

    def complete_scored(
        self, prompt: str, max_tokens: int, run_id: str, workload: str,
        temperature: float = 0.0, seed: int | None = 1234,
        echo: bool = True, n_logprobs: int = 0,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming request that can also ask for token logprobs.

        Two distinct uses, which must not share a request shape:
          * perplexity  -> echo=True  (needs prompt logprobs)
          * MC probe    -> echo=False (needs top-k at the first GENERATED token)

        They differ because with echo=True the `top_logprobs` array is aligned to
        the whole echoed sequence and its first element is `None` (the first
        prompt token has no prior context), which broke the MC parser.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if echo:
            payload["echo"] = True
        if stop:
            payload["stop"] = list(stop)
        payload["logprobs"] = n_logprobs
        if seed is not None:
            payload["seed"] = seed
        payload.update(self.extra_body)

        t0 = time.perf_counter()
        r = self.session.post(f"{self.base_url}/v1/completions",
                              json=payload, timeout=self.timeout)
        r.raise_for_status()
        obj = r.json()
        return {
            "raw": obj,
            "wall_s": time.perf_counter() - t0,
            "prompt_chars": len(prompt),
        }

    def complete_once(
        self, prompt: str, max_tokens: int, run_id: str, workload: str,
        temperature: float = 0.0, seed: int | None = None,
        timeout: float | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming completion with NO logprob parameters.

        This is the path used by generation-based quality evaluation, which must
        remain usable on backends that reject logprob arguments entirely. Unlike
        ``complete_scored`` it never sends ``logprobs`` or ``echo``.

        `stop` is passed through verbatim so a caller can bound generation. Base
        (non-instruct) completion models frequently ignore an "answer with only
        X" instruction and continue generating indefinitely; without a stop
        string the answer is buried in filler and exact-match scoring reports a
        false negative.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = seed
        payload.update(self.extra_body)

        t0 = time.perf_counter()
        r = self.session.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=timeout or self.timeout,
        )
        r.raise_for_status()
        obj = r.json()
        return {
            "raw": obj,
            "wall_s": time.perf_counter() - t0,
            "prompt_chars": len(prompt),
        }

    # Aliases for callers/tests that use different names for the same path.
    complete_generate = complete_once
    complete_nonstreaming = complete_once
    generate = complete_once


# --------------------------------------------------------------------------
# concurrency driver
# --------------------------------------------------------------------------

def run_concurrent(
    client: OpenAICompatClient,
    prompts: list[str],
    max_tokens: int,
    run_id: str,
    workload: str,
    seed: int | None = 1234,
) -> list[RequestResult]:
    """Fire `prompts` simultaneously behind a shared barrier.

    The barrier matters: without it, thread pool startup skews arrival times
    and we would measure Python thread scheduling instead of server-side
    queueing behaviour -- which is exactly the thing under test.
    """
    results: list[RequestResult | None] = [None] * len(prompts)
    barrier = threading.Barrier(len(prompts) + 1)
    errors: list[str] = []

    def worker(i: int, p: str) -> None:
        try:
            barrier.wait()
            results[i] = client.complete_streaming(
                prompt=p, max_tokens=max_tokens, run_id=run_id,
                workload=workload, seed=seed, request_id=f"r{i:04d}",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"worker {i}: {type(exc).__name__}: {exc}")
            try:
                barrier.wait(timeout=30)
            except Exception:
                pass

    threads = [
        threading.Thread(target=worker, args=(i, p), daemon=True)
        for i, p in enumerate(prompts)
    ]
    for t in threads:
        t.start()
    barrier.wait()          # release all workers at once
    for t in threads:
        t.join(timeout=client.timeout + 60)

    out: list[RequestResult] = []
    for i, r in enumerate(results):
        if r is None:
            out.append(RequestResult(
                run_id=run_id, workload=workload, request_id=f"r{i:04d}",
                prompt_chars=len(prompts[i]), target_max_tokens=max_tokens,
                error=errors[i] if i < len(errors) else "no result (thread died)",
            ))
        else:
            out.append(r)
    return out


# --------------------------------------------------------------------------
# aggregation helpers
# --------------------------------------------------------------------------

def _pct(values: Iterable[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def summarize(requests: list[RequestResult], group_wall_s: float) -> dict[str, Any]:
    """Roll a list of requests into run-level summary stats.

    `group_wall_s` is the wall time of the whole concurrent group -- aggregate
    throughput must be computed against that, not the sum of per-request times,
    or concurrent runs get credited with more throughput than the server
    actually delivered.
    """
    ok = [r for r in requests if r.error is None]
    if not ok:
        return {"n_requests": len(requests), "n_ok": 0, "error": "all requests failed"}

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
    e2es = [r.e2e_s for r in ok]
    comp = [r.completion_tokens or 0 for r in ok]
    prompts = [r.prompt_tokens or 0 for r in ok]

    all_itl: list[float] = []
    for r in ok:
        all_itl.extend(r.inter_token_latencies_s)

    total_completion = sum(comp)
    total_prompt = sum(prompts)

    summary = {
        "n_requests": len(requests),
        "n_ok": len(ok),
        "n_errored": len(requests) - len(ok),

        # latency, milliseconds
        "ttft_ms_mean": statistics.mean(ttfts) * 1000 if ttfts else None,
        "ttft_ms_p50": (_pct(ttfts, 0.50) or 0) * 1000 if ttfts else None,
        "ttft_ms_p95": (_pct(ttfts, 0.95) or 0) * 1000 if ttfts else None,
        "ttft_ms_p99": (_pct(ttfts, 0.99) or 0) * 1000 if ttfts else None,
        "ttft_ms_max": max(ttfts) * 1000 if ttfts else None,
        "ttft_ms_std": statistics.pstdev(ttfts) * 1000 if len(ttfts) > 1 else None,

        "tpot_ms_mean": statistics.mean(tpots) * 1000 if tpots else None,
        "tpot_ms_p50": (_pct(tpots, 0.50) or 0) * 1000 if tpots else None,
        "tpot_ms_p95": (_pct(tpots, 0.95) or 0) * 1000 if tpots else None,

        "itl_ms_p50": (_pct(all_itl, 0.50) or 0) * 1000 if all_itl else None,
        "itl_ms_p95": (_pct(all_itl, 0.95) or 0) * 1000 if all_itl else None,
        "itl_ms_p99": (_pct(all_itl, 0.99) or 0) * 1000 if all_itl else None,

        "e2e_ms_mean": statistics.mean(e2es) * 1000 if e2es else None,
        "e2e_ms_p95": (_pct(e2es, 0.95) or 0) * 1000 if e2es else None,

        # throughput
        "group_wall_s": group_wall_s,
        "output_throughput_tok_s": (total_completion / group_wall_s) if group_wall_s else None,
        "total_output_tokens": total_completion,
        "total_prompt_tokens": total_prompt,
        "prefill_throughput_tok_s": (total_prompt / group_wall_s) if group_wall_s else None,
        "goodput_output_tok_s": None,  # filled by caller with SLO definition
    }
    return summary


def records_to_jsonable(requests: list[RequestResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in requests]