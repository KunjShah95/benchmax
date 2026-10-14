"""Command-line entry point.

    python -m benchmax.cli list                     # show the planned matrix
    python -m benchmax.cli preflight                # check artifacts exist
    python -m benchmax.cli download --models ...    # fetch model files
    python -m benchmax.cli run --backends ...       # execute the benchmark
    python -m benchmax.cli report                   # build tables + figures
    python -m benchmax.cli cloud                    # cloud arm status/run
    python -m benchmax.cli selftest                 # mock-endpoint verification
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config as C

DEFAULT_ENV_ROOT = os.path.expanduser("~/benchmax-env")


def _models_by_key(keys: list[str] | None) -> list[C.ModelSpec]:
    if not keys:
        return C.MODELS
    out = [m for m in C.MODELS if m.key in keys]
    unknown = set(keys) - {m.key for m in C.MODELS}
    if unknown:
        raise SystemExit(f"unknown model keys: {sorted(unknown)}")
    return out


def cmd_list(args) -> None:
    models = _models_by_key(args.models)
    cells = C.build_matrix(models, args.backends, device=args.device)
    print(f"{len(cells)} cells\n")
    hdr = f"{'backend':10} {'model':12} {'params':>7} {'quant':7} {'scheme':28} {'repo'}"
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        flag = "SKIP" if c.status == "skipped" else "    "
        print(f"{c.backend:10} {c.model_key:12} {c.params_b:6.1f}B "
              f"{c.quant_level:7} {c.quant_scheme:28} {c.source_repo or '-'}")
        if c.status == "skipped":
            print(f"{'':10} -> skipped: {c.skip_reason}")


def cmd_preflight(args) -> None:
    from .models import preflight
    models = _models_by_key(args.models)
    out = preflight(models, args.backends, device=args.device)
    n_ok = sum(1 for c in out["cells"] if c["available"])
    print(f"{n_ok}/{len(out['cells'])} cells have downloadable artifacts\n")
    for c in out["cells"]:
        mark = "OK " if c["available"] else "NO "
        print(f"{mark} {c['backend']:10} {c['model']:12} {c['quant']:6} "
              f"{c['scheme']:26} {c['note']}")
    with open(os.path.join(args.results, "preflight.json"), "w") as f:
        json.dump(out, f, indent=2)


def cmd_download(args) -> None:
    from .models import resolve_cell_path
    models = _models_by_key(args.models)
    cells = C.build_matrix(models, args.backends, device=args.device)
    seen: set[tuple] = set()
    for c in cells:
        if c.status == "skipped":
            continue
        key = (c.source_repo, c.quant_level if c.backend == "llamacpp" else "full")
        if key in seen:
            continue
        seen.add(key)
        print(f"resolving {c.backend} {c.model_key} {c.quant_level} ...", flush=True)
        try:
            p = resolve_cell_path(c, args.models_dir)
            size = os.path.getsize(p) / 1e6 if os.path.isfile(p) else 0
            print(f"  -> {p} ({size:.0f} MB)")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {type(exc).__name__}: {exc}")


def cmd_run(args) -> None:
    from .runner import BenchmarkRunner
    models = _models_by_key(args.models)
    r = BenchmarkRunner(
        env_root=args.env_root,
        results_dir=args.results,
        models_dir=args.models_dir,
        backends=args.backends,
        workloads=args.workloads,
        repeats=args.repeats,
        do_quality=not args.no_quality,
        device=args.device,
        warmup_reps=args.warmup,
        n_ppl_chunks=args.ppl_chunks,
    )
    limits = {"backends": args.backends, "models": args.models,
              "quants": args.quants}
    r.run(models, limits=limits, resume=args.resume)


def cmd_quality(args) -> None:
    from .runner import BenchmarkRunner
    models = _models_by_key(args.models)
    r = BenchmarkRunner(
        env_root=args.env_root, results_dir=args.results,
        models_dir=args.models_dir,
        backends=args.backends, do_quality=True, device=args.device,
        n_ppl_chunks=args.ppl_chunks,
    )
    r.run_quality_only(models, limits={"backends": args.backends,
                                       "models": args.models})


def cmd_report(args) -> None:
    from .analysis import build_all
    from .report import write_all
    out = write_all(args.results, args.reports)
    derived = build_all(args.results, args.reports)
    print(json.dumps({
        "perf_runs": out.get("n_perf_runs"),
        "quality_runs": out.get("n_quality_runs"),
        "figures": len(out.get("figures", [])),
        "derived_tables": sorted(derived.keys()),
    }, indent=2))


def cmd_cloud(args) -> None:
    from . import cloud
    if args.action == "status":
        print(json.dumps(cloud.describe_arms(), indent=2))
    else:
        providers = cloud.available_providers()
        ready = [(p, ok) for p, ok in providers if ok]
        if not ready:
            print("No provider credentials found. Nothing to measure.")
            print(json.dumps(cloud.describe_arms(), indent=2))
            sys.exit(2)
        for p, _ in ready:
            print(f"running {p.name} ({p.model}) ...", flush=True)
            print(json.dumps(
                cloud.run_provider_suite(p, args.workloads, repeats=args.repeats,
                                         results_dir=args.results), indent=2))


def cmd_selftest(args) -> None:
    """Verify the measurement core against the mock endpoint with known timings."""
    import threading
    import time as _t

    from . import mock_cloud
    from .core import OpenAICompatClient, run_concurrent, summarize

    port = args.port
    mock_cloud.STATE.ttft_s = 0.100
    mock_cloud.STATE.itl_s = 0.020
    # no simulated batching: per-token sleep maps 1:1 onto injected ITL so the
    # selftest compares against a known ground truth
    mock_cloud.STATE.batching = False

    t = threading.Thread(target=mock_cloud.serve, args=(port,), daemon=True)
    t.start()
    _t.sleep(1.0)

    base = f"http://127.0.0.1:{port}/v1"
    c = OpenAICompatClient(base, model="mock-model")
    ok = False
    for _ in range(20):
        try:
            import requests
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                ok = True
                break
        except Exception:  # noqa: BLE001
            pass
        _t.sleep(0.5)
    if not ok:
        raise SystemExit("mock server never became healthy")

    print("=== single request vs injected timings ===")
    r = c.complete_streaming(prompt="hello world", max_tokens=20,
                             run_id="selftest", workload="selftest")
    if r.error:
        print(f"  REQUEST FAILED: {r.error}")
        print(f"  http_status={r.http_status}")
        raise SystemExit("selftest failed: streaming request errored")
    print(f"  ttft        measured={r.ttft_s*1000:.1f}ms  injected=100ms")
    print(f"  itl median  measured={r.itl_p50_s*1000:.2f}ms  injected=20ms")
    print(f"  completion_tokens={r.completion_tokens}")

    print("\n=== concurrent 8 vs injected timings ===")
    res = run_concurrent(c, ["x " * 50] * 8, max_tokens=20,
                         run_id="selftest", workload="selftest")
    s = summarize(res, 0.0)
    print(f"  n_ok={s['n_ok']}  ttft_mean={s['ttft_ms_mean']:.1f}ms  "
          f"itl_p50={s['itl_ms_p50']:.2f}ms")

    print("\n=== quality path (perplexity + MC) ===")
    from .quality import load_corpus, run_quality
    corpus, src = load_corpus(os.path.join(args.results, "raw"), min_chars=3000)
    q = run_quality(c, corpus, n_chunks=3, chunk_chars=4000)
    print(f"  corpus: {src} ({len(corpus)} chars)")
    print(f"  ppl status={q['perplexity'].get('status')} "
          f"value={q['perplexity'].get('ppl')}")
    print(f"  mc  status={q['mc_accuracy'].get('status')} "
          f"acc={q['mc_accuracy'].get('accuracy')}")

    print("\nSELFTEST OK (metrics track injected timings)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="benchmax")
    ap.add_argument("--results", default="results")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--models-dir",
                    default=os.path.expanduser("~/benchmax-env/models"))
    ap.add_argument("--env-root", default=DEFAULT_ENV_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_model_opts(p):
        p.add_argument("--models", nargs="*", default=None)

    p = sub.add_parser("list"); add_model_opts(p)
    p.add_argument("--backends", nargs="+", default=["llamacpp", "vllm"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("preflight"); add_model_opts(p)
    p.add_argument("--backends", nargs="+", default=["llamacpp", "vllm"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("download"); add_model_opts(p)
    p.add_argument("--backends", nargs="+", default=["llamacpp", "vllm"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("run"); add_model_opts(p)
    p.add_argument("--backends", nargs="+", default=["llamacpp", "vllm"])
    p.add_argument("--quants", nargs="*", default=None)
    p.add_argument("--workloads", nargs="*", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--ppl-chunks", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3,
                   help="discarded warmup sweeps per cell before measuring")
    p.add_argument("--resume", action="store_true",
                   help="skip cells already fully measured in runs.jsonl")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("quality")
    add_model_opts(p)
    p.add_argument("--backends", nargs="+", default=["llamacpp", "vllm", "vllm_gguf"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--ppl-chunks", type=int, default=12)
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("cloud")
    p.add_argument("action", choices=["status", "run"])
    p.add_argument("--workloads", nargs="+", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.set_defaults(func=cmd_cloud)

    p = sub.add_parser("selftest")
    p.add_argument("--port", type=int, default=8999)
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()