"""Benchmark orchestrator.

Execution model
---------------
For each matrix cell (backend x model x quantization):

  1. resolve + download the exact model file
  2. start the backend server, record time-to-ready
  3. run a warmup pass (discarded) -- this matters: the first CUDA graphs,
     autotuning sweeps and allocator warmups a server performs would otherwise
     land entirely on the first measured request
  4. run each workload `repeats` times, sampling GPU/host memory throughout
  5. run the quality probe against the same live server
  6. tear down

Results are appended to disk after every single run, so a crash or OOM late in
the matrix costs one cell rather than the whole session.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any

from . import config as C
from . import generation_eval as GE
from . import quality as Q
from .backends import BACKENDS, ResourceSampler, tail
from .core import (OpenAICompatClient, calibrate_overhead, records_to_jsonable,
                   run_concurrent, summarize)
from .workloads import TPOT_SLO_S, TTFT_SLO_S, WORKLOADS, build_prompts


# --------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------

def capture_environment() -> dict[str, Any]:
    """Snapshot everything needed to interpret (or reproduce) the numbers."""
    env: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "logical_cores": os.cpu_count(),
    }

    try:
        out = subprocess.run(["nvidia-smi",
                              "--query-gpu=name,memory.total,driver_version,compute_cap",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20)
        env["gpu"] = out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        env["gpu"] = f"unavailable: {exc}"

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:  # noqa: BLE001
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    env["ram_gib"] = round(kb / 1024 / 1024, 2)
                    break
    except Exception:  # noqa: BLE001
        pass

    # WSL RAM is capped by .wslconfig on the host; the guest cannot see more
    # than that, so record it -- it bounds CPU-path measurements.
    env["wsl_note"] = ("running inside WSL2; visible RAM is the .wslconfig cap, "
                       "which is smaller than host RAM")
    return env


# --------------------------------------------------------------------------
# version capture
# --------------------------------------------------------------------------

def capture_versions(env_root: str) -> dict[str, Any]:
    vers: dict[str, Any] = {}
    llama_bin = os.path.join(env_root, "build-llamacpp", "bin", "llama-server")
    if os.path.exists(llama_bin):
        try:
            commit = subprocess.run(
                ["git", "-C", os.path.join(env_root, "src", "llama.cpp"),
                 "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20
            ).stdout.strip()
            vers["llamacpp"] = {"commit": commit,
                                "build": "GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=89"}
        except Exception as exc:  # noqa: BLE001
            vers["llamacpp"] = {"error": str(exc)}

    for name, venv in [("vllm", "venv-vllm"), ("sglang", "venv-sglang")]:
        py = os.path.join(env_root, venv, "bin", "python")
        if os.path.exists(py):
            try:
                code = (
                    f"import importlib.metadata as m;"
                    f"print(m.version({name!r}))"
                )
                out = subprocess.run([py, "-c", code], capture_output=True,
                                     text=True, timeout=90)
                vers[name] = out.stdout.strip() or out.stderr.strip()[:200]
            except Exception as exc:  # noqa: BLE001
                vers[name] = {"error": str(exc)}
    return vers


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

class BenchmarkRunner:
    def __init__(self, env_root: str, results_dir: str, models_dir: str,
                 backends: list[str], workloads: list[str] | None = None,
                 repeats: int = 3, do_quality: bool = True,
                 device: str = "cuda", warmup_reps: int = 1,
                 n_ppl_chunks: int = 12, ppl_ctx: int = 2048):
        self.env_root = env_root
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.backends = backends
        self.workloads = workloads or list(WORKLOADS.keys())
        self.repeats = repeats
        self.do_quality = do_quality
        self.device = device
        self.warmup_reps = warmup_reps
        self.n_ppl_chunks = n_ppl_chunks
        self.ppl_ctx = ppl_ctx

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

        self.raw_path = os.path.join(results_dir, "raw", "runs.jsonl")
        self.cells_path = os.path.join(results_dir, "raw", "matrix.json")
        self.env_path = os.path.join(results_dir, "raw", "environment.json")
        self.versions_path = os.path.join(results_dir, "raw", "versions.json")
        self.lock_path = os.path.join(results_dir, "raw", ".run.lock")

    # -- single-instance guard --------------------------------------------
    def _acquire_lock(self) -> None:
        """Refuse to run if another benchmark is already active.

        Two concurrent runners were launched accidentally during development and
        both drove the GPU while appending to the same runs.jsonl. Every latency
        number that overlapped was contaminated by the other server, and nothing
        in the output made that visible. A lock makes the failure loud instead
        of silent.
        """
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path, encoding="utf-8") as f:
                    info = json.load(f)
                pid = info.get("pid")
                alive = False
                if pid:
                    try:
                        os.kill(int(pid), 0)
                        alive = True
                    except (OSError, ValueError):
                        alive = False
                if alive:
                    raise SystemExit(
                        f"another benchmark is running (pid {pid}, started "
                        f"{info.get('started')}). Refusing to start a second "
                        f"one -- concurrent servers contaminate every "
                        f"measurement. If you are sure it is gone, delete "
                        f"{self.lock_path}.")
                print(f"[lock] clearing stale lock from dead pid {pid}")
            except json.JSONDecodeError:
                print("[lock] clearing corrupt lock file")

        with open(self.lock_path, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)

    def _release_lock(self) -> None:
        try:
            os.remove(self.lock_path)
        except OSError:
            pass

    # -- persistence ------------------------------------------------------
    def _append_raw(self, record: dict[str, Any]) -> None:
        with open(self.raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()

    def _write_json(self, path: str, obj: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)

    # -- one measured run --------------------------------------------------
    def _measure(self, client: OpenAICompatClient, cell: C.Cell,
                 workload: str, rep: int, sampler: ResourceSampler) -> None:
        w = WORKLOADS[workload]
        prompts = build_prompts(workload, seed=1234)
        run_id = f"{cell.backend}|{cell.model_key}|{cell.quant_level}|{workload}|r{rep}"

        t0 = time.perf_counter()
        results = run_concurrent(
            client, prompts, max_tokens=w["max_tokens"],
            run_id=run_id, workload=workload, seed=1234,
        )
        wall = time.perf_counter() - t0

        mem = sampler.summary()
        summary = summarize(results, wall)

        # goodput: requests meeting BOTH latency objectives, expressed as
        # tokens/s. A backend can post excellent mean throughput while failing
        # this, which is the interesting failure mode.
        good_tokens = 0
        for r in results:
            if r.error is None and r.ttft_s is not None and r.tpot_s is not None:
                if r.ttft_s <= TTFT_SLO_S and r.tpot_s <= TPOT_SLO_S:
                    good_tokens += (r.completion_tokens or 0)
        summary["goodput_output_tok_s"] = good_tokens / wall if wall else None
        summary["slo_ttft_ms"] = TTFT_SLO_S * 1000
        summary["slo_tpot_ms"] = TPOT_SLO_S * 1000

        self._append_raw({
            "kind": "perf_run",
            "run_id": run_id,
            "cell": cell.to_dict(),
            "workload": workload,
            "replicate": rep,
            "workload_spec": w,
            "summary": summary,
            "memory": mem,
            "requests": records_to_jsonable(results),
        })

    def _quality(self, client: OpenAICompatClient, cell: C.Cell,
                 corpus: str, corpus_source: str) -> None:
        # llama-server cannot serve prompt logprobs (it rejects echo=true and
        # returns empty logprobs), so the llama.cpp arm is scored offline from
        # the identical GGUF file. Provenance is recorded so the report never
        # presents the two methods as interchangeable.
        if cell.backend == "llamacpp":
            corpus_path = os.path.join(self.results_dir, "raw",
                                       "wikitext2-valid.txt")
            ppl = Q.perplexity_offline_gguf(
                cell.local_path or "", corpus_path, ctx=self.ppl_ctx)
            res = {"perplexity": ppl, "mc_accuracy": {"status": "skipped",
                    "reason": "top-logprobs not served by llama.cpp "
                              "/v1/completions"}}
        else:
            res = Q.run_quality(client, corpus, n_chunks=self.n_ppl_chunks)

        self._append_raw({
            "kind": "quality_run",
            "run_id": f"{cell.backend}|{cell.model_key}|{cell.quant_level}|quality",
            "cell": cell.to_dict(),
            "corpus_source": corpus_source,
            "corpus_chars": len(corpus),
            "quality": res,
        })

    def _generation_quality(self, client: OpenAICompatClient, cell: C.Cell) -> None:
        dataset = None
        try:
            dataset = GE.load_generation_dataset()
        except Exception:  # noqa: BLE001
            dataset = None
        result = GE.evaluate_generation(
            client, dataset=dataset, max_tokens=64,
            run_id=f"{cell.backend}|{cell.model_key}|{cell.quant_level}|genqual")
        self._append_raw({
            "kind": "generation_quality_run",
            "run_id": (f"{cell.backend}|{cell.model_key}|{cell.quant_level}|"
                       "gen_quality"),
            "cell": cell.to_dict(),
            "quality": result,
            "dataset_version": result.get("dataset_version"),
            "dataset_hash": result.get("dataset_hash"),
        })

    # -- one cell ----------------------------------------------------------
    def run_cell(self, cell: C.Cell, corpus: str, corpus_source: str) -> dict[str, Any]:
        if cell.status == "skipped":
            return cell.to_dict()

        from .models import resolve_cell_path
        try:
            path = resolve_cell_path(cell, self.models_dir)
        except Exception as exc:  # noqa: BLE001
            cell.status = "failed"
            cell.skip_reason = f"download/resolve failed: {type(exc).__name__}: {exc}"
            return cell.to_dict()

        cell.local_path = path
        cell.status = "ready"
        self._write_json(self.cells_path, {
            "matrix": C.matrix_to_dicts([cell]),
            "note": "single-cell update; full matrix written by runner --list",
        })

        backend_cls = BACKENDS[cell.backend]
        backend = backend_cls(self.env_root, os.path.join(self.results_dir, "logs"))

        # device-aware launch args
        if cell.device == "cpu":
            kw = {"n_gpu_layers": 0} if cell.backend == "llamacpp" else {}
        else:
            kw = {}
        if cell.extra_args:
            kw["extra_args"] = list(cell.extra_args)

        print(f"[cell] starting {cell.backend} {cell.model_label} "
              f"{cell.quant_level} ({cell.quant_scheme})", flush=True)

        handle = backend.start(path, quant_label=f"{cell.model_key}_{cell.quant_level}", **kw)

        if not handle.startup_ok:
            log_tail = tail(handle.log_path, 30)
            cell.status = "failed"
            cell.skip_reason = f"server failed to become healthy: {handle.startup_error}"
            self._append_raw({
                "kind": "startup_failure",
                "cell": cell.to_dict(),
                "startup_s": handle.startup_s,
                "log_tail": log_tail,
                "cmd": handle.extra.get("cmd"),
            })
            backend.stop(handle)
            return cell.to_dict()

        # The `model` field sent to /v1/completions is ignored by llama-server and
        # must be the served path for vLLM; using the path works for both.
        client = OpenAICompatClient(handle.base_url, model=path)
        # ollama has no per-configuration process (one long-lived daemon), so
        # host RSS is not attributable to this cell; GPU sampling still is.
        pid = handle.proc.pid if handle.proc is not None else None
        sampler = ResourceSampler(pid, interval=0.5).start()

        # Warmup, discarded. This is not optional bookkeeping: the first
        # measured request in a cell consistently landed ~3x slower than later
        # ones (21.9 / 24.2 / 7.4 ms TTFT for an identical request) because the
        # GPU is still at idle clocks and the server has not finished its own
        # graph/allocator warmup. Sweeping *every* workload rather than just the
        # first also removes the cold-clock penalty from whichever workload
        # happens to be measured first.
        try:
            for _ in range(self.warmup_reps):
                for wl in self.workloads:
                    w = WORKLOADS[wl]
                    run_concurrent(client, build_prompts(wl, seed=9999 + _),
                                   max_tokens=min(32, w["max_tokens"]),
                                   run_id="warmup", workload=wl)
        except Exception:  # noqa: BLE001
            pass

        for wl in self.workloads:
            for rep in range(self.repeats):
                try:
                    self._measure(client, cell, wl, rep, sampler)
                except Exception as exc:  # noqa: BLE001
                    self._append_raw({
                        "kind": "run_error",
                        "run_id": f"{cell.backend}|{cell.model_key}|{cell.quant_level}|{wl}|r{rep}",
                        "cell": cell.to_dict(),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                print(f"  -> {wl} rep {rep} done", flush=True)

        if self.do_quality:
            try:
                self._quality(client, cell, corpus, corpus_source)
                print("  -> quality done", flush=True)
            except Exception as exc:  # noqa: BLE001
                self._append_raw({
                    "kind": "quality_error",
                    "cell": cell.to_dict(),
                    "error": f"{type(exc).__name__}: {exc}",
                })
            try:
                self._generation_quality(client, cell)
                print("  -> generation quality done", flush=True)
            except Exception as exc:  # noqa: BLE001
                self._append_raw({
                    "kind": "generation_quality_error",
                    "cell": cell.to_dict(),
                    "error": f"{type(exc).__name__}: {exc}",
                })

        peak = sampler.stop()
        backend.stop(handle)

        self._append_raw({
            "kind": "cell_summary",
            "cell": cell.to_dict(),
            "startup_s": handle.startup_s,
            "memory_overall": peak,
            "n_workloads": len(self.workloads),
            "repeats": self.repeats,
        })
        print(f"[cell] done {cell.backend} {cell.model_label} {cell.quant_level}",
              flush=True)
        return cell.to_dict()

    def run_quality_only(self, models: list[C.ModelSpec],
                         limits: dict | None = None) -> None:
        """Re-run only the quality probe for cells whose perf data already exists.

        Quality is an independent axis, so a failure in it does not invalidate
        (or require repeating) the latency measurements. This exists because the
        MC probe initially crashed on vLLM's `None` top-logprob entries, which
        cost the quality axis for cells already measured.
        """
        corpus, corpus_source = Q.load_corpus(os.path.join(self.results_dir, "raw"))
        cells = C.build_matrix(models, limits.get("backends", []) if limits else [],
                               device=self.device)

        # only fill gaps: re-measuring a cell that already has a successful
        # quality record would duplicate it in runs.jsonl
        have: set[tuple] = set()
        if os.path.exists(self.raw_path):
            with open(self.raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("kind") != "quality_run":
                        continue
                    ppl = (rec.get("quality") or {}).get("perplexity") or {}
                    if ppl.get("status") == "ok":
                        c = rec.get("cell") or {}
                        have.add((c.get("backend"), c.get("model_key"),
                                  c.get("quant_level")))

        for cell in cells:
            if cell.status == "skipped":
                continue
            if limits and limits.get("models") and cell.model_key not in limits["models"]:
                continue
            k = (cell.backend, cell.model_key, cell.quant_level)
            if k in have:
                print(f"[quality-only] skip {cell.backend} {cell.model_key} "
                      f"{cell.quant_level}: already measured")
                continue

            from .models import resolve_cell_path
            try:
                path = resolve_cell_path(cell, self.models_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"[quality-only] {cell.backend} {cell.model_key} "
                      f"{cell.quant_level}: cannot resolve ({exc})")
                continue
            cell.local_path = path
            cell.status = "ready"

            backend_cls = BACKENDS[cell.backend]
            backend = backend_cls(self.env_root,
                                  os.path.join(self.results_dir, "logs"))
            kw = {}
            if cell.device == "cpu" and cell.backend == "llamacpp":
                kw = {"n_gpu_layers": 0}
            if cell.extra_args:
                kw["extra_args"] = list(cell.extra_args)

            print(f"[quality-only] {cell.backend} {cell.model_key} "
                  f"{cell.quant_level}", flush=True)
            handle = backend.start(path, quant_label=f"{cell.model_key}_{cell.quant_level}",
                                   **kw)
            if not handle.startup_ok:
                print(f"   startup failed: {handle.startup_error}")
                backend.stop(handle)
                continue
            client = OpenAICompatClient(handle.base_url, model=path)
            try:
                self._quality(client, cell, corpus, corpus_source)
                print("   quality ok", flush=True)
            except Exception as exc:  # noqa: BLE001
                self._append_raw({
                    "kind": "quality_error",
                    "cell": cell.to_dict(),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"   quality error: {exc}")
            backend.stop(handle)

    # -- full run ----------------------------------------------------------
    def _completed_keys(self) -> set[tuple]:
        """(backend, model_key, quant_level) cells already fully measured.

        A cell counts as complete only when it has a recorded run for every
        (workload, replicate) pair. Counting unique workload NAMES instead was
        wrong: that set can never exceed len(workloads), so the
        len(workloads) * repeats threshold was never met and `--resume`
        silently re-ran completed cells on every restart -- which is how the
        matrix could restart forever without ever advancing to the vLLM cells.
        """
        if not os.path.exists(self.raw_path):
            return set()
        expected = len(self.workloads) * self.repeats
        per_cell: dict[tuple, set[tuple[str, Any]]] = {}
        with open(self.raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") != "perf_run":
                    continue
                s = rec.get("summary") or {}
                # a run that produced zero successful requests does not count
                if not s.get("n_ok"):
                    continue
                c = rec.get("cell") or {}
                key = (c.get("backend"), c.get("model_key"), c.get("quant_level"))
                per_cell.setdefault(key, set()).add(
                    (rec.get("workload"), rec.get("replicate")))
        return {k for k, v in per_cell.items() if len(v) >= expected}

    def run(self, models: list[C.ModelSpec], limits: dict | None = None,
            resume: bool = False) -> None:
        self._acquire_lock()
        try:
            self._run_locked(models, limits, resume)
        finally:
            self._release_lock()

    def _run_locked(self, models: list[C.ModelSpec], limits: dict | None,
                    resume: bool) -> None:
        env = capture_environment()
        vers = capture_versions(self.env_root)
        corpus, corpus_source = Q.load_corpus(os.path.join(self.results_dir, "raw"))

        self._write_json(self.env_path, env)
        self._write_json(self.versions_path, vers)
        self._append_raw({
            "kind": "session",
            "environment": env,
            "versions": vers,
            "corpus_source": corpus_source,
            "client_overhead": calibrate_overhead(""),
            "workloads": {k: v for k, v in WORKLOADS.items()},
            "slo": {"ttft_ms": TTFT_SLO_S * 1000, "tpot_ms": TPOT_SLO_S * 1000},
            "repeats": self.repeats,
        })

        cells = C.build_matrix(models, self.backends, device=self.device)
        if limits:
            cells = [c for c in cells
                     if (not limits.get("backends") or c.backend in limits["backends"])
                     and (not limits.get("models") or c.model_key in limits["models"])
                     and (not limits.get("quants") or c.quant_level in limits["quants"])]

        self._write_json(self.cells_path, {"matrix": C.matrix_to_dicts(cells)})

        skip: set[tuple] = self._completed_keys() if resume else set()
        if skip:
            print(f"[resume] {len(skip)} cells already complete; skipping them")

        results: list[dict[str, Any]] = []
        for i, cell in enumerate(cells, 1):
            if (cell.backend, cell.model_key, cell.quant_level) in skip:
                print(f"=== [{i}/{len(cells)}] SKIP (done) {cell.backend} "
                      f"{cell.model_label} {cell.quant_level} ===", flush=True)
                results.append(cell.to_dict())
                continue
            print(f"\n=== [{i}/{len(cells)}] {cell.backend} / {cell.model_label} "
                  f"/ {cell.quant_level} / {cell.quant_scheme} ===", flush=True)
            results.append(self.run_cell(cell, corpus, corpus_source))

        # run_cell returns plain dicts, so serialise directly
        self._write_json(self.cells_path, {"matrix": results})
        print(f"\nDONE. raw results: {self.raw_path}")