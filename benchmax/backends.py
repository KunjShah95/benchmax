"""Backend lifecycle management.

Each backend is launched as a real server process, exercised over HTTP, then
torn down. Nothing is imported in-process, because the interesting differences
between these projects are in the *server*: scheduling, batching, KV cache
management, kernel dispatch. An in-process API would hide exactly the
differences we are trying to measure.

Every backend here exposes the OpenAI-compatible surface so `core.py` can stay
backend-agnostic.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(url: str, timeout: float = 900.0, quiet: bool = True,
                    proc=None, log_path: str = "") -> tuple[bool, float, str]:
    """Block until /health returns 200. Returns (ok, seconds_to_ready, tail_log).

    If `proc` is supplied the wait also aborts as soon as the process exits.
    Without that, a backend that dies during startup (bad flags, missing nvcc,
    OOM) burns the entire timeout before anyone notices -- which cost 20 minutes
    per failing cell during development.
    """
    t0 = time.perf_counter()
    deadline = t0 + timeout
    last_err = ""
    while time.perf_counter() < deadline:
        if proc is not None and proc.poll() is not None:
            detail = tail(log_path, 25) if log_path else ""
            return (False, time.perf_counter() - t0,
                    f"process exited rc={proc.returncode} during startup\n{detail}")
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True, time.perf_counter() - t0, ""
            last_err = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}"
        time.sleep(1.0)
    return False, time.perf_counter() - t0, last_err


def nvidia_smi_mem() -> tuple[int, int, int]:
    """(used_MiB, total_MiB, util_pct) for GPU 0."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        used, total, util = out.stdout.strip().split("\n")[0].split(",")
        return int(used.strip()), int(total.strip()), int(util.strip())
    except Exception:  # noqa: BLE001
        return -1, -1, -1


def proc_tree_rss_mb(pid: int) -> float:
    """Resident memory of a process and its children, in MiB."""
    try:
        import psutil
        p = psutil.Process(pid)
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except Exception:  # noqa: BLE001
                pass
        return total / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return -1.0


class ResourceSampler:
    """Background sampler for GPU + host memory during a run.

    Sampling at 2 Hz. Peak values are what matter here: sustained throughput
    numbers hide the fact that vLLM preallocates its KV cache up front while
    llama.cpp grows it lazily, and that difference only shows up in the peak.
    """

    def __init__(self, pid: int | None, interval: float = 0.5):
        self.pid = pid
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            used, total, util = nvidia_smi_mem()
            rss = proc_tree_rss_mb(self.pid) if self.pid else -1.0
            self.samples.append({
                "t": time.time(),
                "gpu_used_mib": used,
                "gpu_total_mib": total,
                "gpu_util_pct": util,
                "host_rss_mib": rss,
            })
            self._stop.wait(self.interval)

    def start(self) -> "ResourceSampler":
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        gpu = [s["gpu_used_mib"] for s in self.samples if s["gpu_used_mib"] >= 0]
        rss = [s["host_rss_mib"] for s in self.samples if s["host_rss_mib"] >= 0]
        util = [s["gpu_util_pct"] for s in self.samples if s["gpu_util_pct"] >= 0]
        return {
            "gpu_peak_mib": max(gpu) if gpu else None,
            "gpu_mean_mib": sum(gpu) / len(gpu) if gpu else None,
            "gpu_util_mean_pct": sum(util) / len(util) if util else None,
            "host_rss_peak_mib": max(rss) if rss else None,
            "host_rss_mean_mib": sum(rss) / len(rss) if rss else None,
            "n_samples": len(self.samples),
        }


# --------------------------------------------------------------------------
# backend base
# --------------------------------------------------------------------------

@dataclass
class ServerHandle:
    backend: str
    model_path: str
    quant_label: str
    port: int
    base_url: str
    proc: Any = None
    log_path: str = ""
    startup_s: float | None = None
    startup_ok: bool = False
    startup_error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Backend:
    name = "base"

    def __init__(self, env_root: str, log_dir: str):
        self.env_root = env_root
        self.log_dir = log_dir
        # Configurations the backend had to degrade to in order to start at
        # all. Reported alongside the results so a measurement taken under a
        # forced fallback is never mistaken for a default configuration.
        self._capability_gaps: list[str] = []
        os.makedirs(log_dir, exist_ok=True)

    @property
    def capability_gaps(self) -> list[str]:
        return list(self._capability_gaps)

    def start(self, model_path: str, quant_label: str, **kw) -> ServerHandle:
        raise NotImplementedError

    def stop(self, h: ServerHandle) -> None:
        raise NotImplementedError

    # -- shared teardown ---------------------------------------------------
    @staticmethod
    def _kill_tree(proc) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            if hasattr(os, "killpg") and proc.pid:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            pass

    def base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # keep HF/torch from grabbing all cores for intra-op threads while the
        # server runs; the server sets its own threading policy at startup
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        return env


def _cuda_libpath(env_root: str) -> str | None:
    """CUDA lib64, if the rootless toolkit is present.

    llama-server links libcudart.so.12 and libcublas.so.12 but WSL does not put
    them on the default search path, so without this the process dies at load
    time with a bare "not found" and every llama.cpp cell fails identically.
    """
    cand = os.path.join(env_root, "cuda-12.8", "lib64")
    return cand if os.path.isdir(cand) else None


def _ensure_dev_symlinks(cuda_home: str) -> list[str]:
    """Create unversioned `.so` links so `ld -l<name>` resolves.

    NVIDIA's pip wheels ship only versioned libraries (`libcudart.so.13`) with
    no development symlink (`libcudart.so`). FlashInfer's JIT links its kernels
    with plain `c++`/`ld`, which looks for the unversioned name, so without
    these links every JIT step fails with "cannot find -lcudart".
    Idempotent, and scoped to the venv's own toolkit.
    """
    libdir = os.path.join(cuda_home, "lib")
    if not os.path.isdir(libdir):
        return []
    made: list[str] = []
    try:
        for entry in os.listdir(libdir):
            if ".so." not in entry:
                continue
            stem = entry.split(".so.")[0]          # libcudart
            link = os.path.join(libdir, stem + ".so")
            if os.path.exists(link) or os.path.islink(link):
                continue
            # point at the highest versioned file
            versions = sorted(
                f for f in os.listdir(libdir)
                if f.startswith(stem + ".so.")
            )
            if not versions:
                continue
            os.symlink(versions[-1], link)
            made.append(link)
    except OSError:
        pass
    return made


def _vllm_cuda_home(venv_bin: str) -> str | None:
    """CUDA toolkit root that matches vLLM's torch build.

    vLLM's wheels install a complete CUDA 13 toolkit under
    `site-packages/nvidia/cu13`, but nothing puts it on CUDA_HOME. Without it,
    torch.compile/Triton aborts at startup with "Could not find nvcc and
    default cuda_home='/usr/local/cuda' doesn't exist" -- which only happens
    once --enforce-eager is off, so it is easy to misread as a vLLM bug.

    Deliberately uses the venv's own cu13 toolkit rather than the rootless
    cuda-12.8 build: torch here is cu130 and mixing an nvcc major version with
    its runtime is asking for a second failure.
    """
    root = os.path.dirname(venv_bin)  # .../venv-vllm
    sp = os.path.join(root, "lib", f"python{sys.version_info.major}."
                             f"{sys.version_info.minor}", "site-packages")
    cand = os.path.join(sp, "nvidia", "cu13")
    if os.path.exists(os.path.join(cand, "bin", "nvcc")):
        # CUDA 13 layout uses lib/; provide lib64 too for anything expecting it
        if not os.path.exists(os.path.join(cand, "lib64")) and \
           os.path.isdir(os.path.join(cand, "lib")):
            try:
                os.symlink("lib", os.path.join(cand, "lib64"))
            except OSError:
                pass
        _ensure_dev_symlinks(cand)
        return cand
    return None


# --------------------------------------------------------------------------
# llama.cpp
# --------------------------------------------------------------------------

class LlamaCppBackend(Backend):
    name = "llamacpp"

    @property
    def binary(self) -> str:
        return os.path.join(self.env_root, "build-llamacpp", "bin", "llama-server")

    def start(self, model_path: str, quant_label: str,
              n_gpu_layers: int = 999, ctx: int = 16384,
              parallel: int = 8, threads: int | None = None,
              extra_args: list[str] | None = None) -> ServerHandle:
        port = free_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = os.path.join(self.log_dir, f"llamacpp_{quant_label}_{port}.log")

        # llama.cpp divides -c (total context) ACROSS the -np slots: it reports
        # n_ctx_per_seq = ctx / parallel. Passing -c 8192 with -np 8 silently
        # yields 1024 tokens per sequence, which is below the prefill workload's
        # prompt length and makes the server reject those requests with HTTP 400.
        # Callers must therefore size ctx as parallel * per-slot context.
        per_seq = ctx // parallel
        if per_seq < 2048:
            raise ValueError(
                f"ctx={ctx} / parallel={parallel} = {per_seq} tokens per slot; "
                f"too small for the prefill workload")

        cmd = [
            self.binary,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-ngl", str(n_gpu_layers),
            "-c", str(ctx),
            "-np", str(parallel),
            "--seed", "1234",
            "--no-webui",
            "--cont-batching",
        ]
        if threads:
            cmd += ["-t", str(threads), "-tb", str(threads)]
        if extra_args:
            cmd += extra_args

        env = self.base_env()
        cuda_lib = _cuda_libpath(self.env_root)
        if cuda_lib:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [cuda_lib, env.get("LD_LIBRARY_PATH", "")]
            ).strip(os.pathsep)
        lf = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        ok, secs, err = wait_for_health(base_url, timeout=900, proc=proc,
                                        log_path=log_path)

        if not ok and proc.poll() is not None:
            # process died rather than being slow: surface the real reason
            # instead of a bare health-check timeout
            err = f"process exited rc={proc.returncode}; see {log_path}"

        return ServerHandle(
            backend=self.name, model_path=model_path, quant_label=quant_label,
            port=port, base_url=base_url, proc=proc, log_path=log_path,
            startup_s=secs, startup_ok=ok, startup_error="" if ok else err,
            extra={"cmd": cmd, "n_gpu_layers": n_gpu_layers, "ctx": ctx,
                   "parallel": parallel, "log_handle": lf},
        )

    def stop(self, h: ServerHandle) -> None:
        self._kill_tree(h.proc)
        try:
            h.extra.pop("log_handle", None).close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# vLLM
# --------------------------------------------------------------------------

class VLLMBackend(Backend):
    name = "vllm"

    def start(self, model_path: str, quant_label: str,
              max_model_len: int = 8192,
              gpu_memory_utilization: float = 0.80,
              max_num_seqs: int = 128,
              dtype: str = "auto",
              extra_args: list[str] | None = None) -> ServerHandle:
        port = free_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = os.path.join(self.log_dir, f"vllm_{quant_label}_{port}.log")

        venv_bin = os.path.join(self.env_root, "venv-vllm", "bin")
        cmd = [
            os.path.join(venv_bin, "vllm"), "serve", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-num-seqs", str(max_num_seqs),
            "--dtype", dtype,
            "--seed", "1234",
            # NOTE: --enforce-eager is deliberately NOT set. With it, vLLM
            # cannot use CUDA graphs and every decode step pays framework
            # overhead; on small models that dominated the measurement
            # (TPOT ~27 ms vs llama.cpp's ~4 ms) and measured an
            # unconfigured vLLM rather than the one users actually deploy.
            # Graph capture cost is handled by the discarded warmup sweep.
        ]
        if extra_args:
            cmd += extra_args

        env = self.base_env()
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cuda_home = _vllm_cuda_home(venv_bin)
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["CUDA_PATH"] = cuda_home
            env["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + env["PATH"]
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [os.path.join(cuda_home, "lib"),
                 os.path.join(cuda_home, "lib64"),
                 env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)

            # FlashInfer JIT-links its kernels with plain c++/ld, which does
            # not consult LD_LIBRARY_PATH. Without these it fails with
            # "cannot find -lcudart" and "cannot find -lcuda"; libcuda.so is
            # only present in the WSL driver directory.
            lib_dirs = [os.path.join(cuda_home, "lib"),
                        os.path.join(cuda_home, "lib64"),
                        "/usr/lib/wsl/lib"]
            env["LIBRARY_PATH"] = os.pathsep.join(
                [d for d in lib_dirs if os.path.isdir(d)]
                + [env.get("LIBRARY_PATH", "")]
            ).strip(os.pathsep)

            # FlashInfer JIT-compiles its sampling kernel at engine start. On
            # this stack (vLLM 0.31 + flashinfer 0.7.0.post1 + the cu13 wheel)
            # that build fails with
            #   "CUDA compiler and CUDA toolkit headers are incompatible"
            # because flashinfer's vendored CCCL headers disagree with the
            # nvcc the cu13 wheel provides. The engine then dies with a bare
            # "Engine core initialization failed" and no usable cause, which
            # looks like a vLLM bug rather than a packaging mismatch.
            #
            # The sampler is a small part of the measurement surface: it does
            # not change prompt/eval kernel selection, so disabling it keeps
            # TTFT/throughput comparable while making the cell startable.
            # It is recorded as a capability gap rather than silently applied.
            if env.get("VLLM_USE_FLASHINFER_SAMPLER") is None:
                env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
                self._capability_gaps.append(
                    "flashinfer sampler disabled (VLLM_USE_FLASHINFER_SAMPLER=0): "
                    "vendored CCCL headers are incompatible with the cu13 nvcc, "
                    "so its JIT sampling build fails and kills engine startup"
                )

        lf = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        ok, secs, err = wait_for_health(base_url, timeout=1200, proc=proc,
                                        log_path=log_path)

        if not ok and proc.poll() is not None:
            err = f"process exited rc={proc.returncode}; see {log_path}"

        return ServerHandle(
            backend=self.name, model_path=model_path, quant_label=quant_label,
            port=port, base_url=base_url, proc=proc, log_path=log_path,
            startup_s=secs, startup_ok=ok, startup_error="" if ok else err,
            extra={"cmd": cmd, "max_model_len": max_model_len,
                   "gpu_memory_utilization": gpu_memory_utilization,
                   "max_num_seqs": max_num_seqs, "log_handle": lf},
        )

    def stop(self, h: ServerHandle) -> None:
        self._kill_tree(h.proc)
        try:
            h.extra.pop("log_handle", None).close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# SGLang
# --------------------------------------------------------------------------

class SGLangBackend(Backend):
    name = "sglang"

    def start(self, model_path: str, quant_label: str,
              context_length: int = 8192,
              mem_fraction_static: float = 0.80,
              extra_args: list[str] | None = None) -> ServerHandle:
        port = free_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = os.path.join(self.log_dir, f"sglang_{quant_label}_{port}.log")

        venv_bin = os.path.join(self.env_root, "venv-sglang", "bin")
        cmd = [
            os.path.join(venv_bin, "python"), "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--context-length", str(context_length),
            "--mem-fraction-static", str(mem_fraction_static),
            "--dtype", "auto",
            "--random-seed", "1234",
        ]
        if extra_args:
            cmd += extra_args

        env = self.base_env()
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        lf = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        ok, secs, err = wait_for_health(base_url, timeout=1200, proc=proc,
                                        log_path=log_path)

        return ServerHandle(
            backend=self.name, model_path=model_path, quant_label=quant_label,
            port=port, base_url=base_url, proc=proc, log_path=log_path,
            startup_s=secs, startup_ok=ok, startup_error="" if ok else err,
            extra={"cmd": cmd, "log_handle": lf},
        )

    def stop(self, h: ServerHandle) -> None:
        self._kill_tree(h.proc)
        try:
            h.extra.pop("log_handle", None).close()
        except Exception:  # noqa: BLE001
            pass


class VLLMGGUFBackend(VLLMBackend):
    """vLLM serving a GGUF file through its experimental GGUF loader.

    Exists to hold quantization format constant while varying the backend. vLLM
    advertises this as experimental; whatever it costs in throughput is itself a
    result, not a bug to be worked around.
    """

    name = "vllm_gguf"

    def start(self, model_path: str, quant_label: str, **kw) -> ServerHandle:
        extra = list(kw.pop("extra_args", []) or [])
        extra += ["--quantization", "gguf"]
        h = super().start(model_path, quant_label, extra_args=extra, **kw)
        h.backend = self.name
        return h


class OllamaBackend(Backend):
    """Ollama as an OpenAI-compatible endpoint.

    Structurally different from the other backends: there is no server process
    to launch per configuration. Ollama keeps one long-lived daemon and loads a
    model on demand, so the meaningful lifecycle costs are *model load* and
    *model unload* rather than server startup. Those are recorded explicitly,
    because they are exactly what a user pays for with this backend.
    """

    name = "ollama"
    base = "http://127.0.0.1:11434"

    def _tags(self) -> set[str]:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=10)
            return {m["name"] for m in r.json().get("models", [])}
        except Exception:  # noqa: BLE001
            return set()

    def _ensure_pulled(self, tag: str) -> tuple[bool, float, str]:
        t0 = time.perf_counter()
        try:
            if tag in self._tags():
                return True, 0.0, "already present"
            rc = subprocess.run(["ollama", "pull", tag], capture_output=True,
                                text=True, timeout=3600)
            if rc.returncode == 0:
                return True, time.perf_counter() - t0, "pulled"
            return False, time.perf_counter() - t0, rc.stderr[-300:]
        except Exception as exc:  # noqa: BLE001
            return False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"

    def start(self, model_path: str, quant_label: str, **kw) -> ServerHandle:
        tag = model_path  # for ollama the "model path" IS the registry tag
        pulled, pull_s, note = self._ensure_pulled(tag)
        h = ServerHandle(
            backend=self.name, model_path=tag, quant_label=quant_label,
            port=11434, base_url=f"{self.base}/v1", proc=None,
            log_path="", startup_s=pull_s, startup_ok=pulled,
            startup_error="" if pulled else note,
            extra={"tag": tag, "pull_s": pull_s, "pull_note": note},
        )
        if not pulled:
            return h

        # Force weights resident and time it. Ollama defers loading to the first
        # request, so without this the first measured TTFT would silently
        # include a multi-second model load and masquerade as a latency result.
        t0 = time.perf_counter()
        try:
            requests.post(
                f"{self.base}/api/generate",
                json={"model": tag, "prompt": "warmup", "stream": False,
                      "keep_alive": "30m", "options": {"num_predict": 1}},
                timeout=900,
            )
            h.extra["load_s"] = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            h.startup_ok = False
            h.startup_error = f"load failed: {type(exc).__name__}: {exc}"
        return h

    def stop(self, h: ServerHandle) -> None:
        t0 = time.perf_counter()
        try:
            subprocess.run(["ollama", "stop", h.extra.get("tag", "")],
                           capture_output=True, timeout=120)
            h.extra["unload_s"] = time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            pass


BACKENDS = {
    "llamacpp": LlamaCppBackend,
    "vllm": VLLMBackend,
    "vllm_gguf": VLLMGGUFBackend,
    "sglang": SGLangBackend,
    "ollama": OllamaBackend,
}


def tail(path: str, n: int = 40) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:  # noqa: BLE001
        return "(no log)"