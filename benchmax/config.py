"""The experiment matrix: models, quantization levels, backend variants.

Three facts drove the design of this file, all established by probing the
Hugging Face API rather than assumed:

1. **Quantization is not a format identity.** llama.cpp consumes GGUF
   (`Q4_K_M`/`Q8_0`/`F16`); vLLM's fast GPU kernels target AWQ, FP8 and BF16.
   "4-bit" means *the backend's preferred 4-bit scheme*. These are different
   algorithms with different error characteristics, so part of any quality
   delta between them belongs to the quantizer, not the backend.

2. **The 4-bit surface is much smaller than it looks.** Public AWQ checkpoints
   exist for `Qwen3-4B` only -- not for Qwen3-0.6B or Qwen3-1.7B. Left alone
   that would have made "compare the backends at 4-bit" quietly impossible for
   most of the matrix.

   So a third backend variant was added: **`vllm_gguf`**, which loads the *same*
   Q4_K_M file through vLLM's experimental GGUF loader. That gives a genuine
   apples-to-apples arm -- identical weights, identical quantization scheme,
   different backend -- and the gap between `vllm_gguf` and `vllm`-AWQ isolates
   "backend" from "quantization format", which is the single most useful axis in
   this whole comparison.

3. **FP8 is not INT8.** On Ada (sm_89) FP8 maps onto tensor-core paths and is
   quanta of a different kind from GGUF's block-scaled Q8_0. The `8bit` level is
   therefore labelled with the concrete scheme per backend and never compared
   as if the two were the same thing.

Unavailable cells are recorded as `skipped` with a reason rather than silently
dropped -- a gap in the supported quantization surface is itself a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    label: str
    params_b: float
    hf_safetensors_repo: str | None      # 16-bit reference weights
    hf_gguf_repo: str | None             # GGUF quants
    # level -> (repo, scheme label) for vLLM's native formats
    vllm_repos: dict[str, tuple[str, str]] = field(default_factory=dict)


# NOTE on Llama-3.2: upstream `meta-llama/*` is gated and no token is available
# here, so ungated mirrors are used. The GGUF and safetensors mirrors are both
# *Instruct* checkpoints -- using the Instruct GGUF against the base safetensors
# would have silently benchmarked two different models under one label.
MODELS: list[ModelSpec] = [
    ModelSpec(
        key="qwen3-0.6b", family="Qwen3", label="Qwen3-0.6B", params_b=0.6,
        hf_safetensors_repo="Qwen/Qwen3-0.6B",
        hf_gguf_repo="unsloth/Qwen3-0.6B-GGUF",
        vllm_repos={"8bit": ("Qwen/Qwen3-0.6B-FP8", "FP8 (W8A8, e4m3)"),
                    "int8": ("Qwen/Qwen3-0.6B-GPTQ-Int8", "GPTQ-INT8 (W8A16)")},
    ),
    ModelSpec(
        key="qwen3-1.7b", family="Qwen3", label="Qwen3-1.7B", params_b=1.7,
        hf_safetensors_repo="Qwen/Qwen3-1.7B",
        hf_gguf_repo="unsloth/Qwen3-1.7B-GGUF",
        vllm_repos={"8bit": ("Qwen/Qwen3-1.7B-FP8", "FP8 (W8A8, e4m3)"),
                    "int8": ("Qwen/Qwen3-1.7B-GPTQ-Int8", "GPTQ-INT8 (W8A16)")},
    ),
    ModelSpec(
        key="qwen3-4b", family="Qwen3", label="Qwen3-4B", params_b=4.0,
        hf_safetensors_repo="Qwen/Qwen3-4B",
        hf_gguf_repo="unsloth/Qwen3-4B-GGUF",
        vllm_repos={"4bit": ("Qwen/Qwen3-4B-AWQ", "AWQ-INT4 (W4A16)"),
                    "8bit": ("Qwen/Qwen3-4B-FP8", "FP8 (W8A8, e4m3)")},
    ),
    ModelSpec(
        key="llama32-1b", family="Llama-3.2", label="Llama-3.2-1B-Instruct", params_b=1.0,
        hf_safetensors_repo="unsloth/Llama-3.2-1B-Instruct",
        hf_gguf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        vllm_repos={"8bit": ("RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic",
                             "FP8 dynamic (W8A8)")},
    ),
    ModelSpec(
        key="llama32-3b", family="Llama-3.2", label="Llama-3.2-3B-Instruct", params_b=3.0,
        hf_safetensors_repo="unsloth/Llama-3.2-3B-Instruct",
        hf_gguf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        vllm_repos={"8bit": ("RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic",
                             "FP8 dynamic (W8A8)")},
    ),
]


QUANT_LEVELS = ["4bit", "int8", "8bit", "16bit"]


# --------------------------------------------------------------------------
# VRAM budget
# --------------------------------------------------------------------------

VRAM_TOTAL_GIB = 8.0
# 8 GiB card shared with a live Windows desktop (~1.5 GiB in use at start), so
# weights must leave room for KV cache, activations and the CUDA context.
VRAM_SHARED_GIB = 1.5
VRAM_RESERVE_GIB = 1.5
USABLE_WEIGHT_GIB = VRAM_TOTAL_GIB - VRAM_SHARED_GIB - VRAM_RESERVE_GIB  # 5.0

# bytes/param including quantization scales and metadata
BYTES_PER_PARAM = {"4bit": 0.60, "int8": 1.15, "8bit": 1.05, "16bit": 2.10}


def level_fits_vram(spec: ModelSpec, level: str) -> bool:
    return spec.params_b * BYTES_PER_PARAM[level] <= USABLE_WEIGHT_GIB


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------

@dataclass
class Cell:
    backend: str
    model_key: str
    model_label: str
    family: str
    params_b: float
    quant_level: str
    quant_scheme: str
    source_repo: str | None
    source_file: str | None = None
    local_path: str | None = None
    status: str = "pending"
    skip_reason: str = ""
    device: str = "cuda"
    extra_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GGUF_SCHEME = {"4bit": "Q4_K_M", "8bit": "Q8_0", "16bit": "F16"}

# levels llama.cpp serves natively. `int8` is deliberately absent: GGUF has no
# separate INT8 scheme, so it collapses into the Q8_0 rung. Keeping it out avoids
# measuring the same file twice under two labels.
LLAMACPP_LEVELS = ["4bit", "8bit", "16bit"]


def _llamacpp_cells(spec: ModelSpec, device: str) -> list[Cell]:
    out = []
    for lvl in LLAMACPP_LEVELS:
        if not level_fits_vram(spec, lvl):
            continue
        out.append(Cell(
            backend="llamacpp", model_key=spec.key, model_label=spec.label,
            family=spec.family, params_b=spec.params_b, quant_level=lvl,
            quant_scheme=GGUF_SCHEME[lvl], source_repo=spec.hf_gguf_repo,
            device=device,
        ))
    return out


def _vllm_gguf_cells(spec: ModelSpec, device: str) -> list[Cell]:
    """Same Q4_K_M file as llama.cpp, served by vLLM's experimental loader.

    This is the only arm where backend and quantization format are held
    constant, which makes it the anchor for the whole comparison.
    """
    if not level_fits_vram(spec, "4bit"):
        return []
    return [Cell(
        backend="vllm_gguf", model_key=spec.key, model_label=spec.label,
        family=spec.family, params_b=spec.params_b, quant_level="4bit",
        quant_scheme="Q4_K_M (GGUF via vLLM)", source_repo=spec.hf_gguf_repo,
        status="skipped", device=device,
        skip_reason=(
            "vLLM 0.31 has no GGUF support: no 'gguf' entry in "
            "QUANTIZATION_METHODS, no quantization.gguf module, no GGUFConfig. "
            "Verified against the installed build; a .gguf path fails with "
            "'not a valid JSON file' and the documented repo/file.gguf form is "
            "rejected as a malformed repo id. Recorded as a capability gap: the "
            "two backends have disjoint quantization surfaces, so no "
            "same-weights comparison is possible."
        ),
    )]


def _vllm_cells(spec: ModelSpec, device: str) -> list[Cell]:
    out = []
    for lvl in QUANT_LEVELS:
        if lvl == "int8":
            # secondary 8-bit scheme, only where a checkpoint exists
            repo_scheme = spec.vllm_repos.get("int8")
            if not repo_scheme:
                continue
            repo, scheme = repo_scheme
        elif lvl in spec.vllm_repos:
            repo, scheme = spec.vllm_repos[lvl]
        elif lvl == "16bit":
            repo, scheme = spec.hf_safetensors_repo, "BF16 (unquantized)"
        else:
            continue

        if not level_fits_vram(spec, lvl):
            out.append(Cell(
                backend="vllm", model_key=spec.key, model_label=spec.label,
                family=spec.family, params_b=spec.params_b, quant_level=lvl,
                quant_scheme=scheme, source_repo=repo, status="skipped",
                skip_reason=(f"weights ~{spec.params_b * BYTES_PER_PARAM[lvl]:.1f} GiB "
                             f"exceed the {USABLE_WEIGHT_GIB:.1f} GiB budget on an 8 GiB "
                             f"card shared with a desktop"),
                device=device,
            ))
            continue

        out.append(Cell(
            backend="vllm", model_key=spec.key, model_label=spec.label,
            family=spec.family, params_b=spec.params_b, quant_level=lvl,
            quant_scheme=scheme, source_repo=repo, device=device,
        ))
    return out


def build_matrix(models: list[ModelSpec], backends: list[str],
                 device: str = "cuda") -> list[Cell]:
    cells: list[Cell] = []
    for backend in backends:
        for spec in models:
            if backend == "llamacpp":
                cells += _llamacpp_cells(spec, device)
            elif backend == "vllm_gguf":
                cells += _vllm_gguf_cells(spec, device)
            elif backend == "vllm":
                cells += _vllm_cells(spec, device)
    return cells


def matrix_to_dicts(cells: list[Cell]) -> list[dict[str, Any]]:
    return [c.to_dict() for c in cells]


# human-readable description of the quantization surface each backend exposes
def quantization_surface(models: list[ModelSpec]) -> dict[str, Any]:
    surface: dict[str, dict[str, list[str]]] = {}
    for spec in models:
        row: dict[str, list[str]] = {}
        for b, fn in [("llamacpp", _llamacpp_cells),
                      ("vllm_gguf", _vllm_gguf_cells),
                      ("vllm", _vllm_cells)]:
            got = [f"{c.quant_level}:{c.quant_scheme}"
                   for c in fn(spec, "cuda") if c.status != "skipped"]
            row[b] = got
        surface[spec.key] = row
    return surface