"""Model resolution and download.

Files are located by querying the Hugging Face API for the actual repo
contents rather than hardcoding filenames, because quant repos get re-uploaded
with sharded filenames and any hardcoded guess silently breaks.

Split GGUF shards (`...-00001-of-00002.gguf`) are downloaded whole and the
first shard is returned; llama.cpp picks up sibling shards automatically.
"""

from __future__ import annotations

import os
import re
from typing import Any

from . import config as C


# GGUF filename suffixes per logical level, in preference order. Matched
# case-insensitively because quant repos are inconsistent about casing
# (`Q4_K_M.gguf` vs `Q4_K_M-00001-of-00002.gguf`).
GGUF_SUFFIX: dict[str, list[str]] = {
    "4bit": ["Q4_K_M.gguf", "Q4_K_S.gguf"],
    "8bit": ["Q8_0.gguf"],
    "int8": ["Q8_0.gguf"],
    "16bit": ["F16.gguf", "BF16.gguf", "f16.gguf"],
}

# backends that consume GGUF rather than safetensors
GGUF_BACKENDS = {"llamacpp", "vllm_gguf"}


def _hf():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    # hf_transfer does parallel chunked range requests. The default single
    # stream sustained well under 1 MB/s here, which is hours over the full
    # matrix; if the package is missing we fall back silently rather than fail.
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except Exception:  # noqa: BLE001
        pass
    return HfApi(), hf_hub_download, snapshot_download


# Match the quantization scheme from the resolved filename so the reported
# label can never drift from the artifact. This caught a real bug: GGUF repos
# ship BF16 rather than F16, and "BF16.gguf".endswith("F16.gguf") is true, so a
# naive suffix preference silently served BF16 while the report claimed F16.
_QUANT_TOKENS = [
    ("q4_k_m", "Q4_K_M"), ("q4_k_s", "Q4_K_S"), ("q4_0", "Q4_0"),
    ("q5_k_m", "Q5_K_M"), ("q5_k_s", "Q5_K_S"),
    ("q6_k", "Q6_K"), ("q8_0", "Q8_0"), ("q8_k", "Q8_K"),
    ("iq4_nl", "IQ4_NL"), ("iq4_xs", "IQ4_XS"),
    ("f32", "F32"), ("bf16", "BF16"), ("f16", "F16"),
]


def quant_from_filename(filename: str) -> str | None:
    low = filename.lower()
    for token, label in _QUANT_TOKENS:
        if token in low:
            return label
    return None


def _pick_gguf(files: list[str], level: str) -> str | None:
    """Choose the best GGUF file for a logical quant level."""
    candidates = [f for f in files if f.lower().endswith(".gguf")]

    # preference order: exact suffix, then family match, then anything
    prefs = GGUF_SUFFIX.get(level, [])
    for pref in prefs:
        exact = [f for f in candidates if f.lower().endswith(pref.lower())]
        if exact:
            return sorted(exact)[0]

    # fall back to matching the quant family token
    fam = {"4bit": ["q4_k_m", "q4_k_s", "q4_0"],
           "int8": ["q8_0"],
           "8bit": ["q8_0"],
           "16bit": ["f16", "bf16"]}[level]
    for token in fam:
        hits = [f for f in candidates if token in f.lower()]
        if hits:
            # prefer the single-file variant over shards
            single = [f for f in hits if "-00001-of-" not in f]
            return sorted(single or hits)[0]
    return None


def _shard_group(filename: str, files: list[str]) -> list[str]:
    """All shards belonging to a split quant file."""
    m = re.match(r"^(.*)-00001-of-(\d+)\.gguf$", filename)
    if not m:
        return [filename]
    stem, total = m.group(1), int(m.group(2))
    out = []
    for i in range(1, total + 1):
        cand = f"{stem}-{i:05d}-of-{total:05d}.gguf"
        if cand in files:
            out.append(cand)
    return out


def resolve_cell_path(cell: C.Cell, models_dir: str) -> str:
    """Return a local path to the model artifact for this cell, downloading if needed."""
    api, hf_hub_download, snapshot_download = _hf()
    os.makedirs(models_dir, exist_ok=True)

    if not cell.source_repo:
        raise ValueError("no source repo resolved for this cell")

    repo = cell.source_repo
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # --- GGUF (llamacpp, vllm_gguf) ---------------------------------------
    if cell.backend in GGUF_BACKENDS:
        files = api.list_repo_files(repo, token=token)
        chosen = _pick_gguf(files, cell.quant_level)
        if chosen is None:
            raise FileNotFoundError(
                f"no GGUF matching level={cell.quant_level} in {repo} "
                f"(available: {sorted({f for f in files if f.endswith('.gguf')})[:12]})")
        shards = _shard_group(chosen, files)
        local: list[str] = []
        for s in shards:
            p = hf_hub_download(repo_id=repo, filename=s,
                                local_dir=os.path.join(models_dir, repo.replace("/", "__")),
                                token=token)
            local.append(p)
        cell.source_file = chosen
        actual = quant_from_filename(chosen)
        if actual:
            cell.quant_scheme = f"{actual} (GGUF)"
        return local[0]

    # --- safetensors repos (vLLM 16-bit, SGLang 16-bit) -------------------
    if not cell.source_file:
        return snapshot_download(
            repo_id=repo,
            local_dir=os.path.join(models_dir, repo.replace("/", "__")),
            token=token,
            # quant configs are small; avoid pulling multi-GB variants
            ignore_patterns=["*.pth", "*.msgpack", "*.h5", "*consolidated*",
                             "original/*", "*.gguf"],
        )

    # --- single-file repos (AWQ etc.) ------------------------------------
    files = api.list_repo_files(repo, token=token)
    want = [f for f in files if f.endswith(".safetensors")]
    if not want:
        raise FileNotFoundError(f"no safetensors in {repo}: {files[:20]}")
    p = hf_hub_download(repo_id=repo, filename=sorted(want)[0],
                        local_dir=os.path.join(models_dir, repo.replace("/", "__")),
                        token=token)
    cell.source_file = sorted(want)[0]
    return p


def preflight(models: list[C.ModelSpec], backends: list[str], device: str = "cuda") -> dict[str, Any]:
    """Check which matrix cells have downloadable artifacts, without downloading."""
    api, _, _ = _hf()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    cells = C.build_matrix(models, backends, device=device)

    report = []
    for c in cells:
        entry: dict[str, Any] = {
            "backend": c.backend, "model": c.model_key, "quant": c.quant_level,
            "scheme": c.quant_scheme, "repo": c.source_repo, "available": None,
            "note": "",
        }
        if c.status == "skipped":
            entry["available"] = False
            entry["note"] = c.skip_reason
        elif not c.source_repo:
            entry["available"] = False
            entry["note"] = "no source repo"
        else:
            try:
                files = api.list_repo_files(c.source_repo, token=token)
                if c.backend in GGUF_BACKENDS:
                    entry["available"] = _pick_gguf(files, c.quant_level) is not None
                    entry["note"] = "" if entry["available"] else "no matching GGUF"
                else:
                    entry["available"] = any(f.endswith(".safetensors") for f in files)
                    entry["note"] = "" if entry["available"] else "no safetensors"
            except Exception as exc:  # noqa: BLE001
                entry["available"] = False
                entry["note"] = f"{type(exc).__name__}: {exc}"
        report.append(entry)
    return {"cells": report}