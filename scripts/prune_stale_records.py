#!/usr/bin/env python
"""Remove stale records from runs.jsonl, keeping a full audit trail.

During setup three classes of record were written that describe *harness
configuration* problems rather than backend behaviour:

  1. startup_failure for vLLM caused by "Could not find nvcc" / FlashInfer
     header mismatch / missing -lcudart   (all fixed before any real run)
  2. startup_failure for the vllm_gguf arm (vLLM has no GGUF support at all)
  3. quality_error from the MC probe crashing on vLLM's None top-logprobs

Classes 1 and 3 are recoverable: those cells were subsequently measured
successfully, or can be re-measured. Class 2 is a genuine capability gap and is
preserved.

Nothing is deleted silently -- removed lines are archived to
runs.jsonl.superseded.jsonl with the reason attached.
"""
import json
import os
import shutil
import sys

SRC = "results/raw/runs.jsonl"
ARCHIVE = "results/raw/runs.jsonl.superseded.jsonl"

VLLM_FIX_MARKERS = (
    "Could not find nvcc",
    "cuda_home",
    "Ninja build failed",
    "cannot find -lcudart",
    "cannot find -lcuda",
    "CUDA compiler and CUDA toolkit headers",
)
GGUF_MARKERS = (
    "not a valid JSON",
    "Repo id must be in the form",
    "GGUFConfig",
    "quantization.gguf",
)
MC_MARKER = "'NoneType' object has no attribute 'items'"

if not os.path.exists(SRC):
    sys.exit(f"missing {SRC}")

kept, removed = [], []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        kind = rec.get("kind")
        blob = json.dumps(rec)
        reason = None
        if kind == "startup_failure":
            if any(m in blob for m in GGUF_MARKERS):
                reason = "vllm_gguf: vLLM has no GGUF support (genuine gap)"
            elif any(m in blob for m in VLLM_FIX_MARKERS):
                reason = "vLLM startup failure from harness env, since fixed"
        elif kind == "quality_error" and MC_MARKER in blob:
            reason = "MC probe crash on None top-logprobs, since fixed"
        elif kind == "quality_error":
            reason = "quality probe error, re-run via `quality` command"

        if reason:
            removed.append({"removed_because": reason, "record": rec})
        else:
            kept.append(line)

with open(SRC, "w", encoding="utf-8") as f:
    for line in kept:
        f.write(line + "\n")

with open(ARCHIVE, "w", encoding="utf-8") as f:
    for entry in removed:
        f.write(json.dumps(entry) + "\n")

from collections import Counter
counts = Counter(e["removed_because"] for e in removed)
print(f"kept {len(kept)} records, moved {len(removed)} to {ARCHIVE}")
for reason, n in counts.most_common():
    print(f"  {n:4}  {reason}")