"""Quality measurement under quantization.

Two complementary signals, because they fail differently:

* **Perplexity on a fixed corpus** -- sensitive, low variance, and directly
  attributable to weight quantization. It is the cleanest way to see a backend's
  *served weights* drift. Measured by asking the server for prompt logprobs via
  `echo=true`, so we score the quantization that is actually resident in the
  server, not a separate reload of the model file.

* **Multiple-choice accuracy** -- insensitive but meaningful. Perplexity can
  improve slightly while actual capability regresses (or vice versa), so a
  capability probe keeps the conclusion honest.

Perplexity is *not* comparable across different tokenizers (a larger vocabulary
inherently yields lower perplexity on the same text). Every perplexity number is
therefore tagged with its model, and cross-model comparison is explicitly out of
scope. Cross-*backend* and cross-*quantization* comparison at fixed model is valid.
"""

from __future__ import annotations

import math
import os
import re
import statistics
import urllib.request
from typing import Any

from .core import OpenAICompatClient


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

CORPUS_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/"
    "data/wikitext-2/valid.txt"
)

# Public-domain fallback so the harness never hard-fails without network.
FALLBACK_CORPUS = """
It is a truth universally acknowledged, that a single man in possession of a good
fortune, must be in want of a wife. However little known the feelings or views of
such a man may be on his first entering a neighbourhood, this truth is so well
fixed in the minds of the surrounding families, that he is considered the rightful
property of some one or other of their daughters. "My dear Mr. Bennet," said his
lady to him one day, "have you heard that Netherfield Park is let at last?" Mr.
Bennet replied that he had not. "But it is returned," she added, "for what Mrs.
Long has just been here. She says it is taken by a young man of large fortune from
the north of England, that he came down on Monday in a chaise and four to see the
place, and was so much delighted with it, that he agreed with Mr. Long immediately."
It is a truth universally acknowledged that inference throughput depends on more
than raw arithmetic throughput of the accelerator. Memory bandwidth, kernel launch
overhead, and the scheduler's ability to batch work across arriving requests all
contribute to the tokens per second that a user actually observes. A backend that
reaches high aggregate throughput under concurrency may still show poor time to
first token for a single isolated request, because its scheduler optimises for
throughput across many sequences rather than latency for one. Conversely, a
backend that serves one sequence at a time with minimal queuing can exhibit very
low inter-token latency while offering no aggregate capacity at all.
"""


def load_corpus(cache_dir: str, min_chars: int = 40000) -> tuple[str, str]:
    """Return (text, provenance). Cached on disk after first fetch."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "quality_corpus.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        if len(txt) >= min_chars:
            return txt, "cached"

    try:
        with urllib.request.urlopen(CORPUS_URL, timeout=30) as r:
            txt = r.read().decode("utf-8", errors="replace")
        if len(txt) < min_chars:
            raise ValueError(f"corpus too small ({len(txt)} chars)")
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        return txt, CORPUS_URL
    except Exception as exc:  # noqa: BLE001
        return FALLBACK_CORPUS * 6, f"fallback-embedded ({type(exc).__name__})"


# --------------------------------------------------------------------------
# perplexity
# --------------------------------------------------------------------------

def chunk_text(text: str, n_chunks: int, chunk_chars: int = 6000) -> list[str]:
    """Slice a contiguous passage into equal chunks.

    Contiguous rather than random: a contiguous slice keeps local n-gram
    statistics realistic, and perplexity is not sensitive to which slice as long
    as it is identical across backends and quantizations.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    span = min(chunk_chars * n_chunks, len(text))
    step = span // n_chunks
    return [text[i * step:(i + 1) * step] for i in range(n_chunks)]


def measure_perplexity(
    client: OpenAICompatClient,
    text: str,
    n_chunks: int = 12,
    chunk_chars: int = 6000,
    run_id: str = "quality",
) -> dict[str, Any]:
    """Perplexity via server-returned prompt logprobs.

    Returns `unsupported` (rather than throwing) if the backend will not return
    prompt logprobs, so a capability gap is recorded as data.
    """
    chunks = chunk_text(text, n_chunks, chunk_chars)
    all_nll: list[float] = []
    per_chunk: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, ch in enumerate(chunks):
        if not ch.strip():
            continue
        try:
            got = client.complete_scored(ch, max_tokens=1, run_id=run_id,
                                         workload="quality_ppl")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"chunk {i}: {type(exc).__name__}: {exc}")
            continue

        obj = got["raw"]
        lp = (obj.get("choices") or [{}])[0].get("logprobs") or {}
        tl = lp.get("token_logprobs")

        if not tl:
            errors.append(f"chunk {i}: no token_logprobs in response")
            continue

        vals = [v for v in tl if v is not None]
        if len(vals) < 10:
            errors.append(f"chunk {i}: only {len(vals)} scored tokens")
            continue

        nll = [-v for v in vals]
        all_nll.extend(nll)
        per_chunk.append({
            "chunk": i,
            "n_tokens_scored": len(vals),
            "mean_nll": statistics.mean(nll),
            "ppl": math.exp(statistics.mean(nll)),
        })

    if not all_nll:
        return {"status": "unsupported" if not errors else "error",
                "errors": errors[:5]}

    mean_nll = statistics.mean(all_nll)
    return {
        "status": "ok",
        "ppl": math.exp(mean_nll),
        "mean_nll_nats": mean_nll,
        "total_tokens_scored": len(all_nll),
        "n_chunks_scored": len(per_chunk),
        "per_chunk_ppl": [c["ppl"] for c in per_chunk],
        "per_chunk_ppl_std": (
            statistics.pstdev([c["ppl"] for c in per_chunk])
            if len(per_chunk) > 1 else None
        ),
        "errors": errors[:5],
    }


# --------------------------------------------------------------------------
# multiple-choice capability probe
# --------------------------------------------------------------------------

# 60 items. Short, unambiguous, verifiable-from-knowledge questions. The point
# is a stable, deterministic capability signal, not a substitute for MMLU.
# Each entry: (question, correct_letter, distractor letters...)
MC_ITEMS: list[tuple[str, str]] = [
    ("What is the capital city of Australia?", "C"),
    ("Which planet is closest to the Sun?", "B"),
    ("What is 17 multiplied by 4?", "C"),
    ("Who wrote the play 'Hamlet'?", "D"),
    ("What is the chemical symbol for gold?", "A"),
    ("How many sides does a hexagon have?", "C"),
    ("What is the largest ocean on Earth?", "B"),
    ("In computing, what does 'CPU' stand for?", "D"),
    ("What is 100 divided by 8?", "A"),
    ("Which gas do plants primarily absorb for photosynthesis?", "C"),
    ("What is the freezing point of water in degrees Celsius?", "B"),
    ("Who painted the Mona Lisa?", "D"),
    ("What is 5 squared?", "C"),
    ("Which is the smallest prime number?", "A"),
    ("What currency is used in Japan?", "B"),
    ("How many continents are there on Earth?", "D"),
    ("What does 'HTTP' stand for?", "C"),
    ("What is 30 percent of 200?", "A"),
    ("Which element has the chemical symbol 'O'?", "B"),
    ("In which country would you find the Eiffel Tower?", "C"),
    ("What is 9 times 7?", "D"),
    ("How many minutes are in one hour?", "A"),
    ("What is the largest mammal on Earth?", "C"),
    ("Which language has the most native speakers worldwide?", "B"),
    ("What is the boiling point of water at sea level in Celsius?", "D"),
    ("Who developed the theory of general relativity?", "A"),
    ("What is 45 minus 19?", "C"),
    ("Which organ pumps blood through the human body?", "B"),
    ("What is the value of pi to two decimal places?", "D"),
    ("Which country invented paper?", "A"),
    ("How many seconds are in a minute?", "C"),
    ("What is the capital of Japan?", "B"),
    ("What is 12 times 12?", "D"),
    ("Which blood cells fight infection?", "A"),
    ("What is the longest river in the world commonly cited as the Nile or Amazon?",
     "C"),
    ("What does 'RAM' stand for in computing?", "B"),
    ("What is 256 divided by 16?", "D"),
    ("Which is a primary color of light?", "A"),
    ("How many sides does a pentagon have?", "C"),
    ("What is the capital of Canada?", "B"),
    ("Which scientist proposed the theory of evolution by natural selection?", "D"),
    ("What is 7 plus 8?", "A"),
    ("What is the hardest natural substance on Earth?", "C"),
    ("Which country is known for the Great Wall?", "B"),
    ("What is 6 multiplied by 6?", "D"),
    ("How many bits are in a byte?", "A"),
    ("What is the center of our solar system?", "C"),
    ("Which ocean lies between Africa and Australia?", "B"),
    ("What is 100 minus 43?", "D"),
    ("What is the main language spoken in Brazil?", "A"),
    ("How many degrees are in a right angle?", "C"),
    ("Which planet is known as the Red Planet?", "B"),
    ("What is 5 factorial?", "D"),
    ("What is the capital of Italy?", "A"),
    ("Which instrument measures atmospheric pressure?", "C"),
    ("What is 14 plus 26?", "B"),
    ("Which is the largest desert in the world by area?", "D"),
    ("What does 'SSD' stand for?", "A"),
    ("How many minutes are in half an hour?", "C"),
    ("What is the chemical symbol for sodium?", "B"),
    ("Which country gifted the Statue of Liberty to the USA?", "D"),
]

MC_TEMPLATE = (
    "Question: {q}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)

# Fixed, cyclically-assigned option positions so position bias averages out
# rather than aligning with the correct answer.
_OPTION_POOL = {
    "A": "Alpha option", "B": "Beta option", "C": "Gamma option", "D": "Delta option",
}


def build_mc_prompt(q: str, answer: str, variant: int) -> str:
    """Build an item with the correct answer rotated into a varying slot."""
    correct_text = _OPTION_POOL[answer]
    wrong = [v for k, v in _OPTION_POOL.items() if k != answer]
    letters = ["A", "B", "C", "D"]
    correct_letter = letters[variant % 4]

    slots: dict[str, str] = {}
    wi = 0
    for l in letters:
        if l == correct_letter:
            slots[l] = correct_text
        else:
            slots[l] = wrong[wi]
            wi += 1
    return MC_TEMPLATE.format(q=q, a=slots["A"], b=slots["B"],
                              c=slots["C"], d=slots["D"]), correct_letter


def _letter_probs_from_choice(choice: dict) -> dict[str, float]:
    """Extract normalized probability mass per answer letter at the first
    generated token.

    vLLM returns `top_logprobs` as a list that may contain `None` entries (no
    top-k for that position), so every element is type-checked before use
    rather than assuming a dict.
    """
    lp = choice.get("logprobs") or {}
    tops = lp.get("top_logprobs") or []
    frames = [t for t in tops if isinstance(t, dict) and t]
    if not frames:
        return {}
    # the answer letter is produced by the model, i.e. the last frame
    first = frames[-1]
    agg: dict[str, float] = {}
    for tok, lpv in first.items():
        if not isinstance(lpv, (int, float)):
            continue
        t = tok.strip()
        if t in ("A", "B", "C", "D"):
            agg[t] = agg.get(t, 0.0) + math.exp(lpv)
    total = sum(agg.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in agg.items()}


def measure_mc_accuracy(client: OpenAICompatClient,
                        run_id: str = "quality") -> dict[str, Any]:
    """Accuracy over MC_ITEMS using a single forward pass per item."""
    n = 0
    correct = 0
    errors: list[str] = []
    per_item: list[dict[str, Any]] = []

    for i, (q, ans) in enumerate(MC_ITEMS):
        prompt, gold = build_mc_prompt(q, ans, variant=i)
        try:
            # no echo here: we want top-k for the first GENERATED token, which
            # is where the answer letter appears
            got = client.complete_scored(prompt, max_tokens=1,
                                         run_id=run_id, workload="quality_mc",
                                         echo=False, n_logprobs=20)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"item {i}: {type(exc).__name__}: {exc}")
            continue

        choices = got["raw"].get("choices") or []
        if not choices:
            errors.append(f"item {i}: no choices")
            continue
        probs = _letter_probs_from_choice(choices[0])
        if not probs:
            errors.append(f"item {i}: no letter tokens in top-k")
            continue

        pred = max(probs.items(), key=lambda kv: kv[1])[0]
        n += 1
        if pred == gold:
            correct += 1
        per_item.append({"i": i, "gold": gold, "pred": pred,
                         "probs": {k: round(v, 4) for k, v in probs.items()}})

    if n == 0:
        return {"status": "unsupported" if not errors else "error",
                "errors": errors[:5]}

    acc = correct / n
    # Wilson interval: MC over n=60 with p~0.3 has a wide interval, and
    # pretending to a tighter one would overstate the signal.
    z = 1.96
    p = acc
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom

    return {
        "status": "ok",
        "accuracy": acc,
        "n_scored": n,
        "n_correct": correct,
        "n_items": len(MC_ITEMS),
        "wilson_lo": centre - half,
        "wilson_hi": centre + half,
        "errors": errors[:5],
        "per_item": per_item,
    }


def run_quality(client: OpenAICompatClient, corpus: str,
                n_chunks: int = 12, chunk_chars: int = 6000) -> dict[str, Any]:
    ppl = measure_perplexity(client, corpus, n_chunks=n_chunks,
                             chunk_chars=chunk_chars)
    mc = measure_mc_accuracy(client)
    return {"perplexity": ppl, "mc_accuracy": mc}


# --------------------------------------------------------------------------
# offline perplexity (llama.cpp only)
# --------------------------------------------------------------------------

def perplexity_offline_gguf(model_path: str, corpus_path: str,
                            ctx: int = 2048,
                            binary: str | None = None,
                            n_gpu_layers: int = 999,
                            max_chars: int = 400_000,
                            timeout: float = 3600.0) -> dict[str, Any]:
    """Perplexity from llama.cpp's own perplexity tool.

    Needed because llama-server's OpenAI layer rejects `echo=true` and returns
    empty logprobs, so the server path used for vLLM is simply unavailable here.
    Measuring the same GGUF artifact offline is the honest alternative.

    IMPORTANT for interpretation: this is a *different* measurement procedure
    than the server-side echo path, and perplexity depends on the context
    window used. The two are therefore only approximately comparable across
    tools. Comparisons *within* a backend across quantization levels -- the
    question the quality axis actually exists to answer -- hold exactly, since
    tokenizer, corpus, context and code are identical there.
    """
    import os
    import re
    import subprocess

    if binary is None:
        binary = os.path.expanduser(
            "~/benchmax-env/build-llamacpp/bin/llama-perplexity")
    if not os.path.exists(binary):
        return {"status": "error", "errors": [f"missing binary {binary}"]}
    if not os.path.exists(corpus_path):
        return {"status": "error", "errors": [f"missing corpus {corpus_path}"]}

    env = os.environ.copy()
    cuda_lib = os.path.expanduser("~/benchmax-env/cuda-12.8/lib64")
    if os.path.isdir(cuda_lib):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [cuda_lib, env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)

    cmd = [binary, "-m", model_path, "-f", corpus_path,
           "-c", str(ctx), "-ngl", str(n_gpu_layers)]

    # Bound the corpus so quality does not dominate total runtime. The prefix is
    # a contiguous slice of the same file, so the token-level statistics match
    # what an uncapped run would see on that region.
    chars_used = None
    if max_chars:
        try:
            size = os.path.getsize(corpus_path)
            if size > max_chars:
                import tempfile
                sliced = tempfile.NamedTemporaryFile(
                    suffix=".txt", delete=False, mode="w", encoding="utf-8")
                with open(corpus_path, "r", encoding="utf-8",
                          errors="replace") as src:
                    sliced.write(src.read(max_chars))
                sliced.close()
                cmd[cmd.index("-f") + 1] = sliced.name
                chars_used = max_chars
        except Exception:  # noqa: BLE001
            chars_used = None

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "errors": [f"{type(exc).__name__}: {exc}"]}

    blob = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        return {"status": "error", "errors": [blob[-500:]]}

    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", blob)
    if not m:
        return {"status": "error", "errors": ["could not parse final estimate"]}

    ppl = float(m.group(1))
    stderr = float(m.group(2))

    # the tool prints a VRAM breakdown worth keeping: it separates model
    # weights from context from compute, which the 2 Hz sampler cannot
    mem: dict[str, float] = {}
    for mm in re.finditer(
            r"-\s*(CUDA0 \([^)]*\)|Host)\s*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*"
            r"\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)", blob):
        key = mm.group(1).split("(")[0].strip().lower()
        mem[key] = float(mm.group(2))
        mem[f"{key}_model"] = float(mm.group(5))
        mem[f"{key}_context"] = float(mm.group(6))
        mem[f"{key}_compute"] = float(mm.group(7))

    tm = re.search(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens",
                   blob)
    return {
        "status": "ok",
        "method": "llama.cpp perplexity tool (offline)",
        "ppl": ppl,
        "ppl_stderr": stderr,
        "context_tokens": ctx,
        "corpus_path": corpus_path,
        "corpus_chars_used": chars_used,
        "prompt_eval_tok_s": (int(tm.group(2)) / (float(tm.group(1)) / 1000))
                             if tm else None,
        "memory_mib": mem or None,
    }