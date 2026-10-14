"""benchmax: controlled LLM serving-backend benchmark harness.

Design goals
------------
1. Every backend is exercised through its OpenAI-compatible HTTP surface, so
   the *serving stack* is measured, not a private Python API. That is what a
   user actually experiences.
2. Latency is decomposed into TTFT (queue + prefill + first decode step) and
   per-token ITL, because those two numbers respond to completely different
   architectural choices (batching vs. kernel efficiency).
3. Raw per-run records are written to disk as JSON *before* any aggregation, so
   numbers can be recomputed or audited later.
4. Nothing is smoothed or averaged away at capture time.
"""

__version__ = "1.0.0"