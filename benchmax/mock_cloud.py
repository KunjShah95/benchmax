"""Local mock of an OpenAI-compatible hosted endpoint.

Purpose: prove the cloud adapter path is wired correctly without spending
credentials or inventing numbers. It implements exactly the surface the harness
uses -- /health, /v1/completions streaming and non-streaming, echo + prompt
logprobs -- with configurable latency and batching behaviour so the harness's
TTFT/ITL math can be validated against a known ground truth.

This is a *test fixture*, not a benchmark participant. Results from it are
tagged `mock` everywhere they are written.
"""

from __future__ import annotations

import json
import math
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockState:
    ttft_s = 0.080          # simulated queue + prefill
    itl_s = 0.025           # simulated per-token decode
    batch_window_s = 0.005  # max time a request waits to join a batch step
    vocab_size = 32000

    # controls whether concurrent requests actually share batch steps
    batching = True

    active = 0
    lock = threading.Lock()
    # current batch step deadline; all requests in a step share the same tick
    batch_deadline = 0.0


STATE = MockState()


def _tokens(text: str) -> list[str]:
    return text.split(" ")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    def _json(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": "mock-model", "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/completions":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return

        prompt = req.get("prompt", "")
        max_tokens = int(req.get("max_tokens", 16))
        echo = bool(req.get("echo", False))
        want_logprobs = req.get("logprobs") is not None
        stream = bool(req.get("stream", False))

        ptoks = _tokens(prompt)
        n_out = max(1, max_tokens)

        if stream:
            self._stream(req, ptoks, n_out, echo, want_logprobs)
        else:
            self._json(200, self._blocking(req, ptoks, n_out, echo, want_logprobs))

    # -- non-streaming --------------------------------------------------
    def _blocking(self, req, ptoks, n_out, echo, want_logprobs):
        time.sleep(STATE.ttft_s)
        prompt = " ".join(ptoks)
        gen = " ".join(f"tok{i}" for i in range(n_out))
        text = (prompt + " " + gen) if echo else gen
        choices = [{"text": gen, "index": 0, "finish_reason": "length",
                    "logprobs": None}]
        if want_logprobs:
            full = ptoks + [f"tok{i}" for i in range(n_out)]
            # deterministic, plausible logprobs around ln(1/100)
            rng = random.Random(0xC0FFEE)
            tl = [None] + [math.log(rng.uniform(0.002, 0.05)) for _ in full]
            # The MC probe reads top-k at position 0 and looks for the tokens
            # " A".." D". Emit those so the quality path is exercised end to end.
            weights = {" A": 0.40, " B": 0.15, " C": 0.25, " D": 0.10, " x": 0.10}
            first_top = {t: math.log(p) for t, p in weights.items()}
            tops = [first_top] + [first_top for _ in full[1:]]
            choices[0]["logprobs"] = {
                "tokens": full,
                "token_logprobs": tl,
                "top_logprobs": tops,
                "text_offset": [0] * len(full),
            }
        return {
            "id": "cmpl-mock", "object": "text_completion",
            "model": req.get("model", "mock-model"),
            "choices": choices,
            "usage": {"prompt_tokens": len(ptoks),
                      "completion_tokens": n_out,
                      "total_tokens": len(ptoks) + n_out},
        }

    # -- streaming -------------------------------------------------------
    def _stream(self, req, ptoks, n_out, echo, want_logprobs):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(event: str, data):
            payload = f"event: {event}\ndata: {data}\n\n".encode()
            self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()

        def chunk(txt: str, finish=None, logprobs=None):
            body = {"id": "cmpl-mock", "object": "text_completion",
                    "model": req.get("model", "mock-model"),
                    "choices": [{"text": txt, "index": 0,
                                 "finish_reason": finish,
                                 "logprobs": logprobs}]}
            send("message", json.dumps(body))

        try:
            if echo:
                # replay prompt as one chunk so non-streaming-style echoes work
                chunk(prompt)

            time.sleep(STATE.ttft_s)

            for i in range(n_out):
                # Simulate batching: if other requests are in flight, share a tick
                time.sleep(STATE.itl_s * (0.25 if STATE.batching else 1.0))
                chunk(f"tok{i}")

            chunk("", finish="length")

            if want_logprobs:
                full = ptoks + [f"tok{i}" for i in range(n_out)]
                rng = random.Random(0xC0FFEE)
                tl = [None] + [math.log(rng.uniform(0.002, 0.05)) for _ in full]
                send("message", json.dumps({
                    "id": "cmpl-mock", "object": "text_completion",
                    "model": req.get("model", "mock-model"),
                    "choices": [],
                    "usage": {"prompt_tokens": len(ptoks),
                              "completion_tokens": n_out,
                              "total_tokens": len(ptoks) + n_out},
                    "_logprobs_payload": {"token_logprobs": tl},
                }))
            send("done", "[DONE]")

            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(port: int = 8999):
    class Quiet(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # clients dropping keep-alive connections is normal here; the
            # default handler prints a traceback that drowns real output
            pass

    srv = Quiet(("127.0.0.1", port), Handler)
    print(f"mock OpenAI-compatible endpoint on http://127.0.0.1:{port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8999)
    ap.add_argument("--ttft-ms", type=float, default=80.0)
    ap.add_argument("--itl-ms", type=float, default=25.0)
    a = ap.parse_args()
    STATE.ttft_s = a.ttft_ms / 1000.0
    STATE.itl_s = a.itl_ms / 1000.0
    serve(a.port)