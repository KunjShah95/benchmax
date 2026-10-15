import socket
import threading
import time

import pytest
import requests

from benchmax.core import OpenAICompatClient
from benchmax import generation_eval as g
from benchmax import mock_cloud


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_url():
    port = _free_port()
    thread = threading.Thread(target=mock_cloud.serve, args=(port,), daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/v1"
    deadline = time.time() + 10.0
    healthy = False
    while time.time() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                healthy = True
                break
        except Exception:
            time.sleep(0.5)
    if not healthy:
        raise RuntimeError(f"mock server failed to start on port {port}")
    yield url


def test_dataset_hash_stable():
    h1 = g.dataset_hash()
    h2 = g.dataset_hash(g.load_generation_dataset())
    assert h1 == h2
    assert len(h1) == 64


def test_normalise_and_exact_match():
    assert g.normalise_answer("  4 \n") == "4"
    assert g.is_exact_match("The Answer Is 4", "4") is False
    assert g.is_exact_match("4", "4") is True


def test_default_stops_are_applied():
    assert "\n" in g.DEFAULT_STOPS

    captured = {}

    class StopClient:
        def complete_once(self, *a, **k):
            captured.update(k)
            return {"raw": {"choices": [{"text": "4"}]}, "wall_s": 0.01}

    g.evaluate_generation(StopClient(), dataset=[
        {"id": "s1", "prompt": "p", "expected": "4"}])
    assert captured.get("stop") == g.DEFAULT_STOPS


def test_stop_none_uses_defaults_and_empty_disables():
    class NoStopClient:
        def __init__(self):
            self.calls = []

        def complete_once(self, *a, **k):
            self.calls.append(k.get("stop"))
            return {"raw": {"choices": [{"text": "4"}]}, "wall_s": 0.01}

    ds = [{"id": "s1", "prompt": "p", "expected": "4"}]
    c1 = NoStopClient()
    g.evaluate_generation(c1, dataset=ds)
    assert c1.calls[0] == g.DEFAULT_STOPS

    c2 = NoStopClient()
    g.evaluate_generation(c2, dataset=ds, stop=[])
    assert c2.calls[0] == []


def test_extract_answer_handles_measured_base_model_shapes():
    # shapes captured from real Qwen3-0.6B /v1/completions output
    assert g.extract_answer(" The answer is Paris.") == "Paris"
    assert g.extract_answer(" Paris") == "Paris"
    # last non-empty line wins, then any leading answer label is stripped
    assert g.extract_answer("chatter\nAnswer: 42") == "42"
    assert g.extract_answer("  **25**  ") == "25"
    assert g.extract_answer("line one\n42") == "42"
    assert g.extract_answer("3.5") == "3.5"
    assert g.extract_answer("") == ""
    assert g.extract_answer(None) == ""


def test_extraction_does_not_rescue_a_wrong_answer():
    assert not g.is_exact_match(g.extract_answer(" The answer is 41."), "42")


def test_generation_eval_runs_against_mock(mock_url):
    client = OpenAICompatClient(mock_url, model="mock-model")
    result = g.evaluate_generation(client, max_tokens=4)
    assert result["dataset_items"] == len(g.load_generation_dataset())
    assert result["n_ok"] == result["dataset_items"]
    assert result["n_failed"] == 0
    assert result["exact_match_accuracy"] == 0.0
    assert result["latency_s_mean"] is not None
    assert result["dataset_hash"]
    assert len(result["per_item"]) == result["dataset_items"]
    for row in result["per_item"]:
        assert row["raw_answer"] is not None
        assert row["latency_s"] is not None
        assert row["completion_tokens"] is not None
        assert row["error"] is None


def test_generation_eval_failed_requests_are_recorded(monkeypatch):
    # Force client to raise on complete_scored
    class BrokenClient:
        def complete_once(self, *a, **k):
            raise RuntimeError("boom")

    result = g.evaluate_generation(BrokenClient())
    assert result["n_ok"] == 0
    assert result["n_failed"] == result["n_items"]
    assert result["status"] == "error"
    assert all(r["error"] for r in result["per_item"])


def test_empty_answer_counted_as_failed(monkeypatch):
    class EmptyClient:
        def complete_once(self, *a, **k):
            return {"raw": {"choices": [{"text": " "}], "usage": None}, "wall_s": 0.1}

    result = g.evaluate_generation(EmptyClient(), dataset=[
        {"id": "e1", "prompt": "p", "expected": "x"}])
    assert result["n_ok"] == 0
    assert result["n_failed"] == 1
    assert result["per_item"][0]["error"] == "empty response"
    assert result["token_usage_available"] is False
    assert result["usage_coverage"] == 0.0
    assert result["total_prompt_tokens"] is None


def test_custom_dataset_version_preserved():
    class OKClient:
        def complete_once(self, *a, **k):
            return {"raw": {"choices": [{"text": "4"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}, "wall_s": 0.1}

    dataset = {"version": "2.5.0", "items": [{"id": "m", "prompt": "p", "expected": "4"}]}
    result = g.evaluate_generation(OKClient(), dataset=dataset)
    assert result["dataset_version"] == "2.5.0"
    assert result["n_with_usage"] == 1
    assert result["total_prompt_tokens"] == 2
    assert result["token_usage_available"] is True
