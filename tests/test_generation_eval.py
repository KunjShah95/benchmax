import threading
import time

import pytest

from benchmax.core import OpenAICompatClient
from benchmax import generation_eval as g
from benchmax import mock_cloud


@pytest.fixture(scope="module")
def mock_url():
    port = 8991
    thread = threading.Thread(target=mock_cloud.serve, args=(port,), daemon=True)
    thread.start()
    time.sleep(1.0)
    url = f"http://127.0.0.1:{port}/v1"
    for _ in range(20):
        try:
            import requests
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
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
        def complete_scored(self, *a, **k):
            raise RuntimeError("boom")

    result = g.evaluate_generation(BrokenClient())
    assert result["n_ok"] == 0
    assert result["n_failed"] == result["n_items"]
    assert result["status"] == "error"
    assert all(r["error"] for r in result["per_item"])
