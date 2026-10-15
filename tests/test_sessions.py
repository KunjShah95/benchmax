import json
import os

from benchmax.report import list_sessions, load_raw, write_all


def _perf(backend, quant, workload, rep, sid=None):
    rec = {
        "kind": "perf_run",
        "workload": workload,
        "replicate": rep,
        "cell": {"backend": backend, "model_key": "m", "model_label": "M",
                 "family": "f", "params_b": 1.0, "quant_level": quant,
                 "quant_scheme": "S", "device": "cuda"},
        "summary": {"n_requests": 1, "n_ok": 1, "ttft_ms_mean": 10.0},
    }
    if sid:
        rec["session_id"] = sid
    return rec


def _session(sid):
    return {"kind": "session", "session_id": sid,
            "environment": {"timestamp_utc": "2026-01-01T00:00:00Z",
                            "gpu": "mock", "python": "3.12"}}


def _write(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_sessions_are_listed_and_filterable(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    _write(raw / "runs.jsonl", [
        _session("s1"),
        _perf("llamacpp", "4bit", "w1_single_short", 0, "s1"),
        _session("s2"),
        _perf("vllm", "8bit", "w1_single_short", 0, "s2"),
    ])

    sessions = list_sessions(str(tmp_path))
    ids = [s["session_id"] for s in sessions]
    assert ids == ["s1", "s2"]
    assert all(s["n_records"] >= 2 for s in sessions)

    perf_all, _, _ = load_raw(str(tmp_path))
    assert len(perf_all) == 2

    perf_s1, _, _ = load_raw(str(tmp_path), session="s1")
    assert len(perf_s1) == 1
    assert perf_s1.iloc[0]["backend"] == "llamacpp"
    assert perf_s1.iloc[0]["session_id"] == "s1"


def test_legacy_records_are_attributed_and_flagged(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    # records written before session stamping existed
    _write(raw / "runs.jsonl", [
        _session("s1"),
        _perf("llamacpp", "4bit", "w1_single_short", 0),
    ])

    perf, _, _ = load_raw(str(tmp_path))
    assert len(perf) == 1
    # attributed to the session that preceded them, not dropped
    assert perf.iloc[0]["session_id"] == "s1"

    sessions = list_sessions(str(tmp_path))
    assert any(s.get("contains_unstamped_records") for s in sessions)


def test_cross_session_collision_is_reported_not_hidden(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    _write(raw / "runs.jsonl", [
        _session("s1"),
        _perf("llamacpp", "4bit", "w1_single_short", 0, "s1"),
        _session("s2"),
        # same cell/workload/replicate measured again in a later session
        _perf("llamacpp", "4bit", "w1_single_short", 0, "s2"),
    ])

    perf, _, _ = load_raw(str(tmp_path))
    # collapsed to one row rather than averaged across sessions
    assert len(perf) == 1
    # and the row is attributable to a single session
    assert perf.iloc[0]["session_id"] in ("s1", "s2")

    out = write_all(str(tmp_path), str(tmp_path / "reports"))
    assert out["session"] is None
    assert os.path.exists(tmp_path / "reports" / "tables" / "sessions.csv")
    assert len(out["sessions"]) == 2


def test_report_to_a_single_session(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    _write(raw / "runs.jsonl", [
        _session("s1"),
        _perf("llamacpp", "4bit", "w1_single_short", 0, "s1"),
        _session("s2"),
        _perf("vllm", "8bit", "w1_single_short", 0, "s2"),
    ])

    out = write_all(str(tmp_path), str(tmp_path / "reports2"), session="s2")
    assert out["session"] == "s2"
    assert out["n_perf_runs"] == 1