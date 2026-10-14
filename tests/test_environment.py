from benchmax.runner import capture_environment, capture_versions
import os


def test_capture_environment_has_core_fields():
    env = capture_environment()
    assert "timestamp_utc" in env
    assert "platform" in env
    assert "python" in env
    assert "logical_cores" in env
    # gpu may be unavailable on host; field must exist either way
    assert "gpu" in env


def test_capture_versions_returns_dict():
    v = capture_versions(os.path.expanduser("~/benchmax-env"))
    assert isinstance(v, dict)
