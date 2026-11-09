from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from harness.recovery import (
    LoopDetector,
    SessionStore,
    call_with_retry,
    format_error_observation,
)


def test_loop_detector_triggers_on_repeated_identical_failures():
    detector = LoopDetector(max_repeats=3)
    assert detector.record_and_check("edit_file", '{"path":"a.py"}', is_error=True) is None
    assert detector.record_and_check("edit_file", '{"path":"a.py"}', is_error=True) is None
    warning = detector.record_and_check("edit_file", '{"path":"a.py"}', is_error=True)
    assert warning is not None
    assert "LOOP_DETECTED" in warning


def test_call_with_retry_recovers_from_transient_error():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("Rate limit")
        return "ok"

    assert call_with_retry(flaky, max_retries=3, base_delay=0.01) == "ok"
    assert attempts["count"] == 3


def test_session_store_save_and_resume(tmp_path: Path):
    store = SessionStore(tmp_path)
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    store.save("sess_01", msgs)
    loaded = store.load("sess_01")
    assert loaded == msgs


class TestStep05Recovery(unittest.TestCase):
    def test_loop_detector(self):
        test_loop_detector_triggers_on_repeated_identical_failures()

    def test_retry(self):
        test_call_with_retry_recovers_from_transient_error()

    def test_session_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_session_store_save_and_resume(Path(tmp))


if __name__ == "__main__":
    unittest.main()
