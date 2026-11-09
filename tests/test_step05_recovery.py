from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from harness.llm import LLMResponse, ToolCall
from harness.recovery import (
    LoopDetector,
    SessionStore,
    call_with_retry,
    inspect_session_state,
)
from harness.runtime import AgentRuntime
from harness.tool import create_coding_tools
from harness.validation import create_default_validation_pipeline
from steps.step05_recovery import run_resilient_agent_loop


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


def test_resume_completed_and_interrupted_sessions_correctly(tmp_path: Path):
    store = SessionStore(tmp_path / ".sessions")
    reg = create_coding_tools(base_dir=tmp_path)
    val = create_default_validation_pipeline()

    # Case 1: Completed session resumed WITHOUT a new prompt must NOT call LLM or re-run old prompt!
    completed_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": "initial task done!"},
    ]
    state, missing = inspect_session_state(completed_msgs)
    assert state == "completed"
    assert missing == []

    class ShouldNotBeCalledLLM:
        def chat(self, messages, tools=None):
            raise AssertionError("LLM should NOT be called when resuming an already-completed session without new prompt!")

    rt_done = AgentRuntime(workspace_root=tmp_path, messages=list(completed_msgs), always_allow_dangerous=True)
    out = run_resilient_agent_loop(
        user_prompt=None,
        llm=ShouldNotBeCalledLLM(),
        registry=reg,
        runtime=rt_done,
        validation=val,
        session_store=store,
        session_id="done_sess",
    )
    assert out == "initial task done!"

    # Case 2: Session interrupted mid-tool-call (assistant emitted tool_calls, but process died before tool ran)
    (tmp_path / "note.txt").write_text("saved_data_123", encoding="utf-8")
    interrupted_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Read note.txt"},
        {
            "role": "assistant",
            "content": "Reading note.txt now.",
            "tool_calls": [
                {
                    "id": "call_interrupted_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "note.txt"}'},
                }
            ],
        },
    ]
    state2, missing2 = inspect_session_state(interrupted_msgs)
    assert state2 == "incomplete_tool_calls"
    assert len(missing2) == 1 and missing2[0].id == "call_interrupted_1"

    class ResumeLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            # Verify the missing tool call was executed and appended before calling LLM!
            assert messages[-1]["role"] == "tool"
            assert "saved_data_123" in messages[-1]["content"]
            return LLMResponse(
                content="Resumed and read: saved_data_123",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "Resumed and read: saved_data_123"},
            )

    rt_interrupted = AgentRuntime(workspace_root=tmp_path, messages=list(interrupted_msgs), always_allow_dangerous=True)
    resume_llm = ResumeLLM()
    out2 = run_resilient_agent_loop(
        user_prompt=None,
        llm=resume_llm,
        registry=reg,
        runtime=rt_interrupted,
        validation=val,
        session_store=store,
        session_id="interrupted_sess",
    )
    assert "saved_data_123" in out2
    assert resume_llm.calls == 1


class TestStep05Recovery(unittest.TestCase):
    def test_loop_detector(self):
        test_loop_detector_triggers_on_repeated_identical_failures()

    def test_retry(self):
        test_call_with_retry_recovers_from_transient_error()

    def test_session_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_session_store_save_and_resume(Path(tmp))

    def test_resume_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_resume_completed_and_interrupted_sessions_correctly(Path(tmp))


if __name__ == "__main__":
    unittest.main()
