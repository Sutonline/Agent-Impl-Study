from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from harness.llm import LLMResponse, ToolCall
from harness.runtime import AgentRuntime
from harness.subagent import register_subagent_tool
from harness.tool import create_coding_tools


class SequencedLLM:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages: list[dict], tools=None) -> LLMResponse:
        self.seen_messages.append([dict(m) for m in messages])
        return self.responses.pop(0)


def test_subagent_runs_in_isolated_context_and_returns_summary(tmp_path: Path):
    (tmp_path / "secret.txt").write_text("CODE_42", encoding="utf-8")
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    reg = create_coding_tools()

    sub_llm = SequencedLLM([
        LLMResponse(
            content="Reading secret.txt",
            tool_calls=[ToolCall(id="sub_1", name="read_file", arguments='{"path": "secret.txt"}')],
            raw_message={
                "role": "assistant",
                "content": "Reading secret.txt",
                "tool_calls": [{
                    "id": "sub_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "secret.txt"}'},
                }],
            },
            total_tokens=30,
        ),
        LLMResponse(
            content="Subagent report: secret code is CODE_42.",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "Subagent report: secret code is CODE_42."},
            total_tokens=40,
        ),
    ])

    register_subagent_tool(reg, llm=sub_llm, runtime=rt)
    parent_messages_before = list(rt.messages)
    result = reg.execute("task", '{"description": "Find code", "prompt": "Read secret.txt and report"}', runtime=rt)
    assert "CODE_42" in result
    # Parent runtime.messages must not be polluted by subagent intermediate tool messages
    assert rt.messages == parent_messages_before


class TestStep08Subagent(unittest.TestCase):
    def test_subagent_isolated_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_subagent_runs_in_isolated_context_and_returns_summary(Path(tmp))


if __name__ == "__main__":
    unittest.main()
