from __future__ import annotations

import unittest

from harness.llm import LLMResponse, ToolCall
from steps.step01_bare_loop import run_bare_loop


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


def test_bare_loop_executes_tool_and_returns_final_answer():
    fake_llm = FakeLLM([
        LLMResponse(
            content="Let me calculate that.",
            tool_calls=[ToolCall(id="call_1", name="calculate", arguments='{"expression": "6 * 7"}')],
            raw_message={
                "role": "assistant",
                "content": "Let me calculate that.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": '{"expression": "6 * 7"}'},
                }],
            },
            total_tokens=42,
        ),
        LLMResponse(
            content="The result of 6 * 7 is 42.",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "The result of 6 * 7 is 42."},
            total_tokens=60,
        ),
    ])
    answer = run_bare_loop("What is 6 * 7?", llm=fake_llm)
    assert "42" in answer
    assert fake_llm.calls == 2


class TestStep01Loop(unittest.TestCase):
    def test_bare_loop(self):
        test_bare_loop_executes_tool_and_returns_final_answer()


if __name__ == "__main__":
    unittest.main()
