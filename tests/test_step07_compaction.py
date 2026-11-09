from __future__ import annotations

import unittest

from harness.compaction import compact_messages, truncate_tool_output
from harness.llm import LLMResponse


class DummySummaryLLM:
    def chat(self, messages: list[dict], tools=None) -> LLMResponse:
        return LLMResponse(
            content="- Created foo.py\n- Next: run tests",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "- Created foo.py\n- Next: run tests"},
            total_tokens=20,
        )


def test_compact_messages_preserves_system_and_never_splits_tool_call_pairs():
    messages = [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "Old task " * 200},
        {"role": "assistant", "content": "Old reply " * 200},
        {
            "role": "assistant",
            "content": "Calling tool",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file content"},
        {"role": "user", "content": "Recent question"},
    ]
    compacted = compact_messages(messages, llm=DummySummaryLLM(), max_tokens=100, keep_recent=2)
    assert compacted[0]["role"] == "system"
    assert "<context_summary>" in compacted[1]["content"]
    # Ensure no orphan 'tool' message appears without its preceding 'assistant' tool_calls message
    for i, msg in enumerate(compacted):
        if msg["role"] == "tool":
            assert i > 0 and compacted[i - 1].get("tool_calls")


def test_truncate_tool_output_keeps_head_and_tail():
    text = "HEAD_" + ("x" * 5000) + "_TAIL"
    out = truncate_tool_output(text, max_chars=400)
    assert out.startswith("HEAD_")
    assert out.endswith("_TAIL")
    assert "TRUNCATED" in out


class TestStep07Compaction(unittest.TestCase):
    def test_compact_messages(self):
        test_compact_messages_preserves_system_and_never_splits_tool_call_pairs()

    def test_truncate_output(self):
        test_truncate_tool_output_keeps_head_and_tail()


if __name__ == "__main__":
    unittest.main()
