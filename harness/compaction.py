from __future__ import annotations

"""
Step 7 核心组件：Context Compaction 上下文裁剪与历史摘要压缩 (harness/compaction.py)

设计机制：
长周期 Coding Agent 会面临两类上下文膨胀问题，需要两级压缩策略：
1. **单次工具大输出截断（Micro-truncation: `truncate_tool_output`）**：
   当 `read_file` 或 `bash` 单次返回上万字符时，保留头部与尾部关键信息（报错堆栈通常在尾部，命令回显在头部），折叠中间冗余内容。
2. **多轮历史消息摘要压缩（Macro-compaction: `compact_messages`）**：
   当整个 `messages` 列表估算 Token 超过阈值 `max_tokens`（或用户手动触发 `/compact`）时：
   - 保留第 0 条 `system` 消息；
   - 保留最近 `keep_recent` 条消息原样不动（维持当前正在进行的推理连贯性）；
   - **协议完整性约束**：切分点 `split_idx` 绝不能落在 `assistant(tool_calls)` 和它对应的 `role="tool"` 消息之间！
     否则保留下来的半截孤儿 `tool` 消息会导致 OpenAI API 直接报 400 Bad Request；
   - 将切分点之前的旧消息序列交给 LLM 浓缩为一条结构化的 `<context_summary>` 消息替换原历史。
"""

import json
from typing import Any

from harness.ui import log_step

SUMMARY_SYSTEM_PROMPT = (
    "You are a context compaction assistant for a coding agent. "
    "Summarize the earlier conversation history concisely into structured bullet points covering:\n"
    "1. Completed actions and modified files;\n"
    "2. Key technical findings, errors encountered, and how they were resolved;\n"
    "3. Current state and remaining tasks.\n"
    "Be factual and preserve exact file paths, function names, and error messages."
)


def truncate_tool_output(output: str, max_chars: int = 4000) -> str:
    """对超长工具输出执行头尾保留 + 中间折叠截断。"""
    if len(output) <= max_chars:
        return output

    half = max_chars // 2
    omitted = len(output) - max_chars
    return (
        f"{output[:half]}\n"
        f"... [TRUNCATED {omitted} characters by Compactor] ...\n"
        f"{output[-half:]}"
    )


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """
    轻量 Token 估算器（按 JSON 序列化字符数 / 4 近似估算，兼容中英文混合场景）。
    """
    total_chars = 0
    for m in messages:
        total_chars += len(str(m.get("content") or ""))
        if m.get("tool_calls"):
            total_chars += len(json.dumps(m["tool_calls"], ensure_ascii=False))
    return max(1, total_chars // 4)


def _find_safe_split_index(non_sys_messages: list[dict[str, Any]], keep_recent: int) -> int:
    """
    计算安全的历史切分下标 `split_idx`（`non_sys_messages[:split_idx]` 将被压缩，`non_sys_messages[split_idx:]` 原样保留）：
    如果 `non_sys_messages[split_idx]` 是 `role == "tool"`，必须把 `split_idx` 往前退到发起该 `tool_calls` 的那条 `assistant` 消息，
    严禁把一对 `assistant(tool_calls) <-> tool` 拆成两半！
    """
    if len(non_sys_messages) <= keep_recent:
        return 0

    split_idx = len(non_sys_messages) - keep_recent
    while split_idx > 0 and non_sys_messages[split_idx].get("role") == "tool":
        split_idx -= 1

    return split_idx


def _format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "unknown").upper()
        content = (m.get("content") or "").strip()
        if m.get("tool_calls"):
            calls_str = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in m["tool_calls"]
            )
            lines.append(f"[{role} TOOL_CALLS]: {calls_str}")
        if content:
            snippet = content if len(content) <= 600 else content[:600] + "...(truncated)"
            lines.append(f"[{role}]: {snippet}")
    return "\n".join(lines)


def compact_messages(
    messages: list[dict[str, Any]],
    llm: Any,
    max_tokens: int = 6000,
    keep_recent: int = 4,
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    当 `estimate_tokens(messages) > max_tokens`（或 `force=True`）时，
    将早期对话历史压缩为一条 `<context_summary>` 消息，同时保证 `system` 消息与最近 `keep_recent` 条消息完整保留。
    """
    current_tokens = estimate_tokens(messages)
    if not force and current_tokens <= max_tokens:
        return messages

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    non_sys = [m for m in messages if m.get("role") != "system"]

    split_idx = _find_safe_split_index(non_sys, keep_recent=keep_recent)
    if split_idx <= 0 or (split_idx < 2 and not force):
        return messages

    old_chunk = non_sys[:split_idx]
    recent_chunk = non_sys[split_idx:]

    transcript = _format_messages_for_summary(old_chunk)
    summary_req = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Please summarize the following {len(old_chunk)} earlier messages:\n\n{transcript}",
        },
    ]

    resp = llm.chat(summary_req, tools=None)
    summary_text = (resp.content or "").strip()

    summary_msg: dict[str, Any] = {
        "role": "user",
        "content": (
            "<context_summary>\n"
            f"The following is a compacted summary of {len(old_chunk)} earlier messages in this session:\n"
            f"{summary_text}\n"
            "</context_summary>"
        ),
    }

    compacted = sys_msgs + [summary_msg] + recent_chunk
    new_tokens = estimate_tokens(compacted)
    log_step(
        "COMPACTION",
        f"Compacted {len(messages)} -> {len(compacted)} messages (tokens: ~{current_tokens} -> ~{new_tokens})",
        color="yellow",
    )
    return compacted
