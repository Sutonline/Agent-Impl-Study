from __future__ import annotations

"""
Step 5 核心组件：Recovery 容错、死循环检测与会话断点恢复 (harness/recovery.py)

设计机制：
1. **工具调用失败指导（Structured Error Observation）**
2. **重复报错死循环检测（`LoopDetector`）**
3. **LLM 接口退避重试（`call_with_retry`）**
4. **会话断点持久化与尾部状态恢复机（`SessionStore` + `inspect_session_state`）**：
   恢复会话时不能无脑重新追加初始 Prompt，必须检查 `messages` 尾部处于哪种状态：
   - `"empty"`：空会话或仅有 system 消息；
   - `"completed"`：最后一条是无 `tool_calls` 的 `assistant` 消息（上一轮任务已正常结束）；
   - `"pending_llm"`：最后一条是 `user` 或已完成全部配对的 `tool` 消息（上次在调 `llm.chat` 前中断）；
   - `"incomplete_tool_calls"`：存在 `assistant.tool_calls`，但缺少对应的 `role="tool"` 结果消息（上次在执行工具中途中断）。
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Literal, TypeVar

from harness.llm import ToolCall
from harness.ui import log_step

T = TypeVar("T")

SessionState = Literal["empty", "completed", "pending_llm", "incomplete_tool_calls"]


def format_error_observation(error_type: str, reason: str, recovery_hint: str = "") -> str:
    """将工具或校验异常格式化为结构化的错误观测文本，供 LLM 自愈。"""
    text = f"[ERROR:{error_type}] {reason}"
    if recovery_hint:
        text += f"\nRECOVERY_HINT: {recovery_hint}"
    return text


def _canonicalize_args(raw_args: str) -> str:
    """将 JSON 参数字符串规范化排序，确保键顺序不同但内容相同的参数生成一致的签名。"""
    try:
        parsed = json.loads(raw_args or "{}")
        if isinstance(parsed, dict):
            return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return (raw_args or "").strip()


@dataclass
class LoopDetector:
    """
    死循环检测器：
    跟踪失败的工具调用签名 `(tool_name, canonical_args)`。
    当连续 `max_repeats` 次出现相同的失败调用时，返回 `[LOOP_DETECTED]` 警告文本追加到工具结果后，强制打断死循环。
    """

    max_repeats: int = 3
    _history: list[tuple[str, str, bool]] = field(default_factory=list)

    def record_and_check(self, tool_name: str, raw_args: str, is_error: bool) -> str | None:
        sig = _canonicalize_args(raw_args)
        self._history.append((tool_name, sig, is_error))

        if not is_error:
            return None

        if len(self._history) < self.max_repeats:
            return None

        recent = self._history[-self.max_repeats :]
        if all(err and name == tool_name and s == sig for name, s, err in recent):
            return (
                f"[LOOP_DETECTED] You have called `{tool_name}` with the exact same arguments "
                f"{self.max_repeats} times in a row and failed every time. "
                "STOP repeating this call! Change your strategy immediately "
                "(e.g., call `read_file` to inspect actual file contents or try a different approach)."
            )
        return None


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> T:
    """
    指数退避重试包装器（用于保护 `llm.chat` 等外部网络调用）：
    遇到瞬态异常时，按 `base_delay * 2^(attempt-1)` 秒等待后重试；超出 `max_retries` 后抛出原始异常。
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            log_step(
                "RECOVERY [RETRY]",
                f"Call failed ({type(e).__name__}: {e}). Retrying {attempt}/{max_retries} in {delay:.2f}s...",
                color="yellow",
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def inspect_session_state(messages: list[dict[str, Any]]) -> tuple[SessionState, list[ToolCall]]:
    """
    检查会话消息列表的尾部状态，返回 `(SessionState, 未执行的 ToolCall 列表)`：
    - `"empty"`：无消息或只有 `system` 消息；
    - `"completed"`：最后一条是 `assistant` 且不含 `tool_calls`（任务已正常结束）；
    - `"incomplete_tool_calls"`：最后一次 `assistant` 发出了 `tool_calls`，但后续缺少对应的 `role="tool"` 结果；
    - `"pending_llm"`：最后一条是 `user` 消息，或所有 `tool_calls` 都已有对应的 `role="tool"` 结果，正等待调用 `llm.chat`。
    """
    non_sys = [m for m in messages if m.get("role") != "system"]
    if not non_sys:
        return "empty", []

    last_msg = non_sys[-1]
    role = last_msg.get("role")

    if role == "user":
        return "pending_llm", []

    if role == "assistant":
        raw_tcs = last_msg.get("tool_calls") or []
        if not raw_tcs:
            return "completed", []
        missing = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )
            for tc in raw_tcs
        ]
        return "incomplete_tool_calls", missing

    if role == "tool":
        # 往回找到最近的一条带 tool_calls 的 assistant 消息，核对是否所有 tool_call_id 都已凑齐
        completed_ids: set[str] = set()
        for m in reversed(non_sys):
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id")
                if tc_id:
                    completed_ids.add(tc_id)
            elif m.get("role") == "assistant":
                raw_tcs = m.get("tool_calls") or []
                missing = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    )
                    for tc in raw_tcs
                    if tc["id"] not in completed_ids
                ]
                if missing:
                    return "incomplete_tool_calls", missing
                return "pending_llm", []
        return "pending_llm", []

    return "pending_llm", []


class SessionStore:
    """
    会话断点持久化存储器：
    将 `messages` 列表以 JSON 格式保存在 `.sessions/<session_id>.json`。
    采用写临时文件 + `os.replace` 原子重命名，防止写盘中途断电损坏已有存档。
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.sessions_dir = (base_dir or (Path.cwd() / ".sessions")).resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"_", "-"})
        if not safe_id:
            raise ValueError(f"Invalid session_id: {session_id!r}")
        return self.sessions_dir / f"{safe_id}.json"

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> Path:
        target = self._session_path(session_id)
        tmp_path = target.with_suffix(".json.tmp")
        state, _ = inspect_session_state(messages)
        payload = {
            "session_id": session_id,
            "state": state,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message_count": len(messages),
            "messages": messages,
        }
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, target)
        return target

    def load(self, session_id: str) -> list[dict[str, Any]]:
        target = self._session_path(session_id)
        if not target.exists():
            raise FileNotFoundError(f"Session checkpoint not found: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        return list(data.get("messages", []))

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.sessions_dir.glob("*.json"))
