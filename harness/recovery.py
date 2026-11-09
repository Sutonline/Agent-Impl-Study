from __future__ import annotations

"""
Step 5 核心组件：Recovery 容错、死循环检测与会话断点恢复 (harness/recovery.py)

设计机制：
在真实运行环境中，Agent 会面临四个层面的故障，需要四类不同的恢复手段：
1. **工具调用失败指导（Structured Error Observation）**：
   不仅告诉 LLM “错了”，还要附上明确的 `RECOVERY_HINT`（例如 `edit_file` 匹配失败时要求先调 `read_file`）。
2. **重复报错死循环检测（`LoopDetector`）**：
   LLM 常见故障是对着同一个错误参数反复重试（如连续 3 次传一模一样的 `old_text` 调 `edit_file`）。
   `LoopDetector` 对 `(tool_name, canonical_json_args)` 计算签名，一旦连续 `max_repeats` 次报相同的错，
   立即返回 `[LOOP_DETECTED]` 强干预警告，打断模型的思维惯性。
3. **LLM 接口退避重试（`call_with_retry`）**：
   针对 HTTP 429 限流、网络抖动超时，使用指数退避（Exponential Backoff）重试，避免因偶发网络波动中断整个任务。
4. **会话断点持久化与恢复（`SessionStore`）**：
   每轮循环结束时通过原子写入（Write-to-temp + `os.replace`）将 `messages` 落盘到 `.sessions/<id>.json`。
   进程被杀或崩溃后，可通过 `load(session_id)` 从断点精确续跑。
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, TypeVar

from harness.ui import log_step

T = TypeVar("T")


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
        payload = {
            "session_id": session_id,
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
