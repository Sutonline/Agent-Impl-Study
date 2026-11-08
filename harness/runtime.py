from __future__ import annotations

"""
Step 3 核心组件：AgentRuntime 运行时容器与 Permission Gate 安全护栏 (harness/runtime.py)

第一性原理视角：
在强化学习或智能体理论中，Agent = Policy (LLM) + Environment (Runtime)。
在 Step 2 中，工具函数直接散装运行在全局环境里：
  - 没有统一的宿主状态（工作区根目录在哪？当前会话改了哪些文件？当前对话历史是什么？）；
  - 子进程没有超时保护和输出截断（一个死循环命令或 100MB 日志就会卡死或撑爆 Agent）；
  - 高危工具（`write_file`, `edit_file`, `bash`）没有人类审批护栏（Permission Gate）。

因此，`AgentRuntime` 就是 Harness 专门为工具打造的安全沙盒与状态容器。
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any, Callable

from harness.ui import log_step

if TYPE_CHECKING:
    from harness.tool import Tool

PermissionCallback = Callable[[str, dict[str, Any]], bool]


def default_terminal_permission_prompt(tool_name: str, args: dict[str, Any]) -> bool:
    """
    默认的终端人工确认交互：
    当工具被标记为 `dangerous=True` 时，暂停并在终端询问用户是否放行。
    """
    preview = json.dumps(args, ensure_ascii=False)
    if len(preview) > 200:
        preview = preview[:200] + "...(truncated)"
    log_step("PERMISSION GATE", f"Agent 申请执行高危工具: {tool_name}({preview})", color="yellow")
    try:
        ans = input("  是否允许执行？[y=允许本次 / n=拒绝 / a=本会话全部允许]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if ans == "a":
        # 返回特殊标记或由调用方处理；这里我们在 AgentRuntime.check_permission 中内联支持 'a'
        return True
    return ans in {"y", "yes"}


@dataclass
class AgentRuntime:
    """
    Agent 运行时环境容器（即 Environment 实例）。
    作为依赖注入对象传给需要感知环境的 Tool、Validation Hook 与 Recovery 模块。
    """

    workspace_root: Path = field(default_factory=lambda: (Path.cwd() / "workspace").resolve())
    cwd: Path | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    modified_files: set[Path] = field(default_factory=set)
    always_allow_dangerous: bool = False
    permission_callback: PermissionCallback | None = None

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        if self.cwd is None:
            self.cwd = self.workspace_root
        else:
            self.cwd = Path(self.cwd).resolve()

    def resolve_path(self, path_str: str) -> Path:
        """将工具传入的相对路径解析为基于当前 runtime.cwd 的绝对路径。"""
        p = Path(path_str)
        base = self.cwd or self.workspace_root
        return p.resolve() if p.is_absolute() else (base / p).resolve()

    def check_permission(self, tool_obj: Tool, args: dict[str, Any]) -> tuple[bool, str]:
        """
        Permission Gate（安全护栏）：
        - 若 `tool_obj.dangerous` 为 False（如 `read_file`），直接放行；
        - 若 `always_allow_dangerous` 为 True，直接放行；
        - 否则调用 `permission_callback` 或在终端弹窗询问用户 `[y/n/a]`。
        """
        if not tool_obj.dangerous or self.always_allow_dangerous:
            return True, ""

        if self.permission_callback is not None:
            allowed = self.permission_callback(tool_obj.name, args)
            if not allowed:
                return (
                    False,
                    f"User denied permission to execute dangerous tool '{tool_obj.name}'. "
                    "Please ask the user for guidance or try a safe alternative.",
                )
            return True, ""

        # 默认终端交互确认（支持输入 'a' 开启本会话 always_allow_dangerous）
        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 200:
            preview = preview[:200] + "...(truncated)"
        log_step("PERMISSION GATE", f"Agent 申请执行高危操作: {tool_obj.name}({preview})", color="yellow")
        try:
            ans = input("  允许执行吗？[y=允许 / n=拒绝 / a=总是允许]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans in {"a", "always"}:
            self.always_allow_dangerous = True
            log_step("PERMISSION GATE", "已开启本会话『总是允许』模式。", color="green")
            return True, ""
        if ans in {"y", "yes"}:
            return True, ""

        return (
            False,
            f"User denied permission to execute '{tool_obj.name}'. Do not repeat the exact same call without user consent.",
        )

    def run_command(
        self,
        command: str,
        timeout: int = 15,
        max_output_chars: int = 8000,
    ) -> str:
        """
        受控子进程执行器：
        1. 限定在 `self.cwd` 目录下运行；
        2. 强制 `timeout` 超时保护（防止死循环命令挂死整个 Agent）；
        3. 强制 `max_output_chars` 截断保护（防止海量日志瞬间撑爆 LLM 上下文窗口）。
        """
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.cwd or self.workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return (
                f"[ERROR] [TIMEOUT] Command timed out after {timeout} seconds: `{command}`. "
                "RECOVERY_HINT: Avoid long-running or interactive blocking commands."
            )

        combined = (proc.stdout or "") + (proc.stderr or "")
        combined = combined.strip()

        if len(combined) > max_output_chars:
            half = max_output_chars // 2
            omitted = len(combined) - max_output_chars
            combined = (
                f"{combined[:half]}\n"
                f"... [TRUNCATED {omitted} characters by AgentRuntime] ...\n"
                f"{combined[-half:]}"
            )

        return f"[exit_code={proc.returncode}]\n{combined}"
