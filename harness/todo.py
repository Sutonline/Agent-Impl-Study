from __future__ import annotations

"""
Step 9 核心组件：Plan & Todo 模式 (harness/todo.py)

第一性原理视角：
为什么在长链路复杂任务中，Coding Agent 容易“写一半忘一半”或“还没读懂代码就乱改一气”？
因为 LLM 缺乏显式的外部任务状态机（State Machine）与读写阶段隔离（Phase Separation）。

Claude Code 的工程解法由两个正交机制组成，且完美复用我们在 Step 4 建立的 `ValidationPipeline`：
1. **Plan / Act 双模态状态机 (`pre_tool_use` 守门员)**：
   - 在 `mode == "plan"` 阶段，只允许调用只读探索工具（`read_file`、`load_skill`、`task`）与规划工具（`todo_write`、`exit_plan_mode`）；
   - 任何试图修改文件（`write_file`、`edit_file`）或执行破坏性 `bash` 命令的调用，都会在 `pre_tool_use` 阶段被直接拦截（返回 `status="denied"`），强制 LLM 先完成代码调研与任务拆解，再调用 `exit_plan_mode(plan_summary)` 切入 `act` 模式。
2. **`TodoWrite` 结构化任务账本 (`pre_stop` 守门员)**：
   - 提供 `todo_write(todos)` 工具让 LLM 维护状态为 `pending | in_progress | completed` 的任务列表；
   - 施加硬性状态不变量：**同一时刻最多只能有 1 个 `in_progress` 任务**（强迫线性聚焦，禁止并发烂尾）；
   - 在 `pre_stop` 阶段挂载守门员：当 LLM 准备退出主循环时，只要存在未完成（`pending` 或 `in_progress`）的 Todo 项，直接否决停止（veto stop），将剩余待办打回主循环逼迫其继续执行。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from harness.runtime import SAFE_READONLY_PREFIXES, extract_command_prefix, split_shell_commands
from harness.tool import Tool, ToolRegistry
from harness.ui import log_step
from harness.validation import PreStopHook, PreToolHook, ValidationResult

if TYPE_CHECKING:
    from harness.runtime import AgentRuntime

TodoStatus = Literal["pending", "in_progress", "completed"]
AgentMode = Literal["plan", "act"]

VALID_STATUSES: set[str] = {"pending", "in_progress", "completed"}
PLAN_ALLOWED_TOOLS: set[str] = {
    "read_file",
    "load_skill",
    "task",
    "todo_write",
    "exit_plan_mode",
}


@dataclass
class TodoItem:
    """单条结构化待办任务。"""

    id: str
    content: str
    status: TodoStatus = "pending"


@dataclass
class TodoManager:
    """
    管理 Plan/Act 模式切换与 Todo 任务账本，并提供挂载到 `ValidationPipeline` 的钩子工厂。
    """

    mode: AgentMode = "act"
    items: list[TodoItem] = field(default_factory=list)
    last_plan_summary: str = ""
    approval_callback: Callable[[str], bool] | None = None

    def update(self, raw_items: list[dict[str, Any]]) -> str:
        """
        全量替换更新当前 Todo 列表，并校验状态不变量：
        1. 每个条目必须包含非空 `id`、`content` 以及合法 `status` (`pending` | `in_progress` | `completed`)；
        2. 同一时刻最多只能有 1 个条目处于 `in_progress` 状态。
        """
        if not isinstance(raw_items, list):
            raise ValueError("`todos` must be a list of todo item objects.")

        parsed: list[TodoItem] = []
        in_progress_count = 0

        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Todo item at index {idx} must be an object.")
            item_id = str(item.get("id", "")).strip()
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).strip()

            if not item_id or not content:
                raise ValueError(f"Todo item at index {idx} requires non-empty 'id' and 'content'.")
            if status not in VALID_STATUSES:
                raise ValueError(
                    f"Invalid status '{status}' for todo '{item_id}'. Must be one of {sorted(VALID_STATUSES)}."
                )
            if status == "in_progress":
                in_progress_count += 1

            parsed.append(TodoItem(id=item_id, content=content, status=status))  # type: ignore[arg-type]

        if in_progress_count > 1:
            raise ValueError(
                f"Only ONE todo item can be 'in_progress' at a time (got {in_progress_count}). "
                "Focus on completing one task before starting the next."
            )

        self.items = parsed
        board = self.render()
        log_step(f"TODO BOARD [mode={self.mode.upper()}]", f"\n{board}", color="cyan")
        return board

    def has_incomplete(self) -> bool:
        """只要存在非 completed 的条目，即视为未完成。"""
        return any(it.status != "completed" for it in self.items)

    def incomplete_items(self) -> list[TodoItem]:
        return [it for it in self.items if it.status != "completed"]

    def render(self) -> str:
        """渲染人类与 LLM 易读的 Markdown 风格任务看板。"""
        if not self.items:
            return "[Todo List is empty]"
        icon_map = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }
        lines = [f"Todo Progress ({ sum(1 for i in self.items if i.status == 'completed') }/{len(self.items)} completed):"]
        for it in self.items:
            marker = icon_map.get(it.status, "[ ]")
            lines.append(f"  {marker} #{it.id}: {it.content} ({it.status})")
        return "\n".join(lines)

    # ========================================================================
    # 生命周期守门员 Hooks (接入 Step 4 ValidationPipeline)
    # ========================================================================

    def make_plan_mode_pre_tool_hook(self) -> PreToolHook:
        """
        [pre_tool_use Hook] Plan 模式只读沙盒守门员：
        当 `self.mode == "plan"` 时：
        - 允许只读工具与规划工具（`read_file`, `load_skill`, `task`, `todo_write`, `exit_plan_mode`）；
        - 允许纯只读的 `bash` 命令（如 `ls`, `pwd`, `git status`）；
        - 拦截一切写文件/改代码/副作用命令（返回 `status="denied"`），强制先完成规划并调用 `exit_plan_mode`。
        """

        def _hook(tool_obj: Tool, args: dict[str, Any], runtime: AgentRuntime) -> ValidationResult:
            if self.mode != "plan":
                return ValidationResult(passed=True)

            if tool_obj.name in PLAN_ALLOWED_TOOLS:
                return ValidationResult(passed=True)

            if tool_obj.name == "bash":
                cmd = str(args.get("command", ""))
                sub_cmds = split_shell_commands(cmd)
                if sub_cmds and all(
                    extract_command_prefix(sub) in SAFE_READONLY_PREFIXES for sub in sub_cmds if sub.strip()
                ):
                    return ValidationResult(passed=True)

            return ValidationResult(
                passed=False,
                status="denied",
                reason=(
                    f"[PLAN_MODE_RESTRICTION] Tool '{tool_obj.name}' is blocked while Agent is in PLAN mode. "
                    "Mutating files or running non-readonly commands is strictly forbidden during planning."
                ),
                recovery_hint=(
                    "1. Use `read_file` to finish inspecting the codebase;\n"
                    "2. Call `todo_write` to break down the implementation steps;\n"
                    "3. Call `exit_plan_mode(plan_summary=...)` to switch to ACT mode before modifying files."
                ),
            )

        return _hook

    def make_todo_pre_stop_hook(self) -> PreStopHook:
        """
        [pre_stop Hook] Todo 完结守门员：
        当 LLM 停止调用工具准备交卷时：
        1. 如果还处于 `plan` 模式，否决停止，要求先调用 `exit_plan_mode` 并执行任务；
        2. 如果 `self.items` 中仍有 `pending` 或 `in_progress` 的条目，否决停止，列出未完成项逼迫 LLM 继续推进。
        """

        def _hook(runtime: AgentRuntime) -> ValidationResult:
            if self.mode == "plan":
                return ValidationResult(
                    passed=False,
                    reason="Agent is still in PLAN mode and has not executed the task yet.",
                    recovery_hint="Call `exit_plan_mode(plan_summary=...)` to enter ACT mode and execute the plan.",
                )

            incomplete = self.incomplete_items()
            if not incomplete:
                return ValidationResult(passed=True)

            pending_desc = ", ".join(f"#{it.id} '{it.content}' ({it.status})" for it in incomplete)
            return ValidationResult(
                passed=False,
                reason=f"Cannot stop yet: {len(incomplete)} todo item(s) remain incomplete: {pending_desc}.",
                recovery_hint=(
                    "Complete the remaining todo items using coding tools, and call `todo_write` to mark all items "
                    "as 'completed' before finishing."
                ),
            )

        return _hook

    # ========================================================================
    # 注册给 LLM 的工具 (`todo_write` 与 `exit_plan_mode`)
    # ========================================================================

    def register_todo_tools(self, registry: ToolRegistry) -> None:
        """向 `ToolRegistry` 注册 `todo_write` 与 `exit_plan_mode` 工具。"""

        def _todo_write(todos: list[dict[str, Any]]) -> str:
            try:
                board = self.update(todos)
                return f"[OK] Todo list updated successfully.\n{board}"
            except ValueError as e:
                return f"[ERROR] Invalid todo_write payload: {e}"

        todo_tool = Tool(
            name="todo_write",
            description=(
                "Create or update the structured Todo list for tracking multi-step tasks. "
                "Always send the full list of todos. At most ONE item may have status='in_progress' at a time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Full list of todo items, each with id, content, and status (pending|in_progress|completed).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Short unique step identifier, e.g. '1', '2'."},
                                "content": {"type": "string", "description": "Clear, actionable description of the step."},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Current status of this step.",
                                },
                            },
                            "required": ["id", "content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
            func=_todo_write,
            dangerous=False,
        )

        def _exit_plan_mode(plan_summary: str) -> str:
            if self.approval_callback is not None:
                approved = self.approval_callback(plan_summary)
                if not approved:
                    return "[PLAN_REJECTED] User rejected the plan. Revise your plan and try again."

            self.last_plan_summary = plan_summary
            self.mode = "act"
            log_step(
                "MODE SWITCH [PLAN -> ACT]",
                f"Plan approved! Switching to ACT mode.\nSummary: {plan_summary}",
                color="green",
            )
            return (
                "[OK] Exited PLAN mode and entered ACT mode. "
                "Write/edit/bash tools are now unlocked. Proceed to execute your todo items step by step."
            )

        exit_plan_tool = Tool(
            name="exit_plan_mode",
            description=(
                "Present your finalized implementation plan summary and transition from read-only PLAN mode "
                "to ACT mode so you can modify files and execute the todo items."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan_summary": {
                        "type": "string",
                        "description": "Concise summary of the root cause analysis and planned file changes.",
                    }
                },
                "required": ["plan_summary"],
            },
            func=_exit_plan_mode,
            dangerous=False,
        )

        registry.register(todo_tool)
        registry.register(exit_plan_tool)
