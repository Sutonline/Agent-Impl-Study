from __future__ import annotations

"""
Step 8 核心整合：完整精简版 Claude Code Agent 类 (harness/agent.py)

将前 8 步的所有 Harness 构件组装为一个内聚的 `Agent` 类：
  Agent = LLMClient
        + ToolRegistry (read_file, write_file, edit_file, bash, load_skill, task)
        + AgentRuntime (Env Isolation, Path Sandbox, Command Prefix Permission Gate)
        + ValidationPipeline (pre_tool_use, post_tool_use, pre_stop Hooks)
        + Recovery (call_with_retry, LoopDetector, SessionStore + inspect_session_state)
        + Context & Skills (AGENTS.md Grounding + Progressive SKILL.md Disclosure)
        + Compaction (truncate_tool_output + compact_messages)
        + Subagent (Isolated `task` dispatcher)
"""

from pathlib import Path
from typing import Any

from harness.compaction import compact_messages, estimate_tokens, truncate_tool_output
from harness.context import build_system_prompt
from harness.llm import LLMClient, LLMResponse
from harness.recovery import LoopDetector, SessionStore, call_with_retry, inspect_session_state
from harness.runtime import AgentRuntime
from harness.skills import SkillManager
from harness.subagent import register_subagent_tool
from harness.todo import AgentMode, TodoManager
from harness.tool import ToolRegistry, create_coding_tools
from harness.ui import log_step
from harness.validation import ValidationPipeline, create_default_validation_pipeline

STATUS_COLOR = {
    "ok": "magenta",
    "error": "red",
    "denied": "red",
}


class Agent:
    """组装全套 Harness 能力的完整 Coding Agent。"""

    def __init__(
        self,
        workspace_root: Path,
        llm: Any | None = None,
        skills_dir: Path | None = None,
        sessions_dir: Path | None = None,
        session_id: str = "default",
        always_allow_dangerous: bool = False,
        pre_stop_command: str | None = None,
        max_context_tokens: int = 6000,
        max_turns: int = 12,
        enable_plan_todo: bool = False,
        initial_mode: AgentMode = "act",
    ) -> None:
        self.llm = llm or LLMClient()
        self.runtime = AgentRuntime(
            workspace_root=workspace_root,
            always_allow_dangerous=always_allow_dangerous,
        )
        self.registry: ToolRegistry = create_coding_tools(base_dir=self.runtime.workspace_root)
        self.validation: ValidationPipeline = create_default_validation_pipeline(pre_stop_command=pre_stop_command)
        self.session_store = SessionStore(base_dir=sessions_dir or (workspace_root.parent / ".sessions"))
        self.session_id = session_id
        self.max_context_tokens = max_context_tokens
        self.max_turns = max_turns

        self.skill_manager: SkillManager | None = None
        if skills_dir and Path(skills_dir).exists():
            self.skill_manager = SkillManager(skills_dir=Path(skills_dir))
            self.skill_manager.register_load_tool(self.registry)

        register_subagent_tool(self.registry, llm=self.llm, runtime=self.runtime)

        self.todo_manager: TodoManager | None = None
        if enable_plan_todo:
            self.todo_manager = TodoManager(mode=initial_mode)
            self.todo_manager.register_todo_tools(self.registry)
            self.validation.add_pre_tool(self.todo_manager.make_plan_mode_pre_tool_hook())
            self.validation.add_pre_stop(self.todo_manager.make_todo_pre_stop_hook())

    def _ensure_system_prompt(self) -> None:
        if not self.runtime.messages:
            sys_prompt = build_system_prompt(self.runtime, skill_manager=self.skill_manager)
            if self.todo_manager is not None:
                sys_prompt += (
                    "\n\n## Plan & Todo Workflow Rules\n"
                    f"- Current Mode: `{self.todo_manager.mode.upper()}`\n"
                    "- When in `PLAN` mode, mutating tools (`write_file`, `edit_file`, non-readonly `bash`) are blocked. "
                    "First inspect files with `read_file`, create a step-by-step checklist with `todo_write` "
                    "(at most ONE item `in_progress` at a time), and call `exit_plan_mode(plan_summary=...)` to enter `ACT` mode.\n"
                    "- Before stopping, every item in `todo_write` must be marked `completed`."
                )
            self.runtime.messages.append({"role": "system", "content": sys_prompt})

    def run_turn(self, user_input: str | None = None) -> str:
        """运行单次用户指令（或从断点恢复），直到通过 `pre_stop` 校验结束。"""
        self._ensure_system_prompt()
        loop_detector = LoopDetector(max_repeats=3)
        state, missing_tool_calls = inspect_session_state(self.runtime.messages)

        if state == "incomplete_tool_calls" and missing_tool_calls:
            log_step("RECOVERY [RESUME]", f"补跑 {len(missing_tool_calls)} 个未完成的工具调用...", color="yellow")
            for call in missing_tool_calls:
                res = self.registry.execute_structured(
                    call.name,
                    call.arguments,
                    runtime=self.runtime,
                    validation=self.validation,
                )
                self.runtime.messages.append({"role": "tool", "tool_call_id": call.id, "content": res.content})
            self.session_store.save(self.session_id, self.runtime.messages)
            state, _ = inspect_session_state(self.runtime.messages)

        if user_input:
            log_step("USER", user_input, color="blue")
            self.runtime.messages.append({"role": "user", "content": user_input})
            self.session_store.save(self.session_id, self.runtime.messages)
        else:
            if state == "completed":
                last_reply = self.runtime.messages[-1].get("content") or ""
                log_step("SESSION RESUME", f"会话已处于完成状态:\n{last_reply}", color="green")
                return last_reply
            if state == "empty":
                return ""

        schemas = self.registry.openai_schemas()

        for turn in range(1, self.max_turns + 1):
            self.runtime.messages = compact_messages(
                self.runtime.messages,
                llm=self.llm,
                max_tokens=self.max_context_tokens,
            )
            est = estimate_tokens(self.runtime.messages)
            log_step(
                "LOOP",
                f"--- Turn {turn}/{self.max_turns} (msgs={len(self.runtime.messages)}, tokens=~{est}) ---",
                color="gray",
            )

            response: LLMResponse = call_with_retry(
                lambda: self.llm.chat(self.runtime.messages, tools=schemas),
                max_retries=3,
                base_delay=0.05,
            )
            self.runtime.messages.append(response.raw_message)
            self.session_store.save(self.session_id, self.runtime.messages)

            if response.content:
                log_step("THINK", response.content, color="cyan")

            if not response.tool_calls:
                stop_check = self.validation.run_pre_stop(self.runtime)
                if stop_check.passed:
                    self.session_store.save(self.session_id, self.runtime.messages)
                    log_step("DONE", "Turn complete.", color="green")
                    return response.content or ""

                feedback = stop_check.format_feedback("pre_stop")
                log_step("VALIDATION [pre_stop VETO]", f"\n{feedback}", color="red")
                self.runtime.messages.append({"role": "user", "content": feedback})
                self.session_store.save(self.session_id, self.runtime.messages)
                continue

            for call in response.tool_calls:
                log_step("TOOL CALL", f"{call.name}({call.arguments})", color="yellow")
                result = self.registry.execute_structured(
                    call.name,
                    call.arguments,
                    runtime=self.runtime,
                    validation=self.validation,
                )
                clipped = truncate_tool_output(result.content, max_chars=4000)
                warn = loop_detector.record_and_check(call.name, call.arguments, is_error=result.is_error)
                content_for_llm = f"{clipped}\n{warn}" if warn else clipped

                log_step(
                    f"TOOL RESULT [{result.status.upper()}]",
                    f"\n{content_for_llm}",
                    color=STATUS_COLOR.get(result.status, "magenta"),
                )
                self.runtime.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content_for_llm,
                    }
                )
                self.session_store.save(self.session_id, self.runtime.messages)

        return "Reached max_turns without finishing."
