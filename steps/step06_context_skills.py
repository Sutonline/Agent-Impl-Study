#!/usr/bin/env python3
from __future__ import annotations

"""
Step 6: Context Grounding（环境锚定）与 Skills（渐进式技能加载）
================================================================

相比 Step 5 的演进（NEW IN STEP 6）：
1. **Context Grounding（环境感知与项目常驻规则 `AGENTS.md` 注入）**：
   - `build_system_prompt(runtime, skill_manager)` 自动采集 OS、工作目录、当前文件列表，并读取工作区下的 `AGENTS.md` / `CLAUDE.md` 注入 `<project_rules>`。
2. **Skills 渐进式披露（Progressive Disclosure）**：
   - 启动时，`SkillManager` 仅扫描 `skills/*/SKILL.md` 的 YAML Frontmatter（`name` + `description`），生成几行简短的 `<available_skills>` 目录放入 System Prompt（不加载冗长的技能正文！）；
   - 当用户的任务命中某个技能（如“帮我做代码审查”）时，LLM 看到目录后主动调用 `load_skill(name="code-reviewer")` 工具，才把完整 SOP 正文按需加载进对话历史。

运行方式：
  1. 查看动态组装出的 System Prompt（验证里面只有技能目录、没有技能正文，且包含了 AGENTS.md）：
     python3 steps/step06_context_skills.py --show-prompt
  2. 免 Key 离线演示（观察：识别技能目录 -> 调用 load_skill("code-reviewer") 按需加载正文 -> 结合 AGENTS.md 规则完成代码审查）：
     python3 steps/step06_context_skills.py --demo
  3. 使用真实 OpenAI API：
     python3 steps/step06_context_skills.py "请使用 code-reviewer 技能审查 workspace/math_utils.py"
"""

from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.context import build_system_prompt
from harness.llm import LLMClient, LLMResponse, ToolCall
from harness.recovery import LoopDetector, call_with_retry
from harness.runtime import AgentRuntime
from harness.skills import SkillManager
from harness.tool import ToolRegistry, create_coding_tools
from harness.ui import log_step
from harness.validation import ValidationPipeline, create_default_validation_pipeline

STATUS_COLOR = {
    "ok": "magenta",
    "error": "red",
    "denied": "red",
}


def run_context_skills_loop(
    user_prompt: str,
    llm: Any,
    registry: ToolRegistry,
    runtime: AgentRuntime,
    validation: ValidationPipeline,
    skill_manager: SkillManager,
    max_turns: int = 10,
) -> str:
    """
    Step 6 的 Agent Loop：
    启动时通过 `build_system_prompt(runtime, skill_manager)` 动态构建带环境锚定与技能目录的 System Prompt。
    """
    if not runtime.messages:
        sys_prompt = build_system_prompt(runtime, skill_manager=skill_manager)
        runtime.messages.append({"role": "system", "content": sys_prompt})

    runtime.messages.append({"role": "user", "content": user_prompt})
    schemas = registry.openai_schemas()
    loop_detector = LoopDetector(max_repeats=3)

    log_step("USER", user_prompt, color="blue")
    log_step(
        "CONTEXT & SKILLS",
        f"Discovered skills: {[s.name for s in skill_manager.discover()]} | Tools: {[t.name for t in registry.list_tools()]}",
        color="gray",
    )

    for turn in range(1, max_turns + 1):
        log_step("LOOP", f"--- Turn {turn} ---", color="gray")
        response: LLMResponse = call_with_retry(
            lambda: llm.chat(runtime.messages, tools=schemas),
            max_retries=3,
            base_delay=0.05,
        )
        runtime.messages.append(response.raw_message)

        if response.content:
            log_step("THINK", response.content, color="cyan")

        if not response.tool_calls:
            stop_check = validation.run_pre_stop(runtime)
            if stop_check.passed:
                log_step("DONE", "Task complete.", color="green")
                return response.content or ""
            feedback = stop_check.format_feedback("pre_stop")
            log_step("VALIDATION [pre_stop VETO]", f"\n{feedback}", color="red")
            runtime.messages.append({"role": "user", "content": feedback})
            continue

        for call in response.tool_calls:
            log_step("TOOL CALL", f"{call.name}({call.arguments})", color="yellow")
            result = registry.execute_structured(
                call.name,
                call.arguments,
                runtime=runtime,
                validation=validation,
            )
            loop_warning = loop_detector.record_and_check(call.name, call.arguments, is_error=result.is_error)
            content_for_llm = f"{result.content}\n{loop_warning}" if loop_warning else result.content

            log_step(
                f"TOOL RESULT [{result.status.upper()}]",
                f"\n{content_for_llm}",
                color=STATUS_COLOR.get(result.status, "magenta"),
            )
            runtime.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content_for_llm,
                }
            )

    return "Reached max_turns without finishing."


class ScriptedSkillsDemoLLM:
    """
    演示 Context Grounding + Skills 渐进式加载的完整调用链：
    - Turn 1: LLM 在 System Prompt 的 `<available_skills>` 中看到 `code-reviewer`，但正文尚未加载；
              于是先调用 `load_skill({"name": "code-reviewer"})` 获取详细审查规程。
    - Turn 2: 拿到 `SKILL.md` 正文后，按照规程第 1 步调用 `read_file({"path": "math_utils.py"})`。
    - Turn 3: 结合 `SKILL.md` 的三维度要求（BUGS / SECURITY / STYLE）与 `<project_rules>` 中 `AGENTS.md` 的命名规范，
              输出符合要求的结构化审查报告 `## [SKILL:CODE-REVIEW-REPORT]`。
    """

    def __init__(self) -> None:
        self.turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.turn += 1
        if self.turn == 1:
            tc_skill = ToolCall(id="call_1_skill", name="load_skill", arguments='{"name": "code-reviewer"}')
            return LLMResponse(
                content="用户要求做代码审查。我看到 `<available_skills>` 中有 `code-reviewer` 技能，先调用 `load_skill` 按需加载它的完整指令。",
                tool_calls=[tc_skill],
                raw_message={
                    "role": "assistant",
                    "content": "用户要求做代码审查。我看到 `<available_skills>` 中有 `code-reviewer` 技能，先调用 `load_skill` 按需加载它的完整指令。",
                    "tool_calls": [{"id": tc_skill.id, "type": "function", "function": {"name": tc_skill.name, "arguments": tc_skill.arguments}}],
                },
            )
        if self.turn == 2:
            tc_read = ToolCall(id="call_2_read", name="read_file", arguments='{"path": "math_utils.py"}')
            return LLMResponse(
                content="已加载 `code-reviewer` 技能正文！按照技能第 1 步要求，现在调用 `read_file` 读取 `math_utils.py`。",
                tool_calls=[tc_read],
                raw_message={
                    "role": "assistant",
                    "content": "已加载 `code-reviewer` 技能正文！按照技能第 1 步要求，现在调用 `read_file` 读取 `math_utils.py`。",
                    "tool_calls": [{"id": tc_read.id, "type": "function", "function": {"name": tc_read.name, "arguments": tc_read.arguments}}],
                },
            )

        report = (
            "## [SKILL:CODE-REVIEW-REPORT]\n"
            "1. **[BUGS]**: `DivideNumbers(a, b)` 未处理 `b == 0` 的除零异常（`ZeroDivisionError`）。\n"
            "2. **[SECURITY]**: 未发现硬编码密钥或危险调用。\n"
            "3. **[STYLE]**: 函数名 `DivideNumbers` 违反了工作区 `AGENTS.md` 中的规则（要求所有函数必须使用 `snake_case` 命名，应改为 `divide_numbers`）。"
        )
        return LLMResponse(
            content=report,
            tool_calls=[],
            raw_message={"role": "assistant", "content": report},
        )


def prepare_demo_workspace(workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "AGENTS.md").write_text(
        "- All Python functions in this workspace MUST use `snake_case` naming.\n"
        "- Division operations MUST explicitly guard against division by zero.\n",
        encoding="utf-8",
    )
    (workspace_dir / "math_utils.py").write_text(
        "def DivideNumbers(a, b):\n"
        "    return a / b\n",
        encoding="utf-8",
    )


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    skills_dir = ROOT_DIR / "skills"
    prepare_demo_workspace(workspace_dir)

    registry = create_coding_tools(base_dir=workspace_dir)
    skill_manager = SkillManager(skills_dir=skills_dir)
    skill_manager.register_load_tool(registry)
    runtime = AgentRuntime(workspace_root=workspace_dir, always_allow_dangerous=True)
    validation = create_default_validation_pipeline()

    args = sys.argv[1:]
    default_prompt = "请审查 math_utils.py 的代码质量与规范"

    if "--show-prompt" in args:
        print(build_system_prompt(runtime, skill_manager=skill_manager))
        return

    if "--demo" in args:
        log_step("DEMO", "运行 Step 6 Context Grounding (AGENTS.md) + Skills 渐进式披露加载演示", color="green")
        run_context_skills_loop(
            default_prompt,
            llm=ScriptedSkillsDemoLLM(),
            registry=registry,
            runtime=runtime,
            validation=validation,
            skill_manager=skill_manager,
        )
        return

    client = LLMClient()
    if not client.api_key:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动运行 --demo 模式。", color="yellow")
        run_context_skills_loop(
            default_prompt,
            llm=ScriptedSkillsDemoLLM(),
            registry=registry,
            runtime=runtime,
            validation=validation,
            skill_manager=skill_manager,
        )
        return

    runtime.always_allow_dangerous = False
    if args:
        run_context_skills_loop(
            " ".join(args),
            llm=client,
            registry=registry,
            runtime=runtime,
            validation=validation,
            skill_manager=skill_manager,
        )
    else:
        print(f"进入 Step 6 交互模式（工作目录：{workspace_dir}，输入 q 退出）：")
        while True:
            try:
                user_in = input("\nStep06 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in or user_in.lower() in {"q", "quit", "exit"}:
                break
            run_context_skills_loop(
                user_in,
                llm=client,
                registry=registry,
                runtime=runtime,
                validation=validation,
                skill_manager=skill_manager,
            )


if __name__ == "__main__":
    main()
