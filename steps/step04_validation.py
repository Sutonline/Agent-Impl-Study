#!/usr/bin/env python3
from __future__ import annotations

"""
Step 4: Validation 三段式生命周期 Hook 管道 (Pre-Tool / Post-Tool / Pre-Stop Hooks)
===================================================================================

相比 Step 3 的核心演进（NEW IN STEP 4）：
1. **引入 `ValidationPipeline` 三段式拦截器**：
   - `pre_tool_use` Hooks：在工具真正落盘或弹窗前执行：
     * `schema_validator`：参数必填项与类型检查；
     * `path_sandbox_validator`：拦截跳出 `workspace_root` 的越界路径；
     * `python_ast_validator`：写入/修改 `.py` 文件前，在内存中执行 `ast.parse()`，有语法错误的坏代码直接拒绝落盘！
   - `post_tool_use` Hooks：工具执行后检查副作用（如 `non_empty_file_post_validator` 防止把代码文件清空为 0 字节）。
   - `pre_stop` Hooks（交卷前守门员）：
     * 当 LLM 不再发出 `tool_calls` 准备退出循环时，若本轮修改过代码文件，自动触发 `validation.run_pre_stop(runtime)`；
     * 若测试命令（如单测断言）失败，Harness **拒绝退出循环（veto stop）**，把报错作为新消息塞回 `messages`，倒逼 LLM 修复！

运行方式：
  1. 免 Key 离线演示（完整走通 3 次拦截自愈：越界拦截 -> 内存 AST 语法错误拦截 -> 单测不过被 pre_stop 打回重修）：
     python3 steps/step04_validation.py --demo
  2. 使用真实 OpenAI API：
     python3 steps/step04_validation.py "在 workspace 下写一个 calc.py 实现 add(a, b) 函数，并确保通过验证"
"""

import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.llm import LLMClient, LLMResponse, ToolCall
from harness.runtime import AgentRuntime
from harness.tool import ToolRegistry, create_coding_tools
from harness.ui import log_step
from harness.validation import ValidationPipeline, create_default_validation_pipeline

STATUS_COLOR = {
    "ok": "magenta",
    "error": "red",
    "denied": "red",
}


def run_validated_agent_loop(
    user_prompt: str,
    llm: Any,
    registry: ToolRegistry,
    runtime: AgentRuntime,
    validation: ValidationPipeline,
    max_turns: int = 10,
) -> str:
    """
    Step 4 的 Agent Loop：
    注意观察第 77~92 行：当 `not response.tool_calls` 时，不再无条件 `break`，
    而是必须先通过 `validation.run_pre_stop(runtime)` 守门员审批！
    """
    if not runtime.messages:
        runtime.messages.append(
            {
                "role": "system",
                "content": (
                    f"You are a coding agent working in {runtime.workspace_root}. "
                    "All tool calls and task completions are checked by an automated ValidationPipeline."
                ),
            }
        )

    runtime.messages.append({"role": "user", "content": user_prompt})
    schemas = registry.openai_schemas()

    log_step("USER", user_prompt, color="blue")

    for turn in range(1, max_turns + 1):
        log_step("LOOP", f"--- Turn {turn} ---", color="gray")
        response: LLMResponse = llm.chat(runtime.messages, tools=schemas)
        runtime.messages.append(response.raw_message)

        if response.content:
            log_step("THINK", response.content, color="cyan")

        # NEW IN STEP 4: 当 LLM 准备交卷（无 tool_calls）时，先过 pre_stop 校验守门员！
        if not response.tool_calls:
            stop_check = validation.run_pre_stop(runtime)
            if stop_check.passed:
                log_step("VALIDATION [pre_stop]", "Pre-stop verification passed! Allowing agent to finish.", color="green")
                log_step("DONE", "Task complete.", color="green")
                return response.content or ""

            # pre_stop 未通过：否决交卷（veto stop），将报错注入 messages 强制进入下一轮 Loop！
            feedback = stop_check.format_feedback("pre_stop")
            log_step("VALIDATION [pre_stop VETO]", f"阻止提前交卷！打回原因:\n{feedback}", color="red")
            runtime.messages.append({"role": "user", "content": feedback})
            continue

        for call in response.tool_calls:
            log_step("TOOL CALL", f"{call.name}({call.arguments})", color="yellow")
            # NEW IN STEP 4: 传入 validation 管道，自动执行 pre_tool_use 与 post_tool_use Hooks
            result = registry.execute_structured(
                call.name,
                call.arguments,
                runtime=runtime,
                validation=validation,
            )
            tag = "VALIDATION BLOCKED" if "VALIDATION_FAILED" in result.content else f"TOOL RESULT [{result.status.upper()}]"
            log_step(tag, f"\n{result.content}", color=STATUS_COLOR.get(result.status, "magenta"))

            runtime.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.content,
                }
            )

    return "Reached max_turns without finishing."


class ScriptedValidationDemoLLM:
    """
    演示三段式 Validation Hooks 的完整拦截与修复闭环：
    - Turn 1: LLM 试图写到工作区外 `../outside.py`，且写的 `calc.py` 漏了冒号 `def add(a, b)\n` ->
              分别被 `path_sandbox_validator` 和 `python_ast_validator` 在 `pre_tool_use` 阶段直接拦截（未落盘）！
    - Turn 2: LLM 修正了语法，但把加法写成了减法 `return a - b`（语法合法，写入成功）。
    - Turn 3: LLM 不再调工具，自信宣称“我写好了！” ->
              触发 `pre_stop` Hook 自动跑测试 `assert calc.add(2, 3) == 5`，断言失败！
              `pre_stop` 否决交卷，将 `AssertionError` 塞回对话历史！
    - Turn 4: LLM 看到 `pre_stop` 打回的报错，调用 `edit_file` 把 `a - b` 改为 `a + b`。
    - Turn 5: LLM 再次申请交卷 -> `pre_stop` 自动重跑测试通过 -> 正式结束！
    """

    def __init__(self) -> None:
        self.turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.turn += 1
        if self.turn == 1:
            tc_escape = ToolCall(
                id="call_1_escape",
                name="write_file",
                arguments=json.dumps({"path": "../outside.py", "content": "x = 1\n"}),
            )
            tc_bad_ast = ToolCall(
                id="call_1_bad_ast",
                name="write_file",
                arguments=json.dumps({"path": "calc.py", "content": "def add(a, b)\n    return a + b\n"}),
            )
            return LLMResponse(
                content="我先尝试写 ../outside.py，再写 calc.py（故意漏掉函数定义末尾的冒号）。",
                tool_calls=[tc_escape, tc_bad_ast],
                raw_message={
                    "role": "assistant",
                    "content": "我先尝试写 ../outside.py，再写 calc.py（故意漏掉函数定义末尾的冒号）。",
                    "tool_calls": [
                        {"id": tc_escape.id, "type": "function", "function": {"name": tc_escape.name, "arguments": tc_escape.arguments}},
                        {"id": tc_bad_ast.id, "type": "function", "function": {"name": tc_bad_ast.name, "arguments": tc_bad_ast.arguments}},
                    ],
                },
            )
        if self.turn == 2:
            # 语法对了，但逻辑故意写错成 a - b
            tc_write_bug = ToolCall(
                id="call_2_buggy",
                name="write_file",
                arguments=json.dumps({"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"}),
            )
            return LLMResponse(
                content="收到 pre_tool_use 的拦截提示：路径不能越界，且刚才漏了冒号。现在我写一个语法正确的 calc.py。",
                tool_calls=[tc_write_bug],
                raw_message={
                    "role": "assistant",
                    "content": "收到 pre_tool_use 的拦截提示：路径不能越界，且刚才漏了冒号。现在我写一个语法正确的 calc.py。",
                    "tool_calls": [
                        {"id": tc_write_bug.id, "type": "function", "function": {"name": tc_write_bug.name, "arguments": tc_write_bug.arguments}},
                    ],
                },
            )
        if self.turn == 3:
            # 故意不跑测试直接申请结束，触发 pre_stop 守门员拦截！
            return LLMResponse(
                content="我已经把 calc.py 写好了，任务完成！",
                tool_calls=[],
                raw_message={"role": "assistant", "content": "我已经把 calc.py 写好了，任务完成！"},
            )
        if self.turn == 4:
            tc_fix = ToolCall(
                id="call_4_fix",
                name="edit_file",
                arguments=json.dumps({"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}),
            )
            return LLMResponse(
                content="被 pre_stop 守门员打回了！测试断言 `calc.add(2, 3) == 5` 失败，因为我误写成了 `a - b`。现在用 edit_file 修复为 `a + b`。",
                tool_calls=[tc_fix],
                raw_message={
                    "role": "assistant",
                    "content": "被 pre_stop 守门员打回了！测试断言 `calc.add(2, 3) == 5` 失败，因为我误写成了 `a - b`。现在用 edit_file 修复为 `a + b`。",
                    "tool_calls": [
                        {"id": tc_fix.id, "type": "function", "function": {"name": tc_fix.name, "arguments": tc_fix.arguments}},
                    ],
                },
            )

        return LLMResponse(
            content="已将 `calc.py` 中的 `return a - b` 修复为 `return a + b`，现再次提交完成。",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "已将 `calc.py` 中的 `return a - b` 修复为 `return a + b`，现再次提交完成。"},
        )


def run_demo(workspace_dir: Path, registry: ToolRegistry) -> None:
    runtime = AgentRuntime(workspace_root=workspace_dir, always_allow_dangerous=True)
    # 配置默认三段式校验管道，并挂载 pre_stop 自动化测试命令：验证 calc.add(2, 3) == 5
    verify_cmd = "python3 -c 'import calc; assert calc.add(2, 3) == 5, f\"Expected 5, got {calc.add(2, 3)}\"'"
    validation = create_default_validation_pipeline(pre_stop_command=verify_cmd)

    prompt = "在 workspace 下编写 calc.py 实现 add(a, b) 函数，要求通过单测验证"
    run_validated_agent_loop(
        prompt,
        llm=ScriptedValidationDemoLLM(),
        registry=registry,
        runtime=runtime,
        validation=validation,
    )


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    registry = create_coding_tools(base_dir=workspace_dir)
    args = sys.argv[1:]

    if "--demo" in args:
        log_step("DEMO", "运行 Step 4 Validation 三段式生命周期 Hook 管道演示", color="green")
        run_demo(workspace_dir, registry)
        return

    client = LLMClient()
    if not client.api_key:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动运行 --demo 模式。", color="yellow")
        run_demo(workspace_dir, registry)
        return

    runtime = AgentRuntime(workspace_root=workspace_dir)
    validation = create_default_validation_pipeline()
    if args:
        run_validated_agent_loop(" ".join(args), llm=client, registry=registry, runtime=runtime, validation=validation)
    else:
        print(f"进入 Step 4 交互模式（工作目录：{workspace_dir}，输入 q 退出）：")
        while True:
            try:
                user_in = input("\nStep04 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in or user_in.lower() in {"q", "quit", "exit"}:
                break
            run_validated_agent_loop(user_in, llm=client, registry=registry, runtime=runtime, validation=validation)


if __name__ == "__main__":
    main()
