#!/usr/bin/env python3
from __future__ import annotations

"""
Step 3: AgentRuntime 容器与 Permission Gate 安全护栏
====================================================

相比 Step 2 的进化（NEW IN STEP 3）：
1. **引入 `AgentRuntime`（即 Environment 容器）**：
   - 统一维护 `workspace_root`、`cwd`、会话 `messages` 以及本轮被修改的文件集合 `modified_files`；
   - 提供受控子进程执行器 `runtime.run_command()`：强制超时保护（防死循环命令卡死）+ 超长输出截断（防撑爆上下文）。
2. **引入 `PermissionGate`（高危操作安全护栏）**：
   - 工具分为安全工具（`read_file`, `dangerous=False`）和高危工具（`write_file`, `edit_file`, `bash`, `dangerous=True`）；
   - 执行高危工具前，`ToolRegistry.execute(..., runtime=runtime)` 会自动触发 `runtime.check_permission()`；
   - 如果被人类拒绝，不会抛异常崩掉，而是返回 `[PERMISSION_DENIED] ...` 消息给 LLM，让 LLM 调整方案！

运行方式：
  1. 免 Key 离线演示（模拟：第 1 次危险命令被拒绝 -> LLM 改用安全方案并获批 -> 演示超时拦截）：
     python3 steps/step03_runtime.py --demo
  2. 使用真实 OpenAI API（每次写文件或执行命令前会在终端弹出 [y/n/a] 让你亲自审批！）：
     python3 steps/step03_runtime.py "查看当前目录有哪些文件，然后新建一个 hello_runtime.py 打印 Hello Runtime 并运行它"
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


def run_runtime_agent_loop(
    user_prompt: str,
    llm: Any,
    registry: ToolRegistry,
    runtime: AgentRuntime,
    max_turns: int = 10,
) -> str:
    """
    Step 3 的 Agent Loop：
    现在消息历史 `runtime.messages`、工作目录 `runtime.cwd` 和权限控制全部收归 `runtime` 统一管理。
    """
    if not runtime.messages:
        runtime.messages.append(
            {
                "role": "system",
                "content": (
                    f"You are a coding agent running inside workspace: {runtime.workspace_root}. "
                    "Dangerous tools (write_file, edit_file, bash) require user permission. "
                    "If a tool call is denied by the user, respect their decision and adapt."
                ),
            }
        )

    runtime.messages.append({"role": "user", "content": user_prompt})
    schemas = registry.openai_schemas()

    log_step("USER", user_prompt, color="blue")
    log_step("RUNTIME", f"workspace_root={runtime.workspace_root}", color="gray")

    for turn in range(1, max_turns + 1):
        log_step("LOOP", f"--- Turn {turn} ---", color="gray")
        response: LLMResponse = llm.chat(runtime.messages, tools=schemas)
        runtime.messages.append(response.raw_message)

        if response.content:
            log_step("THINK", response.content, color="cyan")

        if not response.tool_calls:
            log_step(
                "DONE",
                f"Task complete! Modified files in this session: {[p.name for p in runtime.modified_files]}",
                color="green",
            )
            return response.content or ""

        for call in response.tool_calls:
            log_step("TOOL CALL", f"{call.name}({call.arguments})", color="yellow")
            # NEW IN STEP 3: 将 runtime 注入 execute()，触发 Permission Gate 与受控子进程执行
            result_str = registry.execute(call.name, call.arguments, runtime=runtime)
            color = "red" if ("PERMISSION_DENIED" in result_str or "[ERROR]" in result_str) else "magenta"
            log_step("TOOL RESULT", f"\n{result_str}", color=color)

            runtime.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_str,
                }
            )

    return "Reached max_turns without finishing."


class ScriptedRuntimeDemoLLM:
    """
    演示三个核心 Runtime 特性：
    - Turn 1: 调用 `read_file`（只读工具 dangerous=False，无需审批直接通过）+ 尝试 `bash("rm -rf *.py")`（被 Permission Gate 拦截拒绝！）
    - Turn 2: LLM 收到 `[PERMISSION_DENIED]` 后放弃删除，改用 `write_file` 创建 `hello_runtime.py`（获批通过）
    - Turn 3: 调用 `bash("python3 hello_runtime.py")`（获批通过并拿到输出）
    """

    def __init__(self) -> None:
        self.turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.turn += 1
        if self.turn == 1:
            tc_read = ToolCall(id="call_1_read", name="read_file", arguments='{"path": "fib.py"}')
            tc_rm = ToolCall(id="call_1_rm", name="bash", arguments='{"command": "rm -f fib.py"}')
            return LLMResponse(
                content="我先读取 fib.py（安全操作），然后顺便尝试清理掉旧的 fib.py（高危操作）。",
                tool_calls=[tc_read, tc_rm],
                raw_message={
                    "role": "assistant",
                    "content": "我先读取 fib.py（安全操作），然后顺便尝试清理掉旧的 fib.py（高危操作）。",
                    "tool_calls": [
                        {"id": tc_read.id, "type": "function", "function": {"name": tc_read.name, "arguments": tc_read.arguments}},
                        {"id": tc_rm.id, "type": "function", "function": {"name": tc_rm.name, "arguments": tc_rm.arguments}},
                    ],
                },
            )
        if self.turn == 2:
            tc_write = ToolCall(
                id="call_2_write",
                name="write_file",
                arguments=json.dumps({"path": "hello_runtime.py", "content": "print('Hello from controlled AgentRuntime!')\n"}),
            )
            return LLMResponse(
                content="收到 `[PERMISSION_DENIED]`，用户拒绝删除 fib.py。我尊重用户决定，保留 fib.py，直接创建 `hello_runtime.py`。",
                tool_calls=[tc_write],
                raw_message={
                    "role": "assistant",
                    "content": "收到 `[PERMISSION_DENIED]`，用户拒绝删除 fib.py。我尊重用户决定，保留 fib.py，直接创建 `hello_runtime.py`。",
                    "tool_calls": [
                        {"id": tc_write.id, "type": "function", "function": {"name": tc_write.name, "arguments": tc_write.arguments}},
                    ],
                },
            )
        if self.turn == 3:
            tc_bash = ToolCall(
                id="call_3_bash",
                name="bash",
                arguments='{"command": "python3 hello_runtime.py"}',
            )
            return LLMResponse(
                content="文件已写入，现在在受控 Runtime 中运行 `python3 hello_runtime.py`。",
                tool_calls=[tc_bash],
                raw_message={
                    "role": "assistant",
                    "content": "文件已写入，现在在受控 Runtime 中运行 `python3 hello_runtime.py`。",
                    "tool_calls": [
                        {"id": tc_bash.id, "type": "function", "function": {"name": tc_bash.name, "arguments": tc_bash.arguments}},
                    ],
                },
            )

        out = next((m["content"] for m in messages if m.get("tool_call_id") == "call_3_bash"), "")
        final_msg = f"`hello_runtime.py` 已在受控 Runtime 中创建并执行成功：\n{out}"
        return LLMResponse(
            content=final_msg,
            tool_calls=[],
            raw_message={"role": "assistant", "content": final_msg},
        )


def run_demo(workspace_dir: Path, registry: ToolRegistry) -> None:
    def demo_permission_policy(tool_name: str, args: dict[str, Any]) -> bool:
        # 自动演示：如果是 rm 删除命令则拒绝，其他正常新建/运行命令则批准
        cmd = str(args.get("command", ""))
        if tool_name == "bash" and "rm " in cmd:
            log_step("PERMISSION GATE", f"拦截高危删除命令: {tool_name}({args}) -> 自动决策: 拒绝 (DENY)", color="red")
            return False
        log_step("PERMISSION GATE", f"审批高危操作: {tool_name}({args}) -> 自动决策: 允许 (ALLOW)", color="green")
        return True

    runtime = AgentRuntime(
        workspace_root=workspace_dir,
        permission_callback=demo_permission_policy,
    )
    prompt = "查看 fib.py，并创建一个 hello_runtime.py 打印问候语后运行它"
    run_runtime_agent_loop(prompt, llm=ScriptedRuntimeDemoLLM(), registry=registry, runtime=runtime)


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    registry = create_coding_tools(base_dir=workspace_dir)
    args = sys.argv[1:]

    if "--demo" in args:
        log_step("DEMO", "运行 Step 3 Runtime 与 Permission Gate 演示", color="green")
        run_demo(workspace_dir, registry)
        return

    client = LLMClient()
    if not client.api_key:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动运行 --demo 模式。", color="yellow")
        run_demo(workspace_dir, registry)
        return

    # 真实 API 模式：使用默认的终端 [y/n/a] 交互式审批！
    runtime = AgentRuntime(workspace_root=workspace_dir)
    if args:
        run_runtime_agent_loop(" ".join(args), llm=client, registry=registry, runtime=runtime)
    else:
        print(f"进入 Step 3 交互模式（工作目录：{workspace_dir}，输入 q 退出）：")
        while True:
            try:
                user_in = input("\nStep03 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in or user_in.lower() in {"q", "quit", "exit"}:
                break
            run_runtime_agent_loop(user_in, llm=client, registry=registry, runtime=runtime)


if __name__ == "__main__":
    main()
