#!/usr/bin/env python3
from __future__ import annotations

"""
Step 2: Tool 抽象、注册中心与四大编码工具 (Tool System)
======================================================

相比 Step 1 的进化（NEW IN STEP 2）：
1. **消除手写 JSON Schema**：通过 `@tool` 装饰器 + Python `inspect.signature` 自动从函数签名生成 OpenAI Schema。
2. **消除 if/elif 分发**：通过 `ToolRegistry.execute(name, raw_arguments)` 统一完成 JSON 解析、异常捕获与函数调用。
3. **赋予 Coding Agent 真正的“手和眼”**：
   - `read_file`  （读代码，带行号）
   - `write_file` （创建/覆写代码文件）
   - `edit_file`  （局部精确替换代码）
   - `bash`       （运行代码或测试命令）

运行方式：
  1. 免 Key 离线演示（观察 Agent 自动写代码 -> 改代码 -> 跑代码的全过程）：
     python3 steps/step02_tools.py --demo
  2. 使用真实 OpenAI API 在 workspace/ 下写代码并执行：
     python3 steps/step02_tools.py "在 workspace 下写一个 fib.py 计算前 10 个斐波那契数，并运行它告诉我输出"
"""

import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.llm import LLMClient, LLMResponse, ToolCall
from harness.tool import ToolRegistry, create_coding_tools
from harness.ui import log_step


def run_tool_agent_loop(
    user_prompt: str,
    llm: Any,
    registry: ToolRegistry,
    max_turns: int = 10,
) -> str:
    """
    Step 2 的 Agent Loop：
    对比 Step 1，主循环代码完全不再关心具体有哪些工具！
    所有工具 Schema 来自 `registry.openai_schemas()`，所有执行交给 `registry.execute()`。
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a coding agent working in a workspace directory. "
                "Use read_file, write_file, edit_file, and bash to solve tasks and verify your code."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    schemas = registry.openai_schemas()
    log_step("USER", user_prompt, color="blue")
    log_step("TOOLS", f"Loaded {len(schemas)} tools: {[t.name for t in registry.list_tools()]}", color="gray")

    for turn in range(1, max_turns + 1):
        log_step("LOOP", f"--- Turn {turn} ---", color="gray")
        response: LLMResponse = llm.chat(messages, tools=schemas)
        messages.append(response.raw_message)

        if response.content:
            log_step("THINK", response.content, color="cyan")

        if not response.tool_calls:
            log_step("DONE", "Agent finished task.", color="green")
            return response.content or ""

        for call in response.tool_calls:
            log_step("TOOL CALL", f"{call.name}({call.arguments})", color="yellow")
            # NEW IN STEP 2: 统一通过 ToolRegistry 执行，不再手写 if/elif
            result_str = registry.execute(call.name, call.arguments)
            log_step("TOOL RESULT", f"\n{result_str}", color="magenta")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_str,
                }
            )

    return "Reached max_turns without finishing."


class ScriptedCodingDemoLLM:
    """演示 Agent 使用四大编码工具：write_file -> edit_file -> read_file -> bash 运行验证。"""

    def __init__(self) -> None:
        self.turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.turn += 1
        if self.turn == 1:
            args = json.dumps(
                {
                    "path": "fib.py",
                    "content": (
                        "def fib(n):\n"
                        "    a, b = 0, 1\n"
                        "    res = []\n"
                        "    for _ in range(n):\n"
                        "        res.append(a)\n"
                        "        a, b = b, a + b\n"
                        "    return res\n\n"
                        "if __name__ == '__main__':\n"
                        "    print('FIB_5:', fib(5))\n"
                    ),
                }
            )
            tc = ToolCall(id="call_write_1", name="write_file", arguments=args)
            return LLMResponse(
                content="我先用 write_file 创建 fib.py 文件。",
                tool_calls=[tc],
                raw_message={
                    "role": "assistant",
                    "content": "我先用 write_file 创建 fib.py 文件。",
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}],
                },
            )
        if self.turn == 2:
            args = json.dumps(
                {
                    "path": "fib.py",
                    "old_text": "print('FIB_5:', fib(5))",
                    "new_text": "print('FIB_10:', fib(10))",
                }
            )
            tc = ToolCall(id="call_edit_2", name="edit_file", arguments=args)
            return LLMResponse(
                content="用户要求计算前 10 个斐波那契数，我用 edit_file 把 fib(5) 精确修改为 fib(10)。",
                tool_calls=[tc],
                raw_message={
                    "role": "assistant",
                    "content": "用户要求计算前 10 个斐波那契数，我用 edit_file 把 fib(5) 精确修改为 fib(10)。",
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}],
                },
            )
        if self.turn == 3:
            tc_read = ToolCall(id="call_read_3", name="read_file", arguments='{"path": "fib.py"}')
            tc_bash = ToolCall(id="call_bash_3", name="bash", arguments='{"command": "python3 fib.py"}')
            return LLMResponse(
                content="现在我读取确认一下修改后的 fib.py，并用 bash 运行它验证输出。",
                tool_calls=[tc_read, tc_bash],
                raw_message={
                    "role": "assistant",
                    "content": "现在我读取确认一下修改后的 fib.py，并用 bash 运行它验证输出。",
                    "tool_calls": [
                        {"id": tc_read.id, "type": "function", "function": {"name": tc_read.name, "arguments": tc_read.arguments}},
                        {"id": tc_bash.id, "type": "function", "function": {"name": tc_bash.name, "arguments": tc_bash.arguments}},
                    ],
                },
            )

        bash_obs = next((m["content"] for m in messages if m.get("tool_call_id") == "call_bash_3"), "")
        final_msg = f"我已经创建并修改了 `workspace/fib.py`，运行结果如下：\n{bash_obs}"
        return LLMResponse(
            content=final_msg,
            tool_calls=[],
            raw_message={"role": "assistant", "content": final_msg},
        )


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    registry = create_coding_tools(base_dir=workspace_dir)
    args = sys.argv[1:]
    default_prompt = "在 workspace 下写一个 fib.py 计算前 10 个斐波那契数，并运行它告诉我输出"

    if "--show-schema" in args:
        print(json.dumps(registry.openai_schemas(), indent=2, ensure_ascii=False))
        return

    if "--demo" in args:
        log_step("DEMO", f"在 {workspace_dir} 下演示四大编码工具 (write_file -> edit_file -> read_file & bash)", color="green")
        run_tool_agent_loop(default_prompt, llm=ScriptedCodingDemoLLM(), registry=registry)
        return

    client = LLMClient()
    if not client.api_key:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动运行 --demo 模式。", color="yellow")
        run_tool_agent_loop(default_prompt, llm=ScriptedCodingDemoLLM(), registry=registry)
        return

    if args:
        run_tool_agent_loop(" ".join(args), llm=client, registry=registry)
    else:
        print(f"进入 Step 2 交互模式（工作目录：{workspace_dir}，输入 q 退出）：")
        while True:
            try:
                user_in = input("\nStep02 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in or user_in.lower() in {"q", "quit", "exit"}:
                break
            run_tool_agent_loop(user_in, llm=client, registry=registry)


if __name__ == "__main__":
    main()
