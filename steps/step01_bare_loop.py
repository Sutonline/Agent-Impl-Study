#!/usr/bin/env python3
from __future__ import annotations

"""
Step 1: 裸 LLM 与最小控制循环 (The Bare Loop)
==============================================

第一性原理公式：
    Agent = LLM + Harness
在 Step 1 中，我们的 Harness 只有最核心的一样东西：一个 `while True` 循环！

为什么需要这个循环？
- 裸 LLM 本质上只是一个无状态函数：`messages -> response`。
- 当你问它“现在几点了？如果分钟数是偶数请帮我算 123 * 456”，LLM 自己既没有手表，也不会按计算器。
- LLM 只能输出一个“意图”（`tool_calls`）：“我想调用 `get_current_time`”。
- **Harness 的职责**：
  1. 接住这个 `tool_calls` 意图；
  2. 在真实世界里执行对应的 Python 函数，拿到结果；
  3. 把结果包装成 `role="tool"` 的消息追加进 `messages`；
  4. **再次调用 LLM**（进入下一轮 while 循环），直到 LLM 不再请求工具（`not response.tool_calls`），直接给出最终回答！

运行方式：
  1. 免 Key 离线演示（直观观察多轮 Loop 消息流转）：
     python3 steps/step01_bare_loop.py --demo
  2. 使用真实 OpenAI API 运行：
     export OPENAI_API_KEY="sk-..."
     python3 steps/step01_bare_loop.py "现在几点了？顺便帮我算一下 (42 * 1337) + 99"
  3. 交互模式：
     python3 steps/step01_bare_loop.py
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

# 允许直接运行 `python3 steps/step01_bare_loop.py` 时找到根目录下的 `harness` 包
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.llm import LLMClient, LLMResponse, ToolCall
from harness.ui import log_step

# ============================================================================
# 1. 硬编码的最小工具 Schema 与执行函数（Step 2 我们会用 @tool 装饰器消除这些手写样板代码）
# ============================================================================

NAIVE_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前系统本地时间（格式：YYYY-MM-DD HH:MM:SS）",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算一个四则运算数学表达式并返回精确结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 '(42 * 1337) + 99'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def execute_naive_tool(call: ToolCall) -> str:
    """最原始的工具分发器：用 if/elif 手工解析 JSON 并调用对应逻辑。"""
    args = json.loads(call.arguments or "{}")

    if call.name == "get_current_time":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if call.name == "calculate":
        expr = str(args.get("expression", ""))
        # 仅允许安全数学字符
        allowed = set("0123456789+-*/(). %")
        if not expr or any(ch not in allowed for ch in expr):
            return f"Error: unsafe or invalid expression: {expr}"
        return str(eval(expr, {"__builtins__": {}}, {}))

    return f"Error: unknown tool '{call.name}'"


# ============================================================================
# 2. Agent 的心脏：The Loop（几十行代码看清 Agent 本质）
# ============================================================================


def run_bare_loop(user_prompt: str, llm: Any, max_turns: int = 10) -> str:
    """
    最小 Agent 控制循环：
    不断在 [调用 LLM] -> [执行 Tool] -> [追加结果到 messages] 之间循环，
    直到 LLM 不再返回 tool_calls。
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided tools whenever needed.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    log_step("USER", user_prompt, color="blue")

    for turn in range(1, max_turns + 1):
        log_step("LOOP", f"--- Turn {turn} (messages count: {len(messages)}) ---", color="gray")

        # 1) 把当前完整对话历史 + 工具说明书发给 LLM
        response: LLMResponse = llm.chat(messages, tools=NAIVE_TOOLS_SCHEMA)

        # 2) 把 LLM 的回复原样加入消息历史（无论它是普通回答还是包含 tool_calls）
        messages.append(response.raw_message)

        if response.content:
            log_step("THINK", response.content, color="cyan")

        # 3) 终止条件：如果 LLM 没有发起任何工具调用，说明它认为任务已经完成！
        if not response.tool_calls:
            log_step("DONE", "LLM returned final answer without tool_calls. Loop finished!", color="green")
            return response.content or ""

        # 4) 如果 LLM 发起了 tool_calls，Harness 代它执行每一个工具，并把结果塞回 messages
        for call in response.tool_calls:
            log_step("TOOL CALL", f"{call.name}({call.arguments})", color="yellow")
            result_str = execute_naive_tool(call)
            log_step("TOOL RESULT", f"-> {result_str}", color="magenta")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_str,
                }
            )

    return "Reached max_turns without finishing."


# ============================================================================
# 3. 命令行与内置模拟 Demo 入口
# ============================================================================


class ScriptedDemoLLM:
    """当未配置 OPENAI_API_KEY 或使用 --demo 时，演示一个两轮工具调用的完整消息流。"""

    def __init__(self) -> None:
        self.turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.turn += 1
        if self.turn == 1:
            tc = ToolCall(id="call_time_01", name="get_current_time", arguments="{}")
            return LLMResponse(
                content="我先调用 get_current_time 看看现在几点。",
                tool_calls=[tc],
                raw_message={
                    "role": "assistant",
                    "content": "我先调用 get_current_time 看看现在几点。",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                    ],
                },
            )
        if self.turn == 2:
            tc = ToolCall(id="call_calc_02", name="calculate", arguments='{"expression": "(42 * 1337) + 99"}')
            return LLMResponse(
                content="已经拿到了当前时间，接下来我调用 calculate 计算 (42 * 1337) + 99。",
                tool_calls=[tc],
                raw_message={
                    "role": "assistant",
                    "content": "已经拿到了当前时间，接下来我调用 calculate 计算 (42 * 1337) + 99。",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                    ],
                },
            )
        # 第 3 轮：读取前面两轮塞进 messages 的 tool 结果，生成最终回复
        time_obs = next((m["content"] for m in messages if m.get("tool_call_id") == "call_time_01"), "")
        calc_obs = next((m["content"] for m in messages if m.get("tool_call_id") == "call_calc_02"), "")
        final_text = f"当前系统时间是 {time_obs}，且 (42 * 1337) + 99 的精确计算结果是 {calc_obs}。"
        return LLMResponse(
            content=final_text,
            tool_calls=[],
            raw_message={"role": "assistant", "content": final_text},
        )


def main() -> None:
    args = sys.argv[1:]
    prompt = "现在几点了？顺便帮我算一下 (42 * 1337) + 99"

    if "--demo" in args:
        log_step("DEMO", "运行内置多轮工具调用演示（如果配置了 .env 也可不带 --demo 直接调用真实 OpenAI API）", color="green")
        run_bare_loop(prompt, llm=ScriptedDemoLLM())
        return

    client = LLMClient()
    if not client.api_key:
        log_step(
            "INFO",
            "未检测到 OPENAI_API_KEY，自动切换到 --demo 模拟演示模式（配置 .env 后即可连接真实 OpenAI 模型）。",
            color="yellow",
        )
        run_bare_loop(prompt, llm=ScriptedDemoLLM())
        return

    if args:
        run_bare_loop(" ".join(args), llm=client)
    else:
        print("进入 Step 1 交互模式（输入 q 退出）：")
        while True:
            try:
                user_in = input("\nStep01 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in or user_in.lower() in {"q", "quit", "exit"}:
                break
            run_bare_loop(user_in, llm=client)


if __name__ == "__main__":
    main()
