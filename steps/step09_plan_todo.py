#!/usr/bin/env python3
from __future__ import annotations

"""
Step 9 可独立运行演示脚本：Plan & Todo 模式 (steps/step09_plan_todo.py)

演示目标：
直观展示 Claude Code 的 `Plan / Act` 双阶段隔离与 `TodoWrite` 进度守门员是如何通过 `ValidationPipeline` 运作的：
1. 初始进入 `mode="plan"`；
2. 故意让 LLM 在未完成规划时直接调用 `edit_file` -> 被 `make_plan_mode_pre_tool_hook` (`pre_tool_use`) 当场拦截 (`DENIED`)；
3. LLM 改为先调用 `read_file` 查看源码，调用 `todo_write` 拆解出 2 个步骤，并调用 `exit_plan_mode` 切换至 `ACT` 模式；
4. LLM 完成第 1 步（调用 `edit_file` 修改代码）后，故意忘记执行第 2 步就直接退出（不发 `tool_calls`）；
5. `make_todo_pre_stop_hook` (`pre_stop`) 检测到还有未完成的 Todo 项，当场否决停止 (`VETO`)，逼迫 LLM 跑完第 2 步并将全部 Todo 标记为 `completed` 后才准许交卷。

运行方式：
  1. 离线全链路剧本演示（推荐先跑这个）：
     python3 steps/step09_plan_todo.py --demo

  2. 真实 LLM Plan + Todo 模式运行：
     python3 steps/step09_plan_todo.py "请先阅读 pricing.py，制定包含修改税率和运行验证两步的 Todo 计划，然后执行修改"
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.agent import Agent
from harness.llm import LLMClient, LLMResponse, ToolCall
from harness.ui import log_step


class ScriptedPlanTodoLLM:
    """演示专用剧本式 LLM：完整触发 Plan 拦截 -> 规划建账 -> 切换 Act -> 提前交卷被 pre_stop 拦截 -> 完结 Todo。"""

    def __init__(self) -> None:
        self.turn = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.turn += 1

        if self.turn == 1:
            # 故意在 PLAN 模式下直接改代码 -> 触发 pre_tool_use PLAN_MODE 拦截
            tc = ToolCall(
                id="call_plan_1",
                name="edit_file",
                arguments=json.dumps(
                    {
                        "path": "pricing.py",
                        "old_text": "TAX_RATE = 0.05",
                        "new_text": "TAX_RATE = 0.08",
                    }
                ),
            )
            return LLMResponse(
                content="我直接把 `pricing.py` 里的税率改成 0.08。",
                tool_calls=[tc],
                raw_message={
                    "role": "assistant",
                    "content": "我直接把 `pricing.py` 里的税率改成 0.08。",
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
            # 收到 PLAN_MODE 拦截提示后，按规矩读取文件、建立 Todo 列表并申请退出 PLAN 模式
            t1 = ToolCall(
                id="call_plan_2a",
                name="read_file",
                arguments=json.dumps({"path": "pricing.py"}),
            )
            t2 = ToolCall(
                id="call_plan_2b",
                name="todo_write",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "id": "1",
                                "content": "Update TAX_RATE to 0.08 in pricing.py",
                                "status": "in_progress",
                            },
                            {
                                "id": "2",
                                "content": "Run python3 pricing.py to verify output",
                                "status": "pending",
                            },
                        ]
                    }
                ),
            )
            t3 = ToolCall(
                id="call_plan_2c",
                name="exit_plan_mode",
                arguments=json.dumps(
                    {
                        "plan_summary": "1. Change TAX_RATE from 0.05 to 0.08 in pricing.py; 2. Run pricing.py to verify calculation."
                    }
                ),
            )
            calls = [t1, t2, t3]
            return LLMResponse(
                content="当前处于 `PLAN` 模式，不能直接写文件。我先读取 `pricing.py`，用 `todo_write` 拆解两步计划，再调用 `exit_plan_mode` 切换到 `ACT` 模式。",
                tool_calls=calls,
                raw_message={
                    "role": "assistant",
                    "content": "先完成代码阅读与 Todo 拆解，并申请进入 ACT 模式。",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in calls
                    ],
                },
            )

        if self.turn == 3:
            # 进入 ACT 模式后，修改文件并把 Todo #1 标为 completed、Todo #2 标为 in_progress
            t1 = ToolCall(
                id="call_act_3a",
                name="edit_file",
                arguments=json.dumps(
                    {
                        "path": "pricing.py",
                        "old_text": "TAX_RATE = 0.05",
                        "new_text": "TAX_RATE = 0.08",
                    }
                ),
            )
            t2 = ToolCall(
                id="call_act_3b",
                name="todo_write",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "id": "1",
                                "content": "Update TAX_RATE to 0.08 in pricing.py",
                                "status": "completed",
                            },
                            {
                                "id": "2",
                                "content": "Run python3 pricing.py to verify output",
                                "status": "in_progress",
                            },
                        ]
                    }
                ),
            )
            calls = [t1, t2]
            return LLMResponse(
                content="已进入 `ACT` 模式！现在执行 Todo #1 修改 `pricing.py`，并将进度更新到 Todo #2。",
                tool_calls=calls,
                raw_message={
                    "role": "assistant",
                    "content": "执行 Todo #1 修改文件并更新 Todo 看板。",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in calls
                    ],
                },
            )

        if self.turn == 4:
            # 故意在 Todo #2 仍为 in_progress 时停止调用工具，试图提前交卷 -> 触发 pre_stop Todo 守门员 VETO
            msg = "我已经改好 `pricing.py` 了，任务完成！"
            return LLMResponse(
                content=msg,
                tool_calls=[],
                raw_message={"role": "assistant", "content": msg},
            )

        if self.turn == 5:
            # 被 pre_stop 守门员打回后，老老实实运行验证命令并将全部 Todo 标记为 completed
            t1 = ToolCall(
                id="call_act_5a",
                name="bash",
                arguments=json.dumps({"command": "python3 pricing.py"}),
            )
            t2 = ToolCall(
                id="call_act_5b",
                name="todo_write",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "id": "1",
                                "content": "Update TAX_RATE to 0.08 in pricing.py",
                                "status": "completed",
                            },
                            {
                                "id": "2",
                                "content": "Run python3 pricing.py to verify output",
                                "status": "completed",
                            },
                        ]
                    }
                ),
            )
            calls = [t1, t2]
            return LLMResponse(
                content="收到 `pre_stop` 守门员拦截：Todo #2 尚未完成。现在立即运行 `python3 pricing.py` 验证，并将所有 Todo 项更新为 `completed`。",
                tool_calls=calls,
                raw_message={
                    "role": "assistant",
                    "content": "补做 Todo #2 并更新全部状态为 completed。",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in calls
                    ],
                },
            )

        final_msg = "所有 Todo 步骤已全部执行并验证通过：`pricing.py` 税率已更新为 `0.08`，计算输出 `108.0`。"
        return LLMResponse(
            content=final_msg,
            tool_calls=[],
            raw_message={"role": "assistant", "content": final_msg},
        )


def run_demo() -> None:
    log_step("DEMO", "运行 Step 9 Plan & Todo 模式（读写阶段隔离 + 完结守门员）演示", color="magenta")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "pricing.py").write_text(
            "TAX_RATE = 0.05\n"
            "def total_with_tax(price: float) -> float:\n"
            "    return round(price * (1 + TAX_RATE), 2)\n"
            "if __name__ == '__main__':\n"
            "    print(total_with_tax(100))\n",
            encoding="utf-8",
        )

        agent = Agent(
            workspace_root=ws,
            llm=ScriptedPlanTodoLLM(),
            always_allow_dangerous=True,
            enable_plan_todo=True,
            initial_mode="plan",
        )
        agent.run_turn("把 pricing.py 中的 TAX_RATE 更新为 0.08 并运行脚本验证结果")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 09: Plan & Todo Mode")
    parser.add_argument("prompt", nargs="?", default=None, help="User prompt")
    parser.add_argument("--demo", action="store_true", help="Run deterministic offline demo")
    parser.add_argument(
        "--mode",
        choices=["plan", "act"],
        default="plan",
        help="Initial agent mode (default: plan)",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    client = LLMClient()
    if not client.api_key:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动切换到 --demo 模式。", color="yellow")
        run_demo()
        return

    ws = ROOT_DIR / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    agent = Agent(
        workspace_root=ws,
        llm=client,
        skills_dir=ROOT_DIR / "skills",
        sessions_dir=ROOT_DIR / ".sessions",
        enable_plan_todo=True,
        initial_mode=args.mode,  # type: ignore[arg-type]
    )
    prompt = args.prompt or "请先在 PLAN 模式下阅读当前工作区文件并建立 Todo 计划，再进入 ACT 模式执行。"
    agent.run_turn(prompt)


if __name__ == "__main__":
    main()
