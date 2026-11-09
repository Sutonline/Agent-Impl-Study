#!/usr/bin/env python3
from __future__ import annotations

"""
Step 8: Subagent 子智能体委派与终极版 Mini Claude Code CLI
==========================================================

相比 Step 7 的演进（NEW IN STEP 8）：
1. **引入 `task(description, prompt)` 子智能体委派工具 (`harness/subagent.py`)**：
   - 当主 Agent 遇到需要翻阅多个文件做调研的子任务时，调用 `task` 工具派生一个拥有**全新独立 `messages` 列表**的 Subagent；
   - Subagent 在自己的隔离上下文里执行多轮 `read_file` / `bash` 调研；
   - 调研结束后，Subagent 的中间过程消息直接销毁，**只把最终浓缩报告返回给主 Agent**，保持主 Agent 上下文干净！
2. **整合全套 `Agent` 类 (`harness/agent.py`) 与交互式 CLI**：
   - 支持 `/help`, `/skills`, `/compact`, `/session`, `/quit` 等交互命令。

运行方式：
  1. 免 Key 离线演示（观察：主 Agent 派发 Subagent 连续读取 3 个模块找出端口配置 -> Subagent 销毁中间消息只回传报告 -> 主 Agent 修改配置）：
     python3 steps/step08_subagent_cli.py --demo
  2. 启动完整交互式 Mini Claude Code CLI：
     python3 steps/step08_subagent_cli.py
"""

import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.agent import Agent
from harness.compaction import compact_messages, estimate_tokens
from harness.llm import LLMClient, LLMResponse, ToolCall
from harness.ui import log_step


class ScriptedSubagentDemoLLM:
    """
    演示主从 Agent 协同（Parent Agent <-> Subagent）：
    - 主 Agent Turn 1：收到任务“找出哪个模块定义了 DEFAULT_PORT 并把它改成 9090”。
      为了不把所有模块文件内容塞满主上下文，主 Agent 调用 `task` 工具派发 Subagent 去搜索。
    - Subagent Turn 1..3（运行在独立的 child_runtime.messages 中）：
      依次读取 `mod_auth.py`、`mod_db.py`、`mod_server.py`，发现 `DEFAULT_PORT = 8080` 定义在 `mod_server.py`！
    - Subagent 结束：3 个文件的原始读取记录随子上下文直接销毁，仅向主 Agent 返回一句话报告。
    - 主 Agent Turn 2：拿到报告后，直接精准调用 `edit_file("mod_server.py", "8080", "9090")` 完成修改！
    """

    def __init__(self) -> None:
        self.parent_turn = 0
        self.sub_turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        sys_text = str(messages[0].get("content", "")) if messages else ""

        # 判断当前调用来自 Subagent 还是主 Agent（Subagent 的 system prompt 包含 "You are a focused Subagent"）
        if "You are a focused Subagent" in sys_text:
            return self._chat_subagent(messages)
        return self._chat_parent(messages)

    def _chat_subagent(self, messages: list[dict[str, Any]]) -> LLMResponse:
        self.sub_turn += 1
        if self.sub_turn == 1:
            tc1 = ToolCall(id="sub_read_1", name="read_file", arguments='{"path": "mod_auth.py"}')
            tc2 = ToolCall(id="sub_read_2", name="read_file", arguments='{"path": "mod_db.py"}')
            tc3 = ToolCall(id="sub_read_3", name="read_file", arguments='{"path": "mod_server.py"}')
            return LLMResponse(
                content="我是子智能体，现在在独立上下文中扫描三个模块文件以定位 DEFAULT_PORT。",
                tool_calls=[tc1, tc2, tc3],
                raw_message={
                    "role": "assistant",
                    "content": "我是子智能体，现在在独立上下文中扫描三个模块文件以定位 DEFAULT_PORT。",
                    "tool_calls": [
                        {"id": tc1.id, "type": "function", "function": {"name": tc1.name, "arguments": tc1.arguments}},
                        {"id": tc2.id, "type": "function", "function": {"name": tc2.name, "arguments": tc2.arguments}},
                        {"id": tc3.id, "type": "function", "function": {"name": tc3.name, "arguments": tc3.arguments}},
                    ],
                },
            )

        report = "已在 `mod_server.py` 第 2 行找到目标常量：`DEFAULT_PORT = 8080`（其余两个模块 `mod_auth.py` 与 `mod_db.py` 无关）。"
        return LLMResponse(
            content=report,
            tool_calls=[],
            raw_message={"role": "assistant", "content": report},
        )

    def _chat_parent(self, messages: list[dict[str, Any]]) -> LLMResponse:
        self.parent_turn += 1
        if self.parent_turn == 1:
            args = json.dumps(
                {
                    "description": "Locate DEFAULT_PORT definition",
                    "prompt": "Check mod_auth.py, mod_db.py, and mod_server.py to find where DEFAULT_PORT is defined and its exact line.",
                }
            )
            tc_task = ToolCall(id="parent_call_task_1", name="task", arguments=args)
            return LLMResponse(
                content="为了避免把多个无关模块的内容塞进主对话上下文，我先派发一个 `task` 子智能体去调研 `DEFAULT_PORT` 在哪个文件里。",
                tool_calls=[tc_task],
                raw_message={
                    "role": "assistant",
                    "content": "为了避免把多个无关模块的内容塞进主对话上下文，我先派发一个 `task` 子智能体去调研 `DEFAULT_PORT` 在哪个文件里。",
                    "tool_calls": [{"id": tc_task.id, "type": "function", "function": {"name": tc_task.name, "arguments": tc_task.arguments}}],
                },
            )

        if self.parent_turn == 2:
            args = json.dumps(
                {
                    "path": "mod_server.py",
                    "old_text": "DEFAULT_PORT = 8080",
                    "new_text": "DEFAULT_PORT = 9090",
                }
            )
            tc_edit = ToolCall(id="parent_call_edit_2", name="edit_file", arguments=args)
            return LLMResponse(
                content="收到子智能体的调研报告：`DEFAULT_PORT = 8080` 位于 `mod_server.py`。现在我直接调用 `edit_file` 将其修改为 9090。",
                tool_calls=[tc_edit],
                raw_message={
                    "role": "assistant",
                    "content": "收到子智能体的调研报告：`DEFAULT_PORT = 8080` 位于 `mod_server.py`。现在我直接调用 `edit_file` 将其修改为 9090。",
                    "tool_calls": [{"id": tc_edit.id, "type": "function", "function": {"name": tc_edit.name, "arguments": tc_edit.arguments}}],
                },
            )

        return LLMResponse(
            content="已通过子智能体定位并将 `mod_server.py` 中的 `DEFAULT_PORT` 更新为 `9090`，主上下文全程未混入无关模块源码。",
            tool_calls=[],
            raw_message={
                "role": "assistant",
                "content": "已通过子智能体定位并将 `mod_server.py` 中的 `DEFAULT_PORT` 更新为 `9090`，主上下文全程未混入无关模块源码。",
            },
        )


def prepare_demo_files(workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "mod_auth.py").write_text("# Auth module\nTOKEN_TTL = 3600\n", encoding="utf-8")
    (workspace_dir / "mod_db.py").write_text("# Database module\nPOOL_SIZE = 10\n", encoding="utf-8")
    (workspace_dir / "mod_server.py").write_text("# Server module\nDEFAULT_PORT = 8080\nHOST = '127.0.0.1'\n", encoding="utf-8")


def run_demo(workspace_dir: Path, skills_dir: Path) -> None:
    prepare_demo_files(workspace_dir)
    agent = Agent(
        workspace_root=workspace_dir,
        llm=ScriptedSubagentDemoLLM(),
        skills_dir=skills_dir,
        session_id="demo_step08",
        always_allow_dangerous=True,
    )
    prompt = "在 mod_auth.py, mod_db.py, mod_server.py 中找出哪个模块定义了 DEFAULT_PORT，并把它改成 9090"
    agent.run_turn(prompt)

    log_step(
        "CONTEXT ISOLATION CHECK",
        f"主 Agent 最终 messages 数量仅为 {len(agent.runtime.messages)} 条（子智能体读取 mod_auth.py / mod_db.py 的中间消息已彻底隔离销毁）！",
        color="green",
    )


def print_cli_help() -> None:
    print(
        "\n支持的 CLI 交互命令：\n"
        "  /help     显示此帮助信息\n"
        "  /skills   列出当前已发现的 Skills\n"
        "  /compact  立即强制压缩当前对话上下文\n"
        "  /session  显示当前会话消息数与估算 Token 数\n"
        "  /todos    显示当前 Todo 任务看板与执行模式 (Plan/Act)\n"
        "  /plan     手动切换至只读 PLAN 规划模式 (禁止写操作)\n"
        "  /act      手动切换至 ACT 执行模式 (解锁写操作)\n"
        "  /quit     退出程序\n"
    )


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    skills_dir = ROOT_DIR / "skills"
    args = sys.argv[1:]

    if "--demo" in args:
        log_step("DEMO", "运行 Step 8 Subagent 隔离上下文委派 + 完整 Agent 演示", color="green")
        run_demo(workspace_dir, skills_dir)
        return

    client = LLMClient()
    if not client.api_key:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动运行 --demo 模式。", color="yellow")
        run_demo(workspace_dir, skills_dir)
        return

    plan_flag = "--plan" in args
    filtered_args = [a for a in args if a != "--plan"]

    agent = Agent(
        workspace_root=workspace_dir,
        llm=client,
        skills_dir=skills_dir,
        session_id="step08_cli",
        enable_plan_todo=True,
        initial_mode="plan" if plan_flag else "act",
    )

    if filtered_args:
        agent.run_turn(" ".join(filtered_args))
        return

    mode_label = "PLAN" if plan_flag else "ACT"
    print(f"=== Mini Claude Code (Step 8/9) 已启动 | 工作区: {workspace_dir} | 初始模式: {mode_label} ===")
    print_cli_help()
    while True:
        try:
            mode_indicator = f"[{agent.todo_manager.mode.upper()}] " if agent.todo_manager else ""
            user_in = input(f"\nMiniClaude {mode_indicator}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_in:
            continue
        if user_in.lower() in {"q", "quit", "exit", "/quit"}:
            break
        if user_in == "/help":
            print_cli_help()
            continue
        if user_in == "/skills":
            skills = agent.skill_manager.discover() if agent.skill_manager else []
            for s in skills:
                print(f"  - {s.name}: {s.description}")
            continue
        if user_in == "/compact":
            agent.runtime.messages = compact_messages(agent.runtime.messages, llm=agent.llm, force=True)
            continue
        if user_in == "/session":
            est = estimate_tokens(agent.runtime.messages)
            print(f"  Session ID: {agent.session_id} | Messages: {len(agent.runtime.messages)} | Est. Tokens: ~{est}")
            continue
        if user_in == "/todos":
            if agent.todo_manager:
                print(f"\n当前模式: {agent.todo_manager.mode.upper()}")
                print(agent.todo_manager.render())
            else:
                print("  Plan & Todo 模式未启用。")
            continue
        if user_in == "/plan":
            if agent.todo_manager:
                agent.todo_manager.mode = "plan"
                print("  已切换为 PLAN 规划模式（只读保护已开启，修改文件将被拦截）。")
            continue
        if user_in == "/act":
            if agent.todo_manager:
                agent.todo_manager.mode = "act"
                print("  已切换为 ACT 执行模式（所有工具已解锁）。")
            continue

        agent.run_turn(user_in)


if __name__ == "__main__":
    main()
