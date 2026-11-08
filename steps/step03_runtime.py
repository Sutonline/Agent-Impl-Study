#!/usr/bin/env python3
from __future__ import annotations

"""
Step 3: AgentRuntime（环境隔离沙盒 + 细粒度命令前缀权限引擎 + 结构化状态）
==========================================================================

相比 Step 2 的三大核心进化（NEW IN STEP 3）：
1. **沙盒隔离（Sandbox Isolation）**：
   - **环境变量净室隔离**：`runtime.build_isolated_env()` 自动剥离宿主机的 `OPENAI_API_KEY` 等敏感机密，子进程跑 `env` 也绝不会泄露你的 API Key；
   - **文件系统边界隔离（Filesystem Jail）**：`runtime.resolve_path()` 强制校验路径不得跳出 `workspace_root`；
   - **受控子进程**：强制超时保护（`timeout`）与超长输出截断（`max_output_chars`），并支持生成 macOS `sandbox-exec` 内核级沙盒策略。
2. **工业级命令拆解与前缀权限引擎（Command Prefix Permission Gate）**：
   - 先用 `split_shell_commands()` 按 `&&`, `||`, `;`, `|` 拆解复合命令，用 `has_dangerous_shell_syntax()` 拦截 `$()` 与重定向；
   - 提取 `(二进制 + 子命令)` 前缀（如 `ls`, `git status`, `python3 -m unittest`）：
     * 只读白名单命令（如 `ls -la && git status`）无需弹窗，自动静默放行；
     * 若混入高危子命令（如 `ls -la && rm -f fib.py`），整条拦截！
     * 当用户输入 `a` 记住权限时，仅将当前安全的命令前缀（如 `python3`）加入 `allowed_command_prefixes`，绝不放行 `rm`。
3. **结构化执行结果（`ToolExecutionResult`）**：
   - 主循环通过 `result.status`（`"ok" | "error" | "denied"`）精确识别工具状态，彻底告别靠字符串包含判断意图。

运行方式：
  1. 免 Key 离线演示（完整演示：只读命令免批放行 -> 复合命令混入 rm 被拦截 -> 环境变量隔离防 Key 泄露）：
     python3 steps/step03_runtime.py --demo
  2. 使用真实 OpenAI API 交互体验：
     python3 steps/step03_runtime.py "先用 ls 看看有哪些文件，然后写一个 hello_runtime.py 并运行它"
"""

import json
import os
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

STATUS_COLOR = {
    "ok": "magenta",
    "error": "red",
    "denied": "red",
}


def run_runtime_agent_loop(
    user_prompt: str,
    llm: Any,
    registry: ToolRegistry,
    runtime: AgentRuntime,
    max_turns: int = 10,
) -> str:
    """
    Step 3 的 Agent Loop：
    1. 使用 `registry.execute_structured(..., runtime=runtime)` 拿到结构化的 `ToolExecutionResult`；
    2. 根据 `result.status`（"ok" / "error" / "denied"）精确处理状态，再把 `result.content` 塞进 `runtime.messages`。
    """
    if not runtime.messages:
        runtime.messages.append(
            {
                "role": "system",
                "content": (
                    f"You are a coding agent running inside an isolated workspace: {runtime.workspace_root}. "
                    "Read-only commands (like ls, pwd, git status) are auto-approved. "
                    "Write operations and unapproved commands require user permission. "
                    "If a tool call is denied, respect the user's decision and adapt."
                ),
            }
        )

    runtime.messages.append({"role": "user", "content": user_prompt})
    schemas = registry.openai_schemas()

    log_step("USER", user_prompt, color="blue")
    log_step(
        "RUNTIME",
        f"workspace_root={runtime.workspace_root} | path_jail={runtime.enforce_path_sandbox} | env_isolated=True",
        color="gray",
    )

    for turn in range(1, max_turns + 1):
        log_step("LOOP", f"--- Turn {turn} ---", color="gray")
        response: LLMResponse = llm.chat(runtime.messages, tools=schemas)
        runtime.messages.append(response.raw_message)

        if response.content:
            log_step("THINK", response.content, color="cyan")

        if not response.tool_calls:
            log_step(
                "DONE",
                f"Task complete! Modified files: {[p.name for p in runtime.modified_files]} | "
                f"Allowed prefixes: {sorted(runtime.allowed_command_prefixes)}",
                color="green",
            )
            return response.content or ""

        for call in response.tool_calls:
            log_step("TOOL CALL", f"{call.name}({call.arguments})", color="yellow")

            # NEW IN STEP 3: 返回结构化的 ToolExecutionResult(status, content)
            result = registry.execute_structured(call.name, call.arguments, runtime=runtime)
            log_step(
                f"TOOL RESULT [{result.status.upper()}]",
                f"\n{result.content}",
                color=STATUS_COLOR.get(result.status, "magenta"),
            )

            runtime.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.content,
                }
            )

    return "Reached max_turns without finishing."


class ScriptedRuntimeDemoLLM:
    """
    演示四大硬核特性：
    - Turn 1:
      1) `bash("ls -la && pwd")` —— 纯只读白名单复合命令，无需询问用户，自动放行！
      2) `bash("ls -la && rm -f fib.py")` —— 试图在 `ls` 后面夹带 `rm`，被命令拆解引擎识破并拦截拒绝 (`status="denied"`)！
    - Turn 2:
      LLM 收到拒绝后放弃删除，改用 `write_file` 创建 `env_check.py`（检查子进程能否偷到宿主机的 OPENAI_API_KEY）。
    - Turn 3:
      调用 `bash("python3 env_check.py")` 验证净室环境变量隔离生效！
    """

    def __init__(self) -> None:
        self.turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.turn += 1
        if self.turn == 1:
            tc_safe = ToolCall(id="call_1_safe", name="bash", arguments='{"command": "ls -la && pwd"}')
            tc_sneaky = ToolCall(id="call_1_sneaky", name="bash", arguments='{"command": "ls -la && rm -f fib.py"}')
            return LLMResponse(
                content="我先跑一条只读复合命令 `ls -la && pwd`，再试一条夹带 `rm` 的复合命令 `ls -la && rm -f fib.py`。",
                tool_calls=[tc_safe, tc_sneaky],
                raw_message={
                    "role": "assistant",
                    "content": "我先跑一条只读复合命令 `ls -la && pwd`，再试一条夹带 `rm` 的复合命令 `ls -la && rm -f fib.py`。",
                    "tool_calls": [
                        {"id": tc_safe.id, "type": "function", "function": {"name": tc_safe.name, "arguments": tc_safe.arguments}},
                        {"id": tc_sneaky.id, "type": "function", "function": {"name": tc_sneaky.name, "arguments": tc_sneaky.arguments}},
                    ],
                },
            )
        if self.turn == 2:
            code = (
                "import os, tempfile\n"
                "print('HOST OPENAI_API_KEY visible in child?:', bool(os.environ.get('OPENAI_API_KEY')))\n"
                "print('Isolated TMPDIR:', tempfile.gettempdir())\n"
            )
            tc_write = ToolCall(
                id="call_2_write",
                name="write_file",
                arguments=json.dumps({"path": "env_check.py", "content": code}),
            )
            return LLMResponse(
                content="夹带 `rm` 的复合命令被 Permission Gate 拦截了 (`status=denied`)。我现在创建 `env_check.py` 来验证子进程环境变量隔离。",
                tool_calls=[tc_write],
                raw_message={
                    "role": "assistant",
                    "content": "夹带 `rm` 的复合命令被 Permission Gate 拦截了 (`status=denied`)。我现在创建 `env_check.py` 来验证子进程环境变量隔离。",
                    "tool_calls": [
                        {"id": tc_write.id, "type": "function", "function": {"name": tc_write.name, "arguments": tc_write.arguments}},
                    ],
                },
            )
        if self.turn == 3:
            tc_run = ToolCall(id="call_3_run", name="bash", arguments='{"command": "python3 env_check.py"}')
            return LLMResponse(
                content="现在运行 `python3 env_check.py`（由于 `python3` 前缀已授信，自动放行）。",
                tool_calls=[tc_run],
                raw_message={
                    "role": "assistant",
                    "content": "现在运行 `python3 env_check.py`（由于 `python3` 前缀已授信，自动放行）。",
                    "tool_calls": [
                        {"id": tc_run.id, "type": "function", "function": {"name": tc_run.name, "arguments": tc_run.arguments}},
                    ],
                },
            )

        out = next((m["content"] for m in messages if m.get("tool_call_id") == "call_3_run"), "")
        final_msg = f"验证完成！受控沙盒运行结果如下：\n{out}"
        return LLMResponse(
            content=final_msg,
            tool_calls=[],
            raw_message={"role": "assistant", "content": final_msg},
        )


def run_demo(workspace_dir: Path, registry: ToolRegistry) -> None:
    # 模拟宿主机存在敏感 API Key，验证子进程无法偷取它
    os.environ.setdefault("OPENAI_API_KEY", "sk-demo-host-secret-key")

    def demo_permission_policy(tool_name: str, args: dict[str, Any]) -> bool:
        cmd = str(args.get("command", ""))
        if tool_name == "bash" and "rm " in cmd:
            log_step("PERMISSION GATE", f"发现高危子命令，拒绝执行: {cmd}", color="red")
            return False
        log_step("PERMISSION GATE", f"批准操作: {tool_name}({args})", color="green")
        return True

    runtime = AgentRuntime(
        workspace_root=workspace_dir,
        permission_callback=demo_permission_policy,
    )
    # 模拟用户之前已经对 `python3` 前缀点过一次 "a (Always allow prefix)"
    runtime.allow_command_prefix("python3")

    prompt = "查看目录文件，测试权限拦截与子进程环境变量净室隔离"
    run_runtime_agent_loop(prompt, llm=ScriptedRuntimeDemoLLM(), registry=registry, runtime=runtime)


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    registry = create_coding_tools(base_dir=workspace_dir)
    args = sys.argv[1:]

    if "--demo" in args:
        log_step("DEMO", "运行 Step 3 Runtime（命令拆解前缀引擎 + 沙盒环境隔离 + 结构化结果）演示", color="green")
        run_demo(workspace_dir, registry)
        return

    client = LLMClient()
    if not client.api_key:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动运行 --demo 模式。", color="yellow")
        run_demo(workspace_dir, registry)
        return

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
