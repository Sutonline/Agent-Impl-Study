#!/usr/bin/env python3
from __future__ import annotations

"""
Step 5: Recovery 容错、死循环检测与会话断点恢复 (Self-Correction & Resilience)
==============================================================================

相比 Step 4 的演进（NEW IN STEP 5）：
1. **LLM API 指数退避重试（`call_with_retry`）**：
   - 当 `llm.chat()` 遇到瞬态网络故障或 HTTP 429 限流时自动退避重试，不再直接崩掉进程。
2. **死循环检测与强制干预（`LoopDetector`）**：
   - 记录每次工具调用的 `(tool_name, canonical_args, is_error)`；
   - 当 LLM 连续 3 次用完全相同的参数调用同一个失败工具时，在返回结果中注入 `[LOOP_DETECTED]` 警告，逼迫 LLM 改变策略（例如先 `read_file` 看真实内容，而不是盲猜 `edit_file`）。
3. **会话断点落盘与续跑（`SessionStore`）**：
   - 每轮循环结束自动将 `runtime.messages` 原子写入 `.sessions/<session_id>.json`；
   - 支持通过 `--resume <session_id>` 从上次中断的对话状态继续运行。

运行方式：
  1. 免 Key 离线演示（观察：API 限流自动重试 -> 连续 3 次错误 edit 触发死循环打断 -> 改调 read_file 自愈 -> 落盘保存 Session）：
     python3 steps/step05_recovery.py --demo
  2. 从已保存的 Session 恢复查看：
     python3 steps/step05_recovery.py --resume demo_step05 "刚才我们改了哪个文件？"
"""

import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.llm import LLMClient, LLMResponse, ToolCall
from harness.recovery import LoopDetector, SessionStore, call_with_retry
from harness.runtime import AgentRuntime
from harness.tool import ToolRegistry, create_coding_tools
from harness.ui import log_step
from harness.validation import ValidationPipeline, create_default_validation_pipeline

STATUS_COLOR = {
    "ok": "magenta",
    "error": "red",
    "denied": "red",
}


def run_resilient_agent_loop(
    user_prompt: str,
    llm: Any,
    registry: ToolRegistry,
    runtime: AgentRuntime,
    validation: ValidationPipeline,
    session_store: SessionStore,
    session_id: str = "default",
    max_turns: int = 10,
) -> str:
    """
    Step 5 的 Agent Loop：
    在 Step 4 基础上增加了：
      1) `call_with_retry` 保护 `llm.chat`；
      2) `LoopDetector` 检测重复失败调用并注入打断警告；
      3) `session_store.save` 每轮落盘保存 Checkpoint。
    """
    if not runtime.messages:
        runtime.messages.append(
            {
                "role": "system",
                "content": (
                    f"You are a resilient coding agent working in {runtime.workspace_root}. "
                    "When a tool returns an error or RECOVERY_HINT or LOOP_DETECTED warning, "
                    "adapt your approach immediately."
                ),
            }
        )

    runtime.messages.append({"role": "user", "content": user_prompt})
    session_store.save(session_id, runtime.messages)

    schemas = registry.openai_schemas()
    loop_detector = LoopDetector(max_repeats=3)

    log_step("USER", user_prompt, color="blue")
    log_step("SESSION", f"session_id={session_id} (messages={len(runtime.messages)})", color="gray")

    for turn in range(1, max_turns + 1):
        log_step("LOOP", f"--- Turn {turn} ---", color="gray")

        # NEW IN STEP 5: 用指数退避重试保护 LLM 网络调用
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
                ckpt_path = session_store.save(session_id, runtime.messages)
                log_step("SESSION SAVED", f"Checkpoint persisted to {ckpt_path}", color="green")
                log_step("DONE", "Task complete.", color="green")
                return response.content or ""

            feedback = stop_check.format_feedback("pre_stop")
            log_step("VALIDATION [pre_stop VETO]", f"\n{feedback}", color="red")
            runtime.messages.append({"role": "user", "content": feedback})
            session_store.save(session_id, runtime.messages)
            continue

        for call in response.tool_calls:
            log_step("TOOL CALL", f"{call.name}({call.arguments})", color="yellow")
            result = registry.execute_structured(
                call.name,
                call.arguments,
                runtime=runtime,
                validation=validation,
            )

            # NEW IN STEP 5: 死循环检测器记录本次调用；若连续 3 次同样参数报错，追加 [LOOP_DETECTED] 强警告
            loop_warning = loop_detector.record_and_check(
                call.name,
                call.arguments,
                is_error=result.is_error,
            )
            content_for_llm = result.content
            if loop_warning:
                log_step("RECOVERY [LOOP DETECTED]", loop_warning, color="red")
                content_for_llm = f"{result.content}\n{loop_warning}"

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

        # NEW IN STEP 5: 每轮结束原子写入 Checkpoint
        session_store.save(session_id, runtime.messages)

    return "Reached max_turns without finishing."


class ScriptedRecoveryDemoLLM:
    """
    演示三大 Recovery 场景：
    - Turn 1..3: LLM 没看文件内容，盲猜 `greeting.py` 里写的是 `Hello World`，连续 3 次调用相同的 `edit_file` 失败！
      * 其中 Turn 2 调用 `llm.chat` 时模拟一次 HTTP 429 Rate Limit 异常，由 `call_with_retry` 自动重试恢复；
      * 第 3 次 `edit_file` 失败时，`LoopDetector` 触发 `[LOOP_DETECTED]` 警告！
    - Turn 4: LLM 收到 `[LOOP_DETECTED]` 警告后醒悟，停止盲猜，调用 `read_file("greeting.py")` 查看到真实内容是 `Hello Agent`。
    - Turn 5: LLM 使用正确的 `old_text="Hello Agent"` 调用 `edit_file` 修改成功。
    - Turn 6: 完成任务并持久化 Session。
    """

    def __init__(self) -> None:
        self.turn = 0
        self.simulated_rate_limit_once = False

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        # 模拟在 Turn 2 时发生一次瞬态 API 429 错误
        if self.turn == 1 and not self.simulated_rate_limit_once:
            self.simulated_rate_limit_once = True
            raise RuntimeError("HTTP 429 Too Many Requests (transient rate limit)")

        self.turn += 1
        bad_edit_args = json.dumps(
            {
                "path": "greeting.py",
                "old_text": "print('Hello World')",
                "new_text": "print('Hello Resilient Agent')",
            }
        )

        if self.turn in {1, 2, 3}:
            tc = ToolCall(id=f"call_bad_{self.turn}", name="edit_file", arguments=bad_edit_args)
            return LLMResponse(
                content=f"我盲猜 greeting.py 里是 print('Hello World')，尝试直接替换（第 {self.turn} 次尝试）。",
                tool_calls=[tc],
                raw_message={
                    "role": "assistant",
                    "content": f"我盲猜 greeting.py 里是 print('Hello World')，尝试直接替换（第 {self.turn} 次尝试）。",
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}],
                },
            )

        if self.turn == 4:
            tc_read = ToolCall(id="call_4_read", name="read_file", arguments='{"path": "greeting.py"}')
            return LLMResponse(
                content="收到 `[LOOP_DETECTED]` 警告！连续 3 次盲猜失败，我立刻停止重复调用，先用 `read_file` 查看 `greeting.py` 的真实内容。",
                tool_calls=[tc_read],
                raw_message={
                    "role": "assistant",
                    "content": "收到 `[LOOP_DETECTED]` 警告！连续 3 次盲猜失败，我立刻停止重复调用，先用 `read_file` 查看 `greeting.py` 的真实内容。",
                    "tool_calls": [{"id": tc_read.id, "type": "function", "function": {"name": tc_read.name, "arguments": tc_read.arguments}}],
                },
            )

        if self.turn == 5:
            good_edit_args = json.dumps(
                {
                    "path": "greeting.py",
                    "old_text": "print('Hello Agent')",
                    "new_text": "print('Hello Resilient Agent')",
                }
            )
            tc_good = ToolCall(id="call_5_good", name="edit_file", arguments=good_edit_args)
            return LLMResponse(
                content="通过 read_file 看到原文件实际写的是 `print('Hello Agent')`。现在传入正确的 old_text 完成替换。",
                tool_calls=[tc_good],
                raw_message={
                    "role": "assistant",
                    "content": "通过 read_file 看到原文件实际写的是 `print('Hello Agent')`。现在传入正确的 old_text 完成替换。",
                    "tool_calls": [{"id": tc_good.id, "type": "function", "function": {"name": tc_good.name, "arguments": tc_good.arguments}}],
                },
            )

        return LLMResponse(
            content="已通过自愈流程将 `greeting.py` 更新为 `print('Hello Resilient Agent')`，且会话状态已落盘。",
            tool_calls=[],
            raw_message={
                "role": "assistant",
                "content": "已通过自愈流程将 `greeting.py` 更新为 `print('Hello Resilient Agent')`，且会话状态已落盘。",
            },
        )


def run_demo(workspace_dir: Path, registry: ToolRegistry, session_store: SessionStore) -> None:
    # 预先在 workspace 下放一个 greeting.py，内容故意不是 'Hello World' 而是 'Hello Agent'
    (workspace_dir / "greeting.py").write_text("print('Hello Agent')\n", encoding="utf-8")

    runtime = AgentRuntime(workspace_root=workspace_dir, always_allow_dangerous=True)
    validation = create_default_validation_pipeline()
    prompt = "把 greeting.py 里的问候语改成 'Hello Resilient Agent'"

    run_resilient_agent_loop(
        prompt,
        llm=ScriptedRecoveryDemoLLM(),
        registry=registry,
        runtime=runtime,
        validation=validation,
        session_store=session_store,
        session_id="demo_step05",
    )

    # 演示从磁盘恢复该 Session
    restored_msgs = session_store.load("demo_step05")
    log_step(
        "SESSION RESUME VERIFY",
        f"成功从磁盘加载 session 'demo_step05'，共恢复 {len(restored_msgs)} 条历史消息。",
        color="green",
    )


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    sessions_dir = ROOT_DIR / ".sessions"
    registry = create_coding_tools(base_dir=workspace_dir)
    session_store = SessionStore(base_dir=sessions_dir)
    args = sys.argv[1:]

    if "--demo" in args:
        log_step("DEMO", "运行 Step 5 Recovery（API 退避重试 + 死循环检测打断 + Session 落盘恢复）演示", color="green")
        run_demo(workspace_dir, registry, session_store)
        return

    session_id = "step05_default"
    runtime = AgentRuntime(workspace_root=workspace_dir)
    if "--resume" in args:
        idx = args.index("--resume")
        if idx + 1 < len(args):
            session_id = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
            runtime.messages = session_store.load(session_id)
            log_step("SESSION RESUME", f"已从 {session_id} 恢复 {len(runtime.messages)} 条历史消息", color="green")

    client = LLMClient()
    if not client.api_key:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动运行 --demo 模式。", color="yellow")
        run_demo(workspace_dir, registry, session_store)
        return

    validation = create_default_validation_pipeline()
    if args:
        run_resilient_agent_loop(
            " ".join(args),
            llm=client,
            registry=registry,
            runtime=runtime,
            validation=validation,
            session_store=session_store,
            session_id=session_id,
        )
    else:
        print(f"进入 Step 5 交互模式（session_id={session_id}，输入 q 退出）：")
        while True:
            try:
                user_in = input("\nStep05 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in or user_in.lower() in {"q", "quit", "exit"}:
                break
            run_resilient_agent_loop(
                user_in,
                llm=client,
                registry=registry,
                runtime=runtime,
                validation=validation,
                session_store=session_store,
                session_id=session_id,
            )


if __name__ == "__main__":
    main()
