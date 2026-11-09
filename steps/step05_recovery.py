#!/usr/bin/env python3
from __future__ import annotations

"""
Step 5: Recovery 容错、死循环检测与会话断点状态机恢复 (Self-Correction & Resilience)
====================================================================================

相比 Step 4 的演进（NEW IN STEP 5）：
1. **LLM API 指数退避重试（`call_with_retry`）**
2. **死循环检测与强制干预（`LoopDetector`）**
3. **带尾部状态机的会话断点恢复（`SessionStore` + `inspect_session_state`）**：
   - 当执行 `--resume <session_id>` 时，先通过 `inspect_session_state(runtime.messages)` 判断落盘消息的尾部状态：
     * `incomplete_tool_calls`：上次进程死在工具执行中途 -> 先补跑未完成的 `tool_calls`，然后直接进入 Loop 调 LLM（绝不重复塞入初始 Prompt）；
     * `pending_llm`：上次进程死在工具跑完后、调 LLM 之前 -> 不塞重复 Prompt，直接从下一轮 `llm.chat` 继续；
     * `completed`：上次任务已正常结束 -> 若本次未传新问题，直接打印上次最终回答并进入交互等待；若传了新问题，则作为追问追加后继续。

运行方式：
  1. 免 Key 离线演示（包含死循环自愈 + 模拟半途崩溃后的断点续跑）：
     python3 steps/step05_recovery.py --demo
  2. 恢复已完成的会话（不会重跑旧 Prompt）：
     python3 steps/step05_recovery.py --resume demo_step05
  3. 在已完成的会话基础上追问新问题：
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
from harness.recovery import LoopDetector, SessionStore, call_with_retry, inspect_session_state
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
    user_prompt: str | None,
    llm: Any,
    registry: ToolRegistry,
    runtime: AgentRuntime,
    validation: ValidationPipeline,
    session_store: SessionStore,
    session_id: str = "default",
    max_turns: int = 10,
) -> str:
    """
    Step 5 的 Agent Loop（支持精确断点状态恢复）：
    1. 检查 `runtime.messages` 尾部状态 (`inspect_session_state`)；
    2. 如果存在未跑完的 `tool_calls`（上次崩溃在工具执行阶段），先补跑缺失的工具；
    3. 如果会话已是 `completed` 状态且没有传入新的 `user_prompt`，直接返回上次回答，绝不重跑旧任务！
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

    loop_detector = LoopDetector(max_repeats=3)
    state, missing_tool_calls = inspect_session_state(runtime.messages)

    # 1) 如果上次中断在 assistant 发出 tool_calls 之后、工具尚未全部执行完之前：先补齐缺失的工具调用！
    if state == "incomplete_tool_calls" and missing_tool_calls:
        log_step(
            "RECOVERY [RESUME MID-TURN]",
            f"检测到上次会话在工具执行阶段中断，正在补跑 {len(missing_tool_calls)} 个未完成的 tool_calls...",
            color="yellow",
        )
        for call in missing_tool_calls:
            log_step("TOOL CALL [RESUMED]", f"{call.name}({call.arguments})", color="yellow")
            result = registry.execute_structured(call.name, call.arguments, runtime=runtime, validation=validation)
            log_step(f"TOOL RESULT [{result.status.upper()}]", f"\n{result.content}", color=STATUS_COLOR.get(result.status, "magenta"))
            runtime.messages.append({"role": "tool", "tool_call_id": call.id, "content": result.content})
        session_store.save(session_id, runtime.messages)
        state, _ = inspect_session_state(runtime.messages)

    # 2) 处理本次调用是否传入了新的 user_prompt
    if user_prompt:
        log_step("USER", user_prompt, color="blue")
        runtime.messages.append({"role": "user", "content": user_prompt})
        session_store.save(session_id, runtime.messages)
    else:
        if state == "completed":
            last_reply = runtime.messages[-1].get("content") or ""
            log_step(
                "SESSION RESUME",
                f"会话 '{session_id}' 上一轮任务已处于完成状态 (completed)，无需重跑。上次最终回复：\n{last_reply}",
                color="green",
            )
            return last_reply
        if state == "empty":
            return ""
        log_step(
            "RECOVERY [RESUME PENDING]",
            f"会话 '{session_id}' 处于未完成状态 ({state})，从断点继续调用 LLM...",
            color="yellow",
        )

    schemas = registry.openai_schemas()
    log_step("SESSION", f"session_id={session_id} (messages={len(runtime.messages)})", color="gray")

    for turn in range(1, max_turns + 1):
        log_step("LOOP", f"--- Turn {turn} ---", color="gray")

        response: LLMResponse = call_with_retry(
            lambda: llm.chat(runtime.messages, tools=schemas),
            max_retries=3,
            base_delay=0.05,
        )
        runtime.messages.append(response.raw_message)
        # LLM 响应一拿到立即存盘，即使接下来跑工具时进程被杀，重启也能通过 incomplete_tool_calls 补跑！
        session_store.save(session_id, runtime.messages)

        if response.content:
            log_step("THINK", response.content, color="cyan")

        if not response.tool_calls:
            stop_check = validation.run_pre_stop(runtime)
            if stop_check.passed:
                ckpt_path = session_store.save(session_id, runtime.messages)
                log_step("SESSION SAVED", f"Checkpoint persisted to {ckpt_path} (state=completed)", color="green")
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
            session_store.save(session_id, runtime.messages)

    return "Reached max_turns without finishing."


class ScriptedRecoveryDemoLLM:
    """支持多轮恢复演示与追问回答的内置模拟 LLM。"""

    def __init__(self) -> None:
        self.turn = 0
        self.simulated_rate_limit_once = False

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        # 如果是针对已完成会话的追问（最后一条是 user 且历史中已有超过 5 条消息）
        if len(messages) > 6 and messages[-1].get("role") == "user":
            question = messages[-1].get("content", "")
            reply = f"[基于恢复的会话记忆回答] 你问的是：'{question}'。在之前的会话中，我们读取并修改了 `greeting.py`，将其内容更新为了 `print('Hello Resilient Agent')`。"
            return LLMResponse(
                content=reply,
                tool_calls=[],
                raw_message={"role": "assistant", "content": reply},
            )

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
                content="收到 `[LOOP_DETECTED]` 警告！连续 3 次盲猜失败，我停止重复调用，先用 `read_file` 查看 `greeting.py` 的真实内容。",
                tool_calls=[tc_read],
                raw_message={
                    "role": "assistant",
                    "content": "收到 `[LOOP_DETECTED]` 警告！连续 3 次盲猜失败，我停止重复调用，先用 `read_file` 查看 `greeting.py` 的真实内容。",
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


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    sessions_dir = ROOT_DIR / ".sessions"
    registry = create_coding_tools(base_dir=workspace_dir)
    session_store = SessionStore(base_dir=sessions_dir)
    args = sys.argv[1:]

    use_demo_llm = False
    if "--demo" in args:
        use_demo_llm = True
        args = [a for a in args if a != "--demo"]

    resumed = False
    session_id = "step05_default"
    runtime = AgentRuntime(workspace_root=workspace_dir)

    if "--resume" in args:
        resumed = True
        idx = args.index("--resume")
        if idx + 1 < len(args) and not args[idx + 1].startswith("-"):
            session_id = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
        else:
            args = args[:idx] + args[idx + 1 :]
        runtime.messages = session_store.load(session_id)
        state, _ = inspect_session_state(runtime.messages)
        log_step(
            "SESSION LOADED",
            f"已从 '{session_id}' 恢复 {len(runtime.messages)} 条历史消息 (当前尾部状态: state={state})",
            color="green",
        )

    if use_demo_llm and not resumed:
        log_step("DEMO", "运行 Step 5 Recovery（API 退避重试 + 死循环检测打断 + Session 落盘恢复）演示", color="green")
        run_demo(workspace_dir, registry, session_store)
        return

    client = LLMClient()
    active_llm: Any = ScriptedRecoveryDemoLLM() if (use_demo_llm or not client.api_key) else client

    if not client.api_key and not resumed:
        log_step("INFO", "未检测到 OPENAI_API_KEY，自动运行 --demo 模式。", color="yellow")
        run_demo(workspace_dir, registry, session_store)
        return

    validation = create_default_validation_pipeline()
    user_prompt = " ".join(args).strip() if args else None

    if resumed or user_prompt:
        run_resilient_agent_loop(
            user_prompt=user_prompt,
            llm=active_llm,
            registry=registry,
            runtime=runtime,
            validation=validation,
            session_store=session_store,
            session_id=session_id,
        )
        if not user_prompt:
            return
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
                llm=active_llm,
                registry=registry,
                runtime=runtime,
                validation=validation,
                session_store=session_store,
                session_id=session_id,
            )


if __name__ == "__main__":
    main()
