#!/usr/bin/env python3
from __future__ import annotations

"""
Step 7: Context Compaction 上下文裁剪与历史摘要压缩 (Memory Compaction)
======================================================================

相比 Step 6 的演进（NEW IN STEP 7）：
1. **单次工具输出截断（`truncate_tool_output`）**：
   - 工具返回结果写入 `messages` 前先限制最大长度（保留头尾、折叠中间），防止单个大文件或长日志直接塞满上下文。
2. **多轮历史自动摘要压缩（`compact_messages`）**：
   - 每轮调用 `llm.chat` 前检查 `estimate_tokens(runtime.messages)`；
   - 当超过设定的 `max_context_tokens` 阈值（或用户输入 `/compact`）时：
     * 保留 `system` 消息；
     * 自动对齐切分边界（严禁拆散 `assistant.tool_calls` 与对应的 `role="tool"` 消息）；
     * 调用 LLM 将早期消息压缩为 `<context_summary>`，释放上下文空间。

运行方式：
  1. 免 Key 离线演示（用较低的 Token 阈值触发自动压缩，对比压缩前后的消息结构与 Token 占用）：
     python3 steps/step07_compaction.py --demo
  2. 使用真实 OpenAI API（支持交互中输入 `/compact` 手动触发压缩）：
     python3 steps/step07_compaction.py
"""

import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.compaction import compact_messages, estimate_tokens, truncate_tool_output
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


def run_compacting_agent_loop(
    user_prompt: str,
    llm: Any,
    registry: ToolRegistry,
    runtime: AgentRuntime,
    validation: ValidationPipeline,
    skill_manager: SkillManager | None = None,
    max_context_tokens: int = 4000,
    keep_recent: int = 3,
    max_tool_chars: int = 2000,
    max_turns: int = 10,
) -> str:
    """
    Step 7 的 Agent Loop：
    在每轮调用 LLM 之前执行 `compact_messages()` 检查，在每次工具输出后执行 `truncate_tool_output()`。
    """
    if not runtime.messages:
        sys_prompt = build_system_prompt(runtime, skill_manager=skill_manager)
        runtime.messages.append({"role": "system", "content": sys_prompt})

    runtime.messages.append({"role": "user", "content": user_prompt})
    schemas = registry.openai_schemas()
    loop_detector = LoopDetector(max_repeats=3)

    log_step("USER", user_prompt, color="blue")

    for turn in range(1, max_turns + 1):
        # NEW IN STEP 7: 每轮调用 LLM 前检查是否超过上下文阈值，超限则自动压缩早期历史
        runtime.messages = compact_messages(
            runtime.messages,
            llm=llm,
            max_tokens=max_context_tokens,
            keep_recent=keep_recent,
        )

        est_tokens = estimate_tokens(runtime.messages)
        log_step(
            "LOOP",
            f"--- Turn {turn} (messages={len(runtime.messages)}, est_tokens=~{est_tokens}/{max_context_tokens}) ---",
            color="gray",
        )

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

            # NEW IN STEP 7: 对单次超长工具输出执行头尾保留截断
            clipped_content = truncate_tool_output(result.content, max_chars=max_tool_chars)
            loop_warning = loop_detector.record_and_check(call.name, call.arguments, is_error=result.is_error)
            content_for_llm = f"{clipped_content}\n{loop_warning}" if loop_warning else clipped_content

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


class ScriptedCompactionDemoLLM:
    """
    演示两级上下文控制：
    - Turn 1: 调用 `write_file` 创建一个包含大量日志行的 `big_log.txt`（长内容使总 Token 迅速增长）。
    - Turn 2: 调用 `read_file("big_log.txt")` -> 输出超过 `max_tool_chars=600` 被 `truncate_tool_output` 头尾截断！
    - Turn 3 开始前：由于消息总 Token 超过了 `max_context_tokens=350`，触发 `compact_messages`：
      * Compactor 调用 LLM 生成 `<context_summary>`；
      * 将前几轮冗长的工具调用替换为精简摘要，同时完整保留最近的 `assistant(tool_calls) + tool` 配对消息！
    - Turn 3: LLM 基于压缩后的 `<context_summary>` 与最近消息给出最终回答。
    """

    def __init__(self) -> None:
        self.turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        # 如果 tools is None，说明这是 Compactor 发来的摘要生成请求！
        if tools is None:
            summary = (
                "- Created `big_log.txt` with 40 log lines (HEADER_START to FOOTER_END).\n"
                "- Verified file creation in workspace."
            )
            return LLMResponse(
                content=summary,
                tool_calls=[],
                raw_message={"role": "assistant", "content": summary},
            )

        self.turn += 1
        if self.turn == 1:
            lines = ["HEADER_START: system boot ok"] + [f"LOG_LINE_{i:02d}: " + ("data_" * 12) for i in range(1, 39)] + ["FOOTER_END: all services healthy"]
            args = json.dumps({"path": "big_log.txt", "content": "\n".join(lines) + "\n"})
            tc = ToolCall(id="call_1_write_big", name="write_file", arguments=args)
            return LLMResponse(
                content="我先创建包含 40 行数据的 `big_log.txt`。",
                tool_calls=[tc],
                raw_message={
                    "role": "assistant",
                    "content": "我先创建包含 40 行数据的 `big_log.txt`。",
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}],
                },
            )

        if self.turn == 2:
            tc_read = ToolCall(id="call_2_read_big", name="read_file", arguments='{"path": "big_log.txt"}')
            return LLMResponse(
                content="现在读取 `big_log.txt` 检查头尾内容（将触发大输出截断与下一轮前的自动历史压缩）。",
                tool_calls=[tc_read],
                raw_message={
                    "role": "assistant",
                    "content": "现在读取 `big_log.txt` 检查头尾内容（将触发大输出截断与下一轮前的自动历史压缩）。",
                    "tool_calls": [{"id": tc_read.id, "type": "function", "function": {"name": tc_read.name, "arguments": tc_read.arguments}}],
                },
            )

        has_summary = any("<context_summary>" in str(m.get("content", "")) for m in messages)
        final_ans = (
            f"检查完毕！当前上下文中已包含自动生成的 `<context_summary>` (has_summary={has_summary})，"
            "且从截断后的 `big_log.txt` 首尾确认：头部为 `HEADER_START: system boot ok`，尾部为 `FOOTER_END: all services healthy`。"
        )
        return LLMResponse(
            content=final_ans,
            tool_calls=[],
            raw_message={"role": "assistant", "content": final_ans},
        )


def run_demo(workspace_dir: Path, registry: ToolRegistry) -> None:
    runtime = AgentRuntime(workspace_root=workspace_dir, always_allow_dangerous=True)
    validation = create_default_validation_pipeline()
    prompt = "生成一个大日志文件 big_log.txt，读取它并汇报首尾状态"

    run_compacting_agent_loop(
        prompt,
        llm=ScriptedCompactionDemoLLM(),
        registry=registry,
        runtime=runtime,
        validation=validation,
        max_context_tokens=350,  # 故意设低阈值，方便在 Turn 3 前直观看到自动压缩触发
        keep_recent=2,
        max_tool_chars=600,      # 故意设 600 字符上限，直观看到大输出头尾截断
    )

    log_step("COMPACTED MESSAGES INSPECT", "压缩后的 runtime.messages 角色与内容概览：", color="green")
    for idx, m in enumerate(runtime.messages):
        role = m.get("role")
        preview = str(m.get("content") or "").splitlines()[0] if m.get("content") else "(tool_calls)"
        print(f"  [{idx}] role={role:<9} | {preview[:90]}")


def main() -> None:
    workspace_dir = ROOT_DIR / "workspace"
    skills_dir = ROOT_DIR / "skills"
    registry = create_coding_tools(base_dir=workspace_dir)
    skill_manager = SkillManager(skills_dir=skills_dir)
    skill_manager.register_load_tool(registry)

    args = sys.argv[1:]
    if "--demo" in args:
        log_step("DEMO", "运行 Step 7 Context Compaction（大输出截断 + 历史自动摘要压缩）演示", color="green")
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
        run_compacting_agent_loop(
            " ".join(args),
            llm=client,
            registry=registry,
            runtime=runtime,
            validation=validation,
            skill_manager=skill_manager,
        )
    else:
        print(f"进入 Step 7 交互模式（输入 /compact 手动压缩上下文，输入 q 退出）：")
        while True:
            try:
                user_in = input("\nStep07 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in or user_in.lower() in {"q", "quit", "exit"}:
                break
            if user_in == "/compact":
                runtime.messages = compact_messages(runtime.messages, llm=client, force=True)
                continue
            run_compacting_agent_loop(
                user_in,
                llm=client,
                registry=registry,
                runtime=runtime,
                validation=validation,
                skill_manager=skill_manager,
            )


if __name__ == "__main__":
    main()
