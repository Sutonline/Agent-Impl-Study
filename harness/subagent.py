from __future__ import annotations

"""
Step 8 核心组件：Subagent 子智能体隔离上下文委派 (harness/subagent.py)

第一性原理视角：
为什么需要 Subagent（`task` 工具），而不是让主 Agent 自己去调 `read_file` / `bash`？
- 假设主 Agent 需要“翻阅项目里 15 个模块找出某个配置项是在哪被覆盖的”。
- 如果主 Agent 亲自在主循环里连调 15 次 `read_file`，这 15 个文件的全文都会 permanently 留在主 Agent 的 `runtime.messages` 里，
  瞬间吃掉几万 Token，稀释后续改代码时的注意力。
- **Subagent 的本质是“上下文作用域隔离（Context Scoping / Stack Frame）”**：
  就像编程语言里的函数调用拥有独立的栈帧（局部变量在函数返回时自动销毁）一样：
  1. 主 Agent 调用 `task(description, prompt)`；
  2. Harness 派生一个新的 `child_runtime`，它共享同一个物理工作区 `workspace_root`，但拥有**全新的空 `messages = []`**；
  3. 子 Agent 在自己的独立 `messages` 里翻箱倒柜跑完 $N$ 轮工具调用；
  4. 子 Agent 结束时，它那几十条中间过程消息随 `child_runtime` 直接销毁，**只有最后凝练出的总结字符串**作为 `task` 工具的返回值进到主 Agent 的 `messages`！
"""

from typing import Any

from harness.compaction import truncate_tool_output
from harness.llm import LLMResponse
from harness.recovery import LoopDetector, call_with_retry
from harness.runtime import AgentRuntime
from harness.tool import ToolRegistry, tool
from harness.ui import log_step


def _build_child_registry(parent_registry: ToolRegistry) -> ToolRegistry:
    """
    为子智能体构建工具注册表：
    继承父注册表的基础工具，但移除 `task` 自身，防止子智能体无限递归派生孙智能体。
    """
    child_reg = ToolRegistry()
    for t in parent_registry.list_tools():
        if t.name != "task":
            child_reg.register(t)
    return child_reg


def run_subagent_loop(
    description: str,
    prompt: str,
    llm: Any,
    parent_registry: ToolRegistry,
    parent_runtime: AgentRuntime,
    max_turns: int = 8,
) -> str:
    """在全新的独立 `messages` 列表中运行子智能体循环，仅返回最终文本结论。"""
    child_runtime = AgentRuntime(
        workspace_root=parent_runtime.workspace_root,
        cwd=parent_runtime.cwd,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a focused Subagent working in {parent_runtime.workspace_root}.\n"
                    f"Sub-task description: {description}\n"
                    "Investigate using the provided tools and return a concise, factual summary report to the parent agent."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        always_allow_dangerous=parent_runtime.always_allow_dangerous,
        allowed_command_prefixes=set(parent_runtime.allowed_command_prefixes),
        permission_callback=parent_runtime.permission_callback,
        enforce_path_sandbox=parent_runtime.enforce_path_sandbox,
        use_os_sandbox=parent_runtime.use_os_sandbox,
    )

    child_registry = _build_child_registry(parent_registry)
    schemas = child_registry.openai_schemas()
    loop_detector = LoopDetector(max_repeats=3)

    log_step("SUBAGENT START", f"[{description}] prompt='{prompt}' (isolated messages=2)", color="blue")

    for turn in range(1, max_turns + 1):
        log_step("SUBAGENT LOOP", f"[{description}] Turn {turn}/{max_turns}", color="gray")
        response: LLMResponse = call_with_retry(
            lambda: llm.chat(child_runtime.messages, tools=schemas),
            max_retries=3,
            base_delay=0.05,
        )
        child_runtime.messages.append(response.raw_message)

        if response.content:
            log_step("SUBAGENT THINK", f"[{description}] {response.content}", color="cyan")

        if not response.tool_calls:
            # 同步子智能体可能修改过的文件记录到父 Runtime
            parent_runtime.modified_files.update(child_runtime.modified_files)
            log_step(
                "SUBAGENT DONE",
                f"[{description}] Finished in {turn} turns (discarding {len(child_runtime.messages)} isolated messages).",
                color="green",
            )
            return f"[SUBAGENT REPORT: {description}]\n{response.content or ''}"

        for call in response.tool_calls:
            log_step("SUBAGENT TOOL", f"[{description}] {call.name}({call.arguments})", color="yellow")
            result = child_registry.execute_structured(call.name, call.arguments, runtime=child_runtime)
            clipped = truncate_tool_output(result.content, max_chars=3000)
            warn = loop_detector.record_and_check(call.name, call.arguments, is_error=result.is_error)
            content_for_llm = f"{clipped}\n{warn}" if warn else clipped

            child_runtime.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content_for_llm,
                }
            )

    parent_runtime.modified_files.update(child_runtime.modified_files)
    return f"[SUBAGENT REPORT: {description}] Reached max_turns ({max_turns}) without final answer."


def register_subagent_tool(
    registry: ToolRegistry,
    llm: Any,
    runtime: AgentRuntime,
    max_turns: int = 8,
) -> None:
    """向父级 ToolRegistry 注册 `task` 子智能体派生工具。"""

    @tool(
        name="task",
        description=(
            "Spawn an isolated Subagent with its own clean context window to research files or execute a focused sub-task. "
            "Returns only the subagent's final summary report without polluting the main conversation context."
        ),
        dangerous=False,
    )
    def task(description: str, prompt: str, runtime: AgentRuntime | None = None) -> str:
        active_rt = runtime or _default_rt
        return run_subagent_loop(
            description=description,
            prompt=prompt,
            llm=llm,
            parent_registry=registry,
            parent_runtime=active_rt,
            max_turns=max_turns,
        )

    _default_rt = runtime
    registry.register(task)
