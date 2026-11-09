from __future__ import annotations

"""
Step 2 & Step 3 核心组件：Tool 抽象、结构化执行结果与四大编码工具 (harness/tool.py)

第一性原理视角：
1. **对 LLM 暴露 JSON Schema，对 Harness 隐藏 `runtime` 注入参数**；
2. **结构化执行结果（`ToolExecutionResult`）**：
   Harness 内部绝不靠 `"PERMISSION_DENIED" in result_str` 这种脆弱的字符串匹配来判断工具状态，
   而是通过 `ToolExecutionResult(status="ok" | "error" | "denied", content=...)` 精确传递意图；
   只有最后一步塞进 `messages` 给 LLM 看时，才取 `.content` 字符串。
"""

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from harness.runtime import AgentRuntime
    from harness.validation import ValidationPipeline

TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}

ToolStatus = Literal["ok", "error", "denied"]


@dataclass
class ToolExecutionResult:
    """
    工具执行的结构化返回对象：
    - `status`: 供 Harness 内部精确判断是正常完成 ("ok")、执行/校验出错 ("error") 还是被权限门禁拦截 ("denied")；
    - `content`: 最终序列化后放入 `role="tool"` 消息中喂给 LLM 的观测文本。
    """

    status: ToolStatus
    content: str
    tool_name: str = ""

    @property
    def is_error(self) -> bool:
        return self.status in {"error", "denied"}


@dataclass
class Tool:
    """统一的工具描述符。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    dangerous: bool = False
    accepts_runtime: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(
    name: str | None = None,
    description: str = "",
    dangerous: bool = False,
) -> Callable[[Callable[..., Any]], Tool]:
    """
    装饰器：将一个普通 Python 函数转换为 `Tool` 对象，并自动从类型注解生成 JSON Schema。
    如果函数参数名为 `runtime`（类型为 `AgentRuntime | None`），它会被视为 Harness 内部注入的环境容器，不暴露给 LLM。
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_desc = (description or inspect.getdoc(func) or "").strip()
        sig = inspect.signature(func)

        properties: dict[str, Any] = {}
        required: list[str] = []
        accepts_runtime = False

        for param_name, param in sig.parameters.items():
            if param_name == "runtime":
                accepts_runtime = True
                continue

            ann = param.annotation
            json_type = TYPE_MAP.get(ann, "string")
            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter '{param_name}' ({json_type})",
            }
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters=parameters_schema,
            func=func,
            dangerous=dangerous,
            accepts_runtime=accepts_runtime,
        )

    return decorator


class ToolRegistry:
    """工具注册中心：统一管理 Schema 导出与工具调用分发。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, t: Tool) -> None:
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute_structured(
        self,
        name: str,
        raw_arguments: str,
        runtime: AgentRuntime | None = None,
        validation: ValidationPipeline | None = None,
        **context: Any,
    ) -> ToolExecutionResult:
        """
        结构化执行入口（供 Harness Loop 使用）：
        1. 解析 JSON 参数；
        2. [Step 4] 若提供了 `validation`，先执行 `validation.run_pre_tool(t, args, runtime)`；
        3. [Step 3] 若提供了 `runtime`，执行 `runtime.check_permission(t, args)`；
        4. 调用工具函数；
        5. [Step 4] 若提供了 `validation`，执行 `validation.run_post_tool(t, args, raw_out, runtime)`。
        """
        t = self.get(name)
        if t is None:
            return ToolExecutionResult(
                status="error",
                content=f"[ERROR] Unknown tool: '{name}'",
                tool_name=name,
            )

        try:
            args = json.loads(raw_arguments or "{}")
            if not isinstance(args, dict):
                return ToolExecutionResult(
                    status="error",
                    content=f"[ERROR] Tool arguments must be a JSON object, got: {type(args).__name__}",
                    tool_name=name,
                )
        except json.JSONDecodeError as e:
            return ToolExecutionResult(
                status="error",
                content=f"[ERROR] Invalid JSON arguments for tool '{name}': {e}",
                tool_name=name,
            )

        # Step 4: 执行前拦截钩子 (pre_tool_use Hooks) —— 不通过则短路返回，不弹窗、不落盘
        if validation is not None and runtime is not None:
            pre_res = validation.run_pre_tool(t, args, runtime)
            if not pre_res.passed:
                err_status: ToolStatus = "denied" if getattr(pre_res, "status", "error") == "denied" else "error"
                return ToolExecutionResult(
                    status=err_status,
                    content=pre_res.format_feedback("pre_tool_use"),
                    tool_name=name,
                )

        # Step 3: 如果提供了 AgentRuntime，执行 Permission Gate 拦截检查
        if runtime is not None:
            allowed, deny_reason = runtime.check_permission(t, args)
            if not allowed:
                return ToolExecutionResult(
                    status="denied",
                    content=f"[PERMISSION_DENIED] {deny_reason}",
                    tool_name=name,
                )

        call_kwargs = dict(args)
        if t.accepts_runtime:
            call_kwargs["runtime"] = runtime

        try:
            raw_out = str(t.func(**call_kwargs))
        except Exception as e:
            return ToolExecutionResult(
                status="error",
                content=f"[ERROR] Tool '{name}' execution failed: {type(e).__name__}: {e}",
                tool_name=name,
            )

        # Step 4: 执行后校验钩子 (post_tool_use Hooks)
        if validation is not None and runtime is not None:
            post_res = validation.run_post_tool(t, args, raw_out, runtime)
            if not post_res.passed:
                return ToolExecutionResult(
                    status="error",
                    content=f"{raw_out}\n{post_res.format_feedback('post_tool_use')}",
                    tool_name=name,
                )

        is_err = raw_out.startswith("[ERROR]") or "[exit_code=" in raw_out and not raw_out.startswith("[exit_code=0]")
        return ToolExecutionResult(
            status="error" if is_err else "ok",
            content=raw_out,
            tool_name=name,
        )

    def execute(
        self,
        name: str,
        raw_arguments: str,
        runtime: AgentRuntime | None = None,
        **context: Any,
    ) -> str:
        """便捷字符串入口（兼容直接取文本结果的调用方）。"""
        return self.execute_structured(name, raw_arguments, runtime=runtime, **context).content


# ============================================================================
# Claude Code 核心四大基础工具：read_file, write_file, edit_file, bash
# ============================================================================


def create_coding_tools(base_dir: Path | None = None) -> ToolRegistry:
    """
    创建 Coding Agent 最核心的四个工具：
    - read_file:  眼睛（查看代码，dangerous=False）
    - write_file: 双手（创建/覆写文件，dangerous=True）
    - edit_file:  手术刀（局部精确替换代码，dangerous=True）
    - bash:       执行器（受控沙盒子进程跑命令，dangerous=True）
    """
    default_root = (base_dir or (Path.cwd() / "workspace")).resolve()
    default_root.mkdir(parents=True, exist_ok=True)

    def _resolve_target(path_str: str, runtime: AgentRuntime | None = None) -> Path:
        if runtime is not None:
            return runtime.resolve_path(path_str)
        p = Path(path_str)
        return p if p.is_absolute() else (default_root / p).resolve()

    @tool(
        name="read_file",
        description="Read the text contents of a file at the given path.",
        dangerous=False,
    )
    def read_file(path: str, runtime: AgentRuntime | None = None) -> str:
        target = _resolve_target(path, runtime)
        if not target.exists():
            return f"[ERROR] File not found: {path}. Use bash('ls') to check existing files."
        if target.is_dir():
            return f"[ERROR] Path is a directory, not a file: {path}"
        content = target.read_text(encoding="utf-8")
        lines = content.splitlines()
        numbered = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
        return f"File: {path} ({len(lines)} lines)\n{numbered}"

    @tool(
        name="write_file",
        description="Create or overwrite a file with the provided full content.",
        dangerous=True,
    )
    def write_file(path: str, content: str, runtime: AgentRuntime | None = None) -> str:
        target = _resolve_target(path, runtime)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        if runtime is not None:
            runtime.modified_files.add(target)
        return f"[OK] Wrote {len(content)} characters to {path}"

    @tool(
        name="edit_file",
        description="Replace an exact occurrence of old_text with new_text in a file.",
        dangerous=True,
    )
    def edit_file(path: str, old_text: str, new_text: str, runtime: AgentRuntime | None = None) -> str:
        target = _resolve_target(path, runtime)
        if not target.exists():
            return f"[ERROR] File not found: {path}. Cannot edit a non-existent file."
        content = target.read_text(encoding="utf-8")
        if old_text not in content:
            return (
                f"[ERROR] old_text not found in {path}. "
                "RECOVERY_HINT: Call read_file first to inspect the exact current lines and whitespace."
            )
        count = content.count(old_text)
        if count > 1:
            return (
                f"[ERROR] old_text matched {count} locations in {path}. "
                "RECOVERY_HINT: Include more surrounding context lines in old_text so it matches uniquely."
            )
        updated = content.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        if runtime is not None:
            runtime.modified_files.add(target)
        return f"[OK] Edited {path} successfully"

    @tool(
        name="bash",
        description="Execute a shell command in the isolated workspace runtime and return stdout/stderr.",
        dangerous=True,
    )
    def bash(command: str, runtime: AgentRuntime | None = None) -> str:
        if runtime is not None:
            return runtime.run_command(command)
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(default_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"[exit_code={proc.returncode}]\n{out.strip()}"

    registry = ToolRegistry()
    registry.register(read_file)
    registry.register(write_file)
    registry.register(edit_file)
    registry.register(bash)
    return registry
