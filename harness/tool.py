from __future__ import annotations

"""
Step 2 & Step 3 核心组件：Tool 抽象、自动 Schema 生成与四大编码工具 (harness/tool.py)

第一性原理视角：
Tool 抽象层的本质是一座“双向桥”：
  - 向左（朝向 LLM）：用 `inspect.signature` 把 Python 函数反射成 OpenAI JSON Schema（自动隐藏 `runtime` 参数）；
  - 向右（朝向系统）：把 LLM 返回的 `tool_call(name, arguments_json)` 自动反序列化，并注入宿主 `runtime: AgentRuntime` 实例。
"""

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from harness.runtime import AgentRuntime

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

    def execute(
        self,
        name: str,
        raw_arguments: str,
        runtime: AgentRuntime | None = None,
        **context: Any,
    ) -> str:
        """
        根据工具名和原始 JSON 字符串执行工具：
        1. 解析 JSON 参数；
        2. 若传入了 `runtime: AgentRuntime`，先执行 `runtime.check_permission(tool, args)` 权限门禁；
        3. 若工具函数声明了 `runtime` 参数，将 `runtime` 实例注入调用。
        """
        t = self.get(name)
        if t is None:
            return f"[ERROR] Unknown tool: '{name}'"

        try:
            args = json.loads(raw_arguments or "{}")
            if not isinstance(args, dict):
                return f"[ERROR] Tool arguments must be a JSON object, got: {type(args).__name__}"
        except json.JSONDecodeError as e:
            return f"[ERROR] Invalid JSON arguments for tool '{name}': {e}"

        # Step 3: 如果提供了 AgentRuntime，执行 Permission Gate 拦截检查
        if runtime is not None:
            allowed, deny_reason = runtime.check_permission(t, args)
            if not allowed:
                return f"[PERMISSION_DENIED] {deny_reason}"

        call_kwargs = dict(args)
        if t.accepts_runtime:
            call_kwargs["runtime"] = runtime

        try:
            result = t.func(**call_kwargs)
            return str(result)
        except Exception as e:
            return f"[ERROR] Tool '{name}' execution failed: {type(e).__name__}: {e}"


# ============================================================================
# Claude Code 核心四大基础工具：read_file, write_file, edit_file, bash
# ============================================================================


def create_coding_tools(base_dir: Path | None = None) -> ToolRegistry:
    """
    创建 Coding Agent 最核心的四个工具：
    - read_file:  眼睛（查看代码，dangerous=False）
    - write_file: 双手（创建/覆写文件，dangerous=True）
    - edit_file:  手术刀（局部精确替换代码，dangerous=True）
    - bash:       执行器（受控子进程跑命令，dangerous=True）
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
        description="Execute a shell command in the workspace directory and return stdout/stderr.",
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
