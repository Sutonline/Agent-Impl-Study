from __future__ import annotations

"""
Step 4 核心组件：Validation 三段式生命周期 Hook 管道 (harness/validation.py)

设计机制：
将校验逻辑从主循环和工具函数中解耦，抽象为挂载在 3 个生命周期节点上的可插拔 Hook 链：
  1. `pre_tool_use`  (Tool, args, AgentRuntime) -> ValidationResult
     在工具真正执行、修改磁盘或发起人工权限弹窗【之前】运行。任一 Hook 不通过则立即短路拦截。
  2. `post_tool_use` (Tool, args, output, AgentRuntime) -> ValidationResult
     在工具执行完毕【之后】运行，用于检查执行副作用或追加告警。
  3. `pre_stop`      (AgentRuntime) -> ValidationResult
     当 LLM 不再返回 `tool_calls`、准备退出循环向用户交卷【之前】运行。
     若不通过，主循环拒绝退出（veto stop），将失败原因作为新消息注入 `messages`，逼迫 LLM 修复。
"""

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from harness.runtime import AgentRuntime
    from harness.tool import Tool


@dataclass
class ValidationResult:
    """单次校验的结构化返回结果。"""

    passed: bool
    reason: str = ""
    recovery_hint: str = ""
    status: str = "error"

    def format_feedback(self, stage: str) -> str:
        msg = f"[VALIDATION_FAILED:{stage}] {self.reason}"
        if self.recovery_hint:
            msg += f"\nRECOVERY_HINT: {self.recovery_hint}"
        return msg


PreToolHook = Callable[["Tool", dict[str, Any], "AgentRuntime"], ValidationResult]
PostToolHook = Callable[["Tool", dict[str, Any], str, "AgentRuntime"], ValidationResult]
PreStopHook = Callable[["AgentRuntime"], ValidationResult]


@dataclass
class ValidationPipeline:
    """三段式校验拦截器管道。"""

    pre_tool_hooks: list[PreToolHook] = field(default_factory=list)
    post_tool_hooks: list[PostToolHook] = field(default_factory=list)
    pre_stop_hooks: list[PreStopHook] = field(default_factory=list)

    def add_pre_tool(self, hook: PreToolHook) -> None:
        self.pre_tool_hooks.append(hook)

    def add_post_tool(self, hook: PostToolHook) -> None:
        self.post_tool_hooks.append(hook)

    def add_pre_stop(self, hook: PreStopHook) -> None:
        self.pre_stop_hooks.append(hook)

    def run_pre_tool(self, tool_obj: Tool, args: dict[str, Any], runtime: AgentRuntime) -> ValidationResult:
        for hook in self.pre_tool_hooks:
            res = hook(tool_obj, args, runtime)
            if not res.passed:
                return res
        return ValidationResult(passed=True)

    def run_post_tool(
        self,
        tool_obj: Tool,
        args: dict[str, Any],
        output: str,
        runtime: AgentRuntime,
    ) -> ValidationResult:
        for hook in self.post_tool_hooks:
            res = hook(tool_obj, args, output, runtime)
            if not res.passed:
                return res
        return ValidationResult(passed=True)

    def run_pre_stop(self, runtime: AgentRuntime) -> ValidationResult:
        for hook in self.pre_stop_hooks:
            res = hook(runtime)
            if not res.passed:
                return res
        return ValidationResult(passed=True)


# ============================================================================
# 内置标准校验 Hooks
# ============================================================================

PY_TYPE_CHECK: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def schema_validator(tool_obj: Tool, args: dict[str, Any], runtime: AgentRuntime) -> ValidationResult:
    """
    [pre_tool_use Hook 1] 参数必填项与基础类型校验：
    根据 Tool 的 JSON Schema 检查必填字段是否缺失、字段类型是否匹配。
    """
    schema = tool_obj.parameters
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for req_key in required:
        if req_key not in args:
            return ValidationResult(
                passed=False,
                reason=f"Missing required argument '{req_key}' for tool '{tool_obj.name}'.",
                recovery_hint=f"Provide all required arguments: {required}.",
            )

    for key, val in args.items():
        if key not in properties:
            return ValidationResult(
                passed=False,
                reason=f"Unexpected argument '{key}' for tool '{tool_obj.name}'.",
                recovery_hint=f"Allowed arguments are: {list(properties.keys())}.",
            )
        expected_json_type = properties[key].get("type", "string")
        py_type = PY_TYPE_CHECK.get(expected_json_type)
        # 注意 Python 中 isinstance(True, int) 为 True，需单独排除 bool 冒充 int
        if expected_json_type in {"integer", "number"} and isinstance(val, bool):
            return ValidationResult(
                passed=False,
                reason=f"Argument '{key}' expected {expected_json_type}, got boolean.",
                recovery_hint=f"Pass a valid {expected_json_type} value for '{key}'.",
            )
        if py_type and not isinstance(val, py_type):
            return ValidationResult(
                passed=False,
                reason=f"Argument '{key}' expected {expected_json_type}, got {type(val).__name__}.",
                recovery_hint=f"Pass a valid {expected_json_type} value for '{key}'.",
            )

    return ValidationResult(passed=True)


def path_sandbox_validator(tool_obj: Tool, args: dict[str, Any], runtime: AgentRuntime) -> ValidationResult:
    """
    [pre_tool_use Hook 2] 文件路径越界校验：
    对所有包含 `path` 参数的工具调用，在执行前校验解析后的物理路径是否位于 `runtime.workspace_root` 内。
    """
    if "path" not in args:
        return ValidationResult(passed=True)

    raw_path = str(args["path"])
    try:
        runtime.resolve_path(raw_path, enforce_sandbox=True)
    except PermissionError as e:
        return ValidationResult(
            passed=False,
            reason=f"Path '{raw_path}' is outside workspace '{runtime.workspace_root}' ({e}).",
            recovery_hint="Use a relative path inside the workspace directory without '..' traversal.",
        )
    return ValidationResult(passed=True)


def python_ast_validator(tool_obj: Tool, args: dict[str, Any], runtime: AgentRuntime) -> ValidationResult:
    """
    [pre_tool_use Hook 3] Python 代码内存 AST 语法预校验：
    当调用 `write_file` 或 `edit_file` 修改 `.py` 文件时，在写入磁盘【之前】先在内存中计算目标文本并运行 `ast.parse`。
    一旦存在语法错误，直接短路拦截，保证坏代码绝不落盘污染工作区。
    """
    if tool_obj.name not in {"write_file", "edit_file"}:
        return ValidationResult(passed=True)

    raw_path = str(args.get("path", ""))
    if not raw_path.endswith(".py"):
        return ValidationResult(passed=True)

    try:
        target = runtime.resolve_path(raw_path, enforce_sandbox=True)
    except PermissionError:
        # 路径越界由 path_sandbox_validator 处理
        return ValidationResult(passed=True)

    if tool_obj.name == "write_file":
        candidate_source = str(args.get("content", ""))
    else:
        # edit_file：在内存中模拟替换后的文件内容
        if not target.exists():
            return ValidationResult(passed=True)
        current_source = target.read_text(encoding="utf-8")
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        if old_text not in current_source or current_source.count(old_text) > 1:
            # 匹配问题交给 edit_file 自身返回精细提示
            return ValidationResult(passed=True)
        candidate_source = current_source.replace(old_text, new_text, 1)

    try:
        ast.parse(candidate_source, filename=raw_path)
    except SyntaxError as e:
        line_info = f"line {e.lineno}, column {e.offset}" if e.lineno else "unknown line"
        bad_line = (e.text or "").strip()
        return ValidationResult(
            passed=False,
            reason=f"Python SyntaxError in '{raw_path}' at {line_info}: {e.msg} (near `{bad_line}`). File was NOT modified.",
            recovery_hint="Fix the Python syntax error (check parentheses, colons, indentation) and call the tool again.",
        )

    return ValidationResult(passed=True)


def non_empty_file_post_validator(
    tool_obj: Tool,
    args: dict[str, Any],
    output: str,
    runtime: AgentRuntime,
) -> ValidationResult:
    """
    [post_tool_use Hook] 写入后非空校验：
    防止 LLM 误传空字符串把代码文件清空。
    """
    if tool_obj.name not in {"write_file", "edit_file"} or "path" not in args:
        return ValidationResult(passed=True)

    try:
        target = runtime.resolve_path(str(args["path"]), enforce_sandbox=True)
    except PermissionError:
        return ValidationResult(passed=True)

    if target.exists() and target.stat().st_size == 0:
        return ValidationResult(
            passed=False,
            reason=f"File '{args['path']}' is empty (0 bytes) after {tool_obj.name}.",
            recovery_hint="Write valid non-empty implementation content to the file.",
        )
    return ValidationResult(passed=True)


def make_command_pre_stop_validator(command: str) -> PreStopHook:
    """
    [pre_stop Hook 工厂函数] 交卷前自动化测试/验证守门员：
    当 LLM 不再发起 `tool_calls` 准备结束任务时，如果本次会话修改过文件（`runtime.modified_files` 非空），
    自动在受控 `runtime` 中运行指定的验证命令（如 `pytest` 或断言脚本）。
    若退出码不为 0，则否决停止（veto stop），将报错输出打回主循环。
    """

    def _pre_stop_hook(runtime: AgentRuntime) -> ValidationResult:
        if not runtime.modified_files:
            return ValidationResult(passed=True)

        cmd_out = runtime.run_command(command)
        if cmd_out.startswith("[exit_code=0]"):
            return ValidationResult(passed=True)

        return ValidationResult(
            passed=False,
            reason=f"Verification command `{command}` failed before task completion:\n{cmd_out}",
            recovery_hint="Analyze the test/verification failure above, fix the code using edit_file/write_file, and verify again.",
        )

    return _pre_stop_hook


def create_default_validation_pipeline(pre_stop_command: str | None = None) -> ValidationPipeline:
    """组装包含三段式标准校验器的默认管道。"""
    pipe = ValidationPipeline()
    pipe.add_pre_tool(schema_validator)
    pipe.add_pre_tool(path_sandbox_validator)
    pipe.add_pre_tool(python_ast_validator)
    pipe.add_post_tool(non_empty_file_post_validator)
    if pre_stop_command:
        pipe.add_pre_stop(make_command_pre_stop_validator(pre_stop_command))
    return pipe
