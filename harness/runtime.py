from __future__ import annotations

"""
Step 3 核心组件：AgentRuntime（环境隔离沙盒 + 前缀级权限引擎）(harness/runtime.py)

第一性原理视角：
一个真正工业级的 `AgentRuntime` 不仅是存放 `messages` 的状态容器，更承担两大硬核职责：
1. **沙盒隔离（Sandbox Isolation）**：
   - **环境变量净室隔离（Env Sanitization）**：子进程绝不能继承宿主机的敏感环境变量（如 `OPENAI_API_KEY`、AWS 密钥），只暴露最小白名单变量（`PATH`, `LANG`, `TERM`），并将 `TMPDIR` 关进工作区内。
   - **文件系统边界隔离（Filesystem Jail）**：`resolve_path()` 解析软链接与 `..` 后强制校验路径必须在 `workspace_root` 内部，越界直接拒绝。
   - **OS 内核沙盒策略（macOS Seatbelt `sandbox-exec`）**：提供内核级策略生成，禁止子进程联网或写工作区以外的目录。
2. **细粒度命令权限引擎（Command Prefix Permission Gate）**：
   - 先拦截危险动态语法（`$()`, 反引号, `eval`, 重定向 `>`）；
   - 按 `&&`, `||`, `;`, `|` 将复合命令拆分为原子子命令，逐条检查；
   - 提取 `(程序名 + 子命令)` 前缀（如 `git status`, `python3 -m unittest`）：
     * 命中只读白名单（`ls`, `pwd`, `git status` 等）自动免批放行；
     * 用户选择 `a`（总是允许）时，仅将当前命令前缀记入 `allowed_command_prefixes`，绝不给 `rm` 等高危命令开绿灯。
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import TYPE_CHECKING, Any, Callable

from harness.ui import log_step

if TYPE_CHECKING:
    from harness.tool import Tool

PermissionCallback = Callable[[str, dict[str, Any]], bool]

# 只读安全命令前缀白名单（无需打扰用户，自动放行）
SAFE_READONLY_PREFIXES: set[str] = {
    "ls",
    "pwd",
    "whoami",
    "date",
    "uname",
    "git status",
    "git diff",
    "git log",
    "git branch",
}

# 高危命令黑名单（即使开启前缀记忆，也绝不允许被泛化加入自动放行列表）
HIGH_RISK_BINARIES: set[str] = {
    "rm",
    "sudo",
    "su",
    "chmod",
    "chown",
    "dd",
    "mkfs",
    "kill",
    "pkill",
    "curl",
    "wget",
    "eval",
}

# 需要取 2~3 个词作为前缀的多子命令工具（如 `git status`, `python3 -m unittest`）
MULTI_TOKEN_BINARIES: set[str] = {"git", "npm", "pnpm", "yarn", "cargo", "docker", "kubectl"}


def has_dangerous_shell_syntax(command: str) -> tuple[bool, str]:
    """检查命令是否包含命令替换 `$()`、反引号、eval 或文件重定向等高危动态语法。"""
    if "$(" in command or "`" in command:
        return True, "contains command substitution ($() or backticks)"
    if re.search(r"(^|[;&|]\s*)eval\s+", command):
        return True, "contains dynamic 'eval'"
    if ">" in command:
        return True, "contains output redirection (>)"
    return False, ""


def split_shell_commands(command: str) -> list[str]:
    """按 Shell 连接符 (&&, ||, ;, |) 拆分复合命令为独立的原子子命令列表。"""
    parts = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_command_prefix(subcommand: str) -> str:
    """
    从单条原子命令中提取用于权限匹配的【程序名 + 子命令】前缀：
    - `git status -s` -> `"git status"`
    - `python3 -m unittest discover` -> `"python3 -m unittest"`
    - `python3 -c "..."` -> `"python3 -c"` (含动态代码，单独归类)
    - `python3 fib.py` -> `"python3"`
    - `rm -rf /tmp/a` -> `"rm"`
    """
    try:
        tokens = shlex.split(subcommand)
    except ValueError:
        return subcommand.strip()

    if not tokens:
        return ""

    # 跳过形如 KEY=VAL 的前置环境变量赋值
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    if not tokens:
        return ""

    binary = Path(tokens[0]).name
    if binary in {"python", "python3"} and len(tokens) >= 2:
        if tokens[1] == "-m" and len(tokens) >= 3:
            return f"{binary} -m {tokens[2]}"
        if tokens[1] == "-c":
            return f"{binary} -c"
        return binary

    if binary in MULTI_TOKEN_BINARIES and len(tokens) >= 2 and not tokens[1].startswith("-"):
        return f"{binary} {tokens[1]}"

    return binary


def has_dangerous_flags(subcommand: str) -> bool:
    """扫描单条命令的参数中是否包含破坏性 Flag（如 find -delete / -exec, git --force / --hard）。"""
    try:
        tokens = shlex.split(subcommand)
    except ValueError:
        return True
    dangerous_flags = {"-delete", "-exec", "--force", "-f", "--hard", "--no-preserve-root"}
    binary = Path(tokens[0]).name if tokens else ""
    if binary in {"find", "git"} and any(t in dangerous_flags for t in tokens[1:]):
        return True
    if binary in {"python", "python3"} and "-c" in tokens[1:]:
        return True
    return False


@dataclass
class AgentRuntime:
    """
    Agent 运行时环境与隔离沙盒容器（Environment + Isolation + Permission Engine）。
    """

    workspace_root: Path = field(default_factory=lambda: (Path.cwd() / "workspace").resolve())
    cwd: Path | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    modified_files: set[Path] = field(default_factory=set)
    always_allow_dangerous: bool = False
    allowed_command_prefixes: set[str] = field(default_factory=set)
    permission_callback: PermissionCallback | None = None
    enforce_path_sandbox: bool = True
    use_os_sandbox: bool = False

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        if self.cwd is None:
            self.cwd = self.workspace_root
        else:
            self.cwd = Path(self.cwd).resolve()

    def allow_command_prefix(self, prefix: str) -> None:
        """将指定命令前缀加入当前会话的授信列表（高危二进制除外）。"""
        binary = prefix.split()[0] if prefix.strip() else ""
        if binary and binary not in HIGH_RISK_BINARIES and prefix not in {"python -c", "python3 -c"}:
            self.allowed_command_prefixes.add(prefix)

    def resolve_path(self, path_str: str, enforce_sandbox: bool | None = None) -> Path:
        """
        路径解析与文件系统边界隔离（Filesystem Jail）：
        将相对路径解析为基于 `self.cwd` 的物理路径；
        若开启 `enforce_path_sandbox`，任何跳出 `self.workspace_root` 的路径都会直接抛出 `PermissionError`。
        """
        p = Path(path_str)
        base = self.cwd or self.workspace_root
        resolved = p.resolve() if p.is_absolute() else (base / p).resolve()

        check_jail = self.enforce_path_sandbox if enforce_sandbox is None else enforce_sandbox
        if check_jail:
            try:
                resolved.relative_to(self.workspace_root)
            except ValueError as e:
                raise PermissionError(
                    f"Sandbox violation: path '{path_str}' resolves to '{resolved}', "
                    f"which is outside workspace_root '{self.workspace_root}'."
                ) from e
        return resolved

    def build_isolated_env(self) -> dict[str, str]:
        """
        环境变量净室隔离（Env Sanitization）：
        剥离宿主机所有敏感环境变量（如 OPENAI_API_KEY、云厂商凭据、SSH 代理等），
        仅保留最小系统运行变量，并将 TMPDIR 重定向到工作区内部。
        """
        isolated_tmp = self.workspace_root / ".runtime_tmp"
        isolated_tmp.mkdir(parents=True, exist_ok=True)

        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "TERM": os.environ.get("TERM", "xterm-256color"),
            "HOME": str(self.workspace_root),
            "TMPDIR": str(isolated_tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def build_macos_seatbelt_profile(self) -> str:
        """
        生成 macOS 内核级 `sandbox-exec` (Seatbelt) 隔离策略：
        - 禁止一切网络访问 `(deny network*)`；
        - 默认禁止文件写入，仅允许写入 `workspace_root` 与 `/private/tmp`。
        """
        ws = str(self.workspace_root)
        return (
            "(version 1)\n"
            "(allow default)\n"
            "(deny network*)\n"
            "(deny file-write*)\n"
            f'(allow file-write* (subpath "{ws}"))\n'
            '(allow file-write* (subpath "/private/tmp"))\n'
            '(allow file-write* (literal "/dev/null"))\n'
        )

    def _is_bash_command_auto_allowed(self, command: str) -> tuple[bool, list[str], str]:
        """
        对 bash 命令执行四层安全分析：
        返回 `(是否全部子命令已自动获批, 涉及的前缀列表, 需要人工审批的原因)`。
        """
        has_dyn, dyn_reason = has_dangerous_shell_syntax(command)
        if has_dyn:
            return False, [], f"Dynamic shell syntax ({dyn_reason})"

        subcommands = split_shell_commands(command)
        if not subcommands:
            return True, [], ""

        prefixes: list[str] = []
        unapproved_reasons: list[str] = []

        for sub in subcommands:
            prefix = extract_command_prefix(sub)
            prefixes.append(prefix)
            binary = prefix.split()[0] if prefix else ""

            if binary in HIGH_RISK_BINARIES:
                unapproved_reasons.append(f"high-risk binary '{binary}' in `{sub}`")
                continue
            if has_dangerous_flags(sub):
                unapproved_reasons.append(f"dangerous flag in `{sub}`")
                continue
            if prefix in SAFE_READONLY_PREFIXES or prefix in self.allowed_command_prefixes:
                continue

            unapproved_reasons.append(f"unapproved command prefix '{prefix}' in `{sub}`")

        if unapproved_reasons:
            return False, prefixes, "; ".join(unapproved_reasons)
        return True, prefixes, ""

    def check_permission(self, tool_obj: Tool, args: dict[str, Any]) -> tuple[bool, str]:
        """
        Permission Gate（安全护栏）：
        1. 若 `tool_obj.dangerous` 为 False（如 `read_file`）或 `always_allow_dangerous` 为 True，直接放行；
        2. 若是 `bash` 工具，先运行四层命令前缀拆解检查：
           - 若整条命令的所有子命令都命中 `SAFE_READONLY_PREFIXES` 或 `allowed_command_prefixes`，免审批自动放行！
        3. 否则调用 `permission_callback` 或在终端询问用户 `[y=允许本次 / n=拒绝 / a=总是允许该前缀]`。
        """
        if not tool_obj.dangerous or self.always_allow_dangerous:
            return True, ""

        cmd_prefixes: list[str] = []
        reason_detail = ""
        if tool_obj.name == "bash":
            auto_ok, cmd_prefixes, reason_detail = self._is_bash_command_auto_allowed(str(args.get("command", "")))
            if auto_ok:
                log_step("PERMISSION GATE", f"只读/已授信命令前缀 {cmd_prefixes} -> 自动放行", color="green")
                return True, ""

        if self.permission_callback is not None:
            allowed = self.permission_callback(tool_obj.name, args)
            if not allowed:
                return (
                    False,
                    f"User denied permission to execute '{tool_obj.name}' ({reason_detail or 'policy denied'}). "
                    "Please respect the user's decision and try a safe alternative.",
                )
            return True, ""

        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 200:
            preview = preview[:200] + "...(truncated)"
        hint = f" [原因: {reason_detail}]" if reason_detail else ""
        log_step("PERMISSION GATE", f"拦截高危操作: {tool_obj.name}({preview}){hint}", color="yellow")
        try:
            ans = input("  允许执行吗？[y=允许本次 / n=拒绝 / a=记住并允许该命令前缀]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans in {"a", "always"}:
            if tool_obj.name == "bash" and cmd_prefixes:
                for p in cmd_prefixes:
                    self.allow_command_prefix(p)
                log_step(
                    "PERMISSION GATE",
                    f"已记住授信命令前缀: {sorted(self.allowed_command_prefixes)}（高危命令如 rm 仍会被拦截）",
                    color="green",
                )
            else:
                self.always_allow_dangerous = True
                log_step("PERMISSION GATE", "已开启本会话全部允许模式。", color="green")
            return True, ""

        if ans in {"y", "yes"}:
            return True, ""

        return (
            False,
            f"User denied permission to execute '{tool_obj.name}'. Do not repeat the exact same call without user consent.",
        )

    def run_command(
        self,
        command: str,
        timeout: int = 15,
        max_output_chars: int = 8000,
    ) -> str:
        """
        受控沙盒子进程执行器：
        1. 使用 `build_isolated_env()` 净室环境变量（隔离宿主机 API Key 等机密）；
        2. 限定在 `self.cwd` 目录下运行（可选启用 macOS `sandbox-exec` 内核沙盒）；
        3. 强制 `timeout` 超时保护与 `max_output_chars` 输出截断保护。
        """
        isolated_env = self.build_isolated_env()
        exec_cmd: str | list[str] = command
        use_shell = True

        if self.use_os_sandbox and Path("/usr/bin/sandbox-exec").exists():
            profile = self.build_macos_seatbelt_profile()
            exec_cmd = ["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-c", command]
            use_shell = False

        try:
            proc = subprocess.run(
                exec_cmd,
                shell=use_shell,
                cwd=str(self.cwd or self.workspace_root),
                env=isolated_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return (
                f"[ERROR] [TIMEOUT] Command timed out after {timeout} seconds: `{command}`. "
                "RECOVERY_HINT: Avoid long-running or interactive blocking commands."
            )

        combined = (proc.stdout or "") + (proc.stderr or "")
        combined = combined.strip()

        if len(combined) > max_output_chars:
            half = max_output_chars // 2
            omitted = len(combined) - max_output_chars
            combined = (
                f"{combined[:half]}\n"
                f"... [TRUNCATED {omitted} characters by AgentRuntime] ...\n"
                f"{combined[-half:]}"
            )

        return f"[exit_code={proc.returncode}]\n{combined}"
