from __future__ import annotations

"""轻量终端日志打印工具（使用标准 ANSI 颜色区分 Agent 内部状态）。"""

COLORS = {
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "magenta": "\033[95m",
    "blue": "\033[94m",
    "gray": "\033[90m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def log_step(tag: str, message: str, color: str = "cyan") -> None:
    """在终端打印带颜色标签的 Harness 阶段日志，例如 [THINK], [TOOL CALL], [VALIDATION]。"""
    c = COLORS.get(color, COLORS["cyan"])
    bold = COLORS["bold"]
    reset = COLORS["reset"]
    print(f"{bold}{c}[{tag}]{reset} {message}")
