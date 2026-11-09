from __future__ import annotations

"""
Step 6 核心组件：Context Grounding 环境锚定与动态提示词组装 (harness/context.py)

第一性原理视角：
裸 LLM 不知道自己运行在什么操作系统、当前工作目录在哪、目录里有哪些文件，也不知道这个项目有什么特定规范。
`build_system_prompt` 负责在会话启动时采集真实环境快照并组装成结构化的 System Prompt：
  1. `<environment>`：操作系统、工作目录路径 `workspace_root`、当前目录下的文件列表概览；
  2. `<project_rules>`：自动探测并注入工作区中的 `AGENTS.md` 或 `CLAUDE.md` 常驻规则；
  3. `<available_skills>`：注入 `SkillManager.render_catalog_prompt()` 的轻量级技能目录。
"""

import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.runtime import AgentRuntime
    from harness.skills import SkillManager


def _load_project_rules(runtime: AgentRuntime) -> str:
    """在 workspace_root 中查找 AGENTS.md 或 CLAUDE.md 并读取内容。"""
    chunks: list[str] = []
    for filename in ("AGENTS.md", "CLAUDE.md"):
        candidate = runtime.workspace_root / filename
        if candidate.is_file():
            content = candidate.read_text(encoding="utf-8").strip()
            if content:
                chunks.append(f"### {filename}\n{content}")
    return "\n\n".join(chunks)


def _list_workspace_entries(runtime: AgentRuntime, limit: int = 30) -> str:
    """列出工作区顶层可见文件，帮助模型第一轮就具备环境感知，减少盲目的 ls 调用。"""
    if not runtime.workspace_root.exists():
        return "(empty)"
    entries = sorted(
        p.name + ("/" if p.is_dir() else "")
        for p in runtime.workspace_root.iterdir()
        if not p.name.startswith(".")
    )
    if not entries:
        return "(empty workspace)"
    if len(entries) > limit:
        entries = entries[:limit] + [f"... ({len(entries) - limit} more)"]
    return ", ".join(entries)


def build_system_prompt(
    runtime: AgentRuntime,
    skill_manager: SkillManager | None = None,
) -> str:
    """组装包含环境快照、项目级规则（AGENTS.md）与技能目录的完整 System Prompt。"""
    sections: list[str] = [
        "You are a coding agent operating inside an isolated workspace runtime.",
        (
            "<environment>\n"
            f"- OS: {platform.system()} ({platform.release()})\n"
            f"- Workspace Root: {runtime.workspace_root}\n"
            f"- Working Directory: {runtime.cwd or runtime.workspace_root}\n"
            f"- Workspace Files: {_list_workspace_entries(runtime)}\n"
            "</environment>"
        ),
    ]

    rules_text = _load_project_rules(runtime)
    if rules_text:
        sections.append(
            "<project_rules>\n"
            "The following project rules from the workspace MUST be strictly followed:\n\n"
            f"{rules_text}\n"
            "</project_rules>"
        )

    if skill_manager is not None:
        catalog = skill_manager.render_catalog_prompt()
        if catalog:
            sections.append(catalog)

    return "\n\n".join(sections)
