from __future__ import annotations

"""
Step 6 核心组件：Skills 渐进式披露管理器 (harness/skills.py)

第一性原理视角：
为什么需要 Skills 机制，而不是把所有操作指南全写进 System Prompt？
- 假设你有 20 个领域的 SOP（代码审查、TDD、SQL 优化、K8s 排障、安全审计……），每个 SOP 500 行（约 2,000 Tokens）。
- 如果全部塞进初始 System Prompt，每次对话还没开始就白白消耗 40,000 Tokens，而且长上下文会稀释模型对当前核心指令的注意力。
- **渐进式披露（Progressive Disclosure）**：
  1. **目录层（Catalog Level，常驻 System Prompt）**：启动时只扫描 `skills/*/SKILL.md` 头部的 YAML Frontmatter（`name` + 一句话 `description`），每个 Skill 只占 ~20 Tokens；
  2. **正文层（Body Level，按需 Tool 加载）**：提供 `load_skill(name)` 工具；只有当 LLM 根据目录判断当前任务命中某个 Skill 时，才调用 `load_skill` 把那篇完整的 Markdown 规程加载进当前对话上下文。
"""

from dataclasses import dataclass
from pathlib import Path

from harness.tool import ToolRegistry, tool


@dataclass
class SkillMeta:
    """单个 Skill 的轻量级元数据（仅包含名称、简述与文件路径）。"""

    name: str
    description: str
    path: Path


def parse_skill_markdown(raw_text: str) -> tuple[dict[str, str], str]:
    """
    使用标准库解析 SKILL.md 的 YAML Frontmatter 与正文：
    返回 `(frontmatter_dict, markdown_body)`。
    """
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw_text.strip()

    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return {}, raw_text.strip()

    meta: dict[str, str] = {}
    for line in lines[1:end_idx]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        k, v = stripped.split(":", 1)
        meta[k.strip()] = v.strip().strip('"').strip("'")

    body = "\n".join(lines[end_idx + 1 :]).strip()
    return meta, body


class SkillManager:
    """管理 `skills/*/SKILL.md` 的扫描、目录渲染与按需加载。"""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = Path(skills_dir).resolve()
        self._skills: dict[str, SkillMeta] = {}
        self.discover()

    def discover(self) -> list[SkillMeta]:
        """扫描 skills_dir 下所有子目录中的 SKILL.md，仅提取头部元数据。"""
        self._skills.clear()
        if not self.skills_dir.exists():
            return []

        for skill_file in sorted(self.skills_dir.glob("*/SKILL.md")):
            raw = skill_file.read_text(encoding="utf-8")
            meta, _ = parse_skill_markdown(raw)
            name = meta.get("name") or skill_file.parent.name
            desc = meta.get("description") or "No description provided."
            self._skills[name] = SkillMeta(name=name, description=desc, path=skill_file)

        return list(self._skills.values())

    def render_catalog_prompt(self) -> str:
        """
        生成注入 System Prompt 的轻量级技能目录（不包含任何技能正文！）。
        """
        if not self._skills:
            return ""

        items = "\n".join(f"- **{s.name}**: {s.description}" for s in self._skills.values())
        return (
            "<available_skills>\n"
            "You have access to specialized skills. Their full instructions are NOT loaded yet to save context.\n"
            "Whenever the user's task matches one of the skills below, you MUST call `load_skill(name=...)` FIRST "
            "before executing the task:\n"
            f"{items}\n"
            "</available_skills>"
        )

    def load_skill_body(self, name: str) -> str:
        """按名称读取对应 SKILL.md 的完整正文内容。"""
        meta = self._skills.get(name)
        if meta is None:
            available = ", ".join(sorted(self._skills.keys())) or "none"
            return f"[ERROR] Unknown skill '{name}'. Available skills: {available}"

        raw = meta.path.read_text(encoding="utf-8")
        _, body = parse_skill_markdown(raw)
        return (
            f"<skill_loaded name=\"{meta.name}\" path=\"{meta.path}\">\n"
            f"{body}\n"
            "</skill_loaded>"
        )

    def register_load_tool(self, registry: ToolRegistry) -> None:
        """向 ToolRegistry 注册 `load_skill` 工具（只读安全工具，dangerous=False）。"""

        @tool(
            name="load_skill",
            description="Load the full step-by-step instructions of a skill by name from <available_skills>.",
            dangerous=False,
        )
        def load_skill(name: str) -> str:
            return self.load_skill_body(name)

        registry.register(load_skill)
