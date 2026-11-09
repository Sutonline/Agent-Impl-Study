from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from harness.context import build_system_prompt
from harness.runtime import AgentRuntime
from harness.skills import SkillManager
from harness.tool import ToolRegistry


def test_skill_discovery_and_progressive_loading(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Secret workflow for widgets\n---\n# Detailed Steps\nStep A: Do X\n",
        encoding="utf-8",
    )
    sm = SkillManager(tmp_path / "skills")
    metas = sm.discover()
    assert len(metas) == 1
    assert metas[0].name == "my-skill"

    # Catalog prompt contains description but NOT detailed body
    catalog = sm.render_catalog_prompt()
    assert "Secret workflow for widgets" in catalog
    assert "Step A: Do X" not in catalog

    # Loading the skill returns the full body
    reg = ToolRegistry()
    sm.register_load_tool(reg)
    body = reg.execute("load_skill", '{"name": "my-skill"}')
    assert "Step A: Do X" in body


def test_build_system_prompt_injects_agents_md(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("Always use snake_case for function names.", encoding="utf-8")
    rt = AgentRuntime(workspace_root=tmp_path)
    prompt = build_system_prompt(rt)
    assert "Always use snake_case" in prompt
    assert str(tmp_path) in prompt


class TestStep06ContextSkills(unittest.TestCase):
    def test_skill_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_skill_discovery_and_progressive_loading(Path(tmp))

    def test_system_prompt_grounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_build_system_prompt_injects_agents_md(Path(tmp))


if __name__ == "__main__":
    unittest.main()
