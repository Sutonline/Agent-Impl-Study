from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from harness.runtime import AgentRuntime
from harness.tool import create_coding_tools
from harness.validation import (
    ValidationPipeline,
    make_command_pre_stop_validator,
    path_sandbox_validator,
    python_ast_validator,
    schema_validator,
)


def test_pre_tool_hooks_block_path_escape_and_bad_python_ast(tmp_path: Path):
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    reg = create_coding_tools()
    pipe = ValidationPipeline()
    pipe.add_pre_tool(schema_validator)
    pipe.add_pre_tool(path_sandbox_validator)
    pipe.add_pre_tool(python_ast_validator)

    write_tool = reg.get("write_file")
    assert write_tool is not None

    # 1. Path traversal outside workspace
    res_escape = pipe.run_pre_tool(write_tool, {"path": "../evil.py", "content": "x=1"}, rt)
    assert not res_escape.passed
    assert "outside workspace" in res_escape.reason.lower()

    # 2. Invalid Python syntax on write_file
    res_ast = pipe.run_pre_tool(write_tool, {"path": "broken.py", "content": "def foo(:\n    pass"}, rt)
    assert not res_ast.passed
    assert "syntaxerror" in res_ast.reason.lower()

    # 3. Invalid Python syntax on edit_file simulated in memory (file on disk must remain untouched!)
    (tmp_path / "ok.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    edit_tool = reg.get("edit_file")
    assert edit_tool is not None
    res_edit_ast = pipe.run_pre_tool(
        edit_tool,
        {"path": "ok.py", "old_text": "return a + b", "new_text": "return a +"},
        rt,
    )
    assert not res_edit_ast.passed
    assert "return a + b" in (tmp_path / "ok.py").read_text(encoding="utf-8")


def test_pre_stop_hook_vetoes_stop_when_verification_command_fails(tmp_path: Path):
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    (tmp_path / "app.py").write_text("def inc(x):\n    return x - 1\n", encoding="utf-8")
    rt.modified_files.add(tmp_path / "app.py")

    pipe = ValidationPipeline()
    pipe.add_pre_stop(make_command_pre_stop_validator("python3 -c 'import app; assert app.inc(1) == 2'"))
    res = pipe.run_pre_stop(rt)
    assert not res.passed
    assert "AssertionError" in res.reason


class TestStep04Validation(unittest.TestCase):
    def test_pre_tool_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_pre_tool_hooks_block_path_escape_and_bad_python_ast(Path(tmp))

    def test_pre_stop_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_pre_stop_hook_vetoes_stop_when_verification_command_fails(Path(tmp))


if __name__ == "__main__":
    unittest.main()
