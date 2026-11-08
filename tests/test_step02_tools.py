from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from harness.tool import ToolRegistry, create_coding_tools, tool


def test_tool_decorator_generates_openai_schema():
    @tool(description="Add two integers")
    def add(a: int, b: int, label: str = "sum") -> str:
        return f"{label}: {a + b}"

    reg = ToolRegistry()
    reg.register(add)
    schemas = reg.openai_schemas()
    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert fn["name"] == "add"
    assert fn["parameters"]["properties"]["a"]["type"] == "integer"
    assert set(fn["parameters"]["required"]) == {"a", "b"}
    assert reg.execute("add", '{"a": 3, "b": 4}') == "sum: 7"


def test_coding_tools_read_write_edit_and_malformed_json(tmp_path: Path):
    reg = create_coding_tools(base_dir=tmp_path)
    assert "OK" in reg.execute("write_file", '{"path": "hello.py", "content": "x = 1\\ny = 2\\n"}')
    assert "x = 1" in reg.execute("read_file", '{"path": "hello.py"}')
    assert "OK" in reg.execute("edit_file", '{"path": "hello.py", "old_text": "x = 1", "new_text": "x = 10"}')
    assert "x = 10" in reg.execute("read_file", '{"path": "hello.py"}')
    err = reg.execute("read_file", "{bad json")
    assert "[ERROR]" in err


class TestStep02Tools(unittest.TestCase):
    def test_decorator_schema(self):
        test_tool_decorator_generates_openai_schema()

    def test_coding_tools_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_coding_tools_read_write_edit_and_malformed_json(Path(tmp))


if __name__ == "__main__":
    unittest.main()
