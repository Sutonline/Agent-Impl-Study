from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from harness.runtime import AgentRuntime
from harness.tool import create_coding_tools


def test_runtime_permission_gate_blocks_when_denied(tmp_path: Path):
    rt = AgentRuntime(workspace_root=tmp_path, permission_callback=lambda name, args: False)
    reg = create_coding_tools()
    out = reg.execute("write_file", '{"path": "a.txt", "content": "hi"}', runtime=rt)
    assert "PERMISSION_DENIED" in out
    assert not (tmp_path / "a.txt").exists()


def test_runtime_subprocess_timeout_and_truncation(tmp_path: Path):
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    res_timeout = rt.run_command("python3 -c 'import time; time.sleep(5)'", timeout=1)
    assert "TIMEOUT" in res_timeout
    res_huge = rt.run_command("python3 -c 'print(\"A\" * 5000)'", max_output_chars=500)
    assert "TRUNCATED" in res_huge
    assert len(res_huge) < 1000


class TestStep03Runtime(unittest.TestCase):
    def test_permission_gate_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_runtime_permission_gate_blocks_when_denied(Path(tmp))

    def test_subprocess_timeout_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_runtime_subprocess_timeout_and_truncation(Path(tmp))


if __name__ == "__main__":
    unittest.main()
