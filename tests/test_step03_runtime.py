from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from harness.runtime import AgentRuntime, extract_command_prefix, split_shell_commands
from harness.tool import create_coding_tools


def test_runtime_permission_gate_blocks_when_denied(tmp_path: Path):
    rt = AgentRuntime(workspace_root=tmp_path, permission_callback=lambda name, args: False)
    reg = create_coding_tools()
    res = reg.execute_structured("write_file", '{"path": "a.txt", "content": "hi"}', runtime=rt)
    assert res.status == "denied"
    assert "PERMISSION_DENIED" in res.content
    assert not (tmp_path / "a.txt").exists()


def test_runtime_subprocess_timeout_and_truncation(tmp_path: Path):
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    res_timeout = rt.run_command("python3 -c 'import time; time.sleep(5)'", timeout=1)
    assert "TIMEOUT" in res_timeout
    res_huge = rt.run_command("python3 -c 'print(\"A\" * 5000)'", max_output_chars=500)
    assert "TRUNCATED" in res_huge
    assert len(res_huge) < 1000


def test_command_prefix_and_compound_splitting(tmp_path: Path):
    # 1. Compound command splitting
    cmds = split_shell_commands("ls -la && git status -s; rm -rf /tmp/foo")
    assert cmds == ["ls -la", "git status -s", "rm -rf /tmp/foo"]

    # 2. Prefix extraction: git/python3 -m take 2+ tokens, normal binaries take 1 token
    assert extract_command_prefix("git status -s") == "git status"
    assert extract_command_prefix("python3 -m unittest discover") == "python3 -m unittest"
    assert extract_command_prefix("rm -rf fib.py") == "rm"

    # 3. Safe read-only commands ('ls', 'git status') auto-pass without prompting user!
    prompted = []
    rt = AgentRuntime(
        workspace_root=tmp_path,
        permission_callback=lambda name, args: (prompted.append((name, args)) or False),
    )
    reg = create_coding_tools()
    bash_tool = reg.get("bash")
    allowed_safe, _ = rt.check_permission(bash_tool, {"command": "ls -la && pwd"})
    assert allowed_safe is True
    assert len(prompted) == 0  # Did not even need to ask user!

    # 4. Compound command mixing safe ('ls') and dangerous ('rm') MUST be intercepted!
    allowed_mixed, _ = rt.check_permission(bash_tool, {"command": "ls -la && rm -f a.txt"})
    assert allowed_mixed is False
    assert len(prompted) == 1

    # 5. Granting prefix 'python3' allows future 'python3' commands, but STILL blocks 'rm'!
    rt.allow_command_prefix("python3")
    allowed_py, _ = rt.check_permission(bash_tool, {"command": "python3 fib.py"})
    assert allowed_py is True
    allowed_rm, _ = rt.check_permission(bash_tool, {"command": "rm -f fib.py"})
    assert allowed_rm is False


def test_runtime_environment_and_path_isolation(tmp_path: Path):
    # Set a fake secret in host environment; child process in Runtime must NOT be able to leak it!
    os.environ["OPENAI_API_KEY"] = "sk-super-secret-host-key-do-not-leak"
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    out = rt.run_command("python3 -c 'import os; print(\"LEAK:\", os.environ.get(\"OPENAI_API_KEY\", \"NONE\"))'")
    assert "sk-super-secret-host-key-do-not-leak" not in out
    assert "LEAK: NONE" in out

    # Path isolation: escaping workspace_root raises PermissionError
    try:
        rt.resolve_path("../../etc/passwd")
        assert False, "Should have blocked path escape"
    except PermissionError:
        pass


class TestStep03Runtime(unittest.TestCase):
    def test_permission_gate_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_runtime_permission_gate_blocks_when_denied(Path(tmp))

    def test_subprocess_timeout_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_runtime_subprocess_timeout_and_truncation(Path(tmp))

    def test_command_prefix_and_splitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_command_prefix_and_compound_splitting(Path(tmp))

    def test_env_and_path_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_runtime_environment_and_path_isolation(Path(tmp))


if __name__ == "__main__":
    unittest.main()
