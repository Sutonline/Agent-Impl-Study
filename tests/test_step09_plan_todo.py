from __future__ import annotations

import json
import unittest

from harness.runtime import AgentRuntime
from harness.todo import TodoItem, TodoManager
from harness.tool import create_coding_tools
from harness.validation import create_default_validation_pipeline


class TestStep09PlanTodo(unittest.TestCase):
    def test_plan_mode_blocks_mutating_tools(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

            rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
            reg = create_coding_tools(base_dir=tmp_path)
            pipe = create_default_validation_pipeline()

            todo_mgr = TodoManager(mode="plan")
            todo_mgr.register_todo_tools(reg)
            pipe.add_pre_tool(todo_mgr.make_plan_mode_pre_tool_hook())
            pipe.add_pre_stop(todo_mgr.make_todo_pre_stop_hook())

            # 1. read_file is allowed in plan mode
            read_res = reg.execute_structured("read_file", '{"path": "app.py"}', runtime=rt, validation=pipe)
            self.assertEqual(read_res.status, "ok")
            self.assertIn("x = 1", read_res.content)

            # 2. write_file is blocked in plan mode by pre_tool_use hook
            write_res = reg.execute_structured(
                "write_file",
                '{"path": "app.py", "content": "x = 2\\n"}',
                runtime=rt,
                validation=pipe,
            )
            self.assertEqual(write_res.status, "denied")
            self.assertIn("PLAN_MODE", write_res.content)

            # 3. todo_write is allowed in plan mode
            todo_args = json.dumps(
                {
                    "todos": [
                        {"id": "1", "content": "Update x to 2", "status": "in_progress"},
                        {"id": "2", "content": "Verify x", "status": "pending"},
                    ]
                }
            )
            todo_res = reg.execute_structured("todo_write", todo_args, runtime=rt, validation=pipe)
            self.assertEqual(todo_res.status, "ok")
            self.assertEqual(len(todo_mgr.items), 2)
            self.assertTrue(todo_mgr.has_incomplete())

            # 4. exit_plan_mode switches mode from 'plan' to 'act'
            exit_res = reg.execute_structured(
                "exit_plan_mode",
                '{"plan_summary": "Update x=2 in app.py and verify"}',
                runtime=rt,
                validation=pipe,
            )
            self.assertEqual(exit_res.status, "ok")
            self.assertEqual(todo_mgr.mode, "act")

            # 5. Now write_file succeeds in 'act' mode
            write_res2 = reg.execute_structured(
                "write_file",
                '{"path": "app.py", "content": "x = 2\\n"}',
                runtime=rt,
                validation=pipe,
            )
            self.assertEqual(write_res2.status, "ok")

    def test_todo_pre_stop_gatekeeper_and_single_in_progress_rule(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            rt = AgentRuntime(workspace_root=Path(tmp))
            todo_mgr = TodoManager(mode="act")
            pipe = create_default_validation_pipeline()
            pipe.add_pre_stop(todo_mgr.make_todo_pre_stop_hook())

            # Rule check: at most 1 in_progress item allowed
            with self.assertRaises(ValueError):
                todo_mgr.update(
                    [
                        {"id": "1", "content": "Task A", "status": "in_progress"},
                        {"id": "2", "content": "Task B", "status": "in_progress"},
                    ]
                )

            todo_mgr.update(
                [
                    {"id": "1", "content": "Task A", "status": "completed"},
                    {"id": "2", "content": "Task B", "status": "in_progress"},
                ]
            )
            # pre_stop must veto while Task B is still in_progress
            stop_res = pipe.run_pre_stop(rt)
            self.assertFalse(stop_res.passed)
            self.assertIn("Task B", stop_res.reason)

            # Mark all completed -> pre_stop passes
            todo_mgr.update(
                [
                    {"id": "1", "content": "Task A", "status": "completed"},
                    {"id": "2", "content": "Task B", "status": "completed"},
                ]
            )
            stop_res2 = pipe.run_pre_stop(rt)
            self.assertTrue(stop_res2.passed)


if __name__ == "__main__":
    unittest.main()
