# 从第一性原理构建 Agent (`Agent = LLM + Harness`)

本项目从零出发，仅使用 Python 标准库与 OpenAI 接口，分 9 个独立可运行的步骤构建一个精简版 Claude Code，直观展示 `Harness` 的每一层机制是如何解决裸 LLM 的工程痛点的。

## 演进路线图与运行命令

| 步骤 | 核心模块 (`harness/`) | 独立运行入口 (`steps/`) | 解决的第一性原理问题 |
| :--- | :--- | :--- | :--- |
| **Step 1** | `llm.py`, `ui.py` | `python3 steps/step01_bare_loop.py --demo` | 裸 LLM 只会一问一答，如何通过 `while` 循环与 `tool_calls` 实现多步自主决策？ |
| **Step 2** | `tool.py` | `python3 steps/step02_tools.py --demo` | 如何通过 `@tool` 反射消除手写 JSON Schema，并装上 `read/write/edit/bash` 四大基础工具？ |
| **Step 3** | `runtime.py` | `python3 steps/step03_runtime.py --demo` | 如何实现环境变量净室隔离、路径边界隔离，以及拆解复合命令的前缀级权限门禁（Permission Gate）？ |
| **Step 4** | `validation.py` | `python3 steps/step04_validation.py --demo` | 如何通过 `pre_tool_use`（内存 AST 语法检查）、`post_tool_use`、`pre_stop`（测试守门员）拦截错误与敷衍交卷？ |
| **Step 5** | `recovery.py` | `python3 steps/step05_recovery.py --demo` | 如何处理 API 限流退避重试、连续相同报错的死循环打断（`LoopDetector`）与会话断点状态机恢复（`--resume`）？ |
| **Step 6** | `context.py`, `skills.py` | `python3 steps/step06_context_skills.py --demo` | 如何动态注入环境信息与 `AGENTS.md`，并通过渐进式披露按需加载 `skills/*/SKILL.md`？ |
| **Step 7** | `compaction.py` | `python3 steps/step07_compaction.py --demo` | 如何在不拆散 `tool_calls` 配对的前提下，自动截断单次超长输出并压缩历史长对话？ |
| **Step 8** | `subagent.py`, `agent.py` | `python3 steps/step08_subagent_cli.py --demo` | 如何通过 `task` 工具派生拥有独立上下文窗口的 Subagent，避免探索性脏活污染主对话记忆？ |
| **Step 9** | `todo.py` | `python3 steps/step09_plan_todo.py --demo` | 如何实现 `Plan / Act` 双模式切换（规划期禁止写操作）与 `TodoWrite` 任务完成度守门员？ |

## 运行全部单元测试

```bash
python3 -m unittest discover -s tests -v
```
