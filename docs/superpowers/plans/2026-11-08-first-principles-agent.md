# First-Principles Agent (Agent = LLM + Harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, **stopping after every single Task (Step 1 through Step 8)** so the user can read the code and run it themselves before proceeding to the next Task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a distilled, educational Claude Code equivalent in Python from first principles across 8 self-contained, runnable steps (`Agent = LLM + Harness`).

**Architecture:** Evolves a clean `harness/` package alongside standalone per-step runnable scripts (`steps/step01_*.py` to `steps/step08_*.py`) and unit tests (`tests/test_step01_*.py` to `tests/test_step08_*.py`). Each step adds one foundational Harness mechanism (Loop -> Tool -> Runtime & Permission Gate -> Validation Hook Pipeline -> Recovery -> Context & Skills -> Compaction -> Subagent) without black-box frameworks.

**Tech Stack:** Python 3.10+, `openai` SDK, `python-dotenv`, `pytest` (for automated verification of each step).

**Spec:** `docs/superpowers/specs/2026-11-08-first-principles-agent-design.md`

## Global Constraints

- **编程语言**：Python 3.10+
- **极简依赖**：仅使用官方 `openai` SDK 与 `python-dotenv`（测试使用 `pytest`），不依赖 LangChain / LlamaIndex / AutoGen 等高层封装框架。
- **LLM 接口**：支持通过环境变量或 `.env` 配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`（可选）、`OPENAI_MODEL`（可选，默认 `gpt-4o-mini` 或 `gpt-4o`）。
- **推进节奏硬约束**：严格执行 **Step $N$ 实现 -> 暂停 -> 用户阅读与运行 -> 用户确认 -> Step $N+1$** 的交互闭环，禁止一次连续实现多个 Step。
- **代码可读性**：每个核心模块控制在 80~200 行左右，终端清晰打印 `[THINK]`、`[TOOL CALL]`、`[VALIDATION]`、`[RECOVERY]`、`[SUBAGENT]` 状态。

## Review Focus

1. **Tool call argument JSON parsing failure**: LLM returns malformed JSON in `tool_call.function.arguments`; `ToolRegistry.dispatch` / `Recovery` must catch `json.JSONDecodeError` and return a structured error string rather than crashing the loop.
2. **Path traversal via `..` or symlinks**: LLM passes `../outside.txt` or `/etc/passwd` to file tools; `path_sandbox_validator` must resolve paths with `Path.resolve()` and verify `resolved.is_relative_to(runtime.workspace_root)`.
3. **Subprocess hang or huge stdout in `bash`**: LLM runs a command that hangs or prints megabytes of text; `AgentRuntime.run_command` must enforce `timeout` and truncate stdout/stderr to a safe character ceiling.
4. **Syntax error in `edit_file` partial replacement**: LLM edits a valid `.py` file into invalid Python syntax; `python_ast_validator` must simulate the replacement in memory and run `ast.parse()` *before* writing to disk.
5. **Orphan `tool_calls` message during Context Compaction**: Splitting message history between an `assistant` message with `tool_calls` and its corresponding `tool` result messages causes OpenAI API 400 errors; `Compactor` must never split an assistant `tool_calls` turn from its immediate `tool` responses.

---

### Task 1: Step 1 — Bare LLM & The Minimal Agent Loop (`llm.py` + `step01_bare_loop.py`)

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `harness/__init__.py`
- Create: `harness/ui.py`
- Create: `harness/llm.py`
- Create: `steps/step01_bare_loop.py`
- Test: `tests/test_step01_loop.py`

**Interfaces:**
- Consumes: Environment variables `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.
- Produces:
  - `harness.ui`: `log_step(tag: str, message: str, color: str = "cyan") -> None`
  - `harness.llm`:
    - `@dataclass class ToolCall: id: str; name: str; arguments: str`
    - `@dataclass class LLMResponse: content: str | None; tool_calls: list[ToolCall]; raw_message: dict; total_tokens: int`
    - `class LLMClient:`
      - `__init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None)`
      - `chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse`
  - `steps.step01_bare_loop`:
    - `run_bare_loop(user_prompt: str, llm: LLMClient, max_turns: int = 10) -> str`

- [ ] **Step 1: Write the failing test in `tests/test_step01_loop.py`**

```python
from harness.llm import LLMResponse, ToolCall
from steps.step01_bare_loop import run_bare_loop


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


def test_bare_loop_executes_tool_and_returns_final_answer():
    fake_llm = FakeLLM([
        LLMResponse(
            content="Let me calculate that.",
            tool_calls=[ToolCall(id="call_1", name="calculate", arguments='{"expression": "6 * 7"}')],
            raw_message={
                "role": "assistant",
                "content": "Let me calculate that.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": '{"expression": "6 * 7"}'},
                }],
            },
            total_tokens=42,
        ),
        LLMResponse(
            content="The result of 6 * 7 is 42.",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "The result of 6 * 7 is 42."},
            total_tokens=60,
        ),
    ])
    answer = run_bare_loop("What is 6 * 7?", llm=fake_llm)
    assert "42" in answer
    assert fake_llm.calls == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step01_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness'`

- [ ] **Step 3: Implement `pyproject.toml`, `.env.example`, `harness/ui.py`, `harness/llm.py`, and `steps/step01_bare_loop.py`**

Wrap `openai.OpenAI` in `LLMClient.chat` converting SDK response into `LLMResponse` and `ToolCall`. In `steps/step01_bare_loop.py`, define two minimal toy tools (`get_current_time` and `calculate`), implement `run_bare_loop`, and support both `--demo` and interactive CLI input.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step01_loop.py -v`
Expected: PASS

---

### Task 2: Step 2 — Tool Abstraction, Registry & Core Coding Tools (`tool.py` + `step02_tools.py`)

**Files:**
- Create: `harness/tool.py`
- Create: `steps/step02_tools.py`
- Test: `tests/test_step02_tools.py`

**Interfaces:**
- Consumes: `harness.llm.LLMClient`, `harness.llm.ToolCall`
- Produces:
  - `harness.tool`:
    - `@dataclass class Tool: name: str; description: str; parameters: dict; func: Callable; dangerous: bool = False`
    - `def tool(name: str | None = None, description: str = "", dangerous: bool = False) -> Callable`
    - `class ToolRegistry:`
      - `register(self, t: Tool) -> None`
      - `get(self, name: str) -> Tool | None`
      - `openai_schemas(self) -> list[dict]`
      - `execute(self, name: str, raw_arguments: str, **context) -> str`
    - `def create_coding_tools(base_dir: Path | None = None) -> ToolRegistry` (registers `read_file`, `write_file`, `edit_file`, `bash`)

- [ ] **Step 1: Write the failing tests in `tests/test_step02_tools.py`**

```python
from pathlib import Path
from harness.tool import tool, ToolRegistry, create_coding_tools


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step02_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.tool'`

- [ ] **Step 3: Implement `harness/tool.py` and `steps/step02_tools.py`**

Inspect function signatures (`inspect.signature` + type annotations `str -> "string"`, `int -> "integer"`, `bool -> "boolean"`, `float -> "number"`) to auto-build OpenAI JSON Schema. Catch `json.JSONDecodeError` in `ToolRegistry.execute` and return `[ERROR] Invalid JSON arguments: ...`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step02_tools.py -v`
Expected: PASS

---

### Task 3: Step 3 — AgentRuntime & Permission Gate (`runtime.py` + `step03_runtime.py`)

**Files:**
- Create: `harness/runtime.py`
- Modify: `harness/tool.py` (allow tools to accept optional `runtime: AgentRuntime` parameter)
- Create: `steps/step03_runtime.py`
- Test: `tests/test_step03_runtime.py`

**Interfaces:**
- Consumes: `harness.tool.ToolRegistry`, `harness.tool.Tool`
- Produces:
  - `harness.runtime`:
    - `PermissionCallback = Callable[[str, dict], bool]`
    - `class AgentRuntime:`
      - `workspace_root: Path`
      - `cwd: Path`
      - `messages: list[dict]`
      - `modified_files: set[Path]`
      - `always_allow_dangerous: bool`
      - `permission_callback: PermissionCallback`
      - `resolve_path(self, path_str: str) -> Path`
      - `run_command(self, command: str, timeout: int = 15, max_output_chars: int = 8000) -> str`
      - `check_permission(self, tool_obj: Tool, args: dict) -> tuple[bool, str]`

- [ ] **Step 1: Write the failing tests in `tests/test_step03_runtime.py`**

```python
from pathlib import Path
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step03_runtime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.runtime'`

- [ ] **Step 3: Implement `harness/runtime.py`, update `harness/tool.py`, and create `steps/step03_runtime.py`**

Implement `AgentRuntime` with `resolve_path`, `run_command` (using `subprocess.run(..., timeout=timeout, capture_output=True, text=True)`), and `check_permission` (prompting `[y/n/a]` in terminal by default when `tool.dangerous` is `True`). Update `ToolRegistry.execute` to check permission via `runtime` before invoking the tool function.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step03_runtime.py -v`
Expected: PASS

---

### Task 4: Step 4 — Validation 3-Stage Hook Pipeline (`validation.py` + `step04_validation.py`)

**Files:**
- Create: `harness/validation.py`
- Create: `steps/step04_validation.py`
- Test: `tests/test_step04_validation.py`

**Interfaces:**
- Consumes: `harness.runtime.AgentRuntime`, `harness.tool.Tool`
- Produces:
  - `harness.validation`:
    - `@dataclass class ValidationResult: passed: bool; reason: str = ""; recovery_hint: str = ""`
    - `class ValidationPipeline:`
      - `add_pre_tool(self, hook: Callable[[Tool, dict, AgentRuntime], ValidationResult]) -> None`
      - `add_post_tool(self, hook: Callable[[Tool, dict, str, AgentRuntime], ValidationResult]) -> None`
      - `add_pre_stop(self, hook: Callable[[AgentRuntime], ValidationResult]) -> None`
      - `run_pre_tool(self, tool_obj: Tool, args: dict, runtime: AgentRuntime) -> ValidationResult`
      - `run_post_tool(self, tool_obj: Tool, args: dict, output: str, runtime: AgentRuntime) -> ValidationResult`
      - `run_pre_stop(self, runtime: AgentRuntime) -> ValidationResult`
    - Built-in hooks:
      - `schema_validator(tool_obj: Tool, args: dict, runtime: AgentRuntime) -> ValidationResult`
      - `path_sandbox_validator(tool_obj: Tool, args: dict, runtime: AgentRuntime) -> ValidationResult`
      - `python_ast_validator(tool_obj: Tool, args: dict, runtime: AgentRuntime) -> ValidationResult`
      - `make_command_pre_stop_validator(command: str) -> Callable[[AgentRuntime], ValidationResult]`

- [ ] **Step 1: Write the failing tests in `tests/test_step04_validation.py`**

```python
from pathlib import Path
from harness.runtime import AgentRuntime
from harness.tool import create_coding_tools
from harness.validation import (
    ValidationPipeline,
    schema_validator,
    path_sandbox_validator,
    python_ast_validator,
    make_command_pre_stop_validator,
)


def test_pre_tool_hooks_block_path_escape_and_bad_python_ast(tmp_path: Path):
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    reg = create_coding_tools()
    pipe = ValidationPipeline()
    pipe.add_pre_tool(schema_validator)
    pipe.add_pre_tool(path_sandbox_validator)
    pipe.add_pre_tool(python_ast_validator)

    write_tool = reg.get("write_file")
    # 1. Path traversal outside workspace
    res_escape = pipe.run_pre_tool(write_tool, {"path": "../evil.py", "content": "x=1"}, rt)
    assert not res_escape.passed
    assert "outside workspace" in res_escape.reason.lower()

    # 2. Invalid Python syntax on write_file
    res_ast = pipe.run_pre_tool(write_tool, {"path": "broken.py", "content": "def foo(:\n    pass"}, rt)
    assert not res_ast.passed
    assert "syntaxerror" in res_ast.reason.lower()

    # 3. Invalid Python syntax on edit_file simulated in memory
    (tmp_path / "ok.py").write_text("def add(a, b):\n    return a + b\n")
    edit_tool = reg.get("edit_file")
    res_edit_ast = pipe.run_pre_tool(
        edit_tool,
        {"path": "ok.py", "old_text": "return a + b", "new_text": "return a +"},
        rt,
    )
    assert not res_edit_ast.passed
    assert "return a + b" in (tmp_path / "ok.py").read_text()


def test_pre_stop_hook_vetoes_stop_when_verification_command_fails(tmp_path: Path):
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    (tmp_path / "app.py").write_text("def inc(x):\n    return x - 1\n")
    rt.modified_files.add(tmp_path / "app.py")

    pipe = ValidationPipeline()
    pipe.add_pre_stop(make_command_pre_stop_validator("python3 -c 'import app; assert app.inc(1) == 2'"))
    res = pipe.run_pre_stop(rt)
    assert not res.passed
    assert "AssertionError" in res.reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step04_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.validation'`

- [ ] **Step 3: Implement `harness/validation.py` and `steps/step04_validation.py`**

Implement `ValidationPipeline` and the built-in hooks (`schema_validator`, `path_sandbox_validator`, `python_ast_validator` using `ast.parse`, and `make_command_pre_stop_validator`). Wire `ValidationPipeline` into `ToolRegistry.execute` (or the agent loop) and `steps/step04_validation.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step04_validation.py -v`
Expected: PASS

---

### Task 5: Step 5 — Recovery: Self-Correction, Loop Detection & Session Checkpoint (`recovery.py` + `step05_recovery.py`)

**Files:**
- Create: `harness/recovery.py`
- Create: `steps/step05_recovery.py`
- Test: `tests/test_step05_recovery.py`

**Interfaces:**
- Consumes: `harness.validation.ValidationResult`, `harness.runtime.AgentRuntime`
- Produces:
  - `harness.recovery`:
    - `format_error_observation(error_type: str, reason: str, recovery_hint: str = "") -> str`
    - `class LoopDetector:`
      - `__init__(self, max_repeats: int = 3)`
      - `record_and_check(self, tool_name: str, raw_args: str, is_error: bool) -> str | None`
    - `def call_with_retry(fn: Callable, max_retries: int = 3, base_delay: float = 0.5) -> Any`
    - `class SessionStore:`
      - `save(self, session_id: str, messages: list[dict]) -> Path`
      - `load(self, session_id: str) -> list[dict]`

- [ ] **Step 1: Write the failing tests in `tests/test_step05_recovery.py`**

```python
from pathlib import Path
from harness.recovery import (
    format_error_observation,
    LoopDetector,
    call_with_retry,
    SessionStore,
)


def test_loop_detector_triggers_on_repeated_identical_failures():
    detector = LoopDetector(max_repeats=3)
    assert detector.record_and_check("edit_file", '{"path":"a.py"}', is_error=True) is None
    assert detector.record_and_check("edit_file", '{"path":"a.py"}', is_error=True) is None
    warning = detector.record_and_check("edit_file", '{"path":"a.py"}', is_error=True)
    assert warning is not None
    assert "LOOP_DETECTED" in warning


def test_call_with_retry_recovers_from_transient_error():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("Rate limit")
        return "ok"

    assert call_with_retry(flaky, max_retries=3, base_delay=0.01) == "ok"
    assert attempts["count"] == 3


def test_session_store_save_and_resume(tmp_path: Path):
    store = SessionStore(tmp_path)
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    store.save("sess_01", msgs)
    loaded = store.load("sess_01")
    assert loaded == msgs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step05_recovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.recovery'`

- [ ] **Step 3: Implement `harness/recovery.py` and `steps/step05_recovery.py`**

Implement structured `[ERROR_TYPE] / [REASON] / [RECOVERY_HINT]` formatting, `LoopDetector`, `call_with_retry`, and `SessionStore`. In `edit_file`, when `old_text` is not found in the file, return a recovery hint telling the LLM to call `read_file` first to see exact current contents.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step05_recovery.py -v`
Expected: PASS

---

### Task 6: Step 6 — Context Grounding & Progressive Skills (`context.py` + `skills.py` + `step06_context_skills.py`)

**Files:**
- Create: `harness/context.py`
- Create: `harness/skills.py`
- Create: `skills/code-reviewer/SKILL.md`
- Create: `skills/tdd-helper/SKILL.md`
- Create: `steps/step06_context_skills.py`
- Test: `tests/test_step06_context_skills.py`

**Interfaces:**
- Consumes: `harness.runtime.AgentRuntime`, `harness.tool.ToolRegistry`
- Produces:
  - `harness.skills`:
    - `@dataclass class SkillMeta: name: str; description: str; path: Path`
    - `class SkillManager:`
      - `__init__(self, skills_dir: Path)`
      - `discover(self) -> list[SkillMeta]`
      - `render_catalog_prompt(self) -> str`
      - `load_skill_body(self, name: str) -> str`
      - `register_load_tool(self, registry: ToolRegistry) -> None`
  - `harness.context`:
    - `build_system_prompt(runtime: AgentRuntime, skill_manager: SkillManager | None = None) -> str`

- [ ] **Step 1: Write the failing tests in `tests/test_step06_context_skills.py`**

```python
from pathlib import Path
from harness.runtime import AgentRuntime
from harness.tool import ToolRegistry
from harness.skills import SkillManager
from harness.context import build_system_prompt


def test_skill_discovery_and_progressive_loading(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Secret workflow for widgets\n---\n# Detailed Steps\nStep A: Do X\n"
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
    (tmp_path / "AGENTS.md").write_text("Always use snake_case for function names.")
    rt = AgentRuntime(workspace_root=tmp_path)
    prompt = build_system_prompt(rt)
    assert "Always use snake_case" in prompt
    assert str(tmp_path) in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step06_context_skills.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.skills'`

- [ ] **Step 3: Implement `harness/skills.py`, `harness/context.py`, sample `skills/*/SKILL.md`, and `steps/step06_context_skills.py`**

Parse YAML frontmatter between `---` delimiters using standard library string processing. Register `load_skill` in `ToolRegistry`. Assemble environment snapshot (`os`, `cwd`, `AGENTS.md` / `CLAUDE.md`, `<available_skills>`) in `build_system_prompt`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step06_context_skills.py -v`
Expected: PASS

---

### Task 7: Step 7 — Context Compaction (`compaction.py` + `step07_compaction.py`)

**Files:**
- Create: `harness/compaction.py`
- Create: `steps/step07_compaction.py`
- Test: `tests/test_step07_compaction.py`

**Interfaces:**
- Consumes: `harness.llm.LLMClient`
- Produces:
  - `harness.compaction`:
    - `truncate_tool_output(output: str, max_chars: int = 4000) -> str`
    - `estimate_tokens(messages: list[dict]) -> int`
    - `compact_messages(messages: list[dict], llm: Any, max_tokens: int = 6000, keep_recent: int = 4) -> list[dict]`

- [ ] **Step 1: Write the failing tests in `tests/test_step07_compaction.py`**

```python
from harness.llm import LLMResponse
from harness.compaction import compact_messages, truncate_tool_output


class DummySummaryLLM:
    def chat(self, messages: list[dict], tools=None) -> LLMResponse:
        return LLMResponse(
            content="- Created foo.py\n- Next: run tests",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "- Created foo.py\n- Next: run tests"},
            total_tokens=20,
        )


def test_compact_messages_preserves_system_and_never_splits_tool_call_pairs():
    messages = [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "Old task " * 200},
        {"role": "assistant", "content": "Old reply " * 200},
        {
            "role": "assistant",
            "content": "Calling tool",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file content"},
        {"role": "user", "content": "Recent question"},
    ]
    compacted = compact_messages(messages, llm=DummySummaryLLM(), max_tokens=100, keep_recent=2)
    assert compacted[0]["role"] == "system"
    assert "<context_summary>" in compacted[1]["content"]
    # Ensure no orphan 'tool' message appears without its preceding 'assistant' tool_calls message
    for i, msg in enumerate(compacted):
        if msg["role"] == "tool":
            assert i > 0 and compacted[i - 1].get("tool_calls")


def test_truncate_tool_output_keeps_head_and_tail():
    text = "HEAD_" + ("x" * 5000) + "_TAIL"
    out = truncate_tool_output(text, max_chars=400)
    assert out.startswith("HEAD_")
    assert out.endswith("_TAIL")
    assert "TRUNCATED" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step07_compaction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.compaction'`

- [ ] **Step 3: Implement `harness/compaction.py` and `steps/step07_compaction.py`**

Implement `truncate_tool_output`, `estimate_tokens`, and `compact_messages` (adjusting the split index backward if `messages[split_idx]["role"] == "tool"` so an `assistant` `tool_calls` message is never separated from its `tool` results).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step07_compaction.py -v`
Expected: PASS

---

### Task 8: Step 8 — Subagent Dispatch & Complete Mini Claude Code CLI (`subagent.py` + `step08_subagent_cli.py`)

**Files:**
- Create: `harness/subagent.py`
- Create: `harness/agent.py`
- Create: `steps/step08_subagent_cli.py`
- Create: `README.md`
- Test: `tests/test_step08_subagent.py`

**Interfaces:**
- Consumes: All `harness.*` modules (`llm`, `tool`, `runtime`, `validation`, `recovery`, `context`, `skills`, `compaction`)
- Produces:
  - `harness.agent`:
    - `class Agent:`
      - `run_turn(self, user_input: str) -> str`
  - `harness.subagent`:
    - `register_subagent_tool(registry: ToolRegistry, llm: Any, runtime: AgentRuntime, max_turns: int = 8) -> None`

- [ ] **Step 1: Write the failing tests in `tests/test_step08_subagent.py`**

```python
from pathlib import Path
from harness.llm import LLMResponse, ToolCall
from harness.runtime import AgentRuntime
from harness.tool import create_coding_tools
from harness.subagent import register_subagent_tool


class SequencedLLM:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages: list[dict], tools=None) -> LLMResponse:
        self.seen_messages.append([dict(m) for m in messages])
        return self.responses.pop(0)


def test_subagent_runs_in_isolated_context_and_returns_summary(tmp_path: Path):
    (tmp_path / "secret.txt").write_text("CODE_42")
    rt = AgentRuntime(workspace_root=tmp_path, always_allow_dangerous=True)
    reg = create_coding_tools()

    sub_llm = SequencedLLM([
        LLMResponse(
            content="Reading secret.txt",
            tool_calls=[ToolCall(id="sub_1", name="read_file", arguments='{"path": "secret.txt"}')],
            raw_message={
                "role": "assistant",
                "content": "Reading secret.txt",
                "tool_calls": [{
                    "id": "sub_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "secret.txt"}'},
                }],
            },
            total_tokens=30,
        ),
        LLMResponse(
            content="Subagent report: secret code is CODE_42.",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "Subagent report: secret code is CODE_42."},
            total_tokens=40,
        ),
    ])

    register_subagent_tool(reg, llm=sub_llm, runtime=rt)
    parent_messages_before = list(rt.messages)
    result = reg.execute("task", '{"description": "Find code", "prompt": "Read secret.txt and report"}', runtime=rt)
    assert "CODE_42" in result
    # Parent runtime.messages must not be polluted by subagent intermediate tool messages
    assert rt.messages == parent_messages_before
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step08_subagent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.subagent'`

- [x] **Step 3: Implement `harness/subagent.py`, `harness/agent.py`, `steps/step08_subagent_cli.py`, and `README.md`**

Implement `register_subagent_tool` spawning an isolated `Agent` with a fresh `messages` list and returning only its final answer. Assemble the full `Agent` class in `harness/agent.py` and the interactive CLI in `steps/step08_subagent_cli.py` (supporting `/help`, `/compact`, `/skills`, `/session`, `/quit`).

- [x] **Step 4: Run full test suite to verify all 8 steps pass**

Run: `pytest -v`
Expected: All tests in `tests/test_step01_*.py` through `tests/test_step08_*.py` PASS.

---

### Task 9: Step 9 — Plan & Todo Mode (`todo.py` + `step09_plan_todo.py`)

**Files:**
- Create: `harness/todo.py`
- Create: `steps/step09_plan_todo.py`
- Test: `tests/test_step09_plan_todo.py`

**Interfaces:**
- Consumes: `harness.tool.ToolRegistry`, `harness.validation.ValidationPipeline`, `harness.runtime.AgentRuntime`
- Produces:
  - `harness.todo`:
    - `@dataclass class TodoItem: id: str; content: str; status: Literal["pending", "in_progress", "completed"]`
    - `class TodoManager:`
      - `items: list[TodoItem]`
      - `mode: Literal["plan", "act"]`
      - `update(self, items: list[dict]) -> str`
      - `has_incomplete(self) -> bool`
      - `make_plan_mode_pre_tool_hook(self) -> PreToolHook` (blocks write/edit/bash tools while `mode == "plan"`)
      - `make_todo_pre_stop_hook(self) -> PreStopHook` (vetoes stop while `has_incomplete()` is `True`)
      - `register_todo_tools(self, registry: ToolRegistry) -> None` (registers `todo_write` and `exit_plan_mode`)

- [x] **Step 1: Write the failing tests in `tests/test_step09_plan_todo.py`**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement `harness/todo.py` and `steps/step09_plan_todo.py`**
- [x] **Step 4: Run full test suite to verify all 9 steps pass**


