# 从第一性原理构建 Agent（Agent = LLM + Harness）设计文档

- **日期**：2026-11-08
- **项目路径**：`/opt/docs/study/agent-impl2`
- **工作量规模**：完整教学型工程项目（分 9 个独立可运行演进步骤）

---

## 1. 意图与背景（Intent & Why）

当前业界对智能体的核心共识已经收敛为 **`Agent = LLM + Harness`**：
- **LLM（大脑）**：本质是一个无状态的 Token 预测器（输入消息列表，输出文本或结构化工具调用意图）。它自己既没有手脚，也没有记忆管理、错误恢复或安全边界。
- **Harness（马具/工程骨架）**：包裹在 LLM 外层的控制循环、运行时环境、工具体系、校验机制、容错恢复、技能按需加载、上下文压缩与子智能体调度器。

本项目的核心目的是**通过第一性原理（First Principles）从 0 到 1 亲手构建一个精简版 Claude Code**。为了达到最佳学习效果，项目不是一次性倾倒全部代码，而是拆解为 9 个递进步骤：每一步只针对前一步暴露的真实痛点引入一个核心 Harness 机制；**每完成一步必须暂停**，由学习者亲自阅读代码并运行体验后，再推进下一步。

---

## 2. 目标、非目标与反目标（Goals, Non-Goals & Anti-Goals）

### Goals（目标）
1. **第一性原理递进**：从最原始的 `LLM + while 循环` 出发，分 8 步逐步长出 **Loop、Tool、Runtime、Validation、Recovery、Context & Skills、Compaction、Subagent** 八大核心能力。
2. **每步可独立运行与检验**：每一步都提供独立的入口脚本（`steps/step0X_*.py`），包含可交互模式（REPL）与内置演示任务（`--demo`），方便学习者一行命令跑通观察。
3. **模块化沉淀**：底层稳定能力沉淀在 `harness/` 包中，避免后期单文件膨胀到难以阅读，最终第 8 步自然汇聚为一个精简可用的终端 Coding Agent。
4. **零黑盒框架依赖**：不依赖 LangChain / LlamaIndex / AutoGen 等高层封装框架，仅使用官方 `openai` SDK 与 Python 标准库，让每一行 Harness 逻辑都清晰可见。

### Non-Goals（非目标）
1. **不做复杂全屏 TUI**：不使用 `textual` 或 `curses` 构建复杂全屏界面，仅用简洁的 ANSI 颜色与终端流式/分段打印展示思考、工具调用、校验与恢复过程。
2. **不接入外部扩展协议**：不实现 MCP（Model Context Protocol）客户端/服务端、OAuth 登录、计费统计或多租户数据库。
3. **不做跨语言 AST 解析**：代码静态语法校验聚焦于 Python（使用内置 `ast.parse`）及通用外部验证命令（如 `pytest` / `ruff`）。

### Anti-Goals（反目标 —— 即使代码能跑也视为失败的情况）
1. **禁止“一口气写完所有步骤”**：绝不能在未收到用户确认的情况下连续实现多个 Step。每完成一个 Step 的代码与运行说明，必须立即停下等待用户验收。
2. **禁止过度抽象与炫技**：不能用复杂的元编程、多层设计模式掩盖 Harness 的核心控制流，代码必须直白易读，每个核心模块控制在 80~200 行左右。
3. **禁止吞掉错误或静默失败**：Validation 拦截与 Recovery 恢复的过程必须在终端清晰打印出来，让学习者直观看到“LLM 犯了什么错 -> Harness 如何拦截并引导它自愈”。

---

## 3. 约束条件（Constraints）

- **编程语言**：Python 3.10+
- **LLM 接口**：OpenAI Python SDK（`openai`），支持通过环境变量或 `.env` 配置：
  - `OPENAI_API_KEY`（必填）
  - `OPENAI_BASE_URL`（可选，支持兼容 OpenAI 协议的各类模型接口）
  - `OPENAI_MODEL`（可选，默认如 `gpt-4o` 或用户指定模型）
- **推进节奏硬约束**：严格执行 **Step $N$ 实现 -> 暂停 -> 用户阅读与运行 -> 用户确认 -> Step $N+1$** 的交互闭环。

---

## 4. 方案对比与选型（Approaches）

在规划代码组织方式时，我们评估了三种方案：

| 方案 | 形态描述 | 优势 | 劣势 / 代价 |
| :--- | :--- | :--- | :--- |
| **方案 A：独立单文件快照式** | 每一步都是完全独立的单文件 `step01.py` ~ `step08.py`，下一步复制上一步代码并增量修改。 | 阅读单步时无需跨文件跳转，容易对比前后差异。 | 越往后单文件越长（400~500+ 行），大量基础代码在 8 个文件中重复拷贝。 |
| **方案 B：单一工程原地演进式** | 从第 1 步起直接维护单一 `mini_agent/` 包，每一步在原文件上直接重构覆盖，靠 `git diff` 看变化。 | 零重复代码，最终工程最纯粹。 | 后期文件被改写多轮，学习者无法随时直接运行早期的某一步做对比实验。 |
| **方案 C：演进式模块化 + 每步专属可运行沙盒（已选定）** | 将每步新增的 Harness 机制拆分为清晰的独立模块（`harness/`），同时为每步提供专属运行与实验脚本（`steps/step01_*.py` ~ `steps/step08_*.py`）。 | 兼顾“每步可独立运行、聚焦当步新概念”与“代码结构清晰、最终沉淀出完整精简版 Claude Code”。 | 比纯单文件多一层模块导入关系。 |

**选型结论**：采用 **方案 C**。已学过的底层机制沉淀为清晰的小模块，当前正在学习的机制在对应模块和 `steps/step0X_*.py` 中直观呈现。

---

## 5. 架构与八步演进详细设计（The How）

### 5.1 目录结构设计

```text
/opt/docs/study/agent-impl2/
├── .env.example                 # 环境变量配置模板 (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)
├── pyproject.toml               # 极简依赖声明 (仅 openai, python-dotenv)
├── README.md                    # 学习路线图与各步骤运行指南
├── workspace/                   # 供 Agent 操作的默认安全沙盒工作区（测试文件/示例项目放这里）
├── skills/                      # Step 6 使用的示例技能目录
│   ├── code-reviewer/
│   │   └── SKILL.md
│   └── tdd-helper/
│       └── SKILL.md
├── harness/                     # 核心 Harness 模块库（随 Step 1 -> Step 8 逐步长出）
│   ├── __init__.py
│   ├── llm.py                   # [Step 1] OpenAI 客户端封装与基础消息结构
│   ├── tool.py                  # [Step 2] Tool 抽象、JSON Schema 生成、ToolRegistry 与基础工具集
│   ├── runtime.py               # [Step 3] AgentRuntime（cwd 沙盒、受控子进程、Session、Permission Gate）
│   ├── validation.py            # [Step 4] 前置校验（Schema/路径安全/Python AST）与后置完成校验钩子
│   ├── recovery.py              # [Step 5] 错误转 Observation 自愈、重复调用死循环检测、API 退避重试、断点持久化
│   ├── context.py               # [Step 6] 环境感知（OS/cwd/git/AGENTS.md）与动态 System Prompt 组装
│   ├── skills.py                # [Step 6] SKILL.md 元数据扫描与按需加载工具 (load_skill)
│   ├── compaction.py            # [Step 7] 工具大输出截断与历史上下文自动摘要压缩 (Context Compaction)
│   ├── subagent.py              # [Step 8] 独立上下文的子智能体派生器 (task 工具)
│   ├── todo.py                  # [Step 9] Plan/Act 双模态状态机与 TodoWrite 生命周期守门员
│   └── agent.py                 # 终极整合版 Agent 类
└── steps/                       # 每一步独立可运行的入口与配套实验场景
    ├── step01_bare_loop.py      # Step 1: 最小 ReAct / Tool-Calling 循环
    ├── step02_tools.py          # Step 2: 标准 Tool 抽象与文件/Bash 四大基础工具
    ├── step03_runtime.py        # Step 3: Runtime 容器与 Permission Gate 安全护栏
    ├── step04_validation.py     # Step 4: 前置 AST/路径校验 + 后置自检钩子
    ├── step05_recovery.py       # Step 5: 异常自愈 (Self-Correction)、死循环打断与 Session 恢复
    ├── step06_context_skills.py # Step 6: 环境锚定 (AGENTS.md) 与 Skills 渐进式披露
    ├── step07_compaction.py     # Step 7: 长对话上下文自动压缩与大输出折叠
    ├── step08_subagent_cli.py   # Step 8: Subagent 委派 + 终极版 Mini Claude Code CLI
    └── step09_plan_todo.py      # Step 9: Plan/Act 双模式与 TodoWrite 完结守门员演示
```

### 5.2 八步第一性原理演进明细

#### Step 1：裸 LLM 与最小控制循环（The Loop）
- **第一性原理问题**：LLM 本身只会“一问一答”，怎样让它能连续多步自己决定调什么、看结果、再决定下一步，直到问题解决？
- **核心实现**：
  - `harness/llm.py`：封装最小 OpenAI Chat Completions 调用。
  - `steps/step01_bare_loop.py`：用不到 100 行代码展示核心公式：
    ```python
    while True:
        response = llm.chat(messages, tools)
        if not response.tool_calls:
            break  # LLM 认为任务完成，直接向用户输出结果
        for call in response.tool_calls:
            result = execute_naive_tool(call)
            messages.append(tool_result_message(call.id, result))
    ```
- **可运行实验**：内置 2 个极简硬编码玩具工具（如 `get_current_time`, `calculate`），观察 LLM 如何分多轮调用工具后再给出最终回答。

#### Step 2：Tool 抽象、注册中心与四大编码工具（Tool System）
- **第一性原理问题**：Step 1 中手写 JSON Schema 和 `if/else` 分发工具太脆弱且不可扩展，且 Agent 需要真正操作代码库的“手和眼”。
- **核心实现**：
  - `harness/tool.py`：定义 `@tool` 装饰器与 `Tool` 数据类，利用 Python `inspect` 和类型注解自动生成符合 OpenAI 规范的 JSON Schema；实现 `ToolRegistry`。
  - 实现 Claude Code 最核心的 4 个基础工具：`read_file`、`write_file`、`edit_file`（精确字符串替换）、`bash`。
- **可运行实验**：运行 `steps/step02_tools.py`，让 Agent 在 `workspace/` 下自动创建代码文件并运行它。

#### Step 3：Runtime 运行时容器与安全护栏（Runtime & Permission Gate）
- **第一性原理问题**：Step 2 的工具直接裸跑在全局环境，没有工作目录隔离，命令卡死会拖垮整个进程，危险命令（如 `rm -rf`）没有人工审批拦截。
- **核心实现**：
  - `harness/runtime.py`：构建 `AgentRuntime` 类，统一承载：
    1. `workspace_root` 与当前 `cwd` 状态管理；
    2. 受控子进程执行器（强制超时 `timeout`、环境变量清理、标准输出/错误收集与长度保护）；
    3. `PermissionGate`：将工具划分为 `SAFE`（如 `read_file`）与 `DANGEROUS`（如 `bash`, `write_file`, `edit_file`），在执行危险动作前弹出终端确认提示 `[y/n/always]`。
    4. 结构化事件回调（Event Hook），方便终端清晰打印每一步状态。
- **可运行实验**：运行 `steps/step03_runtime.py`，体验 Agent 执行写文件或 Shell 命令时被 Permission Gate 拦截并等待用户授权的过程。

#### Step 4：Validation 三段式生命周期 Hook 管道（Pre-Tool / Post-Tool / Pre-Stop Hooks）
- **第一性原理问题**：LLM 经常犯低级错误：传错参数类型、尝试越界访问 `/etc/passwd`、生成带语法错误的 Python 代码，或者改完代码不跑测试就宣称“大功告成”。如果把校验写成硬编码 `if/else`，主循环会迅速腐化。
- **核心实现**：
  - `harness/validation.py`：定义统一的 `ValidationResult(passed, reason, recovery_hint)` 与 `ValidationPipeline`，在 Agent 生命周期的 3 个关键节点挂载可插拔 Hook：
    1. **`pre_tool_use` Hooks（工具执行前拦截：不通过则短路，不落盘、不执行）**：
       - `schema_validator`：参数必填项与类型校验；
       - `path_sandbox_validator`：确保任何文件读写路径解析后都在 `runtime.workspace_root` 内；
       - `python_ast_validator`：当 `write_file` / `edit_file` 目标为 `.py` 文件时，先在内存中计算新内容并执行 `ast.parse`，若有 `SyntaxError` 则拒绝写入并指出报错行号。
    2. **`post_tool_use` Hooks（工具执行后校验：补充告警或检查副作用）**：
       - 检查工具执行后的文件状态或输出完整性，必要时附加结构化提示。
    3. **`pre_stop` Hooks（交卷前守门员 / Completion Gate：不通过则打回重做）**：
       - 当 LLM 不再发出 `tool_calls` 准备结束本轮任务时，若本轮修改过代码文件且挂载了验证命令 Hook（如跑单测/语法检查），Harness 自动执行验证；若失败则拒绝结束（veto stop），将报错塞回对话循环逼迫 LLM 继续修复。
- **可运行实验**：运行 `steps/step04_validation.py --demo`，内置模拟场景观察：
  1. 越界路径被 `pre_tool_use` 拦截；
  2. 写入有语法错误的 Python 代码被内存 AST 校验拦截并要求重写；
  3. 故意写错逻辑导致单测失败时，`pre_stop` Hook 阻止 Agent 结束并倒逼其修好测试。

#### Step 5：Recovery 容错与自愈机制（Self-Correction & Resilience）
- **第一性原理问题**：真实环境中工具执行随时会失败（比如 `edit_file` 的 `old_text` 没匹配上、命令报错退出、或者 LLM 陷入死循环反复调同一个报错的参数），怎样让 Agent 具备韧性（Resilience）？
- **核心实现**：
  - `harness/recovery.py`：
    1. **异常转诊断反馈（Error-to-Observation）**：拦截所有工具执行期异常与 Validation 错误，格式化为包含 `[ERROR_TYPE]`、`[REASON]` 和 `[RECOVERY_HINT]`（例如 `edit_file` 匹配失败时提示先调用 `read_file` 查看最新内容）的工具结果消息回传给 LLM。
    2. **死循环检测器（Loop Detector）**：记录最近 $N$ 次工具调用签名（工具名 + 参数哈希）；若发现连续 3 次完全相同的失败调用，主动注入系统级警告打断惯性，或强制暂停交还用户。
    3. **LLM API 退避重试**：针对网络超时/限流实现指数退避重试。
    4. **Session Checkpoint 持久化与恢复**：每步循环将 `messages` 落盘到 `.sessions/<session_id>.json`，支持进程中断后通过 `--resume` 无缝续跑。
- **可运行实验**：运行 `steps/step05_recovery.py --demo`，触发一次 `edit_file` 匹配失败和一次重复调用死循环，直观观察 Harness 如何引导 LLM 自愈以及如何打断死循环。

#### Step 6：Context Grounding 与 Skills 渐进式加载（Context & Skills）
- **第一性原理问题**：如果把所有领域的最佳实践（如何做代码审查、如何写 TDD、项目规范）全塞进初始 System Prompt，每次对话还没开始就消耗上万 Token 且注意力分散。
- **核心实现**：
  - `harness/context.py`：启动时自动采集环境快照（操作系统、工作目录、Git 状态）并自动读取工作区下的 `AGENTS.md` / `CLAUDE.md` 作为项目常驻记忆。
  - `harness/skills.py`：实现 Claude Code 风格的 **Progressive Disclosure（渐进式技能披露）**：
    1. 扫描 `skills/*/SKILL.md` 的 YAML Frontmatter（仅读取 `name` 和 `description`），生成几行简短的 `<available_skills>` 目录注入 System Prompt；
    2. 注册 `load_skill(skill_name)` 工具；当 LLM 判断当前任务需要某项技能时，主动调用 `load_skill` 将对应 `SKILL.md` 的完整指令加载进对话上下文。
- **可运行实验**：运行 `steps/step06_context_skills.py`，向 Agent 提出代码审查或 TDD 任务，观察它如何先识别技能目录、再主动调用 `load_skill` 加载详细规程并按规程执行。

#### Step 7：Context Compaction 上下文压缩管理（Memory Compaction）
- **第一性原理问题**：Coding Agent 读几个大文件、跑几次命令后，消息历史很容易逼近 Token 上限或导致推理变慢变贵。
- **核心实现**：
  - `harness/compaction.py`：
    1. **单次工具大输出修剪（Output Truncation）**：对超长的 `bash` 或 `read_file` 输出进行头尾保留 + 中间折叠（提示总行数并引导按行号范围读取）。
    2. **历史对话自动摘要压缩（Auto-Compaction）**：估算当前 `messages` 的字符/Token 占用；当超过设定的阈值（或用户手动输入 `/compact`）时，保留 System Prompt 与最近 $K$ 条消息，调用 LLM 将早期漫长的工具调用过程压缩为一段结构化的 `<context_summary>`（已完成事项、修改过的文件、当前待办与关键发现）。
- **可运行实验**：运行 `steps/step07_compaction.py --demo`，用较低的阈值触发自动压缩，直观对比压缩前后的消息列表长度与保留的摘要内容。

#### Step 8：Subagent 子智能体委派与终极整合（Subagent & Mini Claude Code）
- **第一性原理问题**：当主 Agent 需要做一件耗时且产生大量中间噪音的子任务（如“翻阅整个目录找出某函数的调用链”）时，如果在主上下文里直接翻找，会迅速污染主对话记忆。
- **核心实现**：
  - `harness/subagent.py`：实现 `task(description, prompt)` 工具。调用时，在同一个 `AgentRuntime` 下创建一个拥有**全新独立 `messages` 列表**的子 Agent 循环（默认仅开放只读/探索类工具或受限迭代次数），子 Agent 在后台跑完多轮探索后，只将最终总结报告作为 `task` 工具的返回值交还给主 Agent。
  - `steps/step08_subagent_cli.py`：将前 7 步所有模块完整串联，支持交互式命令（`/help`, `/compact`, `/skills`, `/session`, `/quit`），成为一个五脏俱全的精简版 Claude Code。
- **可运行实验**：运行 `steps/step08_subagent_cli.py`，下达一个复合任务（先派子智能体调研工作区代码结构，主智能体再根据报告修改代码并通过校验）。

#### Step 9：Plan & Todo 模式（Phase Separation & Todo Lifecycle Gatekeeper）
- **第一性原理问题**：在复杂长链路任务中，LLM 容易“未做深入探索就盲目乱改代码”或者“半途而废、执行到一半遗忘剩余任务”。
- **核心实现**：
  - `harness/todo.py`：
    1. **Plan/Act 双模态状态机**：挂载 `make_plan_mode_pre_tool_hook` 到 `pre_tool_use` 管道。在 `mode == "plan"` 阶段拦截一切破坏性修改操作（`write_file`, `edit_file`, 非只读 `bash`），强制 LLM 仅能使用只读探索与规划工具（`read_file`, `task`, `todo_write`），完成方案构思后调用 `exit_plan_mode` 切换至 `ACT` 模式。
    2. **`todo_write` 任务看板与单一在办约束**：动态维护结构化待办列表，并施加硬性约束：**同一时刻最多只能有 1 个处于 `in_progress` 状态的 Todo 项**，强迫注意力线性聚焦。
    3. **`pre_stop` 完结守门员**：挂载 `make_todo_pre_stop_hook`。当 LLM 停止调用工具准备交卷时，若当前仍在 `plan` 模式或仍有未标记为 `completed` 的 Todo 项，直接否决退出（veto stop），打回主循环强制补齐任务。
  - `steps/step09_plan_todo.py`：提供完整的剧本式离线演示与真实 LLM 交互入口。
- **可运行实验**：运行 `python3 steps/step09_plan_todo.py --demo`，观察 LLM 在 Plan 阶段被拦截、制定计划切换到 Act 阶段、试图提前交卷被 Todo 守门员拦截并最终全量完成的全流程。

---

## 6. 留给实现者自行决定的细节（Left to the Builder）

以下技术细节不需要额外打扰用户，由实现者按清晰、极简原则自行决定：
1. **终端打印样式**：使用标准 ANSI 转义码封装一个 30 行的轻量级打印辅助函数（区分 `[THINK]`、`[TOOL CALL]`、`[VALIDATION]`、`[RECOVERY]`、`[SUBAGENT]` 的颜色）。
2. **YAML Frontmatter 解析**：`SKILL.md` 的头部解析直接使用 Python 标准库字符串切分/正则实现，不额外引入 `PyYAML` 依赖。
3. **Token 估算算法**：为保持零重型依赖，采用字符数折算估算结合 OpenAI 返回的 `usage.total_tokens` 来触发 Step 7 的上下文压缩。
4. **内置 `--demo` 模拟支持**：每个 `steps/step0X_*.py` 除了支持真实交互外，提供清晰的测试 Prompt 建议，让用户复制一行命令就能验证该步特性。
