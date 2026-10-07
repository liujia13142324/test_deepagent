# PRINCIPLES.md —— Deep Agents 四条设计原则 × demo 对照

> 这四条原则是 **deepagents 文档里的官方四原则**。每个 demo 都在文件头的
> `principle=` 里标注了它主要示范哪几条；本文件负责把每条原则讲透——
> 原则是什么、源码/文档里怎么说的、用哪个 demo 证明、以及“反例”（哪些
> 做法其实并不满足该原则）。

四个原则快速总览：

| # | 原则 | 一句话 | 最典型 demo |
|---|------|--------|------------|
| #1 | **Opinionated** 有主见 | 框架替你做好了决定，你有默认但可覆盖 | d01, d05, d10 |
| #2 | **Extensible** 可扩展 | 每个默认都有官方自定义口子 | d03, d11, d12, d13 |
| #3 | **Model-agnostic** 模型无关 | 同样的图换任何支持 tool-calling 的模型都能跑 | d14, d15 |
| #4 | **Production-ready** 可生产 | checkpoint/记忆/权限/部署一次到位 | d07, d08, d09, d16 |

---

## #1 Opinionated —— 有主见：默认就是对的，然后给你离开的出口

**文档原话（概念）**：Deep Agents “has opinions.” 它希望你能直接开工，
而不是先编一个又一个 prompt / 工具 / 子代理胶水。

具体体现（全部由本仓库 demo 验证过）：

- **默认捆绑 Filesystem**：agent 一出厂就能 `read_file` / `write_file` /
  `list_files`，连权限都能用 `permissions=` 声明式配置（d01/d09）。
- **默认捆绑 Sub-agents**：遇到不熟悉的路径/知识领域会自动派生出子代理去
  查（d05），这是“opinionated”里最有分量的一个决定——你不用自己写
  “先搜索再回答”的 prompt。
- **默认捆绑 Skill 渐进式披露**：只把 name+description 列出，需要时 agent
  自己 `read_file` 展开 SKILL.md（d10）。
- **`tools=` 是叠加而不是替换**：给 `tools=[get_weather]`，默认 write_file
  那条还在（d11）——决定是“加上你的”，不是“删掉我的”。

**Opinionated ≠ 无法覆盖**：每个默认都有覆盖口——收窄工具用
`FilesystemMiddleware(tools=...)`，改系统提示用 `system_prompt=` 或
harness profile（d14），这是原则 #1 与 #2 的接缝处。

**反例（耗 token 但没意义）**：自己写“从零实现子代理调度”——框架已经
有 opinion 了，你的代码只是在跟默认打架。

---

## #2 Extensible —— 可扩展：官方自定义口子，而不是让你 fork

文档原话（概念）：中间件目录/工具目录都明确写着
“check the built-in libraries first, then extend if you need to”。

具体体现：

- **AgentMiddleware / AgentRuntime**：`before_agent` / `after_agent` /
  `wrap_model_call` / `wrap_tool_call` 四个钩子，看着就是给扩展准备的
  （d13）。
- **@tool + ToolRuntime 注入**：一个普通 Python 函数 + 参数注入
  `ToolRuntime`（state/store/config），就变成 agent 的工具（d11）。
- **MCP**：不写 SDK 特有代码，`MCPAdapter` 把任意 MCP server 的工具列表
  抬进 agent（d12）。
- **pluggable 后端子类**：`BaseFilesystem` 只有一个方法 `execute()`，
  继承它就能换后端实现（d03）。
- **RubricMiddleware 文档里直接标了 extension point**：`_invoke_grader`
  是“per-call extension point”，专门留给自定义 grader（d15）。

**可扩展的正确姿势**：走框架留下的口子（middleware / tool / subclass /
`_invoke_grader`），而不是改框架源码。改源码 = 升级即阵亡。

---

## #3 Model-agnostic —— 模型无关：图与模型解耦

文档原话（FAQ）：“Any model that supports tool calling works.”

具体体现：

- **模型 = 一个字符串**：`create_deep_agent(model="anthropic:claude-...")`
  换成 `openai:...` / `google_genai:...`，图结构一行不改（d01 / d14）。
- **profiles 切模型**：`ProviderProfile` 在模型符串上按 provider 匹配并
  merge init kwargs；harness profile 再叠 system prompt / 工具覆盖
  （d14）。
- **grader 也是模型无关的**：RubricMiddleware 的 grader 走
  `_strategy_from_model` 自动选 Tool/Provider 结构化输出策略，策略差异全在
  grader 内部，上层只认 GraderResponse（d15）。
- **离线也能验证“模型无关”**：ScriptedModel 把 provider 标识伪装成
  `cheap:scripted-*`，因此 harness 查询、profile merge 等逻辑在没 API key
  时也能跑（shared/offline.py + d14）。

**注意**：模型无关不等于“交互协议无关”——结构化输出策略、thinking、
prompt caching 都是 provider 相关的，框架把这些挡在内部。你的业务代码不该
看见它们。

---

## #4 Production-ready —— 可生产：不是 demo 玩具，是地基

文档把 LangGraph/LangChain 的 persistence / streaming / checkpointing 直接
作为卖点写入原则。

具体体现：

- **会话持久化**：`checkpointer=` + `thread_id`，二次 invoke 自动续上下文
  （d02 / d16）。
- **长期记忆**：`store=` 跨线程读写（d16）；其实 d07 的 memory feature 也
  覆盖了记忆的声明式用法。
- **安全默认**：Filesystem 有版本化子图（write→commit 有源码回溯）、
  permissions 白名单/黑名单/interrupt 三档（d02/d09）。
- **沙箱**：shell 工具可换 Daytona/Modal/Runloop 等沙箱后端（d04）。
- **HITL**：`interrupt()` + 流事件，人工审批嵌入工具调用流（d08）。
- **部署契约**：`build_graph()` 是 langgraph.json 能识别的图入口，
  平台负责 thread/run 注入，图本身不碰——d16 把上面的东西“接线”成一个
  可部署 Runnable。

**可生产的反例**：demo 能跑但没接 checkpointer/store/deploy 入口 —— 一旦
上平台会到处找不到状态。d16 的意义就是：“demo 和能部署的代码是同一份”。