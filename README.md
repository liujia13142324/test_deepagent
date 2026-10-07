# Deep Agents 中文 demo 仓库

用 **16 个可离线运行的 Python 脚本** 把 langchain-ai/deepagents 的核心
feature 逐个演示一遍。每个 demo 都是一份独立的自文档：文件头 docstring 讲
文档依据 + 原理 + 陷阱，运行时输出的 `[ok]`/`[x]` 断言就是源码检验。

四条设计原则的展开和 demo 对照见 [`PRINCIPLES.md`](PRINCIPLES.md)。

## 快速开始

离线跑全部（不联网、不发模型请求、每个 demo 用脚本模型替身）：

```bash
python run_all.py            # 全量（16 个）
python run_all.py 05 12      # 只跑 d05 / d12
```

跑单个（也全部支持 `--offline`）：

```bash
python d01_quickstart.py --offline
```

不传 `--offline` 时，demo 会用回车交互式选择真实模型（读本地 provider
环境变量），输出对照表看当前能连谁（`shared/config.py`）。

## Demo 总览

| # | 文件 | Feature | 原则 |
|---|------|---------|------|
| 01 | `d01_quickstart.py` | 默认装配：Filesystem / Sub-agents / Context 管理 | #1 * #3 |
| 02 | `d02_filesystem.py` | Filesystem 写读 + 版本化提交 | #1 * #2 |
| 03 | `d03_backends.py` | Filesystem 可插拔后端（BaseFilesystem + execute） | #2 * #4 |
| 04 | `d04_shell.py` | 沙箱化 shell 执行 | #1 * #2 * #4 |
| 05 | `d05_subagents.py` | 子代理：声明式 vs 编译式 vs async（导出 build_graph） | #1 * #2 * #3 |
| 06 | `d06_context.py` | Context 管理（系统提示/内存剪裁） | #1 * #2 |
| 07 | `d07_memory.py` | Memory（长期记忆读/写/删除） | #2 * #4 |
| 08 | `d08_hitl.py` | Human-in-the-loop（interrupt + 流事件恢复） | #2 * #4 |
| 09 | `d09_permissions.py` | Filesystem 权限三档：allow / deny / interrupt | #2 * #4 |
| 10 | `d10_skills.py` | Skills（SKILL.md 渐进式披露） | #1 * #2 |
| 11 | `d11_custom_tools.py` | 自定义工具：@tool / dict schema / ToolRuntime | #1 * #2 * #3 |
| 12 | `d12_mcp.py` | MCP：in-process FastMCP + MCPAdapter → 异步工具 | #1 * #2 * #3 * #4 |
| 13 | `d13_middleware.py` | 自定义 AgentMiddleware（4 个钩子 + factory 去重） | #1 * #2 |
| 14 | `d14_profiles.py` | Provider / Harness profile 覆盖 | #3 * #4 |
| 15 | `d15_rubric.py` | RubricMiddleware：自评-修订-再审质量门 | #1 * #2 * #3 |
| 16 | `d16_production.py` | 生产化收敛：checkpointer / store / build_graph | #4 |

每个 demo 文件头部的 docstring 都标注：

- `Feature   :` 属于哪个 feature
- `Principle :` 对应哪条官方原则（见 PRINCIPLES.md）
- `文档依据  ：` 引用的官方文档/源码位置
- `!!! 陷阱`/注意事项：跑 demo 或读源码时踩过的具体坑（例如 dict 工具被
  静默跳过、ScriptedModel 一行一个回复、middleware 同名实例报错等）

## 离线模式是怎么工作的

`shared/offline.py` 提供了一个 `ScriptedModel`（假模型）：
把它替身进 `create_deep_agent`，agent 框架全部真实执行，只有“模型说话”这
一步按剧本回复。这样：

- 图结构、中间件、工具、子代理、rubric 循环全部真实跑通；
- 断言能精确验证框架行为，不需要网 / API key / 花钱。
- `ScriptedModel(model_name=...)` 还能伪装 provider 标识，让 harness profile
  查询等逻辑也可以在离线下被测试（d14）。

约定（写新 demo 时遵守）：

1. `def main(argv=None)`；argv 透传给 `choose_model`/`run_all`。
2. 离线时**一个 agent 用一个独立的 ScriptedModel**（实例共享会 script
   耗尽）；多次 invoke 就把对应次数的 `say(...)` 排进脚本。
3. 断言用 `shared/ui.py` 的 `assert_true/equal/in`（输出 `[ok]`/`[x]`）。
4. 不要用 PowerShell 的 `Get-Content | Set-Content` 回写含中文的 `.py`
   （编码问题）；文件统一 UTF-8。
5. Windows 控制台若出现中文乱码是显示层问题，不影响执行与文件内容。

## 部署（可选）

`langgraph.json` 已经声明了两个图入口，想 `langgraph dev/up` 起本地平台：

```json
{ "graphs": { "agent": "./d16_production.py:build_graph",
              "research": "./d05_subagents.py:build_graph" } }
```

`d16_production.build_graph()` 是个纯函数：不带 `--offline` 时用
`init_chat_model("anthropic:claude-sonnet-4-6")`，并已接好
checkpointer + store + name。生产侧把 `InMemorySaver/InMemoryStore`
换成 Sqlite/Postgres 版本即可。

## 依赖

见 [`requirements.txt`](requirements.txt)。备注：

- `deepagents` 建议直接用 Git 安装（PyPI 的 release 滞后于仓库）：
  `deepagents @ git+https://github.com/langchain-ai/deepagents.git`
- MCP 相关需要 `langchain[mcp]`（MCPAdapter 在 beta，跑起来会打
  LangChainBetaWarning，正常）。
- 想切真实模型：装对应的 `langchain-<provider>`，并在 `.env` 配好
  API key。

## 换机器 / 新环境怎么搭（editable 是「按环境」的，不是「按项目」的）

本仓库里的 demo **本身是可移植的**（没有绝对路径依赖）；但
`pip install -e deepagents` 只存在于**当前这台机器当前这个 Python 环境**
里（它在 site-packages 写 `.pth` 直链到源码目录）。因此换一台机器、
或新建一个 conda env / venv，都必须重新 editable 一次。完整步骤：

```powershell
# 1) 准备 Python 3.12 + deepagents 源码（任选其一）
git clone --depth 1 https://github.com/langchain-ai/deepagents.git D:\workspace_opensource\deepagents
#    或者直接把别人机器上已 check out 的整个 deepagents 目录拷过去

# 2) 在该机器/该环境里，把编译好的依赖装好
pip install -r D:\workspace_python\test_deepagent\requirements.txt
#    注意：requirements.txt 里的 `deepagents @ git+...` 会从 GitHub 装一份 exe 版。
#    如果你走的是下面的 editable 源码流程，就把它注释掉，避免装两遍。

# 3) 用 compat 模式 editable 源码（改源码即时生效，PyCharm 更友好）
python -m pip install -e D:\workspace_opensource\deepagents\libs\deepagents --config-settings editable_mode=compat

# 4) 验证装到了源码目录
python -c "import deepagents; print(deepagents.__file__)"
#    应输出 D:\workspace_opensource\deepagents\libs\deepagents\deepagents\__init__.py
#    而不是 site-packages 里的某个 __editable__ 缓存或拷贝。

# 5) 跑全量回归确认环境 OK
cd D:\workspace_python\test_deepagent
python run_all.py
```

**要点**：

- 步骤 3 的作用域 = 「这台机器 + 这个 Python 解释器」。每台机器 / 每个
  venv 都要单独跑，跑一次即可，除非源码目录被删/挪位置。
- 步骤 2 和 3 的顺序无所谓，但共存时记得把 `requirements.txt` 里的
  deepagents Git URL 注释掉（二选一，避免重复装）。
- `langgraph.json` 的入口是相对路径（`./d16_production.py:build_graph`），
  换机器不用改。
- 想用 PyCharm：解释器选新机器上的同一个 conda base（或新 venv），
  步骤 3 建好的 `.pth` 链接会被自动识别；若识别不到，重跑一次步骤 3
  或 `File → Invalidate Caches / Restart`。
- 源码目录变更后只需**在本机重跑步骤 3** 即可，demo 代码与配置文件都不用动。