"""d01 · Quickstart：为什么 Deep Agents 叫 "opinionated harness"

覆盖 Feature   : （入口）Sub-agents / Filesystem / Shell / Context management 的默认装配
覆盖 Principle : #1 Opinionated、#3 Model-agnostic

---------------------------------------------------------------------------
Principle #1 "Opinionated — defaults tuned for long-horizon, multi-step work"
---------------------------------------------------------------------------
`create_deep_agent()` 一行代码背后，deepagents 装配了下面这一整栈中间件
（顺序即行为，来自 `concepts/middleware-catalog.md`）：

    1. SkillsMiddleware            (给了 skills= 才有)
    2. FilesystemMiddleware        -> ls/read_file/write_file/edit_file/glob/grep
    3. SubAgentMiddleware          -> task
    4. SummarizationMiddleware     -> 自动压缩
    5. PatchToolCallsMiddleware   -> 修复被中断的历史 tool_call
    6. AsyncSubAgentMiddleware     (给了 AsyncSubAgent 才有)
    --- 调用方 middleware 插在这里 ---
    7. HarnessProfile.extra_middleware
    8. _ToolExclusionMiddleware
    9. Anthropic/Bedrock/Fireworks prompt caching
   10. MemoryMiddleware            (给了 memory= 才有)
   11. HumanInTheLoopMiddleware    (给了 interrupt_on= 才有)
   12. UnsupportedContentMiddleware

也就是说：文件系统 + 委派 + 上下文管理 **默认就有**，不需要你逐个 import。
这就是 "batteries-included" 的含义。

---------------------------------------------------------------------------
Principle #3 "Model-agnostic"
---------------------------------------------------------------------------
`model` 接受 `"provider:model"` 字符串或已初始化的 `BaseChatModel`。
本 demo 两种写法都演示一遍：换模型只改字符串 / 传实例，agent 代码不变。

运行：
    python d01_quickstart.py            # 用真实模型（需要 key）
    python d01_quickstart.py --offline  # 无 key 时的确定性回放
"""

from __future__ import annotations

import sys

from deepagents import create_deep_agent
from langchain_core.tools import tool

from shared import config, ui
from shared.offline import ScriptedModel, choose_model, say, tool_call


# --- Feature: Tools —— "bring your own functions" -----------------------------
# 普通 Python 函数 + 类型注解 + docstring 就够了，deepagents 负责转成 BaseTool。
@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气。docstring 就是给模型看的工具说明。"""
    return f"{city} 永远是晴天，25°C。"


def inspect_default_harness(model) -> dict[str, object]:
    """把 opinionated 的默认装配"打印"出来——这是本 demo 的重点。

    我们直接读编译后 graph 的形状，而不是相信文档：
      * `stream_channels` 里有没有 `files`  -> FilesystemMiddleware 生效
      * 工具节点里有哪些工具名             -> 内置工具集
    """
    tools = sorted(agent_nodes := _bound_tool_names(model))
    return {
        "state channels": sorted(model.stream_channels),
        "graph nodes": sorted(model.nodes),
        "exposed tools": tools,
        "tool count": len(tools),
        "graph name": agent_nodes,
    }


def _bound_tool_names(agent) -> list[str]:
    """从编译后的 LangGraph 里取出模型实际可见的工具名。"""
    node = agent.nodes.get("tools")
    bound = getattr(node, "bound", None)
    by_name = getattr(bound, "_tools_by_name", None)
    return sorted(by_name) if by_name else []


def build_agent(model):
    return create_deep_agent(
        model=model,
        tools=[get_weather],
        system_prompt=(
            "你是一个研究助手。"
            "回答前先规划，再读写文件把中间结论落盘，最后给出结论。"
        ),
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    ui.banner(
        "d01 · Quickstart —— batteries-included harness",
        feature="默认装配：Filesystem / Sub-agents / Context management",
        principle="#1 Opinionated · #3 Model-agnostic",
    )

    # ---- 1. 真实模型 or 离线回放 ------------------------------------------
    model = choose_model(
        [
            # 离线脚本：模拟"规划 -> 落盘 -> 回答"的长程多步行为
            tool_call(
                "write_file",
                {
                    "file_path": "/plan.md",
                    "content": "# 计划\n1. 查天气\n2. 写结论\n",
                },
            ),
            tool_call("get_weather", {"city": "旧金山"}),
            say("已把计划落盘到 /plan.md，旧金山今天晴天 25°C。"),
        ],
        argv,
    )
    ui.kv("model (live)", config.model_spec())
    ui.kv("using offline script", isinstance(model, ScriptedModel))

    agent = build_agent(model)

    # ---- 2. 把"默认到底给了什么"打印出来 ---------------------------------
    ui.subtitle("opinionated defaults —— 编译后 graph 的真实形状")
    for key, value in inspect_default_harness(agent).items():
        ui.kv(key, value)
    ui.note("说明：`files` channel + ls/read_file/write_file/edit_file/glob/grep/")
    ui.note("      `task`(sub-agent) 就是 README 里列的默认能力，无需显式开启。")

    # ---- 3. 跑一轮 --------------------------------------------------------
    ui.subtitle("invoke")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "查一下旧金山天气，并把你的计划写下来"}]}
    )
    ui.show_messages(result)
    ui.step("agent 写进虚拟文件系统的产物：")
    ui.show_files(result.get("files"))

    # ---- 4. Model-agnostic 的另一种写法 ----------------------------------
    ui.subtitle("model-agnostic：换成已初始化的 BaseChatModel 实例")
    ui.note("docs FAQ: 任何支持 tool calling 的模型都能用（frontier / open-weight / 本地）。")
    ui.note("`resolve_model` 收到实例会原样返回；收到字符串才走 init_chat_model。")
    try:
        from langchain.chat_models import init_chat_model

        prebuilt = init_chat_model(config.model_spec())
        agent2 = build_agent(prebuilt)
        ui.kv("tools (prebuilt model)", _bound_tool_names(agent2))
        ui.note("工具集完全一致 —— 换 provider 不改变 harness 行为。")
    except Exception as exc:  # noqa: BLE001 —— provider 包没装是正常情况
        ui.note(f"跳过：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
