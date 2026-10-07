"""d12 · MCP 工具 —— 进程内 FastMCP server + MCPAdapter 接进 Deep Agent

覆盖 Feature   : MCP (Model Context Protocol) 工具集成
覆盖 Principle : #1 Opinionated（async 优先）· #2 Extensible · #3 Model-agnostic

---------------------------------------------------------------------------
实现与依据（concepts/tools-filesystem.md「dcode catalog and MCP tools」+
        langchain 的 MCPAdapter docstring，二者都是本机可实测的第一手来源）
---------------------------------------------------------------------------

1. **MCP 是"外部工具源"，不是 filesystem middleware**
   dcode 对每个发现的 MCP 工具用 `langchain.mcp.as_langchain_tool` 包成
   异步 StructuredTool；本 demo 用更现代的 `MCPAdapter`。

2. **MCPAdapter 的取舍（docstring 直接给了三个疑点）**
   - 一个 string target 必须是 http(s) URL —— 即使 fastmcp.Client 支持
     "字符串=本机脚本路径"，MCPAdapter 也会拒绝字符串路径，防止模型
     随便拉起本机进程。要跑本地 stdio 服务器，必须显式传 Path / transport。
   - 服务器 input() 需要人输入时，默认用 LangGraph 的 interrupt() 把，
     由人回答后 resume（`langchain.mcp.elicitation`），本 demo 不演示。

3. **这是"模型无关"最实在的例子**：MCP 服务器根本不知道你在用什么模型，
   甚至不知道是不是 LLM 在调用它 —— 它只认 JSON-RPC 这根线。
"""

from __future__ import annotations

import asyncio
import sys

from deepagents import create_deep_agent
from fastmcp import FastMCP
from langchain.mcp import MCPAdapter

from shared import ui
from shared.offline import bound_tool_names, model_factory, say, tool_call


# ===========================================================================
# 一个"第三方"MCP 服务器：它完全不知道 Deep Agents / LangChain 的存在
# ===========================================================================
def build_legacy_server() -> FastMCP:
    """假装是公司里遗留的"订单域"服务，通过 MCP 对外。

    注释刻意写得像"老服务伪码"：MCP 的价值就是不动这种代码也能接入。
    """
    mcp = FastMCP("legacy-order-service")

    @mcp.tool()
    def lookup_order(order_id: str) -> str:
        """按订单号查询订单状态（遗留系统，channel 是短信/邮件，不是 API）。

        Args:
            order_id: 订单号，例如 ORD-10086。
        """
        return f"开始配送（配送渠道：短信通知 + 智障老系统）"

    @mcp.tool()
    def add_note(order_id: str, note: str) -> str:
        """给订单追加一条仅内部可见的备注。

        Args:
            order_id: 订单号。
            note: 备注内容。
        """
        return f"已写入备注: {note}"

    return mcp


def demo_attach(model) -> list:
    ui.subtitle("1. 进程内 FastMCP server -> MCPAdapter -> 一组异步工具")
    server = build_legacy_server()
    ui.note("FastMCP v4 用 @mcp.tool() 注册；服务器对上面的协议栈一无所知。")

    tools = asyncio.run(_to_tools(server))
    for tool in tools:
        ui.kv("工具", f"name={tool.name}  async={tool.is_async if hasattr(tool,'is_async') else 'n/a'}")
    ui.assert_true(any(getattr(t, "name", "") == "lookup_order" for t in tools), "MCP 工具被发现")

    agent = create_deep_agent(model=model, tools=tools)
    ui.kv("工具集", ", ".join(sorted(t.name for t in tools)))
    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "查一下 ORD-10086"}]})
    )
    ui.show_messages(result)
    tool_msg = next(m for m in result["messages"] if m.type == "tool")
    ui.kv("工具返回（MCP content blocks）", str(tool_msg.content)[:120])
    ui.note("注意：MCP 工具是**异步**的 -> agent 要走 `ainvoke`（或 asyncio.run 包一层）。")
    ui.note("  ToolRuntime/store 等注入规则同 d11 —— 对 MCP 工具照样成立，")
    ui.note("  所以图状态、记忆、HITL 全部不受工具来源影响。")
    return tools


async def _to_tools(server):
    async with MCPAdapter(server) as adapter:
        return await adapter.list_tools()


def demo_principle_mapping() -> None:
    ui.subtitle("2. 对照四条 principle")
    ui.kv("#1 Opinionated", "工具就是一笔异步 StructuredTool；MCP 是协议不是新抽象")
    ui.kv("#2 Extensible", "遗留服务零改动接入；字符串 target 只放行 http(s)，见下")
    ui.kv("#3 Model-agnostic", "服务器不认识模型；换模型只换 model=，工具不动")
    ui.kv("#4 Production-ready", "interrupt 处理输入；连接生命周期用 context manager")

    ui.step("安全设计（来自 MCPAdapter docstring，值得背下来）")
    ui.note("  * str target 必须是 http(s) URL —— 字符串路径会被拒绝。")
    ui.note("    （fastmcp.Client 自己认脚本路径，adapter 刻意收窄，防模型拉起本机进程。）")
    ui.note("  * 本地 stdio 服务器：显式传 Path / transport，别用字符串。")
    ui.note("  * 输入需要人时 -> LangGraph interrupt()，由人回答后 resume。")

    test_model = model_factory([say("只展示，不再真调。")], None)()
    cheap = create_deep_agent(model=test_model)
    cheap_names = bound_tool_names(cheap)
    ui.kv("工具集（对照）", ", ".join(sorted(cheap_names))[:110])
    ui.assert_true("lookup_order" not in cheap_names, "不挂 adapter，工具集里就没有 MCP 工具")
    ui.note("（离线脚本没有真实 MCP 服务器，上面的工具集来自同一进程内的 server；")
    ui.note("  若要用真实 stdio 服务器，让 main(argv) 里的 --offline 传入 URL 即可。）")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d12 * MCP --------- in-process FastMCP -> MCPAdapter -> agent tools",
        feature="MCP tools",
        principle="#1 Opinionated * #2 Extensible * #3 Model-agnostic * #4 Production",
    )

    demo_attach(
        model_factory(
            [
                tool_call("lookup_order", {"order_id": "ORD-10086"}, "c1"),
                say("好的，正在配送。"),
            ],
            argv,
        )()
    )

    demo_principle_mapping()  # 只读工具集，不真跑 agent


if __name__ == "__main__":
    main()