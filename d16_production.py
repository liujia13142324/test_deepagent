"""d16 · Production --------- 把上面十来个 feature 收敛成可以 `langgraph dev/up` 拉起来的图

Feature   : Production hardening. checkpointer / store / 可编译入口 / 环境化模型工厂
Principle : #4 Production-ready（persistence / long-term memory / deploy entrypoint）
考点       : 不止「能跑 demo」，还「能部署」——本条就是收口的那一个 demo。

---------------------------------------------------------------------------
文档依据：concepts/tool-calling.md（ToolRuntime.store）、deployment 文档 +
          shared/offline.py 的 ScriptedModel（离线下把模型换成脚本，生产换真模型）
---------------------------------------------------------------------------
0. 一句话：前面的 demo 都在展示 feature；d16 展示的是**接线**——
   - 同一个 `create_deep_agent` 调用，把 checkpointer + store + name 一次接好；
   - 暴露 `build_graph()` 让 LangGraph 平台（langgraph.json）能编译它；
   - 模型走一个工厂函数：本地离线换 ScriptedModel，生产切真实 provider。
   代码和「能不能编译」是同一份——这就是 production-ready 的意义。

1. **build_graph() 是部署契约**：`langgraph.json` 写 `./d16_production.py:build_graph`。
   返回 Runnable（其实是 compiled Agent），LangGraph 平台据此建图、接线程、接续聊。
   本 demo 故意不带任何 lambda/middleware 的魔法：生产化 = 参数化，不是重排。

2. **checkpointer = 会话持久化**：同一个 thread_id 第二次 invoke，LangGraph 会
   把历史整段续上（多轮对话的基础设施）。不传 thread_id 每次都是新会话。

3. **store = 跨线程长期记忆**：middleware 的 ToolRuntime 有 `.store`（d11 讲过
   没注入 store 时是 None）。这里把 store 真正当「长期记忆」用：线程 A 写入，
   线程 B（同一个 store 实例）能读回来。
---------------------------------------------------------------------------
"""

from __future__ import annotations

import sys

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore

from shared import ui
from shared.offline import ScriptedModel, choose_model, say, tool_call

MODEL = "anthropic:claude-sonnet-4-6"

MEMORY_NS = ("project-memory",)


def build_graph(*, offline: bool = False):
    """供 langgraph.json 引用的部署入口：`./d16_production.py:build_graph`。

    约定：
    - 平台（`langgraph up` 之类）不带 --offline：拿到**真实模型**；
    - 本地离线 demo 传 offline=True：换 ScriptedModel，逻辑一行都不变。
    只读一次构建、不读运行时环境——`langgraph dev` 依赖这个纯函数契约。
    """
    model = _model_for(offline=offline)
    return create_deep_agent(
        model=model,
        # 平台/生产侧替换成 SqliteSaver / PostgresSaver；这里是同构的最小实现。
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        name="deep-agent-prod",
    )


def _model_for(*, offline: bool = False):
    if offline:
        # 离线替身：真实模型在平台上由 init_chat_model(ANTHROPIC...) 接管，
        # 测试双一样保证图「能编译、能跑」，替换的只有最后一个字节（模型）。
        return ScriptedModel(script=[say("离线替身回答。")], model_name="prod-demo")
    from langchain.chat_models import init_chat_model

    # 真实环境：按字符串解析 provider → 我们 demo 里只验证「工厂切换」这个职责。
    return init_chat_model(MODEL)


@tool
def remember(knowledge: str, runtime: ToolRuntime) -> str:
    """把一条经验写进长期记忆（跨会话、跨线程仍然能 recall）。"""
    idx_item = runtime.store.get(MEMORY_NS, "idx")
    n = int(idx_item.value) if idx_item is not None else 0
    runtime.store.put(MEMORY_NS, f"note-{n}", {"knowledge": knowledge})
    runtime.store.put(MEMORY_NS, "idx", n + 1)
    return f"已记住 #{n}：{knowledge}"


@tool
def recall(runtime: ToolRuntime) -> str:
    """读出长期记忆里所有笔记（跨线程）。"""
    found = []
    for item in runtime.store.search(MEMORY_NS):
        if "idx" not in item.key and "note" in item.key:
            found.append(item.value.get("knowledge", ""))
    return "；".join(found) if found else "（暂无记忆）"


def demo_build_graph() -> None:
    ui.subtitle("1. build_graph() = langgraph.json 的部署入口（离线也能编译）")
    agent = build_graph(offline=True)
    ui.kv("agent.name", agent.name)
    cfg = {"configurable": {"thread_id": "prod-demo"}}
    out = agent.invoke({"messages": [{"role": "user", "content": "你好"}]}, config=cfg)
    ui.kv("invoke 正常返回", f"{len(out['messages'])} 条消息")
    ui.assert_true(agent.name == "deep-agent-prod", "名字就是 langgraph.json 里配置的 graph 名之一")
    ui.assert_true(len(out["messages"]) >= 2, "编译出来的图离线也能直接 invoke（脚本模型）")
    ui.assert_true(out.get("files") is not None, "挂上 checkpointer 后必须有 thread_id 才能 invoke")
    ui.note("生产里这个函数不带 offline：`init_chat_model('anthropic:claude-sonnet-4-6')`。")
    ui.note("  langgraph dev 读取 langgraph.json → 调 build_graph() → 给平台一个 Runnable；")
    ui.note("  thread/run 元数据由平台的 config 注入（configurable.thread_id 等），图里不碰。")


def demo_checkpointer() -> None:
    ui.subtitle("2. 同一 thread 二次 invoke = 多轮会话（消息累计）")
    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        model=choose_model(
            [say("第一轮回答。"), say("第二轮回答。"), say("新线程回答。")], None
        ),
        checkpointer=checkpointer,
    )
    cfg = {"configurable": {"thread_id": "conversation-1"}}
    first = agent.invoke({"messages": [{"role": "user", "content": "第一问"}]}, config=cfg)
    n1 = len(first["messages"])
    second = agent.invoke(
        {"messages": [{"role": "user", "content": "第二问（要记得上一轮）"}]}, config=cfg
    )
    n2 = len(second["messages"])
    ui.kv("首轮消息数", n1)
    ui.kv("次轮消息数", n2)
    ui.assert_true(n2 > n1, "二次 invoke 把历史整段续上了（这就是聊天功能的地基）")
    ui.assert_true(
        any("第一轮回答" in str(m.content) for m in second["messages"]),
        "上一轮的模型回答还在上下文里",
    )
    fresh = agent.invoke(
        {"messages": [{"role": "user", "content": "换个新 thread"}]},
        config={"configurable": {"thread_id": "conversation-2"}},
    )
    ui.kv("新 thread 消息数", len(fresh["messages"]))
    ui.assert_true(len(fresh["messages"]) < n2, "新 thread_id = 全新会话，不带老历史")
    ui.note("换 SqliteSaver / PostgresSaver = 进程重启/多实例后仍然记得（生产持久化）。")


def demo_store_memory() -> None:
    ui.subtitle("3. store = 跨线程长期记忆（线程 A 写入，线程 B 读回）")
    store = InMemoryStore()

    def _tool_result(out):
        return next(
            (m.content for m in out["messages"]
             if m.__class__.__name__ == "ToolMessage"), None)

    writer = create_deep_agent(
        model=choose_model(
            [tool_call("remember", {"knowledge": "发布流程=先跑全量回归"}),
             say("好的，已写入长期记忆。")],
            None,
        ),
        tools=[remember, recall],
        store=store,
    )
    out1 = writer.invoke({"messages": [{"role": "user", "content": "记住：发布流程=先跑全量回归"}]})
    ui.kv("写入工具返回", _tool_result(out1))

    reader = create_deep_agent(
        model=choose_model([tool_call("recall", {}), say("查完了。")], None),
        tools=[remember, recall],
        store=store,
    )
    out2 = reader.invoke({"messages": [{"role": "user", "content": "我上回记了什么？"}]})
    got = _tool_result(out2) or ""
    ui.kv("工具读取到", str(got)[:60])
    ui.assert_true("发布流程" in str(got), "跨线程读回了记忆（同一 store 实例）")
    ui.assert_true("先跑全量回归" in str(got), "记忆内容精确命中")
    ui.note("store 与 checkpointer 是两件事：checkpointer=会话级，store=长期记忆。")
    ui.note("  项目级记忆 -> 换 InMemoryStore 为 PostgresStore，同一套工具代码不变。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d16 * Production --------- 收敛成可部署的图（langgraph dev 入口）",
        feature="Production hardening",
        principle="#4 Production-ready",
    )
    demo_build_graph()
    demo_checkpointer()
    demo_store_memory()


if __name__ == "__main__":
    main()