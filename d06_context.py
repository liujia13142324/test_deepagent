"""d06 · Context Management —— 长任务的上下文不会爆炸

覆盖 Feature   : Context management（自动摘要 + 主动压缩 + 大结果外置）
覆盖 Principle : #1 Opinionated（默认带 summarization）· #2 Extensible

---------------------------------------------------------------------------
文档依据（concepts/context-management.md）
---------------------------------------------------------------------------
两套机制，可以叠加使用：

1. **自动摘要**（`SummarizationMiddleware`）—— 超过 `trigger` 阈值时，
   用模型把历史压成一段摘要，并把**原始历史外置到 backend**。
   `create_deep_agent` 默认就带了一个，只是阈值走 profile 默认值。

2. **主动压缩**（`create_summarization_tool_middleware`）—— 给 agent 一个
   `compact_conversation` 工具，让它**自己决定**什么时候压缩。
   带一个 ~50% 触发阈值的资格门：太早调用会被拒绝。

一句话：前者是"框架兜底"，后者是"agent 自主管理"。
"""

from __future__ import annotations

import sys

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import (
    SummarizationMiddleware,
    create_summarization_tool_middleware,
)

from shared import ui
from shared.offline import bound_tool_names, choose_model, say, tool_call

#: 触发阈值与保留量都用 (单位, 数值) 元组表达，单位可以是 messages / tokens / fraction。
TRIGGER = ("messages", 8)  # 超过 8 条消息就触发摘要
KEEP = ("messages", 3)  # 保留最近 3 条原始消息


def demo_defaults(model) -> None:
    ui.subtitle("1. 默认就带 summarization（Opinionated）")
    agent = create_deep_agent(model=model, tools=[])
    ui.kv("state channels", sorted(c for c in agent.stream_channels if c.startswith("_")))
    ui.assert_in("_summarization_event", agent.stream_channels, "默认注入摘要事件通道")
    ui.note("docs: summarization is on by default; you tune thresholds, not enablement.")
    ui.note("所以 d01~d05 的每个 demo 其实都跑在摘要中间件之下。")


def demo_auto_summarization(parent_model, summarizer_model) -> None:
    ui.subtitle("2. 自动摘要：超阈值 -> 模型压缩 + 原始历史外置到 backend")
    backend = StateBackend()
    agent = create_deep_agent(
        model=parent_model,
        tools=[],
        backend=backend,
        middleware=[
            # 放进 middleware= 就是**替换**默认那一个（与 FilesystemMiddleware 同理）
            SummarizationMiddleware(
                model=summarizer_model,  # 摘要可以用更便宜的模型
                backend=backend,  # 原始历史落到这里
                trigger=TRIGGER,
                keep=KEEP,
            )
        ],
    )

    # 关键：要真的**超过阈值**才能看到摘要。预置 10 条历史（>= trigger 8）。
    history: list[dict[str, str]] = [{"role": "user", "content": "目标：清洗 sales.csv 并出图。"}]
    for i in range(1, 10):
        history.append({"role": "user", "content": f"第 {i} 轮：补充约束 {i}。"})
        history.append({"role": "assistant", "content": f"第 {i} 轮：已处理 {i}。"})
    ui.kv("预置历史条数", len(history))

    result = agent.invoke(
        {"messages": history},
        config={"configurable": {"thread_id": "t-sum"}},
    )
    ui.kv("invoke 后消息数", len(result["messages"]))
    ui.step("消息类型分布（human 变少 = 已被摘要吃掉）：")
    kinds: dict[str, int] = {}
    for message in result["messages"]:
        kinds[message.type] = kinds.get(message.type, 0) + 1
    ui.kv("kinds", kinds)
    ui.step("被外置的原始历史（落在 backend 的 files 里，不是 messages）：")
    offloaded = {
        p: d for p, d in (result.get("files") or {}).items() if "conversation_history" in p
    }
    for path, data in offloaded.items():
        body = str(data.get("content", ""))
        ui.kv(path.rsplit("/", 1)[-1], f"{len(body)} chars")
        ui.kv("  首行", body.splitlines()[0])
        ui.kv("  末行", body.splitlines()[-1][:60])
    ui.assert_true(bool(offloaded), "触发了摘要且原始历史被外置到 backend")
    ui.assert_true(
        all("conversation_history" in p for p in (result.get("files") or {})),
        "state.files 里只有被外置的会话历史（不是业务文件）",
    )
    for line in [
        "trigger=('messages', 8) -> 按条数触发；也支持 ('tokens', N) / ('fraction', f)。",
        "keep=('messages', 3)    -> 保留最近 3 条原始消息，其余交给模型压缩。",
        "注意语义：graph **state 里的 messages 不删**（可审计、可 checkpoint），",
        "     被替换的只是**下一次模型请求**的窗口（request.override(messages=...)）。",
        "     原文落到 /conversation_history/session_<id>.md，agent 随时可 read_file 取回。",
        "摘要模型可以单独指定（生产上通常用一个便宜模型跑 summarizer）。",
    ]:
        ui.note(line)


def demo_agent_driven_compaction(parent_model, summarizer_model) -> None:
    ui.subtitle("3. 主动压缩：给 agent 一个 compact_conversation 工具")
    agent = create_deep_agent(
        model=parent_model,
        tools=[],
        backend=StateBackend(),
        middleware=[
            create_summarization_tool_middleware(
                summarizer_model,
                StateBackend(),
                system_prompt="你在做数据清洗。",
            )
        ],
    )
    ui.kv("工具集", bound_tool_names(agent))
    ui.assert_in("compact_conversation", bound_tool_names(agent), "agent 可自主压缩上下文")

    ui.subtitle("3b. 资格门：太早压缩会被拒绝")
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "刚说了两句话就想压缩"},
                {"role": "assistant", "content": "好的。"},
                {"role": "user", "content": "现在压缩"},
            ]
        }
    )
    for message in result["messages"]:
        if message.type == "tool":
            ui.kv(f"{message.name}", str(message.content).replace("\n", " ")[:90])
    for line in [
        "docs: 只有 tool layer 被注册；被包装的 SummarizationMiddleware 只是引擎，",
        "         不会自己按时触发。",
        "资格门在自动触发阈值的 ~50%：过早调用会被拒绝，避免 agent 手贱清空上下文。",
        "想两者都要：同时注册 SummarizationMiddleware 和这个 tool middleware。",
    ]:
        ui.note(line)


def demo_big_tool_output() -> None:
    ui.subtitle("4. 大结果外置：为什么要 grep/glob 而不是把整个文件读进上下文")
    ui.note("docs 的取舍很直接：文件工具会返回**行号 + 窗口**，而不是整份文件。")
    ui.note("配合 (a) 只读需要的行、(b) grep 先定位、(c) 大结果落盘再按需读，")
    ui.note("长任务的上下文增长才是可控的。d02 的 read/grep/glob 就是这条链路的演示。")
    ui.note("另一个正交手段：SubAgentMiddleware —— 把脏活在隔离子 agent 里干完，")
    ui.note("只有最终结论回到父上下文（d05 第 1 节）。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d06 * Context Management ---- auto summarize, self-compact, offload",
        feature="Context management",
        principle="#1 Opinionated * #2 Extensible",
    )

    demo_defaults(choose_model([say("ok")], argv))

    # 父 agent 与 summarizer 必须是两个模型实例，否则脚本游标会互相吃掉。
    demo_auto_summarization(
        choose_model([say("调研轮 1")], argv),
        choose_model([say("## SESSION INTENT\n清洗 CSV\n\n## SUMMARY\n已完成三轮清洗。")], argv),
    )

    demo_agent_driven_compaction(
        choose_model(
            [
                tool_call("compact_conversation", {}, "c1"),
                say("好的，先压缩。"),
            ],
            argv,
        ),
        choose_model([say("## SESSION INTENT\n压缩后的意图\n\n## SUMMARY\n…")], argv),
    )

    demo_big_tool_output()


if __name__ == "__main__":
    main()
