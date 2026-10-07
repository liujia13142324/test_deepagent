"""d08 · Human-in-the-Loop —— 让高危动作停下来等人

覆盖 Feature   : HITL（interrupt_on + 决策类型 + Command resume）
覆盖 Principle : #2 Extensible · #4 Production-ready

---------------------------------------------------------------------------
文档依据（concepts/permissions-hitl.md）
---------------------------------------------------------------------------
`create_deep_agent(interrupt_on={"write_file": True})` 会自动挂上
`HumanInTheLoopMiddleware`，**前提是有 checkpointer**（否则暂停无处可恢复）。

恢复用 LangGraph 的 `Command(resume=...)`，决策类型是四种：

| type      | 语义                                    |
|-----------|-----------------------------------------|
| `approve` | 原样执行                                |
| `edit`    | 用你给的 args 覆盖后执行                |
| `reject`  | 拒绝，模型收到拒绝原因                  |
| `respond` | 不执行工具，直接把你的文本当成最终回答  |

`interrupt_on` 的值还可以是 `InterruptOnConfig`（`allowed_decisions` 白名单），
用来限制审批人只能做某些决策。
"""

from __future__ import annotations

import sys

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from shared import ui
from shared.offline import choose_model, say, tool_call

THREAD = {"configurable": {"thread_id": "hitl-1"}}


def demo_approve(model) -> None:
    ui.subtitle("1. approve —— 批准后原样执行")
    agent = create_deep_agent(
        model=model,
        tools=[],
        interrupt_on={"write_file": True},
        checkpointer=InMemorySaver(),  # 没有它就暂停不了
    )
    pending = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "把结论写进 /report.md"},
            ]
        },
        config=THREAD,
    )
    ui.show_messages(pending)
    ui.assert_true(
        "__interrupt__" in pending,
        "invoke 返回 __interrupt__ 而不是执行结果",
    )

    ui.step("__interrupt__ 的结构（langchain 的 HITLRequest）：")
    for interrupt in pending["__interrupt__"]:
        value = interrupt.value  # dict: {"action_requests": [...], "review_configs": [...]}
        for request in value["action_requests"]:
            ui.kv("action_request.name", request["name"])
            ui.kv("action_request.args", str(request["args"])[:70])
        for config in value["review_configs"]:
            ui.kv("review_config.action", config["action_name"])
            ui.kv("review_config.allowed", config["allowed_decisions"])

    done = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=THREAD)
    ui.show_messages(done)
    ui.assert_in(
        "/report.md",
        (done.get("files") or {}),
        "批准后文件真的写进了 state",
    )


def demo_edit(model) -> None:
    ui.subtitle("2. edit —— 审批人改参数后再执行")
    agent = create_deep_agent(
        model=model,
        tools=[],
        interrupt_on={"write_file": True},
        checkpointer=InMemorySaver(),
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "写到 /tmp/draft.md"}]},
        config=THREAD,
    )
    done = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        # edited_action 是完整的 Action：name + args（可以连工具名一起改）
                        "edited_action": {
                            "name": "write_file",
                            "args": {
                                "file_path": "/tmp/reviewed.md",
                                "content": "# 经人工审阅的版本\n",
                            },
                        },
                    }
                ]
            }
        ),
        config=THREAD,
    )
    ui.show_messages(done)
    ui.assert_in("/tmp/reviewed.md", (done.get("files") or {}), "以审批人给的路径落盘")
    ui.note("典型用途：强制加免责声明、改写路径、注入人工补充内容。")


def demo_reject(model) -> None:
    ui.subtitle("3. reject —— 拒绝，并把原因喂回模型")
    agent = create_deep_agent(
        model=model,
        tools=[],
        # 删除类操作要审批：这里显式只拦 delete
        interrupt_on={"delete": True},
        checkpointer=InMemorySaver(),
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "删掉那个文件"}]},
        config=THREAD,
    )
    done = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "reject", "message": "未经批准不得删除文件，请改用归档。"}
                ]
            }
        ),
        config=THREAD,
    )
    ui.show_messages(done)
    reject_msg = next(
        (m for m in done["messages"] if m.type == "tool" and "未经批准" in str(m.content)),
        None,
    )
    ui.assert_true(reject_msg is not None, "拒绝原因以 ToolMessage 形式回到模型上下文")
    ui.note("这是 HITL 的核心价值：拒绝不是终止，而是把人的意图变成模型的下一步输入。")
    ui.note("不填 message 时，模型只会知道'工具没执行，除非用户要求不要重试'。")


def demo_respond(model) -> None:
    ui.subtitle("4. respond —— 不执行工具，直接用人的回答结束本轮")
    agent = create_deep_agent(
        model=model,
        tools=[],
        interrupt_on={"write_file": True},
        checkpointer=InMemorySaver(),
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "发一封邮件给供应商"}]},
        config=THREAD,
    )
    done = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "respond", "message": "先别发，我需要你先确认价格条款。"}
                ]
            }
        ),
        config=THREAD,
    )
    for message in done["messages"]:
        if message.type == "tool":
            ui.kv(f"tool({message.name})", str(message.content)[:70])
    ui.assert_true(
        not (done.get("files") or {}),
        "respond 不会执行工具，因此没有产生文件",
    )
    ui.note("适用：工具调用本身就是错的（路由错了、参数选错了），人直接接管这一轮。")
    _ = say  # 保留导入，说明脚本里也可能有"被拒绝后模型改口"的轮次


def demo_allowed_decisions() -> None:
    ui.subtitle("5. InterruptOnConfig —— 限制审批人能做的决策")
    ui.note("形状（不实际运行，docs 口径）：")
    ui.note("    interrupt_on={")
    ui.note("        'write_file': {'allowed_decisions': ['approve', 'reject']},")
    ui.note("    }")
    ui.note("这样审批界面里就不给'改参数'和'直接代答'两个选项。")
    ui.note("生产建议：把 allowed_decisions 收窄到业务允许的集合，减少误操作面。")
    ui.note("注意：mode='interrupt' 的 FilesystemPermission（d09）也会自动生成 HITL，")
    ui.note("     两者都靠 checkpointer 生效。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d08 * HITL ---- approve / edit / reject / respond",
        feature="Human-in-the-loop",
        principle="#2 Extensible * #4 Production-ready",
    )

    # 每段都用独立的 checkpointer + thread，避免状态串味。
    demo_approve(
        choose_model(
            [
                tool_call(
                    "write_file",
                    {"file_path": "/report.md", "content": "# 报告\n"},
                    "c1",
                ),
                say("报告已生成。"),
            ],
            argv,
        )
    )
    demo_edit(
        choose_model(
            [
                tool_call("write_file", {"file_path": "/tmp/draft.md", "content": "草稿\n"}, "c1"),
                say("已按人工意见改写。"),
            ],
            argv,
        )
    )
    demo_reject(
        choose_model(
            [
                tool_call("delete", {"file_path": "/report.md"}, "c1"),
                say("明白，那我改成归档。"),
            ],
            argv,
        )
    )
    demo_respond(
        choose_model([say("收到，我先核对条款。")], argv)
    )
    demo_allowed_decisions()


if __name__ == "__main__":
    main()
