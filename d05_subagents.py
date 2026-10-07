"""d05 · Subagents —— 把"上下文隔离"做成一等公民

覆盖 Feature   : Subagents（declarative / compiled / fork / async）
覆盖 Principle : #1 Opinionated（默认带 general-purpose）· #2 Extensible

---------------------------------------------------------------------------
文档依据（concepts/subagents-skills.md）
---------------------------------------------------------------------------
三种声明方式：

1. `SubAgent`（声明式，dict）—— 框架替你 build 一个子 agent
2. `CompiledSubAgent`（预编译）—— 你自己 build，框架只负责调用
3. `AsyncSubAgent` —— 远端 Agent Protocol server（LangSmith Deployment / 自建）

两种上下文模式（`mode` 字段）：

* `isolated`（默认）—— 子 agent **只**看到被委派的那段任务描述，看不到父对话历史。
* `fork` —— 子 agent 接着父对话继续跑（实验特性）。

一句话总结 docs 的动机：subagent 的价值不是"多个人格"，而是**用一次干净的上下文
换掉一次昂贵的污染** —— 子 agent 的中间思考不会回灌父上下文。
"""

from __future__ import annotations

import sys
from typing import Any

from deepagents import CompiledSubAgent, FilesystemMiddleware, SubAgent, create_deep_agent
from langchain.agents import create_agent

from shared import ui
from shared.offline import (
    bound_tool_names,
    choose_model,
    choose_model_lenient,
    file_data,
    model_factory,
    say,
    tool_call,
)

#: 父 agent 的"工作记忆"：这些文件由父 agent 拥有，子 agent 只能按权限访问。
PARENT_FILES = {
    "/plan.md": "# 计划\n1. 收集数据\n2. 写报告\n",
    "/raw/data.csv": "id,value\n1,10\n2,20\n",
}


def researcher_spec(child_model: Any) -> SubAgent:
    """声明式 subagent 的**工厂**。

    为什么要工厂而不是一个模块级常量：`SubAgent["model"]` 可以直接给一个
    `BaseChatModel` 实例。父 agent 和子 agent 各用各的模型实例，父上下文里的
    脚本就不会被子 agent 的轮次消耗掉 —— 这也正是生产里的真实形态
    （给子 agent 换一个更便宜/更快的模型，见 principle #3）。
    """
    return {
        "name": "researcher",
        "description": (
            "调研一个主题并把结论写进 /research/findings.md。"
            "只做调研，不要修改父 agent 的计划文件。"
        ),
        "system_prompt": (
            "你是一个只做调研的子 agent。\n"
            "规则：\n"
            "1. 用 write_file 把结论落盘到 /research/findings.md，不要只回一句话。\n"
            "2. 不要碰 /plan.md（那是父 agent 的计划）。\n"
            "3. 最终回复要极简，因为父 agent 只会看到它。"
        ),
        "model": child_model,
        # 显式给 FilesystemMiddleware 就是**替换**子 agent 的默认文件系统中间件。
        "middleware": [FilesystemMiddleware(tools=["read_file", "write_file"])],
    }


# ===========================================================================
# 1. 声明式 subagent：isolated 上下文
# ===========================================================================
def demo_declarative_subagent(parent_model, child_model) -> None:
    ui.subtitle("1. 声明式 SubAgent（isolated）—— 只看得到被委派的任务")
    agent = create_deep_agent(
        model=parent_model,
        tools=[],
        subagents=[researcher_spec(child_model)],
    )
    result = agent.invoke(
        {
            "files": {p: file_data(body) for p, body in PARENT_FILES.items()},
            "messages": [
                {"role": "user", "content": "让 researcher 调研 Deep Agents 的文件系统，任务描述：调研并落盘。"}
            ],
        }
    )
    ui.show_messages(result)
    ui.show_files(result.get("files"))

    ui.step("父子上下文隔离的证据：")
    ui.kv("父 agent 可见工具", bound_tool_names(agent))
    ui.step("父 state 里的消息条数（子 agent 的中间过程不会回灌）：")
    for msg in result["messages"]:
        kind = type(msg).__name__
        if kind == "AIMessage" and getattr(msg, "tool_calls", None):
            ui.kv("  AIMessage", f"-> {[c['name'] for c in msg.tool_calls]}")
        elif kind == "ToolMessage":
            ui.kv("  ToolMessage", f"({msg.name}) {str(msg.content)[:56]}")
        else:
            ui.kv(f"  {kind}", str(getattr(msg, "content", ""))[:56].replace("\n", " "))
    ui.note("isolated 模式下子 agent 的 messages 不进父 state —— 父只拿到一条 task ToolMessage。")
    ui.note("这就是 subagent 的核心收益：用一次干净上下文换掉一次昂贵污染。")

    ui.step("子 agent 写了哪些文件（与父 agent 共享同一个 backend）：")
    for path, data in (result.get("files") or {}).items():
        if path.startswith("/research"):
            ui.kv(path, str(data.get("content", "")).replace("\n", " ")[:50])
    ui.note("注意：文件是**共享**的，消息历史是**隔离**的。这两件事经常被搞混。")


# ===========================================================================
# 2. fork 模式：接着父对话继续
# ===========================================================================
def demo_fork_subagent(parent_model, child_model) -> None:
    ui.subtitle("2. mode='fork' —— 子 agent 继承父对话历史（实验特性）")
    forked: SubAgent = {
        "name": "critic",
        "description": "审阅父 agent 已经在做的任务，利用完整上下文继续推进。",
        "system_prompt": "你是审阅者，直接指出父 agent 计划的漏洞并给出修正。",
        "mode": "fork",  # 继承父对话；注意 skills 在 fork 下被禁止
        "model": child_model,
    }
    agent = create_deep_agent(model=parent_model, tools=[], subagents=[forked])
    result = agent.invoke({"messages": [{"role": "user", "content": "先写个计划，然后让 critic 审阅。"}]})
    ui.show_messages(result)
    ui.kv("fork 下的工具集", bound_tool_names(agent))
    for line in [
        "isolated -> 子 agent 只收到 task 描述，父 history 不进它的 prompt。",
        "fork      -> 子 agent 继承父 history，system_prompt 是**追加**而不是替换。",
        "fork 下禁止 skills（会与继承的 prompt 语义冲突）。",
        "isolated 下需要显式传文件/结论给子 agent —— 落盘 + 传路径是最稳的做法。",
    ]:
        ui.note(line)


# ===========================================================================
# 3. 预编译 subagent：自己 build，框架只管调用
# ===========================================================================
def build_reviewer(argv: list[str] | None = None) -> Any:
    """用**普通** `create_agent` 编译一个 reviewer，不带 deepagents 的默认栈。

    适合：子 agent 需要跟父 agent 完全不同的 harness（不同 provider、
    不同 state schema、不同的中间件链）。

    注意 argv 要透传进来：否则从 `run_all.py`（`main(["--offline"])`）调用时
    它会去连真实 provider。
    """
    return create_agent(
        model=choose_model([say("审阅结论：计划缺少验收标准。")], argv),
        tools=[],
        system_prompt="你是一个严苛的审阅者，只回一句话。",
    )


def reviewer_spec(argv: list[str] | None = None) -> CompiledSubAgent:
    return {
        "name": "reviewer",
        "description": "审阅一段文字，只返回一句结论。",
        "runnable": build_reviewer(argv),
    }


def demo_compiled_subagent(parent_model, argv) -> None:
    ui.subtitle("3. CompiledSubAgent —— 自己 build，框架只负责调用与回传")
    agent = create_deep_agent(model=parent_model, tools=[], subagents=[reviewer_spec(argv)])
    result = agent.invoke({"messages": [{"role": "user", "content": "用 reviewer 审阅'先收集数据'"}]})
    ui.show_messages(result)
    for line in [
        "runnable 的 state schema 必须含 messages —— 框架靠它把结果回传给父 agent。",
        "回传规则：structured_response 非空 -> JSON 序列化；否则取最后一条非空 AIMessage。",
        "CompiledSubAgent **不继承** create_deep_agent(state_schema=...) 等配置。",
        "要结构化回传就配 response_format（见 d15 rubric）。",
    ]:
        ui.note(line)


# ===========================================================================
# 4. subagent 级 middleware / tools / model 覆盖
# ===========================================================================
def demo_subagent_overrides(parent_model) -> None:
    ui.subtitle("4. 每个 subagent 可以有自己的 tools / model / middleware")
    narrow: SubAgent = {
        "name": "reader",
        "description": "只能读文件的只读角色。",
        "tools": [],  # 显式给空 -> 不继承父 agent 的 tools
        "middleware": [FilesystemMiddleware(tools=["read_file", "ls"])],
    }
    agent = create_deep_agent(model=parent_model, tools=[], subagents=[narrow])
    ui.kv("父 agent 工具", bound_tool_names(agent))
    ui.note("SubAgent 省略 tools -> 继承父 agent 的工具；显式给 [] -> 一个都不给。")
    ui.note("SubAgent 省略 middleware -> 仍然会拿到**默认**中间件栈（在你自定义的之前）。")
    ui.note("model='openai:gpt-5.5' 这种字符串可以单独换 provider（principle #3）。")


# ===========================================================================
# 5. 默认就有的 general-purpose subagent
# ===========================================================================
def demo_default_subagent() -> None:
    ui.subtitle("5. general-purpose subagent：Opinionated 的体现（开箱即用）")
    agent = create_deep_agent(model=choose_model_lenient([], None), tools=[])
    ui.kv("默认工具集", bound_tool_names(agent))
    ui.assert_in("task", bound_tool_names(agent), "默认带 task 工具")
    ui.note("docs: create_deep_agent ships with a general-purpose subagent, so `task`")
    ui.note("is available even when you pass no subagents at all.")
    ui.note("GeneralPurposeSubagentProfile 可以改它的描述/工具/model —— 见 d14 profiles。")
    ui.assert_true(
        not any(name in bound_tool_names(agent) for name in ("researcher", "critic")),
        "没有声明 subagents 时不应出现自定义 subagent 名字",
    )


# ===========================================================================
# 6. AsyncSubAgent：远端 Agent Protocol
# ===========================================================================
def demo_async_subagent() -> None:
    ui.subtitle("6. AsyncSubAgent —— 远端 server 上的子 agent")
    ui.note("形状（不真正连接，docs 逐字）：")
    ui.note("    subagent: AsyncSubAgent = {")
    ui.note("        'name': 'researcher',")
    ui.note("        'description': 'Runs long research jobs remotely.',")
    ui.note("        'graph_id': 'researcher',   # 远端 graph 名 / assistant id")
    ui.note("        'url': 'http://localhost:8123',  # 省略则走进程内 ASGI 传输")
    ui.note("    }")
    ui.note("要点：")
    ui.note("  * 兼容 LangSmith Deployment（托管）与自建 Agent Protocol server。")
    ui.note("  * 鉴权由 SDK 自动处理（LANGGRAPH_API_KEY / LANGSMITH_API_KEY / LANGCHAIN_API_KEY）。")
    ui.note("  * 省略 url 的进程内 ASGI 传输**只能**从异步入口（ainvoke）进入，")
    ui.note("    同步 invoke 路径需要一个可达的 URL。")
    ui.note("  * 它们作为后台任务运行，主 agent 可以监控和更新。")


# ===========================================================================
# 7. langgraph.json 部署入口
# ===========================================================================
def build_graph():
    """`langgraph.json` 引用的生产入口（见 d16）。"""
    return create_deep_agent(
        model=choose_model_lenient([], None),
        tools=[],
        subagents=[researcher_spec(choose_model_lenient([], None))],
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d05 * Subagents ---- isolated vs fork, declarative vs compiled vs async",
        feature="Subagents",
        principle="#1 Opinionated * #2 Extensible * #3 Model-agnostic",
    )

    demo_declarative_subagent(
        model_factory(
            [
                tool_call(
                    "task",
                    {
                        "description": "调研 Deep Agents 的文件系统设计，落盘到 /research/findings.md",
                        "subagent_type": "researcher",
                    },
                    "c1",
                ),
                say("调研完成，报告已生成。"),
            ],
            argv,
        )(),
        # 子 agent 用自己的模型实例：脚本不共享，也可以在生产里换成更便宜的模型。
        model_factory(
            [
                tool_call(
                    "write_file",
                    {
                        "file_path": "/research/findings.md",
                        "content": "# 调研结论\n- 文件系统是最优的长程记忆\n- StateBackend 随 thread 存活\n",
                    },
                    "k1",
                ),
                say("结论已落盘。"),
            ],
            argv,
        )(),
    )

    demo_fork_subagent(
        model_factory(
            [
                tool_call(
                    "task",
                    {"description": "审阅当前计划", "subagent_type": "critic"},
                    "c1",
                ),
                say("已采纳审阅意见。"),
            ],
            argv,
        )(),
        model_factory([say("计划缺少验收标准，建议补充。")], argv)(),
    )

    demo_compiled_subagent(
        model_factory(
            [
                tool_call(
                    "task",
                    {"description": "审阅'先收集数据'", "subagent_type": "reviewer"},
                    "c1",
                ),
                say("审阅完成。"),
            ],
            argv,
        )(),
        argv,
    )

    demo_subagent_overrides(model_factory([say("确认配置。")], argv)())
    demo_default_subagent()
    demo_async_subagent()


if __name__ == "__main__":
    main()
