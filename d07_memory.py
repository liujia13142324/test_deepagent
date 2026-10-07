"""d07 · Memory —— 让 agent 跨会话记住东西

覆盖 Feature   : Memory（AGENTS.md 注入 + edit_file 写回）
覆盖 Principle : #2 Extensible · #3 Model-agnostic · #4 Production-ready

---------------------------------------------------------------------------
文档依据（concepts/subagents-skills.md 的 memory 章节 + MemoryMiddleware 源码）
---------------------------------------------------------------------------
`create_deep_agent(memory=["/AGENTS.md", "/docs/"])` 做的事：

1. `MemoryMiddleware` 在 system prompt 末尾插入 `<agent_memory>...</agent_memory>`。
2. 提示词里同时给出一整套**记忆准则**（何时记、怎么记、什么不能记）。
3. agent 用内置的 `edit_file` 直接改这些文件 —— 记忆就是普通文件，没有额外 API。

所以"记忆"这件事被刻意做成了 opinionated + 无魔法：
* 加载：框架在 prompt 注入；
* 写入：模型自己调 edit_file；
* 隔离：backend 决定文件存在哪（StateBackend=thread 内，StoreBackend=跨会话）。

与 d03 的 StoreBackend 组合起来就是完整的"跨会话长期记忆"。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

from shared import ui
from shared.offline import choose_model, model_factory, say, tool_call

ASSETS = Path(__file__).parent / "assets"
MEM_DIR = Path(tempfile.gettempdir()) / "deepagents_demo_memory"


def demo_memory_from_assets(model) -> None:
    ui.subtitle("1. memory=[路径...] —— 注入 <agent_memory>，并告诉模型怎么改")
    agent = create_deep_agent(
        model=model,
        tools=[],
        memory=[str(ASSETS / "AGENTS.md")],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "记住：我更喜欢表格输出。"}]})
    ui.show_messages(result)
    ui.kv("memory 源", str(ASSETS / "AGENTS.md"))
    ui.step("assets/AGENTS.md 内容（会被整段塞进 system prompt）：")
    for line in (ASSETS / "AGENTS.md").read_text(encoding="utf-8").splitlines()[:8]:
        ui.note("  " + line)

    for line in [
        "docs 的加载路径：files -> <agent_memory> 块 -> system prompt。",
        "默认 system_prompt 里还带一整套 memory_guidelines（何时写/不写记忆、",
        "  'never store API keys'、'prefer the user over stale memory' 等）。",
        "写入侧没有新 API：agent 用内置 edit_file 改这些文件，所以记忆可被 git 追踪。",
    ]:
        ui.note(line)


def demo_memory_round_trip(parent_model) -> None:
    ui.subtitle("2. 写回：agent 用 edit_file 落一条新记忆")
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    (MEM_DIR / "AGENTS.md").write_text("# 记忆\n\n- 偏好：中文回答\n", encoding="utf-8")
    ui.kv("写入前", (MEM_DIR / "AGENTS.md").read_text(encoding="utf-8").strip().replace("\n", " / "))

    agent = create_deep_agent(
        model=parent_model,
        tools=[],
        memory=[str(MEM_DIR / "AGENTS.md")],
        # 用真实磁盘后端，否则 edit_file 只会改 state 里的副本
        backend=FilesystemBackend(root_dir=MEM_DIR, virtual_mode=True),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "以后报告都用表格，这是长期偏好。"}]}
    )
    ui.show_messages(result)
    ui.kv("写入后", (MEM_DIR / "AGENTS.md").read_text(encoding="utf-8").strip().replace("\n", " / "))
    ui.assert_in("表格", (MEM_DIR / "AGENTS.md").read_text(encoding="utf-8"), "记忆已落盘")
    ui.note("同一个 backend 既承载 agent 产物，也承载记忆文件 —— 这是 docs 推荐形态：")
    ui.note("  agent 读得到自己写的东西，不需要另一套 API。")
    ui.note("Sources labeled 'Deepagents'（私有）vs 'Agents'（全机器共享）是 skills 侧的概念，")
    ui.note("memory= 不区分；需要共享就把同一个文件挂到多个 agent 上。")


def demo_cross_session_memory(parent_model_factory) -> None:
    ui.subtitle("3. 跨会话：StoreBackend + context 命名空间（Principle #4）")
    store = InMemoryStore()

    def make_agent():
        return create_deep_agent(
            model=parent_model_factory(),
            tools=[],
            memory=["/memories/prefs.md"],
            backend=StoreBackend(namespace=lambda rt: ("users", rt.context.user, "memories")),
            store=store,
            context_schema=_UserContext,
        )

    agent = make_agent()
    agent.invoke(
        {"messages": [{"role": "user", "content": "记住我的邮箱 a@b.com"}]},
        context=_UserContext(user="u1"),
    )
    ui.kv("store 条目数（写入后）", len(list(store.search(("users", "u1")))))

    # 全新 agent + 全新 thread，但同一个 store：记忆还在
    fresh = make_agent()
    ui.kv("新 agent 的记忆文件数", len(list(store.search(("users", "u1")))))
    ui.assert_true(
        bool(list(store.search(("users", "u1")))),
        "同一个 store + 同一个 user 命名空间 -> 记忆可跨 agent/跨 thread 复用",
    )
    for line in [
        "三层组合：memory=[路径] 负责'注入'，StoreBackend 负责'跨会话存活'，",
        "context_schema + namespace 负责'按用户隔离'。",
        "生产建议：长期记忆用 Store/Postgres，thread 内草稿用 StateBackend，",
        "  代码库用 FilesystemBackend，三者用 CompositeBackend 挂到不同前缀（d03 第 4 节）。",
    ]:
        ui.note(line)


class _UserContext:
    """极简 context（也可以用 dataclass；这里只演示 runtime.context 的用法）。"""

    def __init__(self, user: str) -> None:
        self.user = user


def demo_custom_memory_prompt() -> None:
    ui.subtitle("4. 自定义记忆准则（MemoryMiddleware 可替换）")
    from deepagents.middleware import MemoryMiddleware

    agent = create_deep_agent(
        model=choose_model([say("ok")], None),
        tools=[],
        backend=FilesystemBackend(root_dir=MEM_DIR, virtual_mode=True),
        middleware=[
            MemoryMiddleware(
                backend=FilesystemBackend(root_dir=MEM_DIR, virtual_mode=True),
                sources=[str(MEM_DIR / "AGENTS.md")],
                system_prompt="<记忆>\n{agent_memory}\n</记忆>\n只记业务规则，不记闲聊。",
            )
        ],
    )
    ui.kv("工具集（记忆写入仍走 edit_file）", [t for t in _tools(agent) if "file" in t])
    ui.note("放进 middleware= 会**替换** create_deep_agent 自动加的 MemoryMiddleware，")
    ui.note("所以自定义 system_prompt 时要自己保证 {agent_memory} 占位符还在。")
    ui.note("add_cache_control=True（自动注入时）只对 Anthropic 生效，可安全常开。")


def _tools(agent) -> list[str]:
    from shared.offline import bound_tool_names

    return bound_tool_names(agent)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d07 * Memory ---- AGENTS.md injection + edit_file write-back",
        feature="Memory",
        principle="#2 Extensible * #4 Production-ready",
    )

    demo_memory_from_assets(choose_model([say("已了解你的偏好。")], argv))
    demo_memory_round_trip(
        model_factory(
            [
                tool_call(
                    "edit_file",
                    {
                        "file_path": "/AGENTS.md",
                        "old_string": "- 偏好：中文回答",
                        "new_string": "- 偏好：中文回答\n- 偏好：报告用表格",
                    },
                    "c1",
                ),
                say("已记住：报告用表格。"),
            ],
            argv,
        )()
    )
    demo_cross_session_memory(
        model_factory(
            [
                tool_call(
                    "write_file",
                    {"file_path": "/memories/prefs.md", "content": "邮箱：a@b.com\n"},
                    "c1",
                ),
                say("已保存邮箱。"),
            ],
            argv,
        )
    )
    demo_custom_memory_prompt()


if __name__ == "__main__":
    main()
