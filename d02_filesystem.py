"""d02 · Filesystem：read / write / edit / search，默认落在 graph state

覆盖 Feature   : Filesystem（read, write, edit, search）
覆盖 Principle : #1 Opinionated · #2 Extensible

---------------------------------------------------------------------------
文档依据（concepts/tools-filesystem.md + concepts/backends.md）
---------------------------------------------------------------------------
`FilesystemMiddleware` 按 allowlist 暴露 8 个工具：

    ls  read_file  write_file  edit_file  delete  glob  grep  execute

两条硬约束（本 demo 会现场验证）：

1. **allowlist 必须包含 `read_file`**，否则 middleware 在构造时直接拒绝。
   原因见 context-management 的不变量 #1：
   "every advertised path must be readable through the configured backend's
   read_file tool" —— 任何"已落盘/已卸载"的指针都必须能被读回来。
2. `execute` / `delete` 还要过后端能力检查：
   - `execute` 只有后端实现 `SandboxBackendProtocol` 才会真正可用；
   - `delete` 是可选能力，用 `_supports_delete` 探测而不是直接调。

另外几个容易踩的语义（docs 明确写了）：

* **edit 要求 old_string 唯一**，除非 `replace_all=True`。
* **grep 是字面量匹配，不是正则**。要真正的正则请用 `execute` 跑 `rg`。
* `glob` 是 root 相对：无 `/` 的 pattern 匹配任意深度下的 basename；
  以 `/` 开头则锚定到 root；`.` 开头的隐藏文件需要 dot-prefixed pattern。
"""

from __future__ import annotations

import sys

from deepagents import FilesystemMiddleware, create_deep_agent

from shared import ui
from shared.offline import choose_model, probe_model, say, seed_files, tool_call

SEED_FILES = seed_files(
    {
        "/notes/ideas.md": "用文件系统当长期工作记忆\n中间产物不要都塞进 context\n",
        "/notes/todo.md": "- [ ] 补测试\n- [x] 写文档\n",
        "/data/orders.csv": "id,amount\n1,120\n2,80\n3,240\n",
    }
)


def _tool_names(agent) -> list[str]:
    """从编译后的 LangGraph 里取出模型实际可见的工具名。"""
    node = agent.nodes.get("tools")
    bound = getattr(node, "bound", None)
    by_name = getattr(bound, "_tools_by_name", None)
    return sorted(by_name) if by_name else []


def demo_default_state_backend(model) -> None:
    """A. 默认 StateBackend：文件 = graph state 的一个字段（零外部依赖）。"""
    ui.subtitle("A. 默认 StateBackend：文件就是 graph state 的一个字段")
    agent = create_deep_agent(model=model, tools=[])

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "整理一下 /notes 下的文件并写一份汇总"}],
            # StateBackend 的初始文件通过 graph 输入注入
            # （docs: "Seed state-backed files in graph input"）——
            # 因为在 graph 之外直接调用 StateBackend 会抛 RuntimeError。
            "files": SEED_FILES,
        }
    )
    ui.show_messages(result)
    ui.step("state['files'] 里的全部文件：")
    ui.show_files(result.get("files"), limit=10)


def demo_search(model) -> None:
    """B. glob / grep / ls：三种搜索语义。"""
    ui.subtitle("B. ls / glob / grep：搜索语义")
    agent = create_deep_agent(model=model, tools=[])
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "列出根目录，glob 所有 md，再 grep 关键词"}],
            "files": SEED_FILES,
        }
    )
    ui.show_messages(result, limit=12)


def demo_write_and_edit(model) -> None:
    """C. write_file / edit_file：写入与"唯一匹配"编辑。"""
    ui.subtitle("C. write_file / edit_file")
    agent = create_deep_agent(model=model, tools=[])
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "更新待办并生成汇总"}],
            "files": SEED_FILES,
        }
    )
    ui.show_messages(result, limit=12)
    todo = (result.get("files") or {}).get("/notes/todo.md", {})
    body = todo.get("content") if isinstance(todo, dict) else ""
    ui.step("edit 之后的 /notes/todo.md：")
    print("   " + str(body).replace("\n", "\n   "))


def demo_constraints() -> None:
    """D/E/F：不依赖任何模型，纯 API 层断言 —— Extensible 的证据。"""
    ui.subtitle("D. 约束 1：FilesystemMiddleware 的 allowlist 必须包含 read_file")
    try:
        FilesystemMiddleware(tools=["ls", "write_file"])
    except ValueError as exc:
        ui.kv("ValueError", str(exc)[:120])
        ui.note("docs: FilesystemMiddleware rejects a tool allowlist without read_file")
    else:  # pragma: no cover
        ui.note("未抛错（若上游放宽了这条约束，这里会变化）")

    ui.subtitle("E. 约束 2：默认 StateBackend 不提供真正的 shell 能力")
    agent = create_deep_agent(model=probe_model(), tools=[])
    ui.kv("暴露的工具集", _tool_names(agent))
    ui.note("execute 工具名会被暴露，但 StateBackend 未实现 SandboxBackendProtocol，")
    ui.note("调用它只会得到一句错误提示。真正的 shell 能力见 d04_shell.py。")

    ui.subtitle("F. 替换内置 FilesystemMiddleware = principle #2 Extensible")
    agent2 = create_deep_agent(
        model=probe_model(),
        tools=[],
        middleware=[FilesystemMiddleware(tools=["read_file", "write_file"])],
    )
    ui.kv("自定义 allowlist 工具集", _tool_names(agent2))
    ui.note("放进 middleware= 的 FilesystemMiddleware 会**替换**默认实例（不是叠加），")
    ui.note("所以收窄能力必须走这里，而 tools= 是**叠加**语义、永远删不掉内置工具。")

    ui.subtitle("G. 工具级收窄 vs 请求级隐藏（容易搞混的两件事）")
    ui.note("tools=[...]            -> 构建期移除，模型根本看不到这个工具")
    ui.note("HarnessProfile.excluded_tools -> 请求期过滤；被调用时返回 unavailable 错误，")
    ui.note("                           文档明确写了 'this is consistency behavior,")
    ui.note("                           not a security boundary'。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d02 * Filesystem ---- read / write / edit / search",
        feature="Filesystem",
        principle="#1 Opinionated * #2 Extensible",
    )

    demo_default_state_backend(
        choose_model(
            [
                tool_call("ls", {"path": "/notes"}, "c1"),
                tool_call("read_file", {"file_path": "/notes/ideas.md"}, "c2"),
                say("已汇总：文件系统是最佳的长程工作记忆载体。"),
            ],
            argv,
        )
    )

    demo_search(
        choose_model(
            [
                tool_call("ls", {"path": "/"}, "c1"),
                tool_call("glob", {"pattern": "*.md"}, "c2"),
                tool_call("grep", {"pattern": "测试"}, "c3"),
                say("共 2 个 md 文件，1 处包含该关键词。"),
            ],
            argv,
        )
    )

    demo_write_and_edit(
        choose_model(
            [
                tool_call(
                    "write_file",
                    {
                        "file_path": "/notes/summary.md",
                        "content": "# 汇总\n- 用 FS 做记忆\n- 产物落盘\n",
                    },
                    "c1",
                ),
                tool_call(
                    "edit_file",
                    {
                        "file_path": "/notes/todo.md",
                        "old_string": "- [ ] 补测试",
                        "new_string": "- [x] 补测试",
                    },
                    "c2",
                ),
                say("已更新待办并生成汇总。"),
            ],
            argv,
        )
    )

    demo_constraints()


if __name__ == "__main__":
    main()
