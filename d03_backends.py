"""d03 · Backends：文件存哪、活多久、能不能执行命令，全由后端决定

覆盖 Feature   : Filesystem（pluggable local / sandboxed / remote backends）
覆盖 Principle : #2 Extensible · #4 Production-ready

---------------------------------------------------------------------------
文档依据（concepts/backends.md）
---------------------------------------------------------------------------
> A backend is the implementation boundary behind agent file operations.
> It selects where files live, how long they persist, how virtual paths route,
> and whether a shell exists. It is not a model-visible tool and not an
> authorization policy.

官方 storage / capability 矩阵（逐字）：

| Backend            | 存储位置                       | Shell 能力 |
|--------------------|--------------------------------|------------|
| StateBackend       | LangGraph `files` state, 同一 thread | 无   |
| StoreBackend       | 带 namespace 的 BaseStore, 跨 thread  | 无   |
| FilesystemBackend  | 本地磁盘（受 root 约束）        | 无         |
| LocalShellBackend  | 本地磁盘 + 宿主进程环境          | 有（无限制）|
| LangSmithSandbox   | 隔离的 LangSmith 沙箱            | 有         |
| ContextHubBackend  | 远端 LangSmith Hub 仓库（有 commit）| 无      |
| CompositeBackend   | default + 路由挂载              | 仅当 default 有 |

最重要的一条安全规则（逐字）：

> The backend's execution environment establishes the boundary, not model
> intent, a tool description, a virtual prefix, or a path allowlist.
> A local shell remains a host shell even when its file API presents a
> virtual root.

也就是说 `LocalShellBackend(virtual_mode=True)` 只是**文件操作**看起来被限制在
一个虚拟根下，`execute()` 依然是**宿主机 shell**。
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    ContextHubBackend,
    FilesystemBackend,
    LocalShellBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.backends.protocol import (
    BackendProtocol,
    FileInfo,
    LsResult,
    ReadResult,
    WriteResult,
)
from langgraph.store.memory import InMemoryStore

from shared import ui
from shared.offline import choose_model, file_data, model_factory, probe_model, say, tool_call

DEMO_DIR = Path(tempfile.gettempdir()) / "deepagents_demo_workspace"


# ===========================================================================
# 1. StateBackend —— 默认。文件在 graph state 里，零外部依赖
# ===========================================================================
def demo_state_backend(model) -> None:
    ui.subtitle("1. StateBackend（默认）—— 文件 = graph state，单 thread 内有效")
    agent = create_deep_agent(model=model, tools=[])
    result = agent.invoke({"messages": [{"role": "user", "content": "写个 README"}]})
    ui.kv("state channels", sorted(agent.stream_channels))
    ui.show_files(result.get("files"))
    ui.note("docs: checkpointed within a conversation thread, not across threads.")
    ui.note("在 graph 之外直接调用 StateBackend 会抛 RuntimeError。")


# ===========================================================================
# 2. StoreBackend —— 跨 thread / 跨会话的命名空间持久化
# ===========================================================================
def demo_store_backend(model_factory) -> None:
    """`model_factory` 而不是 `model`：每个 agent 用独立的模型实例。

    离线回放模式下模型是有状态的（脚本游标），复用实例会耗尽脚本；
    真实模型则是无状态的。这里统一成 factory，避免这种隐式耦合。
    """
    ui.subtitle("2. StoreBackend —— 跨会话记忆（Principle: Persistent memory）")
    store = InMemoryStore()

    @dataclass
    class UserContext:
        """`context_schema` 定义的是**不可变的 run 级上下文**。

        docs: 命名空间依赖 runtime 时，正确的做法是注入 context，
        而不是去翻 RunnableConfig（`Runtime` 上并没有 `.config` 属性）。
        """

        user_id: str
        scope: str = "private"

    def namespace(runtime) -> tuple[str, ...]:
        """docs: namespace 组件会做安全字符集校验，避免 glob 通配扩大查询范围。"""
        ctx = runtime.context
        return ("users", ctx.user_id, "memories", ctx.scope)

    def build(model):
        return create_deep_agent(
            model=model,
            tools=[],
            backend=StoreBackend(namespace=namespace),
            store=store,  # docs: StoreBackend 必须配一个 store
            context_schema=UserContext,
        )

    # 注意：设了 context_schema 之后，invoke(context=...) 收的是 context 对象本身，
    # 不是 configurable 字典（LangGraph 会用 context_schema(**context) 构造）。
    result = build(model_factory()).invoke(
        {"messages": [{"role": "user", "content": "记住：我的花名叫小明"}]},
        context=UserContext(user_id="u1", scope="shared"),
    )
    ui.show_messages(result)

    # 换一个 scope：说明 namespace 来自 context，因此是逻辑隔离的
    other = build(model_factory()).invoke(
        {"messages": [{"role": "user", "content": "我叫什么"}]},
        context=UserContext(user_id="u1", scope="private"),
    )
    ui.step("换一个 scope（private）读同一个 store —— 逻辑隔离生效：")
    ui.show_messages(other)

    ui.step("store 里的实际条目（跨会话持久化的证据）：")
    for item in store.search(("users", "u1")):
        ui.kv("/".join(str(p) for p in item.namespace), str(item.value)[:60])


# ===========================================================================
# 3. FilesystemBackend —— 真实本地磁盘，虚拟路径受 root 约束
# ===========================================================================
def demo_filesystem_backend(model) -> None:
    ui.subtitle("3. FilesystemBackend —— 真实本地磁盘（virtual_mode 路径护栏）")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    (DEMO_DIR / "preloaded.txt").write_text("这个文件是磁盘上真实存在的。\n", encoding="utf-8")

    agent = create_deep_agent(
        model=model,
        tools=[],
        backend=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "看看有哪些文件并写个索引"}]})
    ui.show_messages(result)
    ui.kv("root_dir", str(DEMO_DIR))
    ui.note("virtual_mode=True: 虚拟路径 /x 映射到 root_dir/x，阻断 .. 逃逸。")
    for path in sorted(DEMO_DIR.glob("*")):
        ui.kv("disk", path.name)

    ui.subtitle("3b. 路径解析与失败语义（docs: blocks traversal, verifies containment）")
    backend = FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True)
    listing = backend.ls("/")
    ui.kv("ls('/')", [e["path"] for e in (listing.entries or [])][:3])

    # 1) 缺文件：返回结构化结果（error 字段），**不抛异常** —— docs: failures are data.
    ui.kv("read 不存在的文件", backend.read("/nope.txt").error)

    # 2) 路径穿越：`_resolve_path` 直接抛 ValueError（这是边界校验，不是业务错误）
    try:
        backend.read("/../../Windows/win.ini")
    except ValueError as exc:
        ui.kv("路径穿越 -> ValueError", str(exc))

    # 3) virtual_mode=False：路径原样交给操作系统，root_dir 不再是护栏
    passthrough = FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=False)
    real = passthrough.read(str(DEMO_DIR / "preloaded.txt"))
    ui.kv("virtual_mode=False", (real.file_data or {}).get("content", "").strip()[:20])
    ui.note("virtual_mode=True : /x -> root_dir/x，阻断 .. 逃逸（ValueError）。")
    ui.note("virtual_mode=False: 路径原样透传给 OS —— /preloaded.txt 会变成 C:\\preloaded.txt。")
    ui.note("护栏是**文件系统边界**，不是授权策略：真正的策略要用 permissions（d11）。")


# ===========================================================================
# 4. CompositeBackend —— 多个作用域共存，按最长前缀路由
# ===========================================================================
def demo_composite_backend(model) -> None:
    ui.subtitle("4. CompositeBackend —— 多个存储作用域共存")
    store = InMemoryStore()

    @dataclass
    class UserContext:
        user_id: str = "u1"

    agent = create_deep_agent(
        model=model,
        tools=[],
        backend=CompositeBackend(
            default=StateBackend(),  # 默认：thread 内临时文件
            routes={
                "/memories/": StoreBackend(  # 路由：跨会话长期记忆
                    namespace=lambda rt: ("users", rt.context.user_id, "memories")
                ),
                "/repo/": FilesystemBackend(  # 路由：真实代码库
                    root_dir=DEMO_DIR, virtual_mode=True
                ),
            },
            artifacts_root="/artifacts",  # 卸载/归档产物的落盘根
        ),
        store=store,
        context_schema=UserContext,
    )
    # 显式传 context：设了 context_schema 之后不传的话 runtime.context 是 None。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "存档 + 写索引"}]},
        context=UserContext(),
    )
    ui.show_messages(result)
    ui.show_files(result.get("files"))

    ui.subtitle("4b. 路由规则（docs 逐条）")
    for line in [
        "前缀按**最长优先**排序；匹配后剥掉前缀再委派（route root 变成 /）。",
        "未匹配 -> 走 default；返回结果再把前缀加回去。",
        "ls('/') = default 的根 + 合成出来的 route 目录。",
        "execute **不可路由**：只委派给 default；default 不是沙箱就抛 NotImplementedError。",
        "  => 把 StoreBackend 挂在 /memories/ 并不会让 /memories 变成 shell 可见。",
    ]:
        ui.note(line)

    ui.subtitle("4c. execute 的不可路由性（现场验证）")
    cb = CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=lambda rt: ("x",))},
    )
    try:
        cb.execute("ls /memories")
    except NotImplementedError as exc:
        ui.kv("NotImplementedError", str(exc)[:80])
    ui.note("正是 docs 说的：pairing a sandbox default with a StoreBackend at")
    ui.note("/memories/ does not make /memories/ shell-visible.")


# ===========================================================================
# 5. 远端 / 沙箱后端 —— 无凭据时优雅跳过
# ===========================================================================
def demo_remote_backends() -> None:
    ui.subtitle("5. ContextHubBackend / LangSmithSandbox（需要 LangSmith 凭据）")
    hub = ContextHubBackend("acme/research-agent")
    ui.kv("ContextHubBackend", type(hub).__name__)
    ui.note("构造是惰性的：真正读写时才需要 LANGSMITH_API_KEY，失败以结构化 error 返回。")
    ui.note("带 commit 的远端仓库语义：一次会话里的写入形成 commit，可 diff / revert。")

    ui.note("LangSmithSandbox 需要先创建远端沙箱对象：")
    ui.note("    from deepagents.backends import LangSmithSandbox")
    ui.note("    backend = LangSmithSandbox(sandbox=<LangSmith sandbox object>)")
    ui.note("它同时实现 SandboxBackendProtocol，所以带 execute —— 隔离来自沙箱本身，")
    ui.note("而不是来自文件工具的路径校验（docs 原文口径）。")
    ui.note("第三方 partner 后端同理，见 d04_shell.py。")


# ===========================================================================
# 6. 自定义后端 —— principle #2 Extensible 的最终形态
# ===========================================================================
class InMemoryDictBackend(BackendProtocol):
    """最小可用的自定义后端：只实现 read / write / ls。

    docs: "uniform, deliberately partial interface" —— 未实现的方法默认抛
    NotImplementedError，所以你可以只实现自己需要的那几个。

    签名以 `deepagents/backends/protocol.py` 为准（已逐一核对）：

    * `ls(path) -> LsResult`，`LsResult.entries` 是 `list[FileInfo]`（TypedDict）
    * `read(file_path, offset, limit) -> ReadResult`，正文在 `ReadResult.file_data`
      这个 `FileData` TypedDict 里，分页字段由 `__post_init__` 校验
    * `write(file_path, content) -> WriteResult(path=...)`
    """

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def write(self, file_path: str, content: str) -> WriteResult:
        self.files[file_path] = content
        return WriteResult(path=file_path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if file_path not in self.files:
            return ReadResult(error="file_not_found")
        lines = self.files[file_path].splitlines()
        start = max(offset, 0)
        window = lines[start : start + max(limit, 0)]
        # 注意 ReadResult.__post_init__ 的不变量：start_line / end_line / next_offset
        # 必须"要么都给、要么都不给"，否则构造时就抛 ValueError。
        if not window:
            return ReadResult(file_data=file_data(""), total_lines=len(lines))
        return ReadResult(
            file_data=file_data("\n".join(window)),
            total_lines=len(lines),
            start_line=start + 1,
            end_line=start + len(window),
            next_offset=start + len(window),
        )

    def ls(self, path: str) -> LsResult:
        prefix = path if path.endswith("/") else path + "/"
        return LsResult(
            entries=[
                FileInfo(path=p, is_dir=False, size=len(c))
                for p, c in sorted(self.files.items())
                if p.startswith(prefix)
            ],
        )


def demo_custom_backend(model) -> None:
    ui.subtitle("6. 自定义后端：实现 BackendProtocol 即可接入 harness")
    backend = InMemoryDictBackend()
    agent = create_deep_agent(model=model, tools=[], backend=backend)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "写文件、列目录、读回内容"}]}
    )
    ui.show_messages(result)
    ui.kv("自定义后端里的文件", list(backend.files))
    ui.step("读回内容：")
    print("   " + (backend.read("/memo.md").file_data or {})["content"].rstrip())
    ui.note("只实现 read/write/ls 就能跑；grep/glob/execute 保持默认 NotImplementedError，")
    ui.note("middleware 会把不支持的工具处理成友好错误，而不是让整个 graph 崩溃。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d03 * Backends ---- local / state / store / composite / custom",
        feature="Filesystem (pluggable backends)",
        principle="#2 Extensible * #4 Production-ready",
    )

    demo_state_backend(
        choose_model(
            [
                tool_call("write_file", {"file_path": "/README.md", "content": "# hi\n"}, "c1"),
                say("已写入。"),
            ],
            argv,
        )
    )

    demo_store_backend(
        model_factory(
            [
                tool_call("write_file", {"file_path": "/profile.md", "content": "花名：小明\n"}, "c1"),
                say("记住了。"),
            ],
            argv,
        )
    )

    demo_filesystem_backend(
        choose_model(
            [
                tool_call("ls", {"path": "/"}, "c1"),
                tool_call("write_file", {"file_path": "/index.md", "content": "# 索引\n- preloaded.txt\n"}, "c2"),
                say("索引已生成。"),
            ],
            argv,
        )
    )

    demo_composite_backend(
        choose_model(
            [
                tool_call("write_file", {"file_path": "/memories/花名.md", "content": "小明\n"}, "c1"),
                tool_call("write_file", {"file_path": "/tmp/scratch.md", "content": "临时草稿\n"}, "c2"),
                say("已分别写入长期记忆与临时文件。"),
            ],
            argv,
        )
    )

    demo_remote_backends()
    demo_custom_backend(
        choose_model(
            [
                tool_call("write_file", {"file_path": "/memo.md", "content": "自定义后端 OK\n"}, "c1"),
                tool_call("ls", {"path": "/"}, "c2"),
                tool_call("read_file", {"file_path": "/memo.md"}, "c3"),
                say("读写正常。"),
            ],
            argv,
        )
    )
    _ = probe_model  # 保持导入可用（形状断言示例）


if __name__ == "__main__":
    main()
