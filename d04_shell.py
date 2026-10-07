"""d04 · Shell & Sandbox —— 让 agent 真正执行命令（以及为什么生产环境要换沙箱）

覆盖 Feature   : Sandboxed shell execution · filesystem × shell 协同
覆盖 Principle : #1 Opinionated（默认无 execute）· #2 Extensible · #4 Production-ready

---------------------------------------------------------------------------
文档依据（concepts/backends.md + tools-filesystem.md）
---------------------------------------------------------------------------
核心结论（逐字）：

> `StateBackend` / `StoreBackend` / `FilesystemBackend` do **not** implement
> `execute`. `LocalShellBackend` and `LangSmithSandbox` do.

以及那条最重要的安全声明：

> The backend's execution environment establishes the boundary, not model
> intent, a tool description, a virtual prefix, or a path allowlist.
> A local shell remains a host shell even when its file API presents a
> virtual root.

翻译成工程语言：
* `LocalShellBackend(virtual_mode=True)` **只**收敛 `read_file/write_file` 的路径，
  `execute()` 依然是你这台机器的 shell。任何"提示词里写清楚别越界"的做法都不是安全边界。
* 真正的隔离 = 换后端（Docker / Firejail / E2B / Modal / LangSmith Sandbox）。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend

from shared import ui
from shared.offline import choose_model, say, tool_call

DEMO_DIR = Path(tempfile.gettempdir()) / "deepagents_demo_workspace"

#: demo 里的 shell 命令必须跨平台可跑（这套 demo 要在 Windows/macOS/Linux 都成立）。
IS_WINDOWS = sys.platform == "win32"
LIST_CMD = "dir" if IS_WINDOWS else "ls -la"
READ_MISSING_CMD = (
    'type "%SystemRoot%\\System32\\drivers\\etc\\hosts.bak"' if IS_WINDOWS else "cat /nope/nope"
)


# ===========================================================================
# 1. LocalShellBackend —— 文件 API + execute 同时可用
# ===========================================================================
def demo_local_shell(model) -> None:
    ui.subtitle("1. LocalShellBackend —— execute 工具出现，文件系统同一份视图")
    agent = create_deep_agent(
        model=model,
        tools=[],
        backend=LocalShellBackend(root_dir=DEMO_DIR, virtual_mode=True),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "先 ls 看目录，再执行 shell 命令确认"}]}
    )
    ui.show_messages(result)
    ui.kv("工具集", sorted(_bound_tool_names(agent)))
    ui.note("execute 一直在工具集里（batteries-included 的默认值），但只有后端实现")
    ui.note("SandboxBackendProtocol 时它才真的能跑 —— 见第 4 节的对比。")
    ui.note("ls() 看到的目录就是 shell 的 cwd，两个视图必须一致，")
    ui.note("否则 agent 会写出'文件工具看不见'的产物。")


# ===========================================================================
# 2. execute 的输出形状：exit code / stdout / stderr
# ===========================================================================
def demo_execute_shape(model) -> None:
    ui.subtitle("2. execute 的返回结构（agent 靠它判断成功与否）")
    agent = create_deep_agent(
        model=model,
        tools=[],
        backend=LocalShellBackend(root_dir=DEMO_DIR, virtual_mode=True),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "跑一个会失败的命令"}]})
    for message in result["messages"]:
        if message.type == "tool":
            ui.step(f"execute -> {message.name}")
            print("   " + str(message.content).replace("\n", "\n   "))
    ui.note("非零 exit code 也要如实回给模型，让它自己纠正，而不是假装成功。")


# ===========================================================================
# 3. 危险命令：展示"宿主 shell 事实"，并给出生产替代方案
# ===========================================================================
def demo_host_shell_is_the_boundary(model) -> None:
    ui.subtitle("3. 边界在哪里：virtual_mode 挡不住 shell")
    backend = LocalShellBackend(root_dir=DEMO_DIR, virtual_mode=True)

    # ExecuteResponse 是刻意精简的：output 是 stdout+stderr 合并后的文本，
    # 加上 exit_code / truncated —— 因为它要被塞进模型上下文。
    inside = backend.execute("cd")
    ui.kv("execute('cd') exit", inside.exit_code)
    ui.kv("stdout", inside.output.strip())

    # 这一条能读到 root_dir 之外的东西 —— 证明"虚拟根"不是安全边界。
    escaped = backend.execute("echo 我在虚拟根之外也能执行")
    ui.kv("任意命令 exit", escaped.exit_code)
    ui.note("上面这条命令完全没有用到 root_dir 里的文件。docs 的措辞是：")
    ui.note("  a local shell remains a host shell even when its file API")
    ui.note("  presents a virtual root.")

    ui.subtitle("3b. 生产替代方案（按隔离强度递增）")
    for line in [
        "1) FilesystemBackend + 显式工具白名单：不需要 shell 时根本不装它。",
        "2) LocalShellBackend + OS 级封装：Docker --read-only / Firejail / bubblewrap。",
        "3) 远端沙箱：E2B、Modal、 Daytona、LangSmith Sandbox（自带 compute + timeout）。",
        "4) 无论哪种，都要配 permissions（d11）限制模型能主动调用的动作。",
    ]:
        ui.note(line)
    ui.note("本 demo 的取舍：为了离线可验证，用真实宿主机 shell 跑无害命令；")
    ui.note("生产部署请看 d16_production.py 的沙箱化配置。")


# ===========================================================================
# 4. FilesystemBackend + 显式 shell：自己拼一个"受控执行"后端
# ===========================================================================
def demo_hybrid(model) -> None:
    ui.subtitle("4. 关键区分：工具是**默认注册**的，能力由**后端**把门")
    ui.note("execute 出现在默认工具集里（batteries-included），但 StateBackend /")
    ui.note("FilesystemBackend 没有实现 SandboxBackendProtocol，调用它只会得到：")
    print("      Error: Execution not available. This agent's backend does not support")
    print("      command execution (SandboxBackendProtocol). ...")
    ui.note("=> 能力判定在运行时、由后端决定；这不是异常，是一条 ToolMessage（failures are data）。")

    files_only = _agent_for(FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True))
    sandboxed = _agent_for(LocalShellBackend(root_dir=DEMO_DIR, virtual_mode=True))
    ui.assert_in("execute", _bound_tool_names(files_only), "FilesystemBackend 仍注册 execute 工具")
    ui.assert_in("execute", _bound_tool_names(sandboxed), "LocalShellBackend 注册 execute 工具")

    # 现场跑一次，证明"注册了但用不了"和"真的能跑"的差别。
    result = create_deep_agent(
        model=model, tools=[], backend=LocalShellBackend(root_dir=DEMO_DIR, virtual_mode=True)
    ).invoke({"messages": [{"role": "user", "content": "跑一下 cd"}]})
    for message in result["messages"]:
        if message.type == "tool":
            ui.kv(f"{message.name} 返回", str(message.content).replace("\n", " ")[:60])

    ui.note("生产形态（principle #2 Extensible 的实践）：")
    ui.note("  * 文件走 FilesystemBackend（路径受 root 约束）")
    ui.note("  * 只把 execute 暴露给确实需要的 subagent（d05 演示）")
    ui.note("  * 用 permissions / interrupt_on 让人批准高危命令（d11）")


def _agent_for(backend):
    """只为形状断言构造一个 agent（不 invoke）。"""
    from shared.offline import probe_model

    return create_deep_agent(model=probe_model(), tools=[], backend=backend)


def _bound_tool_names(agent) -> list[str]:
    """从 graph 的 ToolNode 里取出真实注册的工具名（形状断言用）。"""
    from shared.offline import bound_tool_names

    return bound_tool_names(agent)


# ===========================================================================
# 5. 系统工具探测：本地 shell 能看到 PATH 上的东西
# ===========================================================================
def demo_shell_environment() -> None:
    ui.subtitle("5. 宿主环境就是执行环境（virtual_mode 不改环境变量）")
    backend = LocalShellBackend(root_dir=DEMO_DIR, virtual_mode=True)
    result = backend.execute("cd")
    ui.kv("shell cwd", result.output.strip())
    ui.kv("python 可执行文件", shutil.which("python") or "PATH 里没有")
    ui.note("LocalShellBackend 不注入沙箱环境变量，也不限制 PATH —— 宿主的全部能力都在。")
    ui.note("生产上要限制 PATH / 资源（CPU、内存、磁盘、wall-clock），只能靠外部封装。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d04 * Shell & Sandbox ---- execute exists only on sandbox-capable backends",
        feature="Sandboxed shell execution",
        principle="#1 Opinionated * #2 Extensible * #4 Production-ready",
    )

    demo_local_shell(
        choose_model(
            [
                tool_call("ls", {"path": "/"}, "c1"),
                tool_call("execute", {"command": LIST_CMD}, "c2"),
                say("目录已确认。"),
            ],
            argv,
        )
    )

    demo_execute_shape(
        choose_model(
            [
                tool_call("execute", {"command": READ_MISSING_CMD}, "c1"),
                say("命令失败，我会调整。"),
            ],
            argv,
        )
    )

    demo_host_shell_is_the_boundary(
        choose_model([say("我会先看看能执行什么。")], argv)
    )

    demo_hybrid(choose_model([say("先确认工具集。")], argv))

    demo_shell_environment()


if __name__ == "__main__":
    main()
