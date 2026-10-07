"""d09 · Permissions —— 用声明式规则约束文件访问

覆盖 Feature   : Filesystem permissions（allow / deny / interrupt）
覆盖 Principle : #2 Extensible · #4 Production-ready

---------------------------------------------------------------------------
文档依据（concepts/permissions-hitl.md + FilesystemPermission 源码）
---------------------------------------------------------------------------
`create_deep_agent(permissions=[FilesystemPermission(...)])`：

```python
FilesystemPermission(
    operations=["read", "write"],   # 只有 'read' / 'write' 两个操作维度
    paths=["/project/**", "/secrets/**"],
    mode="allow" | "deny" | "interrupt",
)
```

规则：
* 按**声明顺序**求值，**第一个命中的规则获胜**（所以 deny 要写在 allow 前面）。
* 路径必须以 `/` 开头，禁止 `..`，禁止 `~`（构造时就 `ValueError`/`NotImplementedError`）。
* `mode="interrupt"` 会自动生成对应的 HITL 配置（同样需要 checkpointer）。
* bulk 工具（`ls`/`glob`/`grep`）按"搜索子树是否可能碰到该前缀"判断，
  所以模式最好带**字面前缀锚点**（`/secrets/**` 而不是 `/**/secrets`）。

安全边界在哪（docs 的核心提醒）：
* 权限是**策略层**（模型不能自己绕过）；
* backend 的 root_dir / 沙箱是**边界层**（进程层面根本碰不到）；
* 两者要一起用：策略层防"模型做错事"，边界层防"进程被打穿"。
"""

from __future__ import annotations

import sys

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.middleware import FilesystemMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from shared import ui
from shared.offline import bound_tool_names, choose_model, say, tool_call

READ_ONLY = [
    FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
    FilesystemPermission(operations=["read", "write"], paths=["/project/**"], mode="allow"),
]

READ_WRITE_PROMPTED = [
    FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt"),
    FilesystemPermission(operations=["read", "write"], paths=["/secrets/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/project/**"], mode="allow"),
]


def demo_validation() -> None:
    ui.subtitle("1. 路径模式在构造时就校验（fail fast）")
    for bad, expected in [
        ("secrets/**", "ValueError: 必须以 / 开头"),
        ("/a/../b", "ValueError: 不允许 .."),
    ]:
        try:
            FilesystemPermission(operations=["read"], paths=[bad])
        except (ValueError, NotImplementedError) as exc:
            ui.kv(bad, f"{type(exc).__name__}: {str(exc)[:56]}")
            ui.note(f"预期：{expected}")
    try:
        FilesystemPermission(operations=["read"], paths=["/~/secrets"])
    except NotImplementedError as exc:
        ui.kv("/~/secrets", f"NotImplementedError: {str(exc)[:50]}")


def demo_deny(model) -> None:
    ui.subtitle("2. deny —— 写 /secrets 被拒")
    agent = create_deep_agent(model=model, tools=[], permissions=READ_ONLY)
    result = agent.invoke({"messages": [{"role": "user", "content": "把密钥写进 /secrets/key.txt"}]})
    ui.show_messages(result)
    denied = next((m for m in result["messages"] if m.type == "tool"), None)
    ui.assert_true(
        denied is not None and "denied" in str(denied.content).lower(),
        "被 deny 的调用返回 permission denied 错误而不是执行",
    )
    ui.assert_true(not (result.get("files") or {}), "被拒的文件没有落进 state")

    ui.subtitle("2b. 同一套规则下 /project 是允许的")
    ok = create_deep_agent(
        model=choose_model(
            [
                tool_call(
                    "write_file",
                    {"file_path": "/project/notes.md", "content": "# 允许\n"},
                    "c1",
                ),
                say("已写入。"),
            ],
            None,
        ),
        tools=[],
        permissions=READ_ONLY,
    ).invoke({"messages": [{"role": "user", "content": "写 /project/notes.md"}]})
    ui.show_messages(ok)
    ui.assert_in("/project/notes.md", (ok.get("files") or {}), "/project/** 命中 allow")
    ui.note("第一个命中即生效：/secrets/** 的 deny 写在前面，所以它不会被后面的 allow 覆盖。")


def demo_interrupt(model) -> None:
    ui.subtitle("3. interrupt —— 敏感目录要人点头（自动接上 HITL）")
    agent = create_deep_agent(
        model=model,
        tools=[],
        permissions=READ_WRITE_PROMPTED,
        checkpointer=InMemorySaver(),  # interrupt 模式必须有 checkpointer
    )
    config = {"configurable": {"thread_id": "perm-1"}}
    pending = agent.invoke(
        {"messages": [{"role": "user", "content": "把新密钥写进 /secrets/key.txt"}]},
        config=config,
    )
    ui.assert_true("__interrupt__" in pending, "mode='interrupt' 触发了暂停")
    if "__interrupt__" in pending:
        for request in pending["__interrupt__"][0].value["action_requests"]:
            ui.kv("pending action", request["name"])
            ui.kv("args", str(request["args"])[:60])
    ui.note("不需要另外写 interrupt_on：permissions 里的 mode='interrupt' 会自动生成配置，")
    ui.note("两套机制可以混用（interrupt_on 拦工具，permissions 拦路径）。")
    _ = say


def demo_subagent_inheritance() -> None:
    ui.subtitle("4. 父子权限：省略=继承，指定=整体替换")
    from deepagents import SubAgent

    strict_child: SubAgent = {
        "name": "reader",
        "description": "只读子 agent。",
        "permissions": [FilesystemPermission(operations=["read"], paths=["/**"], mode="allow")],
    }
    agent = create_deep_agent(
        model=choose_model([say("ok")], None),
        tools=[],
        permissions=READ_ONLY,
        subagents=[strict_child],
    )
    ui.kv("父 agent 工具", bound_tool_names(agent))
    ui.note("SubAgent 省略 permissions -> 继承父规则；一旦指定 -> **完全替换**父规则。")
    ui.note("这是一个容易踩的坑：想让子 agent 更严，必须把父的规则一起抄过去，")
    ui.note("否则它会变成'只带自己那一条规则'，其余路径默认不受限。")
    ui.note("规则求值是线性 first-match，没有隐式 deny-all 兜底。")


def demo_vs_filesystem_middleware() -> None:
    ui.subtitle("5. permissions vs FilesystemMiddleware(tools=...) —— 两种收窄")
    agent = create_deep_agent(
        model=choose_model([say("ok")], None),
        tools=[],
        permissions=READ_ONLY,
        middleware=[FilesystemMiddleware(tools=["read_file", "ls", "grep"])],
    )
    ui.kv("工具集（无 write_file/edit_file）", bound_tool_names(agent))
    ui.assert_true(
        "write_file" not in bound_tool_names(agent),
        "tools allowlist 把写工具整个拿掉",
    )
    ui.note("FilesystemMiddleware(tools=...) —— 收窄**能力**（模型看不到这个工具）")
    ui.note("permissions=[...]                  —— 收窄**路径**（工具在，但某些路径不给碰）")
    ui.note("两者正交：先用 allowlist 砍掉不用的工具，再用 permissions 守住敏感路径。")
    ui.note("docs 明确提醒：permissions 也不是万能的，")
    ui.note("  后端 root_dir / 沙箱才是进程级的硬边界（见 d03、d04）。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d09 * Permissions ---- allow / deny / interrupt, first match wins",
        feature="Filesystem permissions",
        principle="#2 Extensible * #4 Production-ready",
    )

    demo_validation()
    demo_deny(
        choose_model(
            [
                tool_call(
                    "write_file",
                    {"file_path": "/secrets/key.txt", "content": "sk-live-xxx\n"},
                    "c1",
                ),
                say("好的，我不碰密钥目录。"),
            ],
            argv,
        )
    )
    demo_interrupt(
        choose_model(
            [
                tool_call(
                    "write_file",
                    {"file_path": "/secrets/key.txt", "content": "sk-live-xxx\n"},
                    "c1",
                ),
                say("已按批准写入。"),
            ],
            argv,
        )
    )
    demo_subagent_inheritance()
    demo_vs_filesystem_middleware()


if __name__ == "__main__":
    main()
