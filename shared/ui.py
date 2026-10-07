"""控制台渲染 —— 让每个 demo 的输出直接对应文档里的那条 principle。

官方文档的 demo 全部靠"看 agent 干了什么"来理解。这里把
`agent.stream(...)` 的事件流转成可读文本，行为不变，只是可读性更好
（Principle: Production-ready 的 streaming 能力被显式使用）。
"""

from __future__ import annotations

from typing import Any, Iterable

BAR = "=" * 78
SUB = "-" * 78


def _ascii_safe(text: str) -> str:
    """部分 Windows 终端字体对 em-dash / 箭头支持不好，统一降级为 ASCII。"""
    return (
        text.replace("—", "--")
        .replace("–", "-")
        .replace("→", "->")
        .replace("←", "<-")
        .replace("·", "*")
        .replace("✅", "[ok]")
        .replace("❌", "[x]")
    )


def banner(title: str, feature: str = "", principle: str = "") -> None:
    """打印 demo 头部：编号 + 覆盖的 feature + 对应的 principle。"""
    print(f"\n{BAR}\n{_ascii_safe(title)}\n{BAR}")
    if feature:
        print(f"Feature   : {_ascii_safe(feature)}")
    if principle:
        print(f"Principle : {_ascii_safe(principle)}")


def step(text: str) -> None:
    print(f"\n>> {text}")


def note(text: str) -> None:
    print(f"   {text}")


def kv(key: str, value: Any) -> None:
    print(f"   {key:<22} {value}")


def show_files(files: dict[str, Any] | None, limit: int = 5) -> None:
    """打印 StateBackend 落在 graph state 里的 `files`。"""
    if not files:
        print("   (files 为空)")
        return
    for path in list(files)[:limit]:
        data = files[path]
        body = data.get("content") if isinstance(data, dict) else data
        text = body if isinstance(body, str) else str(body)
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        print(f"   {path:<34} {len(text):>6} chars | {first[:60]}")
    if len(files) > limit:
        print(f"   ... 另有 {len(files) - limit} 个文件")


def show_messages(result: dict[str, Any], limit: int = 8) -> None:
    """按 role/工具调用打印消息流。"""
    for msg in result.get("messages", [])[-limit:]:
        kind = type(msg).__name__
        if kind == "AIMessage" and getattr(msg, "tool_calls", None):
            calls = ", ".join(c["name"] for c in msg.tool_calls)
            print(f"   {kind:<14} -> {calls}")
        else:
            body = str(getattr(msg, "content", "")).replace("\n", " ")
            print(f"   {kind:<14} {body[:80]}")


def show_stream(events: Iterable[dict[str, Any]], keys: tuple[str, ...] = ("write_file", "edit_file", "task")) -> None:
    """精简 `stream_mode="updates"`：只打印关键工具节点的事件。"""
    for chunk in events:
        for node, payload in chunk.items():
            if node == "model":
                for update in payload.get("messages", []) or []:
                    if getattr(update, "tool_calls", None):
                        print(f"   model  -> {[c['name'] for c in update.tool_calls]}")
                    elif getattr(update, "content", ""):
                        print(f"   model  <- {str(update.content)[:70]}")
            elif node == "tools":
                for msg in payload.get("messages", []) or []:
                    if any(k in str(msg.name) for k in keys):
                        print(f"   tools  <- {msg.name}: {str(msg.content)[:70]}")


def subtitle(text: str) -> None:
    print(f"\n{SUB}\n{_ascii_safe(text)}\n{SUB}")


# --------------------------------------------------------------------------
# 断言：官方 workflow 要求"每个 demo 都要有可验证的断言"，
# 这里统一渲染成 [ok] / [x]，让 run_all.py 的日志一眼看出哪条 principle 塌了。
# --------------------------------------------------------------------------


def assert_true(condition: bool, label: str) -> bool:
    print(f"   [{'ok' if condition else 'x'}] {label}")
    return bool(condition)


def assert_equal(actual: Any, expected: Any, label: str) -> bool:
    ok = actual == expected
    print(f"   [{'ok' if ok else 'x'}] {label}: {actual!r}")
    if not ok:
        print(f"        expected: {expected!r}")
    return ok


def assert_in(needle: Any, haystack: Any, label: str) -> bool:
    ok = needle in haystack
    hay = repr(haystack)
    if len(hay) > 72:
        hay = hay[:69] + "..."
    print(f"   [{'ok' if ok else 'x'}] {label}: {needle!r} in {hay}")
    return ok
