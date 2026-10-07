"""d11 · Custom Tools —— 从一个普通 Python 函数到模型可用的工具

覆盖 Feature   : Custom tools（@tool / 结构化输入输出 / ToolRuntime 注入）
覆盖 Principle : #1 Opinionated（工具即函数）· #2 Extensible · #3 Model-agnostic

---------------------------------------------------------------------------
文档依据（concepts/tools-filesystem.md + middleware-catalog.md）
---------------------------------------------------------------------------
三条主线：

1. **一个普通函数 = 一个工具**
   `@tool` 用 docstring 当 description、用类型注解当 JSON schema。
   这就是 opinionated：没有独立的 schema 文件，没有注册表。

2. **`ToolRuntime` 自动注入**
   在函数签名里写 `runtime: ToolRuntime`，执行时会被替换成运行时对象：
   - `state` 当前 graph state（含 `files`）
   - `config` RunnableConfig
   - `store` BaseStore（跨会话）
   - `tool_call_id` / `context` / `execution_info`
   注意：这是 `ToolRuntime`，**不是** middleware/node 拿到的 `Runtime`
   （后者没有 `config`，见 d03 第 2 节）。

3. **工具是"叠加"语义，收窄才需要 middleware**
   `tools=[...]` 永远是往默认集上加；要删掉内置工具必须走
   `FilesystemMiddleware(tools=[...])`（见 d02 的 F/G 节）。
"""

from __future__ import annotations

import sys
from typing import Literal

from deepagents import create_deep_agent
from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore

from shared import ui
from shared.offline import bound_tool_names, choose_model, say, tool_call


# ===========================================================================
# 1. 最简单：函数 + docstring + 类型注解 = 工具
# ===========================================================================
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询某个城市的当前天气。

    Args:
        city: 城市名，例如 "Beijing"。
        unit: 温度单位，"celsius" 或 "fahrenheit"。

    Returns:
        一段人类可读的天气描述。
    """
    return f"{city} 26 度，多云（单位：{unit}）"


@tool
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b


def demo_basics(model) -> None:
    ui.subtitle("1. @tool：docstring -> description，类型注解 -> args schema")
    agent = create_deep_agent(model=model, tools=[get_weather, add])
    result = agent.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？"}]})
    ui.show_messages(result)
    ui.kv("工具集", bound_tool_names(agent))

    ui.step("自动生成的参数 schema（供绑定到模型用）：")
    for key, spec in get_weather.args.items():
        ui.kv(key, f"type={spec['type']} required={'required' in spec}")
    ui.kv("描述前 40 字", get_weather.description[:40])
    ui.assert_in("get_weather", bound_tool_names(agent), "函数已注册成工具")
    ui.note("三条推论：")
    ui.note("  * 重命名函数 = 重命名工具（模型看到的就是函数名）。")
    ui.note("  * 改 docstring = 改模型的使用说明（等于改提示词）。")
    ui.note("  * 缺类型注解 -> schema 不完整，模型会瞎填；必填项靠没有默认值表达。")


#: OpenAI 风格的 JSON schema dict（d11 本章 + d14 用）
lookup_schema = {
    "name": "lookup_order",
    "description": "按订单号查订单状态。",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "订单号"}},
        "required": ["order_id"],
    },
}


def demo_dict_schema(model) -> None:
    ui.subtitle("2. dict 形式：只被 profile 覆盖层使用，不会被注册成工具")
    # 形状：profile 的 tool_descriptions 覆盖就是靠 `name` + `description` 定位并改写 dict 工具。
    lookup = lookup_schema
    ping = StructuredTool.from_function(
        func=lambda: "pong",
        name="ping",
        description="连通性检查。",
    )
    agent = create_deep_agent(model=model, tools=[lookup, ping])
    names = bound_tool_names(agent)
    ui.kv("工具集", names)
    ui.assert_in("ping", names, "StructuredTool 被注册")
    ui.assert_true("lookup_order" not in names, "裸 dict 没有被注册成可调用工具")
    ui.note("实测（langchain create_agent 与 deepagents create_deep_agent 一致）：")
    ui.note("  dict 输入会被**静默跳过**，不会进 _tools_by_name。")
    ui.note("  它存在的意义是 `_apply_tool_description_overrides` 要能就地改写")
    ui.note("  dict 工具的 description（profiles 的 tool_descriptions 就走这条路，d14）。")

    ui.subtitle("2b. 想把 schema dict 变成真工具：自己包一层（Extensible）")
    ready = tool_from_schema(lookup, handler=lambda order_id: f"订单 {order_id}: 已发货")
    ready_agent = create_deep_agent(model=model, tools=[ready])
    ui.assert_in("lookup_order", bound_tool_names(ready_agent), "适配后被注册")
    ui.note("本工程选择 `StructuredTool.from_function` 而不是第三方转换器：")
    ui.note("没有隐式依赖、schema 可以在运行时由配置中心下发。")


def tool_from_schema(schema: dict, handler):
    """把 OpenAI 风格的 JSON schema dict 转成可注册的 `StructuredTool`。

    这是本工程自己写的适配器（约 20 行），演示 Extensible：
    上游**没有**提供 `create_tool_from_openai_schema` 这样的官方入口，
    所以自己补；同时保留 schema 与 handler 分离，便于远端下发。
    """
    properties = schema.get("parameters", {}).get("properties", {})
    required = set(schema.get("parameters", {}).get("required", []))

    def run(**kwargs):
        missing = required - kwargs.keys()
        if missing:
            raise ValueError(f"缺少必填参数：{sorted(missing)}")
        return handler(**kwargs)

    run.__name__ = schema["name"]
    run.__doc__ = schema.get("description", "")
    return StructuredTool.from_function(
        func=run,
        name=schema["name"],
        description=schema.get("description", ""),
    )


def demo_runtime_tool(model) -> None:
    ui.subtitle("3. ToolRuntime 注入：工具能读 state / store / config")
    calls: dict[str, object] = {}

    @tool
    def remember(key: str, value: str, runtime: ToolRuntime = None) -> str:
        """把一个键值对记到长期 store 里，并同时写进 thread state。

        Args:
            key: 键名。
            value: 值。
        """
        # ToolRuntime 由执行系统注入，不是用户传的
        config = getattr(runtime, "config", None)
        thread_id = (config.get("configurable") or {}).get("thread_id", "?")
        calls["thread_id"] = thread_id
        calls["tool_call_id"] = getattr(runtime, "tool_call_id", None)
        calls["has_store"] = runtime.store is not None
        if runtime.store is not None:
            runtime.store.put(("notes", key), key, {"value": value})
        # runtime.state 是**当前** state（含 files）——同一次调用里能读到刚写的文件
        calls["state_keys"] = sorted(runtime.state)
        return f"已记住 {key}={value}（thread={thread_id}）"

    agent = create_deep_agent(model=model, tools=[remember], store=InMemoryStore())
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "记下 preference=中文"}]},
        config={"configurable": {"thread_id": "tools-1"}},
    )
    ui.show_messages(result)
    for k, v in calls.items():
        ui.kv(k, str(v)[:70])
    ui.assert_in("thread_id", calls, "工具拿到了 config（因此知道当前 thread）")
    ui.assert_true(calls.get("has_store") is True, "工具拿到了 store（可跨会话写入）")
    ui.assert_in("files", calls.get("state_keys", []), "runtime.state 里能看到 files 通道")
    ui.note("store 必须真的传给 create_deep_agent(store=...)，否则 runtime.store 是 None。")
    ui.note("对比 d03：middleware/node 拿的是 `Runtime`（无 config，要走 context_schema），")
    ui.note("工具拿的是 `ToolRuntime`（有 config/state/store）——两个对象别搞混。")


def demo_error_shapes(model_factory1, model_factory2) -> None:
    ui.subtitle("4. 三种错误，形态完全不同（这条最容易想当然）")

    @tool
    def fail_loudly(reason: str) -> str:
        """故意抛异常，用于演示运行时错误的传播行为。"""
        raise ValueError(f"工具内部错误：{reason}")

    # (1) 模型**传参**不合法 -> ToolInvocationError -> 友好 ToolMessage，模型可自我纠正
    # 注意：工具调用失败后 agent 可能会再发起一次模型调用，因此脚本里要有承接回复。
    agent = create_deep_agent(model=model_factory1(), tools=[get_weather])
    bad_args = agent.invoke(
        {"messages": [{"role": "user", "content": "调用 get_weather"}]}
    )
    validation = next((m for m in bad_args["messages"] if m.type == "tool"), None)
    ui.kv("① 传参校验错误", str(validation.content).replace("\n", " ")[:78] if validation else "未触发")
    ui.assert_true(
        validation is not None and "Please fix the error" in str(validation.content),
        "参数校验失败 -> ToolMessage，模型可以改参数重试",
    )

    # (2) 业务错误由 backend / middleware 以**结构化 error** 返回（failures are data）
    ui.kv("② 业务错误", "ReadResult(error=...) / 'permission denied for write' / 'file not found'")
    ui.note("这些不抛异常，直接是 ToolMessage —— 见 d03 第 3b 节、d09 第 2 节。")

    # (3) 工具**内部**抛异常 -> 默认**冒泡**，整个 graph 运行失败
    boom = create_deep_agent(model=model_factory2(), tools=[fail_loudly])
    try:
        boom.invoke({"messages": [{"role": "user", "content": "调用 fail_loudly"}]})
    except ValueError as exc:
        ui.kv("③ 运行时异常", f"{type(exc).__name__}: {str(exc)[:40]}")
        ui.assert_true(True, "工具内部异常默认**不会**被吞掉，会冒泡到调用方")
    else:  # pragma: no cover
        ui.assert_true(False, "异常应当冒泡")
    ui.note("langchain 的默认处理逻辑（实测）：")
    ui.note("  * ToolInvocationError（传参不合 schema）-> 返回错误消息给模型。")
    ui.note("  * 其它任何异常 -> `raise e` 冒泡，本轮 run 失败。")
    ui.note("生产建议：业务失败走**返回值**（结构化 error），")
    ui.note("  只有不可预期的 bug 才让它抛出 —— 别让模型对着 traceback 猜。")


def demo_tools_are_additive() -> None:
    ui.subtitle("5. tools= 是叠加，收窄必须走 middleware（容易踩的坑）")
    agent = create_deep_agent(model=choose_model([say("ok")], None), tools=[get_weather])
    names = bound_tool_names(agent)
    ui.kv("工具集", names)
    ui.assert_in("get_weather", names, "自定义工具加进来了")
    ui.assert_in("write_file", names, "默认文件工具也还在（叠加语义）")
    ui.note("tools=[get_weather] **没有**让 write_file 消失。")
    ui.note("要收窄：middleware=[FilesystemMiddleware(tools=['read_file','ls'])]（d02/d09）。")
    ui.note("要屏蔽**单个工具名**而非删除：HarnessProfile.excluded_tools（请求期过滤，")
    ui.note("  被调用时返回 unavailable 错误，docs 明确说这不是安全边界）。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d11 * Custom Tools ---- @tool, dict schema, ToolRuntime",
        feature="Custom tools",
        principle="#1 Opinionated * #2 Extensible * #3 Model-agnostic",
    )

    demo_basics(
        choose_model(
            [
                tool_call("get_weather", {"city": "Beijing", "unit": "celsius"}, "c1"),
                say("北京 26 度，多云。"),
            ],
            argv,
        )
    )

    demo_dict_schema(choose_model([say("确认工具集。")], argv))

    demo_runtime_tool(
        choose_model(
            [
                tool_call("remember", {"key": "preference", "value": "中文"}, "c1"),
                say("已记住。"),
            ],
            argv,
        )
    )

    demo_error_shapes(
        lambda: choose_model(
            [
                # 故意缺必填参数 city，触发 ToolInvocationError
                tool_call("get_weather", {"unit": "celsius"}, "c1"),
                say("收到错误，换参数重试。"),
            ],
            argv,
        ),
        lambda: choose_model(
            [
                tool_call("fail_loudly", {"reason": "演示"}, "c1"),
                say("这次不该走到这里。"),
            ],
            argv,
        ),
    )

    demo_tools_are_additive()


if __name__ == "__main__":
    main()
