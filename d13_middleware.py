"""d13 · Middleware —— 用 AgentMiddleware 钩子改造每一轮模型调用的"中间件"能力

覆盖 Feature   : Custom middleware（before_agent / wrap_model_call / wrap_tool_call）
覆盖 Principle : #1 Opinionated（生命周期固定）· #2 Extensible（按钩子插入）
覆盖要点       : 与 d02(FilesystemMiddleware) / d09(权限) / d10(技能/模型请求) 互补，
              本 demo 专注**用户自己写的**中间件，和生命周期顺序。

---------------------------------------------------------------------------
依据（concepts/middleware-catalog.md「Lifecycle」节 + AgentMiddleware 源码）
---------------------------------------------------------------------------
middleware-catalog 给出的生命周期边界（背下来）：
  运行开始 -> before_agent 做一次性的 state 初始化/历史修复
          -> wrap_model_call 每次请求都生效（改 messages/prompts/tools）
          -> 模型调用
          -> 有工具调用 -> wrap_tool_call 中介执行或改写结果 -> 回到 wrap_model_call
          -> 没工具调用 -> 结束（rubric 开启则可能追加反馈再来一轮）

判断"该用 middleware 还是 tools="的标准（同一节的原文）：
  - 工具=：只在模型选中它之后才跑，适合独立的、消费者特定的活。
  - 中间件=：行为要改 prompts、改广告出去的工具集、改 messages/state、改循环控制
            时，就必须用中间件。它不在"执行"阶段，而在"请求"。全程生效。
"""

from __future__ import annotations

import sys

from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from shared import ui
from shared.offline import choose_model, say, tool_call


# ===========================================================================
# 1. before_agent：跑前做一次性 state 初始化；并拿到 Runtime（不是 ToolRuntime）
# ===========================================================================
class RunTicker(AgentMiddleware):
    """每次 run 开始数一刀，并把 Runtime 形态记下来供断言。

    `before_agent(state, runtime)` 每轮 run 调用一次。
    注意这里的 `Runtime`（langchain.runtime）和 d11 工具里的
    `ToolRuntime`（langgraph.prebuilt）是**两个东西**。
    """

    def __init__(self):
        super().__init__()
        self.runs = 0
        self.runtime_info = {}

    def before_agent(self, state, runtime):
        self.runs += 1
        state["run_count"] = (state.get("run_count") or 0) + 1
        self.runtime_info = {
            "type": type(runtime).__name__,
            "has_config": hasattr(runtime, "config"),
            "has_store": runtime.store is not None,
        }
        return {}

    def wrap_model_call(self, request, handler):
        return handler(request)


# ===========================================================================
# 2. wrap_model_call：偷偷往 system prompt 塞进一句固定规则（模板化提示词）
# ===========================================================================
class HouseStyle(AgentMiddleware):
    """把公司规范追加进每一次 model call 的 system prompt。

    mid=... 用于验证顺序（第 3 节）。注意我们只"加内容"、"不拦截"，
    否则就可能破坏其它中间件已经写好的 prompt 片段。
    """

    def __init__(self, rule: str, mid: str, captured):
        super().__init__()
        self.rule, self.mid, self.captured = rule, mid, captured

    @property
    def name(self) -> str:
        # 基类的 name 是只读 property（默认返回类名），且 factory.py 用
        # {m.name for m ...} 去重：同名 = "重复实例" 直接抛错。
        # 官方文档说 name "can be overridden for custom naming" —— 用子类 property 覆盖。
        return f"HouseStyle_{self.mid}"

    def wrap_model_call(self, request, handler):
        old = str(request.system_prompt or "")
        request = request.override(
            system_prompt=f"{old}\n\n[内部规则]\n{self.rule}\n".strip()
        )
        self.captured.append((self.mid, str(request.system_prompt)))
        return handler(request)


# ===========================================================================
# 3. wrap_tool_call：改写工具参数（比 d02 的"结果超限截断"再往前一步）
# ===========================================================================
class UnitsForcer(AgentMiddleware):
    """把 get_weather 的 unit 参数强制改写成 fahrenheit，模型不知道。

    这就是"middleware 不是 tool"最典型的差别：模型**以为**自己传了
    celsius，执行到的其实是 fahrenheit —— 中介层在工具执行前做了手脚。
    """

    def __init__(self, forced_units: str):
        super().__init__()
        self.forced_units = forced_units
        self.args_seen = []

    def wrap_tool_call(self, request, handler):
        call = dict(request.tool_call)
        if call.get("name") == "get_weather":
            call = {**call, "args": {**call["args"], "unit": self.forced_units}}
            request = request.override(tool_call=call)
            self.args_seen.append(dict(call["args"]))
        return handler(request)


def demo_before_agent() -> None:
    ui.subtitle("1. before_agent：一次性初始化 state，并拿到 Runtime")
    ticker = RunTicker()
    agent = create_deep_agent(
        model=choose_model([say("跑完了。"), say("再来一份。")], None), middleware=[ticker]
    )
    agent.invoke({"messages": [{"role": "user", "content": "开始"}]})
    agent.invoke({"messages": [{"role": "user", "content": "再来"}]})
    ui.kv("run_count（两次 run 累加）", ticker.runs)
    ui.kv("Runtime 形态", str(ticker.runtime_info))
    ui.assert_true(ticker.runs == 2, "before_agent 每轮 run 都生效")
    ui.assert_true(ticker.runtime_info.get("type") == "Runtime", "注入的是 Runtime")
    ui.assert_true(ticker.runtime_info.get("has_config") is False, "Runtime **没有** config")
    ui.note("对照 d11：工具拿到的是 `ToolRuntime`（有 config/state/store），")
    ui.note("middleware 拿到的是 `Runtime`（没 config）。")
    ui.note("  —— 别指望中间件里直接翻 config；要穿上下文，走 runtime.context。")
    ui.note("  （context 由 context_schema + runtime 提供，见 d06 摘要、d03。）")


def demo_wrap_model_call() -> None:
    ui.subtitle("2. wrap_model_call：每轮模型请求都带上一句内部规则")
    captured = []
    house = HouseStyle("所有回复必须以「收到」+ 一句话结尾。", "house", captured)
    agent = create_deep_agent(
        model=choose_model(
            [
                tool_call("get_weather", {"city": "Paris", "unit": "celsius"}, "c1"),
                say("收到。"),
            ],
            None,
        ),
        tools=[__import__("d11_custom_tools", fromlist=["get_weather"]).get_weather],
        middleware=[house],
    )
    agent.invoke({"messages": [{"role": "user", "content": "巴黎天气？"}]})
    ui.kv("prompt 长度", f"{len(captured[0][1])} 字符")
    ui.assert_in("内部规则", captured[0][1], "规则被追加进了 system prompt")
    ui.assert_true(len(captured) >= 2, "每轮模型调用都生效（不只是一次）")
    ui.note("这就是 memory/skills/summarization 等内置中间件干的事：")
    ui.note("  * 加载时读资源，请求时注入 prompt 片段（见 d07/d10）。")
    ui.note("  * 只 add、不 rewrite 整段 prompt —— 中间件彼此串联时不会打架。")


def demo_wrap_tool_call() -> None:
    ui.subtitle("3. wrap_tool_call：在工具执行前改写参数（模型蒙在鼓里）")
    forcer = UnitsForcer("fahrenheit")
    agent = create_deep_agent(
        model=choose_model(
            [
                tool_call("get_weather", {"city": "Paris", "unit": "celsius"}, "c1"),
                say("收到，按摄氏度？其实我不会读。"),
            ],
            None,
        ),
        tools=[__import__("d11_custom_tools", fromlist=["get_weather"]).get_weather],
        middleware=[forcer],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "巴黎天气"}]})
    ui.show_messages(result)
    ui.kv("模型发出的参数", str({"unit": "celsius"}))
    ui.kv("实际执行的参数", str(forcer.args_seen))
    ui.assert_true(forcer.args_seen and forcer.args_seen[0]["unit"] == "fahrenheit",
                   "模型传 celsius，执行层拿到 fahrenheit")
    ui.note("防线视角：参数改写是中间件的权力，但**不是**安全边界 ——")
    ui.note("它管不了直接把参数从图外用 batch 提交的调用方（见 d09 第 5 节）。")
    ui.note("结果侧的另一半：FilesystemMiddleware 在 wrap_tool_call 里 evict")
    ui.note("超大的 read_file 结果以免污染上下文 —— 见 d02 的 E 节。")


def demo_ordering() -> None:
    ui.subtitle("4. 顺序：middleware 按列表顺序串联（先列先执行）")
    captured = []
    agent = create_deep_agent(
        model=choose_model([say("好。")], None),
        middleware=[
            HouseStyle("第一条。", "A", captured),
            HouseStyle("第二条。", "B", captured),
        ],
    )
    agent.invoke({"messages": [{"role": "user", "content": "开始"}]})
    order = [mid for mid, _ in captured]
    idx_b = order.index("B") if "B" in order else -1
    idx_a = order.index("A") if "A" in order else -1
    ui.kv("调用顺序", " -> ".join(order[:2]))
    ui.assert_true(idx_a == 0 and idx_b == 1, "A 先执行，B 后执行")
    ui.note("可验证：从 request 里 read 出来的 prompt 是 A->B 叠加的。")
    ui.note("附加 invariants（middleware-catalog 原文）：")
    ui.note("  * create_deep_agent 会在你的列表**最后**自动追加 UnsupportedContentMiddleware，")
    ui.note("    因为它要检查最终选定的模型（见 d14/d15 的模型解析）。")
    ui.note("  * 想进子 agent 的私有字段用 PrivateStateAttr 标注，构造期解析并剥离。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d13 * Middleware ---- before_agent / wrap_model_call / wrap_tool_call",
        feature="Custom middleware",
        principle="#1 Opinionated * #2 Extensible",
    )
    demo_before_agent()
    demo_wrap_model_call()
    demo_wrap_tool_call()
    demo_ordering()


if __name__ == "__main__":
    main()