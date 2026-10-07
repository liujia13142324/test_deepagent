"""离线确定性模型 —— 让"全部 feature"在没有任何 API key 的机器上也能被验证。

这不是 deepagents 的 API，而是本 demo 工程自己的测试替身：实现
`BaseChatModel.bind_tools` + `_generate`，按脚本逐条吐出预设的
`AIMessage`（可以是纯文本，也可以是 tool_calls）。

它对应官方 `workflows/build-a-deep-agent.md` 第 8 节的要求：

    Start with fake-model assembly tests. ... Then use a scripted fake model to
    exercise the modified loop.

用法：

    model = ScriptedModel(script=[
        tool_call("write_file", {"file_path": "/a.txt", "content": "hi"}),
        say("已完成"),
    ])
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from shared import config

# --------------------------------------------------------------------------
# 构造脚本片段的小工具（让 demo 里的模型行为一眼可读）
# --------------------------------------------------------------------------


def say(text: str) -> AIMessage:
    """一次纯文本回复（不调用工具）。"""
    return AIMessage(content=text)


def tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    """一次工具调用。`call_id` 必须显式给，LangGraph 靠它配对 ToolMessage。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id or f"call_{name}"}],
    )


def think_then(text: str) -> AIMessage:
    """带 reasoning 痕迹的终局回复，便于观察 thinking 模型行为。"""
    return AIMessage(content=text)


def file_data(content: str) -> dict[str, Any]:
    """构造 StateBackend 期望的 `FileData` 形状。

    `StateBackend` 的文件存在 graph state 的 `files: dict[str, FileData]` 里，
    注入时必须给全字段，直接塞裸字符串会在 `ls()` 里炸掉：

        {"content": ..., "encoding": ..., "created_at": ..., "modified_at": ...}
    """
    stamp = "2026-01-01T00:00:00+00:00"
    return {
        "content": content,
        "encoding": "utf-8",
        "created_at": stamp,
        "modified_at": stamp,
    }


def seed_files(mapping: dict[str, str]) -> dict[str, dict[str, Any]]:
    """把 `{path: 文本}` 转成可直接塞进 graph 输入的 `files`。"""
    return {path: file_data(body) for path, body in mapping.items()}


class ScriptedModel(BaseChatModel):
    """按脚本回放的 tool-calling 模型。

    * `bind_tools` 必须实现，否则 harness 的 `bind_tools` 会抛 NotImplementedError。
    * 每个实例独立计数，**不要**在不同 agent 之间共享同一个实例
      （deepagents 会为 summarizer / rubric grader 单独 resolve model）。
    """

    script: list[AIMessage] = []
    name_: str = "scripted"
    fallback: str | None = None
    model_name: str = "scripted-sonnet"

    # 游标：私有属性，pydantic 不把它当模型字段（不会进 schema / 不会序列化）
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-deepagents-demo"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"turns": len(self.script)}

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        # 让 deepagents 的 `get_model_provider` / `get_model_identifier`
        # （d14 profiles 的 harness 解析依赖它们）把脚本模型识别成
        # provider="cheap"、identifier="scripted-sonnet"，
        # 这样 `create_deep_agent(model=<scripted>)` 能命中我们注册的
        # harness profile（默认 provider 元数据拿不到东西 -> 空 profile）。
        return {"ls_provider": "cheap", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # noqa: D102
        # 真正的模型在这里把工具 schema 绑上去；脚本模型只需要"知道有这些工具"，
        # 因此把 tools 当作普通 kwarg 透传，_generate 里忽略即可。
        return self.bind(tools=list(tools or []), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = self._cursor
        if idx >= len(self.script):
            if self.fallback is not None:
                # 兜底分支：summarization / rubric 这类 middleware 会消耗额外的
                # 模型调用。显式声明 fallback，表示"这些额外轮次不影响主流程断言"。
                return ChatResult(generations=[ChatGeneration(message=say(self.fallback))])
            raise RuntimeError(
                f"ScriptedModel 脚本已耗尽（已回放 {idx} 轮）。"
                "通常是 summarization / rubric 额外多消耗了一轮，"
                "请在脚本里补上对应的回复，或设置 fallback= 显式接受兜底。"
            )
        self._cursor = idx + 1
        return ChatResult(generations=[ChatGeneration(message=self.script[idx])])


def bound_tool_names(agent) -> list[str]:
    """从编译后的 graph 里取出模型**实际可见**的工具名。

    这是官方 `workflows/build-a-deep-agent.md` 第 8 节推荐的假模型断言方式：
    组装一次 graph，拿工具列表做形状断言，不消耗任何 token。

    实现细节：LangGraph 把工具节点放在 `agent.nodes["tools"]`，
    其 `.bound` 是 `ToolNode`，工具表在私有属性 `_tools_by_name` 上。
    """
    node = agent.nodes.get("tools")
    bound = getattr(node, "bound", None)
    by_name = getattr(bound, "_tools_by_name", None)
    return sorted(by_name) if by_name else []


def probe_model() -> ScriptedModel:
    """只要"能把 graph 组装出来做形状断言"，不需要真的推理。

    用它代替 `create_deep_agent(model="openai:...")`，可以避免 demo 依赖
    某个具体 provider 包——这本身就是 principle #3 的直接收益。
    """
    return ScriptedModel(script=[])


def _resolve_flags(argv: Sequence[str] | None) -> Sequence[str]:
    """把调用方传入的 argv 与真实 sys.argv 合并，供 flag 判定使用。"""
    import sys

    return list(argv or []) + list(sys.argv)


def choose_model(script: list[AIMessage], argv: Sequence[str] | None = None) -> BaseChatModel | str:
    """离线则返回 `ScriptedModel`，否则返回 `provider:model` 字符串。

    这就是 principle #3（Model-agnostic）在工程上的落地方式：
    **agent 的组装代码完全不感知底层是脚本回放还是真实 LLM。**

    `fallback` 用于容忍 summarization / rubric 这类 middleware 的额外模型调用。
    """
    return _build_model(script, argv, fallback=None)


def choose_model_lenient(
    script: list[AIMessage], argv: Sequence[str] | None = None, fallback: str = "（兜底回复）"
) -> BaseChatModel | str:
    """同 `choose_model`，但脚本耗尽后返回固定兜底文本而不是抛错。"""
    return _build_model(script, argv, fallback=fallback)


def _build_model(
    script: list[AIMessage], argv: Sequence[str] | None, fallback: str | None
) -> BaseChatModel | str:
    # argv 优先于 sys.argv：这样 run_all.py 里 main(["--offline"]) 能真正生效。
    if config.should_go_offline(list(argv or [])):
        return ScriptedModel(script=list(script), fallback=fallback)
    return config.model_spec()


def model_factory(
    script: list[AIMessage], argv: Sequence[str] | None = None, fallback: str | None = None
):
    """返回一个**每次调用都新建模型**的工厂。

    为什么要工厂而不是直接给 model：
    `ScriptedModel` 是有状态游标，而一次 demo 里经常会建多个 agent
    （例如"同一个 store、两个不同 thread"），复用实例会耗尽脚本。
    真实模型无状态，但用工厂写法两种模式共用同一份代码。

    用法::

        agent1 = build(model_factory([...]))
        agent2 = build(model_factory([...]))
    """
    return lambda: _build_model(list(script), argv, fallback)
