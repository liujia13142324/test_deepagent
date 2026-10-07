"""d14 · Profiles —— ProviderProfile 管"怎么构造模型"，HarnessProfile 管"agent 怎么跑"

覆盖 Feature   : Provider profiles + Harness profiles（profiles-models.md）
覆盖 Principle : #4 Production-ready（注册表 = 配置解耦）· #3 Model-agnostic（同栈跑新模型）

---------------------------------------------------------------------------
依据（concepts/profiles-models.md）
---------------------------------------------------------------------------
两层 profile，职责必须分开（本文档最核心的一句话）：

  * `ProviderProfile`：模型**构造阶段**的叠加层。
      `resolve_model("openai:gpt-5")` -> `init_chat_model(..., **apply_provider_profile(spec))`。
      控制 init_kwargs、加 attribution header（NVIDIA/OpenRouter）、
      OpenAI Responses API 默认值等 —— "模型是怎么被建出来的"。

  * `HarnessProfile`：模型**已构造后**的运行时叠加层。
      只影响提示词三段（USER / BASE / SUFFIX 用空行分隔拼接）、
      广告出去的工具可见性（excluded_tools）、附加 middleware、
      以及自动 general-purpose 子代理的开关与描述。

  * 关键解析细节（我们实测）：create_deep_agent 传的是**构造好的模型对象**时，
    用该对象的 provider + identifier 元数据查 harness 注册表；
    传的是**字符串 spec** 时，直接用 spec 查。
    因此本 demo 用 `ScriptedModel(model_name="profile-demo")`（见 shared/offline.py
    的 `_get_ls_params` override）来离线命中注册表 —— 这就是 principle #3 的收益：
    profile 的解析不关心底层是不是真 LLM。
"""

from __future__ import annotations

import sys

from deepagents import (
    FilesystemMiddleware,
    HarnessProfile,
    ProviderProfile,
    create_deep_agent,
    register_harness_profile,
    register_provider_profile,
)
from langchain.agents.middleware import AgentMiddleware

from shared import ui
from shared.offline import ScriptedModel, bound_tool_names, choose_model, say, tool_call


class PromptSilk(AgentMiddleware):
    """记录每轮模型请求看到的 system prompt 与工具 description。"""

    name = "d14_prompt_silk"

    def __init__(self):
        super().__init__()
        self.prompts = []
        self.tool_descriptions = {}

    def wrap_model_call(self, request, handler):
        self.prompts.append(str(request.system_prompt or ""))
        for tool in getattr(request, "tools", []) or []:
            if isinstance(tool, dict):  # 模型请求里的工具可能是 dict，不是 BaseTool
                name, desc = tool.get("name", ""), tool.get("description", "")
            else:
                name, desc = tool.name, getattr(tool, "description", "")
            self.tool_descriptions[name] = desc
        return handler(request)


def demo_provider_profiles() -> None:
    ui.subtitle("1. ProviderProfile = 构造阶段叠加（init_kwargs / attribution）")
    from deepagents.profiles.provider.provider_profiles import apply_provider_profile

    builtin = apply_provider_profile("openai:gpt-5")
    ui.kv("openai 内置叠加", str(list(builtin.keys()))[:80])

    register_provider_profile(
        "d14-demo",
        ProviderProfile(init_kwargs={"temperature": 0.2, "probs_file": "d14.trace"}),
    )
    out = apply_provider_profile("d14-demo:scripted-sonnet")
    ui.kv("自定义 ProviderProfile", str(out))

    # 注册两个 provider profile 时做"增量覆盖"：第二轮只补 d14b，第一轮的字段保留。
    register_provider_profile(
        "d14-demo", ProviderProfile(init_kwargs={"extra_field": 42})
    )
    merged = apply_provider_profile("d14-demo:scripted-sonnet")
    ui.kv("再注册后（合并）", str(sorted(merged.keys())))
    ui.assert_in("temperature", merged, "先注册的字段还在（additive merge）")
    ui.assert_in("extra_field", merged, "后注册的字段也进了")

    ui.note("这是例子，不是推荐：ProviderProfile 是为云厂商特有的构造参数服务的，")
    ui.note("比如 OpenRouter/NVIDIA 的 attribution header、Responses API 默认关闭。")
    ui.note("自己给模型编 temperature 属于微调，通常应该直接构造 model 对象。")


def demo_harness_profiles() -> None:
    ui.subtitle("2. HarnessProfile = 运行时叠加（USER/BASE/SUFFIX + 工具覆盖层）")
    # 跟 d11 的"裸 dict 不会被注册成工具、但会被 profile 覆盖层改写"接上：
    # 把 d11 那张 lookup_order 的 description 改掉，证明覆盖层真的在干活。
    silk = PromptSilk()
    register_harness_profile(
        "cheap:profile-demo",
        HarnessProfile(
            base_system_prompt="（profile BASE）本环境不允许讨论天气。",
            system_prompt_suffix="（profile SUFFIX）所有回复一律用中文。",
            tool_description_overrides={"delete": "删除文件（Override'd by profile.）"},
        ),
    )
    # 关键：这个 ScriptedModel 带上 model_name -> provider=cheap / id=profile-demo
    model: ScriptedModel = choose_model([say("收到。")], None)
    model.model_name = "profile-demo"

    agent = create_deep_agent(
        model=model,
        system_prompt="（USER）你是订单助理。",
        middleware=[silk],
    )
    agent.invoke({"messages": [{"role": "user", "content": "干活"}]})
    prompt = silk.prompts[0]
    ui.kv("prompt 组装", " / ".join(
        s.strip()[:22] for s in prompt.replace("\n\n", "\n").split("\n") if s
    ))
    ui.assert_true(prompt.strip().startswith("（USER）"), "USER 在最前")
    ui.assert_in("（profile BASE）", prompt, "BASE 在中间（替换 authored base）")
    ui.assert_in("（profile SUFFIX）", prompt, "SUFFIX 在后")
    ui.kv("delete description", silk.tool_descriptions.get("delete", "")[:50])
    ui.assert_in("Override'd by profile", silk.tool_descriptions.get("delete", ""),
                 "内置工具的 description 被 profile 覆盖")
    ui.note("顺序 = caller USER -> profile BASE -> profile SUFFIX，空行分隔。")
    ui.note("BASE 是 replace 语义：配置了就替换 authored base；SUFFIX 永远 append。")
    ui.note("d11 那个细节在这里有了意义：profile 的覆盖层会 copy+rewrite dict 工具")
    ui.note("（deepagents._tools），但裸 dict 本就不被注册，所以真正能落地改的是")
    ui.note("BaseTool/内置工具 —— 对用户自定义 @tool 同样生效。")


def demo_excluded_tools() -> None:
    ui.subtitle("3. excluded_tools = 请求期过滤，**不是**构造期移除，也不是安全边界")
    silk = PromptSilk()
    register_harness_profile(
        "cheap:profile-demo",
        HarnessProfile(excluded_tools=frozenset({"write_file"})),
    )
    model: ScriptedModel = choose_model(
        [
            tool_call("write_file", {"file_path": "/tmp/x.txt", "content": "hi"}, "c1"),
            say("我换个方式。"),
        ],
        None,
    )
    model.model_name = "profile-demo"
    agent = create_deep_agent(model=model, middleware=[silk])
    ui.kv("注册表工具集（_tools_by_name）", ", ".join(bound_tool_names(agent)))
    ui.assert_in("write_file", bound_tool_names(agent), "工具**仍然注册着**（没被构造期移除）")
    result = agent.invoke({"messages": [{"role": "user", "content": "写个文件"}]})
    blocker = next((m for m in result["messages"] if m.type == "tool"), None)
    ui.kv("模型调 write_file 后", str(blocker.content).replace("\n", " ")[:70] if blocker else "无")
    ui.assert_true(
        blocker is not None and "not available" in str(blocker.content),
        "请求期被 _ToolExclusionMiddleware 拒绝",
    )
    ui.note("看 docs 原文：exclusion 不是 security boundary；模型还是可以拿到工具")
    ui.note("（executor 残留），只是广告/请求期不让它碰 —— 换个入口就能绕过。")
    ui.note("高危工具请用 d09 的 permissions 加白名单，别只靠在 harness 上 excluden。")


def demo_excluded_middleware() -> None:
    ui.subtitle("4. excluded_middleware = 保护核心机制，乱排会直接炸")
    try:
        HarnessProfile(excluded_middleware=frozenset({FilesystemMiddleware}))
    except ValueError as exc:
        ui.kv("构造期拦截", f"HarnessProfile.__post_init__ 抛 ValueError")
        ui.assert_true(True, "protected 中间件（Filesystem/SubAgent）构造期就被拒")
    else:  # pragma: no cover
        ui.assert_true(False, "应该被拦截")
    ui.note("FilesystemMiddleware 和 SubAgentMiddleware 是核心脚手架：")
    ui.note("（1）HarnessProfile.__post_init__ 直接拒绝（required scaffolding 不能删）；")
    ui.note("（2）create_deep_agent 组装时还会再验一遍 _validate_excluded_middleware_config。")
    ui.note("要取消自动 task：disable general-purpose subagent 且不提供同步子代理即可。")
    ui.note("另一条：HarnessProfileConfig 只含声明式字段（middleware 只支持名字字符串），")
    ui.note("runtime-only 的 extra_middleware 不能导出到 config（导出会抛错而不是甩掉）。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d14 * Profiles ------ ProviderProfile + HarnessProfile layered overlays",
        feature="Profiles",
        principle="#3 Model-agnostic * #4 Production-ready",
    )
    demo_provider_profiles()
    demo_harness_profiles()
    demo_excluded_tools()
    demo_excluded_middleware()


if __name__ == "__main__":
    main()