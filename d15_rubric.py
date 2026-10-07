"""d15 · RubricMiddleware —— 让 agent 按你自己的标准"自评-修订-再交卷"

覆盖 Feature   : Rubric (self-evaluated iteration / quality gate)
覆盖 Principle : #1 Opinionated（内置 grader 流程）· #2 Extensible（_invoke_grader 是扩展点）
覆盖要点       : 与前几章关系——输出不是"模型说什么就交什么"，而是"过一遍检查再交"。

---------------------------------------------------------------------------
依据（deepagents.middleware.rubric 源码 docstring + 本机实测）
---------------------------------------------------------------------------
1. **触发**：调用方在 invocation state 里给 `rubric` 字符串；不给，before_agent
   和 after_agent 都当 no-op —— 所以担心性能/副作用的话，`RubricMiddleware`
   可以无条件放在栈里。

2. **流程**：模型自然停（没有更多工具调用）-> 中间件另起一个 **独立的 grader 子代理**
   对着 transcript 打分。grader 返回 `needs_revision` 时，把 feedback 注入成一条
   `HumanMessage`（不是把 grader 的话直接塞给用户），然后 `jump_to='model'` 让
   agent 按反馈重写；直到 `satisfied` / `failed` / 达到 `max_iterations`。

3. **grader 用的是它自己的模型**（构造时传入 model=），和主 agent 互相独立
   —— 生产里可以"主模型大、评审模型小"，或主模型是内部的、评审用最强的。

4. **结束形态**：不满足时框架**不改写**最后一条 AIMessage，只会：
   - 在 state 里写 `_rubric_status`（satisfied / failed / max_iterations_reached /
     grader_error），
   - 逐个触发 `on_evaluation` 回调 / `rubric_evaluation_end` 流事件。

本章为了可离线、可断言，使用官方文档点名的扩展点 `_invoke_grader` 来替代
grader 子代理本身（`_grade` 里的覆盖重试、判定门、跳回逻辑仍全部走真实代码）。
"""

from __future__ import annotations

import sys

from deepagents import create_deep_agent
from deepagents.middleware import RubricMiddleware
from deepagents.middleware.rubric import GraderResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from shared import ui
from shared.offline import ScriptedModel, choose_model, say


class _ScriptedGrader(RubricMiddleware):
    """确定性 grader：verdicts 里依次取结果，替代真实 grader 子代理。

    真实场景直接用 `RubricMiddleware(model="openai:gpt-5.1")` 让它起一个
    真正的评审子代理；这里 override `_invoke_grader`（官方文档里的
    "per-call extension point"），把判定序列写死，保证离线可复现。
    """

    def __init__(self, verdicts, max_iterations: int = 3, on_evaluation=None):
        super().__init__(model=ScriptedModel(script=[]), max_iterations=max_iterations,
                         on_evaluation=on_evaluation)
        self.verdicts = list(verdicts)
        self.calls = 0

    def _invoke_grader(self, state, iteration, correction=None, *, context=None):
        result, explanation = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        criterion = {"name": "criteria", "passed": result == "satisfied"}
        if result != "satisfied":
            # CriterionFail 必填 gap（判定分支的差分校验字段）
            criterion["gap"] = explanation
        return GraderResponse(
            result=result,
            explanation=explanation,
            criteria=[criterion],
        )


def demo_loop() -> None:
    ui.subtitle("1. 典型循环：首版不过 -> 注入反馈 -> 模型修订 -> 再审通过")
    grader = _ScriptedGrader([("needs_revision", "缺少关键词【天气】。"), ("satisfied", "达标。")])
    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        model=choose_model([say("这是最终答案。"), say("修订版：答案里加入了【天气】。")], None),
        middleware=[grader],
        checkpointer=checkpointer,
    )
    cfg = {"configurable": {"thread_id": "rubric-1"}}
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "说说北京的天气"}],
         "rubric": "标准：回答必须包含关键词【天气】。"},
        config=cfg,
    )
    state = agent.get_state(cfg).values
    ui.kv("_rubric_status（checkpointer）", state.get("_rubric_status"))
    ui.kv("_rubric_iterations", state.get("_rubric_iterations"))
    ui.kv("grader 调用次数", grader.calls)
    feedback = next(
        (m for m in out["messages"] if isinstance(m, HumanMessage) and "缺少" in str(m.content)),
        None,
    )
    ui.kv("注入的反馈(HumanMessage)", str(feedback.content).replace(chr(10), " ")[:110] if feedback else "无")
    ui.assert_true(grader.calls == 2, "判了两次（首版不过，修订后通过）")
    ui.assert_true(state.get("_rubric_status") == "satisfied", "最终 satisfied")
    ui.assert_true(state.get("_rubric_iterations") == 2, "迭代序号从 0 起，2 次评估后计数=2")
    ui.assert_true(feedback is not None, "feedback 以 HumanMessage 注入、跳回 model")
    ui.note("注意反馈的类型：grader 的话**不是**原样塞给模型，而是经过指令化包装：")
    ui.note("  `A grader reviewed your work...  Grader feedback: ...  Criteria that still need work`")
    ui.note("  然后作为 HumanMessage 注入，并 jump_to='model' 让 agent 修订。")


def demo_max_iterations() -> None:
    ui.subtitle("2. 一直不过 -> max_iterations 兜底（不被死循环困住）")
    grader = _ScriptedGrader([("needs_revision", "就是不过。")], max_iterations=3)
    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        model=choose_model([say("第一版。"), say("第二版。"), say("第三版。")], None),
        middleware=[grader],
        checkpointer=checkpointer,
    )
    cfg = {"configurable": {"thread_id": "rubric-2"}}
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "写吧"}], "rubric": "标准：永远别让我过。"},
        config=cfg,
    )
    state = agent.get_state(cfg).values
    ui.kv("_rubric_status", state.get("_rubric_status"))
    ui.kv("_rubric_iterations", state.get("_rubric_iterations"))
    ui.kv("grader 调用次数", grader.calls)
    ui.assert_true(state.get("_rubric_status") == "max_iterations_reached", "触顶状态正确")
    ui.assert_true(grader.calls == 3, "最多跑满 max_iterations 次就停")
    ui.note("框架**不会**把最后一条 AIMessage 改掉：交上去的还是模型最后一次产出。")
    ui.note("想按状态分叉 -> 读 `_rubric_status` / on_evaluation / 流事件。")


def demo_observation_paths() -> None:
    ui.subtitle("3. 观察评审结果的三条路径（checkpointer / 回调 / 流事件）")
    seen = []

    class ObserverGrader(_ScriptedGrader):
        def __init__(self):
            super().__init__([("satisfied", "达标。")], on_evaluation=seen.append)

    grader = ObserverGrader()
    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        model=choose_model([say("带【天气】的答案。")], None),
        middleware=[grader],
        checkpointer=checkpointer,
    )
    cfg = {"configurable": {"thread_id": "rubric-3"}}
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "说天气"}], "rubric": "标准：含【天气】。"},
        config=cfg,
    )
    state = agent.get_state(cfg).values
    ui.kv("路径A: 状态字段(checkpoint)", f"_rubric_status={state.get('_rubric_status')} "
                                       f"_rubric_iterations={state.get('_rubric_iterations')}")
    ev = seen[0] if seen else {}
    ui.kv("路径B: on_evaluation", f"run_id={ev.get('grading_run_id','')[-6:]} "
                                  f"result={ev.get('result')} unverified={ev.get('unverified')}")
    ui.assert_true(state.get("_rubric_status") == "satisfied", "checkpointer 上能读到私有字段")
    ui.assert_true(bool(seen) and seen[0].get("result") == "satisfied", "回调拿到 RubricEvaluation")
    ui.assert_in("grading_run_id", seen[0] if seen else {}, "回调里有 run id（重跑可追踪）")
    ui.note("路径C: stream 事件 `rubric_evaluation_end`（文档写的名字，stream 模式可订阅）。")
    ui.note("私有字段（PrivateStateAttr）不进公开 state 输出，这正是要挂 checkpointer 的原因。")


def demo_unconditional_safe_and_notes() -> None:
    ui.subtitle("4. 无 rubric = no-op；grader 模型要独立实例")
    grader = _ScriptedGrader([("satisfied", "不会触发。")])
    agent = create_deep_agent(
        model=choose_model([say("普通回答。")], None),
        middleware=[grader],
    )
    out = agent.invoke({"messages": [{"role": "user", "content": "随便聊聊"}]})
    ui.kv("没有 rubric 时", f"status={out.get('_rubric_status')} grader 调用={grader.calls}")
    ui.assert_true(grader.calls == 0, "不传 rubric -> after_agent 直接 return None")
    ui.note("strategy 备注（源码函数 _strategy_from_model，模型无关的又一佐证）：")
    ui.note("  * grader 的 structured-output 策略 = ToolStrategy / ProviderStrategy，")
    ui.note("    由模型的 profile 与标识符自动选择（Claude/OpenAI 走各自原生路径）。")
    ui.note("  * 我们 override _invoke_grader 时**完全不碰聊天协议**——这正说明了中间件")
    ui.note("    把它该有的边界守住了：策略适配永远在 grader 内部，上层只认 GraderResponse。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d15 * Rubric --------- self-evaluated revision loop (quality gate)",
        feature="Rubric / self-eval iteration",
        principle="#1 Opinionated * #2 Extensible * #3 Model-agnostic",
    )
    demo_loop()
    demo_max_iterations()
    demo_observation_paths()
    demo_unconditional_safe_and_notes()


if __name__ == "__main__":
    main()