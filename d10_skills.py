"""d10 · Skills —— 渐进式披露的"能力包"

覆盖 Feature   : Skills（SKILL.md + 渐进式披露 + 多来源）
覆盖 Principle : #1 Opinionated（自带使用说明）· #2 Extensible

---------------------------------------------------------------------------
文档依据（concepts/subagents-skills.md + SkillsMiddleware 源码）
---------------------------------------------------------------------------
`create_deep_agent(skills=["/path/to/skills/"])` 的工作方式：

1. 扫描目录里所有 `SKILL.md`，抽出 **name + description**，
   把一张清单注入 system prompt（**只给名字和描述**）。
2. agent 判断某个 skill 匹配任务后，用 `read_file` 自己去读完整 SKILL.md。
3. SKILL.md 里可以引用脚本/参考文件，agent 再按需读。

这就是 progressive disclosure：常驻上下文只有几十个 token，
需要时才把几百上千行 instructions 拉进来。

prompt 里还会明确告诉 agent：
* 用 `read_file(file_path=..., limit=1000)` 读 skill 文件（默认 100 行不够用）；
* 技能脚本要用**绝对路径**执行。

本工程的 skill 样例见 `assets/skills/`：
* `quarterly-report/` —— 写季度报告的流程型 skill
* `csv-audit/` —— 数据审计的检查清单型 skill
"""

from __future__ import annotations

import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware

from shared import ui
from shared.offline import choose_model, say, seed_files, tool_call

SKILL_DIR = Path(__file__).parent / "assets" / "skills"
PROJECT_ROOT = Path(__file__).parent

#: 关键实现细节：SkillsMiddleware 是**从 backend 读**技能的，不是从宿主机文件系统。
#: 默认 StateBackend 里没有 /assets/skills，所以要显式给一个 FilesystemBackend，
#: 并用**虚拟路径**（相对 root_dir）作为 source。
VIRTUAL_SKILLS = "/assets/skills"


class PromptSpy(AgentMiddleware):
    """抓取每次模型请求的 system prompt —— 用来"看见"注入内容。

    这同时是 d13 middleware 的最小示例：`wrap_model_call` 拿到 request，
    改一改再 `return handler(request)`。
    """

    def __init__(self) -> None:
        self.captured: list[str] = []

    def wrap_model_call(self, request, handler):
        self.captured.append(str(request.system_prompt or ""))
        return handler(request)


def demo_skill_registry() -> None:
    ui.subtitle("1. 扫描 skills 目录：只有 name + description 进上下文")
    found = sorted(p.parent.name for p in SKILL_DIR.glob("*/SKILL.md"))
    ui.kv("发现的 skill", found)
    for name in found:
        header = (SKILL_DIR / name / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
        for line in header.strip().splitlines():
            if line.strip().startswith(("name:", "description:")):
                ui.kv("  " + line.split(":")[0], line.split(":", 1)[1].strip()[:64])
    ui.note("frontmatter 里的 name/description 就是模型看到的全部常驻信息。")


def demo_injection(model) -> None:
    ui.subtitle("2. 注入 system prompt（用 PromptSpy 抓下来看）")
    spy = PromptSpy()
    agent = create_deep_agent(
        model=model,
        tools=[],
        skills=[VIRTUAL_SKILLS],
        backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True),
        middleware=[spy],
    )
    agent.invoke({"messages": [{"role": "user", "content": "帮我出一份季度报告"}]})
    prompt = spy.captured[0]
    ui.kv("system prompt 长度", f"{len(prompt)} chars")
    ui.step("注入的技能清单片段：")
    for line in prompt.splitlines():
        if "skill" in line.lower() or line.strip().startswith(("- ", "* ")):
            ui.note("  " + line[:96])
    ui.assert_in("quarterly-report", prompt, "技能名出现在 system prompt")
    ui.assert_in("csv-audit", prompt, "两个技能都在清单里")
    ui.assert_true(
        "Detailed Instructions" not in prompt and "## Workflow" not in prompt,
        "SKILL.md 的正文没有进上下文（这正是渐进式披露）",
    )
    ui.note("注意 SkillsMiddleware 是从 **backend** 读技能的：默认 StateBackend 里没有")
    ui.note("/assets/skills，所以必须配一个 FilesystemBackend 才能发现（docs 实现细节）。")


def demo_progressive_disclosure(model) -> None:
    ui.subtitle("3. 第二步：agent 自己去 read_file 拉完整指令")
    result = create_deep_agent(
        model=model,
        tools=[],
        skills=[VIRTUAL_SKILLS],
        backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True),
    ).invoke({"messages": [{"role": "user", "content": "先读 quarterly-report 技能再动手"}]})
    ui.show_messages(result)
    read_msg = next((m for m in result["messages"] if m.type == "tool"), None)
    ui.assert_true(read_msg is not None, "agent 调用了 read_file 读取 SKILL.md")
    if read_msg is not None:
        body = str(read_msg.content)
        ui.kv("读到的正文长度", f"{len(body)} chars")
        ui.step("正文首部：")
        for line in body.splitlines()[:5]:
            ui.note("  " + line[:90])
    ui.note("注意 read_file 用了 limit=1000：默认窗口 100 行对 skill 文件不够用，")
    ui.note("这也是 SkillsMiddleware 默认 system_prompt 里特意叮嘱的一句。")
    ui.note("另一个实现细节：技能元数据按 **thread 缓存**在 state['skills_metadata']，")
    ui.note("  增删改技能后要传 skills_metadata=None（或 update_state）才会重新扫描。")


def demo_multiple_sources() -> None:
    ui.subtitle("4. 多来源与来源标签")
    ui.note("skills=[...] 每项可以是裸路径，也可以是 (路径, 标签) 元组：")
    ui.note("    skills=['/skills/user/', ('/repo/.claude/skills', 'Project')]")
    ui.note("标签会渲染成 '**{label} Skills**'；后加载的来源覆盖先加载的同名技能。")
    ui.note("docs 的来源语义：")
    ui.note("  * 'Deepagents' —— 这个 agent 工具私有的技能")
    ui.note("  * 'Agents'    —— 这台机器上所有 agent 工具共享的技能")
    ui.note("同名冲突时以哪个为准取决于来源顺序，写文档时要约定好。")
    ui.note("限制：mode='fork' 的 subagent 禁止配 skills（与继承的 prompt 冲突，d05）。")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ui.banner(
        "d10 * Skills ---- progressive disclosure via SKILL.md",
        feature="Skills",
        principle="#1 Opinionated * #2 Extensible",
    )

    demo_skill_registry()

    demo_injection(choose_model([say("我会先看技能清单。")], argv))

    demo_progressive_disclosure(
        choose_model(
            [
                tool_call(
                    "read_file",
                    {"file_path": f"{VIRTUAL_SKILLS}/quarterly-report/SKILL.md", "limit": 1000},
                    "c1",
                ),
                say("已读取技能，开始执行流程。"),
            ],
            argv,
        )
    )

    demo_multiple_sources()
    _ = seed_files  # d02/d03 共用的 fixture 工具


if __name__ == "__main__":
    main()
