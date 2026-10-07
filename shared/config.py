"""模型解析 —— Principle: Model-agnostic。

官方 FAQ 原文：

    Yes. Any model that supports tool calling works — frontier APIs
    (OpenAI, Anthropic, Google), open-weight models hosted on providers like
    Baseten or Fireworks, and self-hosted models via Ollama, vLLM, or
    llama.cpp.

以及 docs 里的实现约束（`concepts/profiles-models.md`）：

* `create_deep_agent` **先** resolve model，**再** 选 harness profile。
* `model` 接受 `provider:model` 字符串（走 `init_chat_model`）或已初始化的
  `BaseChatModel` 实例。
* 只有**第一个冒号**是结构分隔符（模型名本身可以带冒号，比如 Bedrock ARN）。
* 不要传 `model=None`：它会落到 `ChatAnthropic(claude-sonnet-4-6)`，
  并已在 `deepagents==1.0.0` 标记为将移除。
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from dotenv import load_dotenv

load_dotenv(override=False)

#: 默认模型。用 `DEEPAGENTS_MODEL` 覆盖即可切换 provider。
DEFAULT_MODEL_SPEC = "anthropic:claude-sonnet-4-6"

#: `provider:model` -> 该 provider 需要的环境变量。
PROVIDER_KEY_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_CLOUD_PROJECT",
    "ollama": "",  # 本地模型不需要 key
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
}


def model_spec() -> str:
    """返回 `provider:model` 形式的模型标识。

    注意"只有第一个冒号是结构分隔符"这一实现细节：
    `bedrock:arn:aws:bedrock:...::model/x` 也能正确拆出 provider。
    """
    return os.environ.get("DEEPAGENTS_MODEL", DEFAULT_MODEL_SPEC)


def provider_of(spec: str) -> str:
    """拆出 provider（第一个冒号之前）。"""
    return spec.split(":", 1)[0] if ":" in spec else spec


def has_credentials(spec: str | None = None) -> bool:
    """该 provider 的 key 是否就位（本地 provider 视为永远就位）。"""
    spec = spec or model_spec()
    var = PROVIDER_KEY_VARS.get(provider_of(spec))
    if var == "":
        return True
    return bool(var and os.environ.get(var))


def tracing_enabled() -> bool:
    """Principle: Production-ready —— 追踪只靠环境变量，业务代码不感知。"""
    return os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}


def should_go_offline(flags: Sequence[str] | None = None) -> bool:
    """决定走"离线确定性回放"还是"真实 LLM 调用"。

    调用方一律写成：

        model = choose_model(SCRIPT, argv)

    这样同一个 demo 文件既能真跑（`python d01_quickstart.py`），
    也能在 CI / 无 key 环境里被 `run_all.py` 完整验证（`--offline`）。

    优先级：`--offline` > `--online` > `DEEPAGENTS_FORCE_OFFLINE` > 有无 key。
    `flags` 显式传入时**优先**于 `sys.argv`，这样 `run_all.py` 逐个调用
    `main(["--offline"])` 才能真正生效（否则真实 sys.argv 里没有这个 flag）。
    """
    import sys

    pool = list(flags or []) + list(sys.argv)
    if "--offline" in pool:
        return True
    if "--online" in pool:
        return False
    if os.environ.get("DEEPAGENTS_FORCE_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return True
    return not has_credentials()
