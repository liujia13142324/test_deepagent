"""共享基础设施：模型解析、离线回放、控制台渲染。

这个包本身也在演示 deepagents 的两条 principle：

* **Model-agnostic** —— `config.model_spec()` 只是一个字符串；provider 包没装、
  key 没有，就把演示降级成"离线回放"，不改变任何 agent 代码。
* **Production-ready** —— LangSmith 追踪只靠环境变量打开，业务代码零侵入。
"""

from __future__ import annotations

import sys

# 某些字符（em-dash、箭头、emoji）在 GBK 控制台上无法编码。
# 保留控制台原生编码，只把错误处理改成 replace，避免抛 UnicodeEncodeError；
# 真正不可显示的字符由 shared.ui._ascii_safe 提前降级成 ASCII。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 —— 非 tty 时可能不支持
        pass

__all__ = ["config", "offline", "ui"]
