#!/usr/bin/env python3
"""run_all.py · 全量回归入口 —— 与 README 的“一条命令跑完”约定对应

逐个把 d01..d16 当作子进程跑 `python dXX_*.py --offline`，
收集退出码与 `[x]`（失败断言）计数：exit!=0 或出现 [x] 都算失败。

用子进程而非 in-process import，原因：
- 每个 demo 里都可能注册 profile / middleware / 建同名图，进程隔离最稳；
- 失败是“这一个 demo”失败，而不是连带污染后面的。

用法：
    python run_all.py            # 全量
    python run_all.py 05 12      # 只跑指定的两个
"""

from __future__ import annotations

import glob
import re
import subprocess
import sys

DEMO_GLOB = "d*.py"
EXCLUDE = {"run_all.py"}


def demo_files() -> list[str]:
    return sorted(
        [f for f in glob.glob(DEMO_GLOB) if f not in EXCLUDE and not f.startswith("_")]
    )


def run_one(path: str, offline: bool = True) -> tuple[str, int, int]:
    cmd = [sys.executable, path] + (["--offline"] if offline else [])
    proc = subprocess.run(cmd, capture_output=True)
    output = proc.stdout.decode("utf-8", errors="replace")
    output += "\n" + proc.stderr.decode("utf-8", errors="replace")
    # 只数“断言失败”的 [x]：行首（带缩进）出现 [x]；文件内容里的勾选框
    # 形如 `- [x] 补测试`，行首是 `-`，不会误报。
    failed_asserts = len(re.findall(r"(?m)^[ \t]*\[x\]", output))
    return path, proc.returncode, failed_asserts


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    files = demo_files()
    if argv:
        wanted = {"d%s_" % n for n in argv}
        files = [f for f in files if f[:3] in wanted]
    print("=" * 78)
    print("Deep Agents demo 全量回归  （共 %d 个）" % len(files))
    print("=" * 78)
    failed: list[tuple[str, int]] = []
    for path in files:
        code, xcount = run_one(path)[1:]
        mark = "OK " if code == 0 and xcount == 0 else "** FAIL **"
        print(f"  [{mark}] {path:28s} exit={code} failed_asserts={xcount}")
        if code != 0 or xcount:
            failed.append((path, xcount))
    print("-" * 78)
    if failed:
        print("失败 %d 个：" % len(failed))
        for path, xcount in failed:
            print(f"  - {path}  ([x] 数 = {xcount})")
        sys.exit(1)
    print("全部通过  (%d 个 demo，exit=0 且无 [x])" % len(files))


if __name__ == "__main__":
    main()