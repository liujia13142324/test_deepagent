# 团队长期记忆（Principle: Persistent memory）

这个文件通过 `create_deep_agent(memory=["/memory/AGENTS.md"], ...)` 加载，
`MemoryMiddleware` 会在 `before_agent` 读取它，并把内容注入 system prompt。

文档中的关键语义（`concepts/middleware-catalog.md` / `concepts/subagents-skills.md`）：

1. **一次性加载**：内容进入私有 state 字段 `memory_contents`，每个 thread 只读一次。
2. **每次请求都注入**：`system_prompt=None` 可以关掉 prompt 注入，但**不会**关掉加载。
3. **默认来源**是 `AGENTS.md`；显示名由路径自动推导。
4. **信任模型很重要**：内存被注入的 prompt 明确告诉模型——把它当作
   "磁盘上的参考资料"，而不是"凌驾于用户之上的隐藏系统指令"。

## 约定

- 输出所有面向用户的数字时，保留 2 位小数。
- 代码示例用 TypeScript 优先，Python 作为备选。
- 任何对外承诺（日期、价格、SLA）必须先经 `write_file` 落盘再确认。

<!-- HTML 注释在渲染时会被 MemoryMiddleware 剥掉，不会进 prompt。 -->

## 反面清单

- 不要把密钥、token、密码写进任何文件、记忆或 system prompt。
- 不要在没有 `read_file` 验证的情况下断言某个文件的内容。
