# 第 1 课设计 spec — 一个 request 的完整旅程：从 `llm.generate` 到 token 输出

- 日期：2026-06-25
- 状态：设计已确认（用户 approve），待 spec 复核
- 课程：nano-vllm 推理全流程课程（v2，8 课，M1/M2）
- 关联：`course/README.md`、`course/lessons/L01.md`（v2 大纲）、本 spec 驱动第 1 课的**可运行内容**实现

---

## 1. 背景与上下文

- repo 下已存在 v2 课程大纲（`course/README.md` + `course/lessons/L01.md`–`L08.md`），8 课、已收敛（5 轮 review pass）。v2 是"大纲 + 覆盖内容"层，**可运行 lab 脚手架当时明确不在范围内**。
- 受众：**算子开发者**（系统 / 内核 / 推理优化背景，懂推理原理、会读 Python、但不熟 vLLM / nano-vllm）。与 v2 一致（用户已澄清："算子开发，不是算法开发"——最初任务描述里的"算法开发"是误写）。
- 本次任务：把第 1 课从"大纲"推进为"可运行内容"——正式 TUTORIAL + 可跑 lab + 可跑验证。

## 2. 方向决策（关键，已与用户确认）

**走 v2 的 trace / observe / explain，不挖空核心函数。** 理由：

1. 受众是算子开发者，最关心整条执行流与成本 / 决策落点；trace / observe 直接暴露这些。
2. nano-vllm 的精髓（paged KV、GQA、TP 分片不传输）适合"看它跑"来理解；拆空重写反而丢整体直觉，且卡壳风险高。
3. 用户曾拒绝 v1 的"逐函数留空"粒度，v2 高粒度是有意收敛的结果。

"代码框架" = `lab.py` 提供 **monkeypatch hook 壳** + `# TODO` 标记 + **一个自包含小 wrapper 桩**；**框架源码一行不改**。

用户最初任务描述里的"动手补全关键逻辑 / 待补全函数 / 单测"被重新理解为：**轻量动手（插桩 + 一个自包含小函数）+ 可运行验证**，而非挖空核心逻辑重写。这条重新理解已与用户对齐。

## 3. 目标 / 非目标

**目标**（学完第 1 课，算子开发者能）：

- 说清一个 request 从 `llm.generate` 到输出 token 的完整旅程；
- 看懂 `step()` 三段式（schedule → run → postprocess）；
- 理解 `Sequence` 作为主进程 ↔ GPU worker 的唯一 IPC 载体（`__getstate__` / `__setstate__` 精简序列化）；
- 解释 `num_tokens` 正负号如何同时编码 prefill / decode 双语义并驱动两条吞吐曲线。

**非目标**：

- 不重写 / 挖空任何 nano-vllm 核心函数；
- 不讲 KV cache 内部、连续批处理、TP（留给 L2–L7）；
- 不做性能优化（profiling 留给 L8）。

## 4. 受众与前置依赖

- 受众前提：懂 LLM 推理原理、会读 Python（不必精通）、不熟 vLLM。
- 环境前置（lab 真跑 nano-vllm 必需，base 环境当前缺，需学生先装）：
  - `uv`（本机已装 0.8.9）；`uv sync` 用根 `pyproject.toml`（已声明 `torch>=2.4` / `triton>=3.0` / `transformers>=4.51` / `flash-attn` / `xxhash`）。
  - 下载 Qwen3-0.6B 到 `~/huggingface/Qwen3-0.6B/`。
  - `flash-attn` 若 `uv sync` 装不上，给 pip wheel 备选命令（实现时按目标 CUDA / torch 版本定）。
  - 一张 CUDA 卡（`enforce_eager=True`, `tensor_parallel_size=1`）。
- `lab.py` 第一步 **fail-fast 探环境**：缺 torch / flash_attn / 模型时打清晰提示退出，不甩 import 报错。

## 5. 文件布局（最小化）

```
course/lessons/lesson1/
├─ TUTORIAL.md            # 正式讲义 + §0 环境初始化
├─ lab.py                 # 3 任务（monkeypatch hook + TODO + 小 wrapper），源码不动
└─ solution/ANSWERS.md    # Task2 explain 参考答案 + 期望 trace（学生不可见）
```

不新建 pyproject（复用根）。环境用 `uv sync`。

## 6. TUTORIAL.md 结构

- **§0 环境初始化**：`uv sync` → 下模型 → `uv run python example.py` 跑通（smoke）。
- **§1 目标 & 受众前提**（同 §3 目标 + 受众说明，明确"假设你懂推理原理、会读 Python，但不熟 vLLM"）。
- **§2 全景图**：`generate → add_request → [while not is_finished: step()] → schedule → run → postprocess → decode → output` 总图 + Sequence 状态机图 `WAITING → RUNNING → FINISHED`。
- **§3 逐段讲（锚 file:line，本 session 已核验）**：
  - `generate()` `llm_engine.py:60-90`
  - `add_request → Sequence` `llm_engine.py:43-48`, `sequence.py:18-31`
  - `step()` 三段式 `llm_engine.py:49-55`
  - IPC：`Sequence.__getstate__` / `__setstate__` `sequence.py:72-83`（decode 传单个 `last_token`，prefill 传完整 `token_ids`）
  - `postprocess` 收尾 `scheduler.py:81-92`（`hash_blocks` → `num_cached_tokens` 累加 → prefill 未完 `continue`（`scheduler.py:86-87`）→ `append_token` → eos / max_tokens 终止（`scheduler.py:89`）→ `FINISHED` + `deallocate`）
  - **关键非显然点（本课重点观察）**：prefill 步若一次算完（`num_cached_tokens == num_tokens`），`continue` 不触发，`append_token` 照常执行——即 **prefill 步也会产出第 1 个 completion token**（`scheduler.py:86-88`）。推论：seq 参与的每个 step（含那唯一一次 prefill）都 append 恰好 1 个 token ⇒ **总参与步数 == `num_completion_tokens`**。
  - `num_tokens` 正负号 `llm_engine.py:51`，驱动两条吞吐曲线 `llm_engine.py:76-79`
  - 三态机 `sequence.py:8-11`
- **§4 进 lab**：指向 `lab.py` 三个任务。
- **§5 图示补充**：状态机图、step 时序图、prefill vs decode 的 `num_scheduled_tokens` 对比。

## 7. lab.py 三任务

**通用机制**：`lab.py` 顶部 monkeypatch `Scheduler.postprocess` 为 `traced_postprocess`，源码不改。捕获点选在 `postprocess` **入口**——此时每个 seq 的 `num_scheduled_tokens` 尚未被清零（`scheduler.py:85` 在 postprocess 内部才清），可完整记录本步数据。

- **Task 1 · trace**：`traced_postprocess` 壳已给，`# TODO` 处让学生记录每步 `(is_prefill, [(seq.seq_id, seq.num_scheduled_tokens, seq.status) for seq in seqs], token_ids)` 进 `_trace`；跑 example.py 两条 prompt，打印每个 seq 的状态轨迹表。**预期观察**：两条 prompt 在**同一步**被一起 prefill（`schedule` 的 while 循环把 waiting 里能装的都装进来，`scheduler.py:30`），之后交替 decode；且 prefill 那步 `token_ids` 非空（产出第 1 个 token，见 §3 关键点）。
- **Task 2 · observe + explain**：单条 prompt `enforce_eager=True`。从 `_trace` 标注：prefill 步 `num_scheduled_tokens == num_prompt_tokens`；decode 步 `== 1`。学生在 lab 注释区写 explain（`num_tokens` 正负号双语义 + 驱动两条吞吐曲线）。
- **Task 3 · 小 wrapper**：实现 `summarize_request(seq) -> (num_prompt_tokens, num_completion_tokens, total_steps)`，`total_steps` 从 `_trace` 数该 seq 参与的步数；`generate` 返回前对每条 seq 打印。

## 8. 验证（内建，不另起 pytest 目录）

`uv run python lab.py` 跑完末尾断言：

- **Task1**：每 seq 轨迹 = `WAITING → RUNNING → FINISHED`；`is_prefill=True` 步每 seq 恰好 1 次；**总参与步数 == `num_completion_tokens`**（每步含 prefill 都 append 一个 token，见 §3），故 decode 步数 == `num_completion_tokens - 1`。
- **Task2**：prefill 步 `num_scheduled_tokens == num_prompt_tokens`；decode 步 `num_scheduled_tokens == 1`。
- **Task3**：`num_completion_tokens == len(seq.completion_token_ids)` 且 `≤ max_tokens`；`total_steps == num_completion_tokens`（见 §3 关键点；对本课短 prompt 单次 prefill 成立）。

全过打印 `All checks passed ✓`。Task2 explain 不可自动判 → `solution/ANSWERS.md` 参考 + TUTORIAL 判分要点。

## 9. 角色流程与执行模型

三角色循环（写进本 spec 当执行模型，在 writing-plans / 执行阶段落地）：

1. **框架开发工程师**：按本 spec 产出 `TUTORIAL.md` + `lab.py` + `solution/ANSWERS.md`，并真跑 `uv run python lab.py` 自证 `All checks passed ✓`。
2. **课程架构审核工程师**：审目标清晰度 / 知识递进 / tut:lab 比例 / lab 是否真助理解 / 每任务输入·输出·验证是否明确 / file:line 锚点是否准（通读对应源码段）。
3. **学生**：真跑 lab，从学习者视角记卡点（概念跳步 / 接口不清 / 验证不足 / 环境配置 / 错误提示不友好）。
4. 反馈 → 工程师改 → 审核再审 → 学生再试，到无明显问题。

**enactment 待 spec 复核时定**：(a) 真用 subagent 跑三角色循环（更忠实、更费 token）；(b) 我在本会话一人分饰三角（更轻）。

## 10. 成功标准（交付定义）

第 1 课可交付 = 同时满足：

- `TUTORIAL.md` 含 §0–§5，锚点 file:line 准；
- `lab.py` 三任务 TODO 清晰，`uv run python lab.py` 在装好依赖的机器上跑通并打印 `All checks passed ✓`；
- fail-fast 环境探测可用；
- 至少一轮"审核 pass + 学生试课无阻塞性卡点"。

## 11. 开放项 / 待决

- **enactment**：subagent 循环 vs 一人分饰（§9）。
- **Task1 插桩方式**：默认 monkeypatch hook（源码不改、可分发）；备选直接在源码加 print（更直白但改源码）。默认 monkeypatch，若审核 / 学生反馈对"Python 不算精通"者太难则切换。
- **flash-attn 安装兜底命令**的精确形式（依目标 CUDA / torch 版本，实现时定）。
- **多卡（tp>1）**不在第 1 课范围（tp=1 即可）。

## 12. 源码锚点清单（本 session 已逐个核验）

| 内容 | 位置 |
|---|---|
| `generate` | `llm_engine.py:60-90` |
| `add_request` | `llm_engine.py:43-48` |
| `step` 三段式 / `num_tokens` 正负号 | `llm_engine.py:49-55`（51 行） |
| 吞吐双曲线 | `llm_engine.py:76-79` |
| `SequenceStatus` 三态 | `sequence.py:8-11` |
| `Sequence` 数据结构 | `sequence.py:18-31` |
| `num_completion_tokens` | `sequence.py:43-45` |
| IPC `__getstate__` / `__setstate__` | `sequence.py:72-83` |
| `postprocess` 收尾（`continue`@86-87，终止@89） | `scheduler.py:81-92` |
| prefill 单次吃完（`num_scheduled_tokens=min(num_tokens, remaining)`） | `scheduler.py:46` |
| decode `num_scheduled_tokens=1` | `scheduler.py:67` |
| `max_num_batched_tokens` 默认 16384 / `max_num_seqs` 512 | `config.py:9-10` |
| `kvcache_block_size` 默认 256 | `config.py:17` |
