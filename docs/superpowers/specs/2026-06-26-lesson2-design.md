# 第 2 课设计 spec — KV 缓存的接口与物理形态：从 flash-attention 调用到 paged 张量

- 日期：2026-06-26
- 状态：设计已确认（用户 approve，含 D1=A / D2=A 两项决策），待 spec 复核
- 课程：nano-vllm 推理全流程课程（v2，8 课，M1/M2）
- 关联：`course/README.md`、`course/lessons/L02.md`（v2 大纲）、`course/lessons/lesson1/`（L1 范式）、`course/ROLES.md`（八维质量）；本 spec 驱动第 2 课的**可运行内容**实现

---

## 1. 背景与上下文

- repo 下已有 v2 课程大纲（`course/README.md` + `course/lessons/L01.md`–`L08.md`），8 课、已收敛（5 轮 review pass）。第 2 课大纲 `course/lessons/L02.md` 在**内容层已相当完整**（Q1/Q2/Q3、锚定源码、涵盖概念、验证方式都已写定）。
- **L1 已完整实现并合并 main**（`course/lessons/lesson1/`：TUTORIAL + lab + solution），是本课要复刻的**范式标准**（文件布局 / altitude / 验证双层 / 运行时语言）。
- 受众：**算子开发者**（系统 / 内核 / 推理优化背景，懂 LLM 推理原理、会读 Python、不熟 vLLM / nano-vllm）。与 v2 一致。
- 本次任务：把第 2 课从"大纲"推进为"可运行内容"——正式 TUTORIAL + 可跑 lab + 可跑验证，**严格沿用 L1 的标准**（用户原话："继续用第一节课的标准开发第二节课"）。
- 本 session 已**逐条 grep 核对** L02 全部 `file:line` 锚点对源码准确（见 §12），无漂移。

## 2. 方向决策（关键，已与用户确认）

**走 v2 的 trace / observe / explain + 一个小实现，不挖空核心函数。** 与 L1 同构。

1. 受众是算子开发者，最关心 flash-attention 两阶段调用契约与 paged 物理形态；trace / observe 直接暴露这些。
2. nano-vllm 的 KV 缓存精髓（paged 张量、slot_mapping 散射、store_kvcache 哨兵、prefix-cache 分支）适合"看它跑 + 手写一个几何函数"来理解；拆空重写丢整体直觉且卡壳风险高。
3. 用户曾拒绝 v1"逐函数留空"粒度，v2 高粒度是有意收敛的结果；本课"继续用 L1 的标准"= hook + 一个小实现 + observe/explain。

**大纲里 Q1/Q2/Q3 全是 observe + explain（连 Q1b 的 cu_seqlens 也是"手算核对"，不是函数实现）。** 本 spec 的关键补充：按 L1 标准补**一个动手 TODO 小实现** = `prefill_slot_mapping` 几何（用户已选），既补上 L1 式的"动手填一个小函数 + GPU-free 可判分点"，又恰好落在 L02 最难也最值的概念（paged slot_mapping 三段偏移几何）上。

"代码框架" = `lab.py` 提供 **monkeypatch hook 壳** + `# TODO` 标记 + **一个自包含小函数桩**；**框架源码一行不改**。

## 3. 目标 / 非目标

**目标**（学完第 2 课，算子开发者能）：

- 说清 flash-attention 在 prefill / decode 两阶段的**两个不同调用契约**：prefill 调 `flash_attn_varlen_func`（`q,k,v` packed + `cu_seqlens_q/k` + `max_seqlen_q/k` + 可选 `block_table`，`attention.py:67-70`）；decode 调 `flash_attn_with_kvcache`（`q.unsqueeze(1)` + `k_cache/v_cache` + `cache_seqlens=context_lens` + `block_table`，`attention.py:72-74`）。
- 解释 paged KV 张量物理形态 `(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)`（`model_runner.py:115`），其中 `num_kv_heads = num_key_value_heads // world_size`（TP 下每 rank 只存自己的 kv heads，`model_runner.py:110`），`num_kvcache_blocks` 由剩余显存反推（`model_runner.py:113`，**非构造期旋钮**）。
- 说清 `slot_mapping` 把 token 散射进 paged 缓存的几何：`block_id*block_size + 块内偏移`，首块再 `+= start%block_size`，末块按 `end - i*block_size` 截断（`model_runner.py:151-161`）；decode 单点 `block_table[-1]*block_size + last_block_num_tokens - 1`（`model_runner.py:181`）。
- 解释 Triton `store_kvcache` kernel 的程序模型与 `slot==-1` 哨兵（`attention.py:10-30/23`）——为何 CUDA-graph 静态缓冲必须用 `-1` 占位、去掉会写到哪里（`model_runner.py:206-207`）。
- 解释 `Context`（`context.py:6-27`）作为**调度（主进程）↔ 计算（Attention 层）的解耦桥梁**：把 per-step 元数据放进全局单例，让调度决策与 GPU 计算解耦。
- 前缀缓存命中时 prefill 走 `block_table` 路径（`k/v` 换成 `k_cache/v_cache`，`attention.py:65-66`）——为 L4 埋点（本课只点到接口分支，不展开内容寻址）。

**非目标**：

- 不重写 / 挖空任何 nano-vllm 核心函数（含 flash-attention 调用、store_kvcache kernel、prepare_prefill/decode）。
- 不讲 Block / BlockManager / Sequence↔Block 映射（留给 L3）、不讲内容寻址前缀缓存（留给 L4）、不讲连续批处理（L5）、不讲模型前向结构 / TP（L6/L7）、不做 profiling 优化（L8）。
- KV cache 的"按 kv_head 分片、rank 间不传输"这一**纠错点**留给 L7 专题；本课只在 `num_kv_heads // world_size`（`model_runner.py:110`）处点到"每 rank 只存自己的 kv heads"，不展开跨 rank 通信。

## 4. 受众与前置依赖

- 受众前提：懂 LLM 推理原理（prefill/decode、KV cache 概念不用再讲）、会读 Python（不必精通）、不熟 vLLM。**默认已学完 L1**（知道 step() 三段式、Sequence、主进程↔worker IPC）。
- 环境前置（与 L1 完全相同；本课 §0 指向 L1 §0 通用工序，不重复）：
  - `.venv` 已就绪：torch 2.8.0+cu128 + flash-attn 2.8.3（源码编译）+ nano-vllm `-e --no-deps` + Qwen3-0.6B 在 `~/huggingface/Qwen3-0.6B/`。
  - 一张 CUDA 卡（`tensor_parallel_size=1`）。
  - **Q3b 额外要求**：`enforce_eager=False` 可用（`capture_cudagraph` 能成功捕获）——本卡为 H200（`nvidia-smi` 误显 L20X），Qwen3-0.6B 极小、显存充裕，预期可成；**若捕获失败，Q3b 自动降级为 explain-from-code**（见 §8 D1）。
- `lab.py` 第一步 **fail-fast 探环境**（复用 L1 的 `_check_env` 思路）：缺 torch / flash_attn / 模型时打清晰提示退出，不甩 import 报错。

## 5. 文件布局（复刻 L1）

```
course/lessons/lesson2/
├─ TUTORIAL.md            # 正式讲义 + §0 环境（指向 L1 §0）
├─ lab.py                 # 3 任务（monkeypatch hook + TODO + 小实现），源码不动；顶层不 import nanovllm/torch
└─ solution/
   ├─ lab_solved.py       # 参考答案（= lab.py 填好）
   ├─ test_checks.py      # 合成 trace 单测（GPU-free，仅标准库）
   └─ ANSWERS.md          # Task3 explain 参考答案 + 判分要点 + 期望 trace（学生不可见）
```

约定（同 L1 / CLAUDE.md）：`lab.py` / `lab_solved.py` **顶层不 import nanovllm/torch**（让验证逻辑能 GPU-free 单测）；nanovllm import 放 `main()` 内。不新建 pyproject（复用根）。环境用 `.venv`。

## 6. TUTORIAL.md 结构（镜像 L1 的 §0–§5）

- **§0 环境初始化**：指向 `course/lessons/lesson1/TUTORIAL.md §0` 的通用工序（不重复造）；点明 L02 Q3b 额外需要 `enforce_eager=False`（cudagraph 可用）。
- **§1 学完你能**：4 条目标（对齐 §3 目标）。
- **§2 全景图**（mermaid）：prefill / decode 两条 flash-attention 调用路径 + paged KV 张量 `(2,L,num_blocks,bs,kv_heads,D)` + **Context 作为调度(主进程)↔计算(Attention层)的解耦桥梁**（L02 结构主轴）+ slot_mapping 散射示意。**禁止中英文混排 ASCII 框图**（ROLES ②）。
- **§3 逐段讲（逐条核过的 file:line）**：
  - prefill varlen 调用 `attention.py:64-70`（含前缀缓存命中分支 `65-66`）
  - decode kvcache 调用 `attention.py:72-74`（`q.unsqueeze(1)` + `cache_seqlens=context.context_lens` + `block_table`）
  - paged 张量 + `num_kv_heads//world_size` + `block_bytes` + `num_kvcache_blocks` 反推 `model_runner.py:110/112/113/115`（点到每 rank 只存自己 kv heads，不展开跨 rank）
  - `store_kvcache` kernel + `slot==-1` 哨兵 `attention.py:10-30/23` + launcher 断言 `36-39`
  - slot_mapping 几何 `model_runner.py:151-161`（首块 `+= start%block_size`、末块截断）+ decode 单点 `model_runner.py:181`
  - Context `context.py:5-27`（解耦桥梁）
  - cudagraph `fill_(-1)` 哨兵 `model_runner.py:206-207`（Q3b）
- **§4 进 lab**：与 `lab.py` **1:1**，每 Task 显式写"run `X` → 期望 `All checks passed ✓`"（ROLES ④）。
- **§5 图示补充**：prefill vs decode 的 FA-args 对比表（markdown 表格）+ slot_mapping 三段偏移几何表 + `-1` 哨兵机制说明。

讲义语气（ROLES ③）：预先验证、直接可跑的 step-by-step；**禁止**开发过程叙述（"为什么不直接 X / 这里有个坑 / 踩过 / 会失败"）。

## 7. lab.py 三任务（镜像 L1 的 hook + 小实现 + observe/explain）

**通用机制**：`lab.py` 顶部 monkeypatch 两个点，源码不改：
- `Attention.forward` → `traced_attention_forward`（抓 FA 真实调用）。
- `Scheduler.add` → `traced_add`（按 id 记 `_seqs`，同 L1，供 Task2 取 `seq.block_table`）。

捕获点选在 `Attention.forward` **入口**（`attention.py:59-60` 之后、`store_kvcache`/FA 调用之前）——此时 `context` 已由 `prepare_prefill/decode` 设好、`q/k/v` 尚未被消费。

### Task 1 · instrument hook — `traced_attention_forward`

**问题**：`Attention.forward` **每层每步都被调**（`num_hidden_layers` 次/步）。同一步各层的 `context` 与 `q/k/v` shape 完全相同，layer 0 足够代表。

**解法**：模块级 `_layer_order: dict[int,int]`（`id(self)→idx`），hook 内 `idx = _layer_order.setdefault(id(self), len(_layer_order))`，**仅 `idx == 0` 时记录**。这样每步恰好一条 trace（prefill 步→varlen 调用记录；decode 步→kvcache 调用记录）。

**输入**：`(self, q, k, v)`（`Attention.forward` 原签名）。
**记录结构**（建议；先取局部 `is_prefill = context.is_prefill` 复用）：
```python
is_prefill = context.is_prefill
{"is_prefill": is_prefill,
 "q_shape": tuple(q.shape), "k_shape": tuple(k.shape), "v_shape": tuple(v.shape),
 "cu_seqlens_q": context.cu_seqlens_q.tolist() if is_prefill else None,
 "cu_seqlens_k": context.cu_seqlens_k.tolist() if is_prefill else None,
 "max_seqlen_q": context.max_seqlen_q, "max_seqlen_k": context.max_seqlen_k,
 "context_lens": context.context_lens.tolist() if not is_prefill else None,
 "block_tables": (tuple(context.block_tables.shape) if context.block_tables is not None else None),
 "slot_mapping": context.slot_mapping.tolist()}
```
**输出**：调用原始 `_orig_attention_forward(self, q, k, v)` 并 `return` 其返回值；副作用是向 `_trace` 追加一条 dict（仅 layer 0）。

### Task 2 · 小实现 — `prefill_slot_mapping`（用户已定）

**输入**：`(block_table: list[int], block_size: int, start: int, num_tokens: int)`。`block_table` 是该 seq 的物理 block_id 列表；`start` = `num_cached_tokens`（fresh prefill 为 0）；`num_tokens` = `num_scheduled_tokens`（fresh prefill 为 prompt 长度）。
**要做的事**：忠实复现 `model_runner.py:151-161` 的散射几何，返回物理 slot 列表：
```python
def prefill_slot_mapping(block_table, block_size, start, num_tokens):
    """TODO(student) — Task 2：复现 model_runner.py:151-161 的 prefill slot 散射。
    返回每个 token 的物理槽 block_id*block_size + 块内偏移。
    首块起点再 += start % block_size；末块终点按 end - i*block_size 截断。"""
```
**输出**：`list[int]`，长度 == `num_tokens`。
**端到端自洽闭环**：Run 1 用**两条 prompt**（seq_A 长 ~600 跨 ≥3 块、seq_B 短，对齐大纲 Q1b 的 `[0, a, a+b]`），fresh prefill `start=0`。captured `slot_mapping` 是两 seq 槽按序拼接（seq_A 在前），故取 seq_A 的真实 `block_table` / `start=0` / `num_tokens=cu_seqlens_q[1]`（=a）喂进函数，断言输出 **== `captured_slot_mapping[:a]`**（seq_A 那段切片）。手写几何 ↔ 真机 trace 对拍。

> **`start>0`（首块偏移分支）分工**：单 prompt fresh prefill 触发不到（`start=0`）；该分支由 **GPU-free 合成单测**覆盖（喂 `start=300` 的 chunked case）。TUTORIAL 显式说明此分工——函数必须通用，live run 展示常见 `start=0`，单测覆盖 `start>0`。

### Task 3 · observe + explain（无代码，写注释 + ANSWERS 判分）

- **Observe（`run_checks` 机判）**：prefill `q.shape` 是 packed `(total_tokens, H, D)`（非按 seq padded）；decode `q.shape` 是 `(num_seqs, H, D)`（`attention.py:72` 的 `unsqueeze(1)` 在调用 FA 前注入 query 维）；手算 `cu_seqlens_q` 与打印值一致。
- **Explain（rubric，对照 `ANSWERS.md`，不机判）**：① 为何 prefill 用 varlen（packed、无 padding）而非 padded batched attention？② decode 为何要 `q.unsqueeze(1)`（注入 `seq_len=1` 的 query 维）而 prefill 不需要？`cache_seqlens=context.context_lens` 与 `block_table=context.block_tables` 各自职责？③ prefill vs decode 在 FA 用法上的根本差异？④ `-1` 哨兵为何必要、去掉会写到哪里？

学生把 explain 写在 `lab.py` 的 `=== Task 3 (observe + explain) ===` 注释块里。

## 8. 验证（镜像 L1 的双层）

### GPU-free 合成单测（`solution/test_checks.py`）

用 `types.SimpleNamespace` 模拟、仅标准库、不碰 GPU：
- **`prefill_slot_mapping` 两 case**（用**非连续** block_id 测散射，否则槽恰好连续会掩盖 `block_id*block_size` 映射）：
  - fresh：`block_table=[7,3,9], block_size=256, start=0, num_tokens=600` → 期望 `list(range(1792,2048)) + list(range(768,1024)) + list(range(2304,2432))`（256+256+88；首块 offset=0、末块截断 88）。
  - chunked：`block_table=[7,3,9], block_size=256, start=300, num_tokens=256` → 期望 `list(range(812,1024)) + list(range(2304,2348))`（212+44；首块 `+= start%block_size=44`、末块截断到 `end - i*block_size`，此 case 专测首块偏移分支）。
- **trace 记录 / `run_checks` 逻辑**：用合成 `_trace` 喂，含"喂错 slot / 喂错 cu_seqlens 故意 FAIL"的反例（同 L1 的 `catches_corrupt_*`）。

### 端到端（`solution/lab_solved.py`）

```bash
.venv/bin/python course/lessons/lesson2/solution/lab_solved.py   # 期望末行 All checks passed ✓
```

**两次运行分工**（D1=A）：
- **Run 1 `enforce_eager=True`**（**两条 prompt**：seq_A ~600 token 跨 ≥3 块 + seq_B 短；`max_tokens` 设小如 4）：Attention.forward hook 抓 prefill(varlen) + decode(kvcache) 调用 + prefill `slot_mapping`（Q1/Q2/Q3a）。Task2 端到端闭环（seq_A 切片）在此 run 完成。
- **Run 2 `enforce_eager=False`**（同样两条 prompt）：读 `llm.model_runner.graph_vars["slot_mapping"]`（graph decode 后 `[bs:]` 段仍是 `-1`，即哨兵活体证据），打印佐证 Q3b。`graph_vars` 是持久 dict（`model_runner.py:250-257`），graph decode 后其中 `slot_mapping` 状态 = `fill_(-1)` 后被 `[:bs]` 覆盖 → 尾部 `-1` 可见。
- **D1 fallback**：若 Run 2 的 `capture_cudagraph`（`model_runner.py:222-257`）在本卡抛错或 `graph_vars` 不可达，**Q3b 自动降级为 explain-from-code**（读 `model_runner.py:206-207` + `attention.py:23` 解释哨兵机制与越界后果），Run 2 跳过；`run_checks` 不依赖 Run 2（Q3b 不进机判）。lab 在 Run 2 失败时打一条说明而非崩溃。

`run_checks` 命名断言（机判；Task3 explain 不进）：
- `Task1 captured prefill varlen call`
- `Task1 captured decode kvcache call`
- `Task1 prefill q.shape is packed (total,H,D)`
- `Task2 prefill_slot_mapping matches captured slot_mapping`
- `Task3 prefill cu_seqlens_q == [0, a, a+b]`
- `Task3 decode q.shape is (num_seqs,H,D)`

全过打印 `All checks passed ✓`；有 FAIL 打印哪条挂了、`SystemExit(1)` 退出。`lab.py` 未填 TODO 时 `raise NotImplementedError`（fail-closed，同 L1）。

### 运行时语言

**运行时打印保持英文**（与 L1 一致；`lab.py` 注释 / docstring / 讲义正文用中文）。符合 ROLES ①"运行时输出语言是每课决策"——L02 决策 = 沿用 L01 的英文输出。

## 9. 角色流程与执行模型（D2=A：subagent 三角色循环）

三角色循环（CLAUDE.md 标准流程 + ROLES.md），**真用 subagent 跑**（D2=A，用户已选）：

1. **框架开发工程师**：按本 spec 产出 `TUTORIAL.md` + `lab.py` + `solution/`，自检八维 + 真跑 `lab_solved.py` 自证 `All checks passed ✓` + 真跑 `test_checks.py` 自证 GPU-free 单测全过。
2. **课程架构审核工程师（reviewer subagent）**：prompt **从 ROLES.md 取**，对照 nano-vllm 源码 + 本 spec，**逐维**（八维）审计全部产物，输出优先级化问题清单（阻塞 / 重要 / 次要，每条带 `file:line` + 改法）。重点查 ⑤ 正确性（每个 `file:line` 逐条 grep、lab.py 内部行号漂移）+ ⑧ 锚点。
3. **学生（student subagent，盲测）**：persona = 算子开发者（系统/内核/推理优化背景，懂推理原理、会读 Python、不熟 vLLM），**只许看 `TUTORIAL.md` + `lab.py`**（不许看 `solution/`）。从学习者视角报卡点（概念跳步 / 接口不清 / 验证不足 / 错误提示不友好）。
4. 反馈 → 工程师改 → reviewer 再审 → student 再试，到无阻塞 / 重要项。

执行节奏（在 writing-plans 阶段细化为 task）：subagent-driven-development 按 task 推进；reviewer / student 在"产物完整可跑"后集中派一次，迭代到收敛。

## 10. 成功标准（交付定义）

第 2 课可交付 = 同时满足：

- `TUTORIAL.md` 含 §0–§5，锚点 `file:line` 准（逐条 grep 核对源码，含 lab.py 内部行号）。
- `lab.py` 三任务 TODO 清晰，`.venv/bin/python course/lessons/lesson2/solution/lab_solved.py` 在装好依赖的机器上跑通并打印 `All checks passed ✓`。
- `python course/lessons/lesson2/solution/test_checks.py`（GPU-free）全过。
- fail-fast 环境探测可用；未填 TODO 时 fail-closed。
- **两轮**：reviewer subagent 按八维审**无阻塞 / 重要项** + student subagent 盲测**无阻塞性卡点**。
- Run 2（`enforce_eager=False`）能跑通并展示 `-1` 哨兵；若本卡 `capture_cudagraph` 不可用，Q3b 已按 D1 fallback 降级并在 lab/ANSWERS 说明（不视为阻塞）。

## 11. 开放项 / 待决

- **D1 fallback 触发判定**：实现时实测 `capture_cudagraph` 在本卡是否成功；`graph_vars` 经 `llm.model_runner.graph_vars` 是否可达（属性路径在实现时核对 `llm.py`/`llm_engine.py`）。若不可达，按 fallback 处理。
- **Task2 端到端 `(block_table, start, num_tokens)` 的捕获点**：倾向 post-run 从 `_seqs`（`traced_add` 记录）读 `seq.block_table` + 用 `start=0`（fresh）+ `num_tokens = len(prompt token_ids)`（lab 自己 tokenize，长度已知）重建；若发现 `seq.block_table` 在 FINISHED/deallocate 后被清，改 hook `ModelRunner.prepare_prefill` 入口捕获（plan 阶段定）。
- **prompt 构造**：Run 1 用两条英文 prompt——seq_A tokenize 到 >512（跨 ≥3 个 block_size=256 块，展示首/中/末三段），seq_B 短；精确长度不必硬编码（Task2 闭环用 captured `cu_seqlens_q[1]` 自校准切片边界）。
- **多卡（tp>1）**不在第 2 课范围（tp=1 即可）。
- enactment（D2=A）已在 §9 定为 subagent 循环，不再悬置。

## 12. 源码锚点清单（本 session 已逐个 grep 核验，全部准确）

| 内容 | 位置 |
|---|---|
| `Attention.forward`（prefill varlen / decode kvcache 双路径） | `nanovllm/layers/attention.py:59-75` |
| prefill varlen 调用 | `attention.py:67-70`（`64` 起分支） |
| 前缀缓存命中分支（`k/v` 换 `k_cache/v_cache`） | `attention.py:65-66` |
| decode kvcache 调用（`q.unsqueeze(1)` + `cache_seqlens` + `block_table`） | `attention.py:72-74` |
| Triton `store_kvcache_kernel` + `slot==-1` 哨兵 | `attention.py:10-30`（`23`） |
| `store_kvcache` launcher 断言 | `attention.py:36-39` |
| paged KV 张量形状 | `model_runner.py:115` |
| `num_kv_heads = num_key_value_heads // world_size` | `model_runner.py:110` |
| `block_bytes` 计算 | `model_runner.py:112` |
| `num_kvcache_blocks` 由剩余显存反推 + `assert>0` + 写回 config | `model_runner.py:113-114` |
| kv_cache 挂到每层 Attention | `model_runner.py:116-121` |
| `prepare_prefill` slot_mapping 三段偏移几何 | `model_runner.py:151-161` |
| 前缀缓存信号 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` | `model_runner.py:162` |
| `prepare_decode` 单点 slot | `model_runner.py:181` |
| eager/graph 分流（`enforce_eager or >512` 走 eager） | `model_runner.py:197-198` |
| graph.replay 分支 | `model_runner.py:199-212` |
| cudagraph `slot_mapping.fill_(-1)` + `[:bs]` 覆盖 | `model_runner.py:206-207` |
| `graph_bs` 桶 + `graph_vars` 静态缓冲 | `model_runner.py:234 / 250-257` |
| `Context` dataclass + `set_context`/`get_context`/`reset_context` | `nanovllm/utils/context.py:5-27` |
| `block_size` 默认 256 | `nanovllm/config.py:17` |
| `gpu_memory_utilization` 默认 0.9 | `nanovllm/config.py:12` |
