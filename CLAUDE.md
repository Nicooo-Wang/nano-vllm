# CLAUDE.md — nano-vllm 课程开发

本项目是 **nano-vllm 教学课程**：把 nano-vllm（~1450 行的极简 vLLM 复刻，仅支持 Qwen3）的 request→output 全流程教给**算子 / 内核开发者**。课程在 `course/` 下。

## 受众（重要，别搞错）
- **算子开发者**（系统 / 内核 / 推理优化背景）——**不是**"算法开发者"（这是用户明确纠正过的）。
- 懂 LLM 推理原理、会读 Python（不必精通）、**不熟 vLLM / nano-vllm**。
- 推论：讲义**不要假设 vLLM 熟悉度**；lab 走 trace/observe/explain，**不挖空核心函数**。

## 课程现状
- `course/README.md`：8 课总纲（v2，已收敛）。`course/lessons/L01.md`–`L08.md`：各课大纲。
- **L01 已完整实现**（`course/lessons/lesson1/`，已合并 main）。**L02–L08 仍是大纲**，是接下来的工作。
- 主题：L1 请求全流程 · L2+L3 KV cache（flash-attn 接口 → paged 抽象 → Block/BlockManager → Sequence↔Block）· L4 前缀缓存 · L5 连续批处理 · L6 模型前向 · L7 张量并行 · L8 profiling。
- L7 关键正确性点：KV cache **按 kv_head 分片、rank 间不传输**；唯一跨 rank 通信是 RowParallel all_reduce / ParallelLMHead gather / rank0-only sampler。

## 开发一节课的标准流程
1. **brainstorm**（`superpowers:brainstorming`）→ spec：`docs/superpowers/specs/YYYY-MM-DD-lessonN-design.md`
2. **writing-plans** → plan：`docs/superpowers/plans/YYYY-MM-DD-lessonN.md`
3. **subagent-driven-development** → 按 task 实现，每 task 过 implementer + reviewer
4. **三角色循环**（工程师 → 审核 → 学生盲测 → 改 → 复审）直到无阻塞 / 重要项
5. **finishing-a-development-branch** → fast-forward 合并回 main + `git push origin main` + 删分支（用户已确认的课程分支惯例）
- L01 的 spec / plan 可当模板。

## 质量标准：`course/ROLES.md`（每节课必读必用）
三角色定义 + **八维质量 checklist**。派 reviewer / student 的 prompt **一律从 ROLES.md 取**，按八维验收。踩坑换来的重点：
- **③ 讲义语气**：讲义是预先验证、直接可跑的 step-by-step；**禁止**开发过程叙述（"为什么不直接 X / 这里有个坑 / 踩过 / 会失败"）。学生要 how，不要 dev story。
- **② 图示**：结构图用 **mermaid**，对比 / 时序表用 **markdown 表格**；**禁止中英文混排的 ASCII 框图**（双宽度字符会错位）。
- **① 语言**：受众是中文 → 注释 / docstring / 讲义正文用中文；代码标识符 / 路径 / `file:line` / 命令保留原文。运行时输出语言是每课决策（L01 保持英文）。
- **④ 讲义↔lab 对应**：讲义 step-by-step 与 `lab.py` **1:1**；**显式**写出"如何验证做对"（run X → 期望 `All checks passed ✓`）。
- **⑤ 正确性 + ⑧ 锚点**：每个 `file:line` **逐条 grep 核对源码**；**文件改动后 lab.py 内部行号会漂移，必须重核**（L01 漏过一次，漂了 ~30 行）。
- **⑦ 残留物**：交付前 grep 确认无遗留 `TODO`（除有意的 `TODO(student)`）/ debug print / 占位返回。

## lab 风格（altitude）
trace / observe / explain + **至多一个小实现**；**不逐函数留空**（v1 的 function-level blanks 已被用户否决）。验证：合成 trace 单测（GPU-free，仅标准库）+ 真实 H200 端到端跑通。范式见 `course/lessons/lesson1/`。

## 每节课文件布局
```
course/lessons/lessonN/
├─ TUTORIAL.md            # 讲义 + §0 环境（指向通用工序）
├─ lab.py                 # 学生版（引导式 TODO）
└─ solution/
   ├─ lab_solved.py       # 参考答案（= lab.py 填好）
   ├─ test_checks.py      # 合成 trace 单测（GPU-free，仅标准库）
   └─ ANSWERS.md          # 答案 + 判分要点
```
约定：`lab.py` / `lab_solved.py` **顶层不 import nanovllm/torch**（让验证逻辑能 GPU-free 单测）；nanovllm import 放 `main()` 内。

## 环境（已就绪，别重踩）
- `.venv` 已建好：torch 2.8.0+cu128 + flash-attn 2.8.3（源码编译）+ nano-vllm `-e --no-deps` + Qwen3-0.6B（已下载到 `~/huggingface/Qwen3-0.6B/`）。GPU 是 8×"L20X"（**实为 H200**，nvidia-smi 误显）。
- `uv sync` 单独**会失败**（flash-attn 隔离构建拿不到 torch + uv 默认 torch 对驱动太新）。完整工序见 `course/lessons/lesson1/TUTORIAL.md §0`。
- `uv pip install` 在本机会**混淆 conda-base 与 .venv** → 用 `--python "$(pwd)/.venv/bin/python"` 强制目标。
- 跑 lab：`.venv/bin/python course/lessons/lessonN/solution/lab_solved.py`。

## 验证
- 合成单测：`python course/lessons/lessonN/solution/test_checks.py`（GPU-free）。
- 端到端：`.venv/bin/python course/lessons/lessonN/solution/lab_solved.py` → 期望 `All checks passed ✓`。
- 学生版 `lab.py` 未填 TODO 时应 **fail-closed**（`NotImplementedError`）。
- 关键教学不变式必须在真实硬件上成立（如 L1 的 `total_steps == num_completion_tokens`）。

## Git
- 每节课一个分支 `course/lessonN`；完成后 fast-forward 合并回 main + `git push origin main` + 删分支。
- `.superpowers/`（SDD scratch：ledger / brief / review 包）已 gitignore，别提交。
