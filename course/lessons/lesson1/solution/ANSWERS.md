# Lesson 1 参考答案（answer key）

> 面向工程师 / reviewer 的判分参考，**非学生可见文件**。
> 所有代码与行号均取自本仓库实际源码：
> - 参考实现：`course/lessons/lesson1/solution/lab_solved.py`
> - 学生作业：`course/lessons/lesson1/lab.py`（其中两处 `TODO(student)`）
> - 引擎源码（只读）：`nanovllm/engine/llm_engine.py`、`nanovllm/engine/scheduler.py`

---

## Task 1：`traced_postprocess` 参考实现

### 1.1 参考代码（逐字摘自 `solution/lab_solved.py:21-36`）

```python
def traced_postprocess(self, seqs, token_ids, is_prefill):
    """TODO(student): record this step into _trace, then call the original.

    Each seq still carries its num_scheduled_tokens at entry (postprocess
    resets it to 0 internally at scheduler.py:85), so capture it in `before`.
    """
    before = [(s.seq_id, s.num_scheduled_tokens, s.status.name) for s in seqs]
    result = _orig_postprocess(self, seqs, token_ids, is_prefill)
    after = [(s.seq_id, s.status.name, s.is_finished) for s in seqs]
    _trace.append({
        "is_prefill": is_prefill,
        "before": before,
        "token_ids": list(token_ids),
        "after": after,
    })
    return result
```

**关键点（判分时检查）：**
1. `before` 必须在调用 `_orig_postprocess` **之前**采集 `num_scheduled_tokens`——因为原始 `postprocess` 会在 `scheduler.py:85` 把它清零（`seq.num_scheduled_tokens = 0`），之后就读不到了。
2. `before` / `after` 都要记，区分"这一步进栈时 seq 的状态"与"出栈后的状态"（用来观测 FINISHED 转移、`running.remove`）。
3. `token_ids` 用 `list(...)` 拷贝，避免外部后续 mutate 污染 trace。
4. 必须 `return result`——钩子不能吞掉原始返回值。

### 1.2 example.py 两条 prompt 的期望 trace 形态

example.py 用同一份 `SamplingParams(max_tokens=64)` 提交两条 prompt（seq0、seq1）。两条 prompt 在 `add_request` 时就被加入 `scheduler.waiting`，下一步 `schedule()` 把它们**一起**送进 prefill。**这里没有任何"关闭 chunked prefill"的开关**——chunked prefill 不是靠 flag 控制的，而是靠 `max_num_batched_tokens` 预算：本课 smoke 的两条 prompt 加起来远小于预算（16384），一次 prefill 就能算完整条 prompt，所以 chunked 分支（`scheduler.py:42` 的 `if remaining < num_tokens and scheduled_seqs: break`）根本不会触发。期望 trace 形态如下（`nst` = num_scheduled_tokens，`<promptN>` 表示 prompt N 的 token 数）：

| step | 阶段 | seqs | nst | 说明 |
|------|------|------|-----|------|
| 0 | PREFILL | `[0, 1]` | `{0: <prompt0>, 1: <prompt1>}` | 两条 prompt 同一步 prefill；每个 seq 的 nst = 自己的 prompt 长度 |
| 1..k | DECODE | `[0, 1]` | `{0: 1, 1: 1}` | 每步每 seq 各产 1 token |
| k+1.. | DECODE | `[1]` | `{1: 1}` | 某步 seq0 命中 eos（或达到 max_tokens）→ `scheduler.py:90-92` 置 FINISHED 并 `running.remove(seq)`；之后只剩 seq1 |
| 末步 | DECODE | `[1]` | `{1: 1}` | seq1 达到 max_tokens=64 结束 |

**核查点：**
- `_trace` 里 **PREFILL 行有且仅有一行**（Task1 check `n_prefill_steps == 1`）。
- 每个 seq 最终都到达 FINISHED（`after` 中 `is_finished=True` 至少出现一次）。
- **不会出现** "PREFILL seqs=[0]" 紧接 "PREFILL seqs=[1]"——两条 prompt 是**同一 prefill step**（这是连续 batch 的关键现象）。

---

## Task 2：为什么 `total_steps == num_completion_tokens`（explain）

### 2.1 题目

lab 的 Task 2（observe + explain）只问一件事：**为什么每条 request 的参与步数 `total_steps` 正好等于它的 `num_completion_tokens`？**

- Observe 部分（`run_checks` 自动验证，不判分）：trace 里两条 prompt 在**同一 step 0 PREFILL** 里一起算；prefill 步每条 seq 的 `num_scheduled_tokens` == 各自 prompt 长度；每个 DECODE 步每条 seq 的 `num_scheduled_tokens` == 1。
- Explain 部分（**对照本文 §2.2 自检，`run_checks` 不判分**）：用自然语言讲清 `total_steps == num_completion_tokens` 的因果。

### 2.2 参考解释（核心：prefill 步本身吐第 1 个 token）

关键在 `nanovllm/engine/scheduler.py:86-88`：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
seq.append_token(token_id)
```

`continue` **只在 chunked prefill 还没算完整条 prompt 时**触发（`num_cached_tokens < num_tokens`）。本课 smoke 的 prompt 很短，整条 prompt 的长度远小于 `max_num_batched_tokens`，一次 prefill 就能算完，于是 `num_cached_tokens == num_tokens`，`continue` **不触发**——直接走到 `seq.append_token(token_id)`。也就是说：**prefill 步不仅算了 prompt，还顺手采样并 append 了第 1 个 completion token**。

由此：一条 seq 参与的**每一步**（含 prefill 步）都恰好 `append_token` 一次：
- prefill 步：`num_scheduled_tokens = 整条 prompt 长度`，但 `append_token` 只跑一次 → 贡献 1 个 completion token。
- 每个 decode 步：`num_scheduled_tokens = 1`，`append_token` 跑一次 → 贡献 1 个 completion token。

所以：

```
total_steps  ==  num_completion_tokens  ==  len(completion_token_ids)
```

**多条 seq 共享 prefill step 的桥接**：即便两条 seq 被 `scheduler.py:30` 的 `while` 拉进同一个 prefill step，这条不变式仍是**按 seq 各自计**的——`total_steps` 数的是该 seq 在 trace 的 `before` 里出现的次数，那个共享的 prefill step 对每条 seq 都算一次参与，每参与一次就 append 一个 token，所以每条 seq 的步数仍等于它自己的 completion token 数。

> **顺带一句（`num_tokens` 的正负号）**：上面解释里提到的 `num_tokens = sum(...) if is_prefill else -len(seqs)`（`llm_engine.py:51`）只是用符号位给吞吐进度条分流（prefill 步累进 `prefill_throughput`、decode 步累进 `decode_throughput`，见 `llm_engine.py:76-79`），与 `total_steps` 不变式无直接关系——别把它和不變式混为一谈。

### 2.3 判分要点（rubric）

学生只要**用自然语言讲清因果**即判通过：

| 必答点 | 通过标准 |
|--------|----------|
| **① prefill 步也 append 一个 token** | 明确指出：prefill 步的 `continue`（`scheduler.py:86-87`）在"整条 prompt 一次算完"时**不触发**，于是 `append_token`（`scheduler.py:88`）照跑，prefill 步贡献 1 个 completion token。 |
| **② 由此步数 == completion token 数** | 明确推出：seq 参与的每一步都 append 一个，故 `total_steps == num_completion_tokens`。 |

**加分项（非必需）**：能说明"多条 seq 共享 prefill step 时不变式按各自计"；或点明 smoke prompt 短到 fit 进 `max_num_batched_tokens`、所以 chunked 分支不触发。

**常见错误（扣分）**：以为步数 = completion token 数 **+ 1**（把 prefill 步误当成"只算 prompt、不产 token"）；或答成 `num_tokens` 正负号的解释（那是另一回事，与本题无关）。

---

## Task 3：`summarize_request` 参考实现

### 3.1 参考代码（逐字摘自 `solution/lab_solved.py:39-49`）

```python
def summarize_request(seq):
    """TODO(student): return (num_prompt_tokens, num_completion_tokens, total_steps).

    total_steps = number of engine steps this seq participated in (count _trace).
    It should equal num_completion_tokens — the prefill step itself emits the
    first token (scheduler.py:86-88), so every participating step appends one.
    """
    total_steps = sum(
        1 for r in _trace if any(b[0] == seq.seq_id for b in r["before"])
    )
    return (seq.num_prompt_tokens, seq.num_completion_tokens, total_steps)
```

### 3.2 为什么 `total_steps == num_completion_tokens`？

直觉上"prompt 在 prefill 步处理、completion 在 decode 步生成"，似乎步数应等于 completion token 数 **加上 prefill 那 1 步**。但在 nano-vllm 的调度里，**prefill 步本身也会吐出第一个 completion token**——证据在 `nanovllm/engine/scheduler.py:86-88`：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
seq.append_token(token_id)
```

`continue` 只在"chunked prefill 还没处理完整个 prompt"时触发（`num_cached_tokens < num_tokens`）。lesson 1 的 smoke prompt 很短，整条 prompt 的长度远小于 `max_num_batched_tokens`（16384），一次 prefill 就能算完整条 prompt，于是 `num_cached_tokens == num_tokens`；同时 `scheduler.schedule` 的 prefill 分支里只有"第一条 seq"才允许 chunked（`scheduler.py:42` 的 `if remaining < num_tokens and scheduled_seqs: break`），这里 `remaining` 充足、该 break 不触发，所以根本不会走到 chunked-prefill-only-for-the-first-seq 那条分支。`continue` 不触发——直接走到 `seq.append_token(token_id)`，**prefill 步也产 1 个 token**。

由此：seq 参与的**每一步**（含 prefill 步）都 `append_token` 一次，所以

```
total_steps  ==  num_completion_tokens  ==  len(completion_token_ids)
```

（`run_checks` 中 `Task3 summarize_request total_steps==num_completion_tokens` 与 `Task1 trace recorded all steps (count==num_completion_tokens)` 校验的就是这一等式；`completion == len(completion_token_ids)` 校验的是 `num_completion_tokens` 确实在 `append_token` 时被同步自增。）

### 3.3 TODO → 参考答案映射

`lab.py` 只有两处 `TODO(student)`，**一一对应**到 `lab_solved.py` 的两个函数体：

| `lab.py` 位置 | TODO | `lab_solved.py` 对应实现 |
|---------------|------|--------------------------|
| `def traced_postprocess(...)`（Task 1，约 `lab.py:21`） | TODO(student) — Task 1 (trace)：记录本步到 `_trace`，再调用原始 `postprocess` | `lab_solved.py:21-36` 整个函数体（见上 §1.1） |
| `def summarize_request(seq)`（Task 3，约 `lab.py:35`） | TODO(student) — Task 3 (small wrapper)：返回 `(num_prompt_tokens, num_completion_tokens, total_steps)` | `lab_solved.py:39-49` 整个函数体（见上 §3.1） |

Task 2（observe/explain）**不写代码**，只是在 example.py 跑完后观察进度条 / `_trace`，回答"`num_tokens` 正负号双重语义"——见本文 §2。因此 `lab.py` 只有两个代码 TODO，与 `lab_solved.py` 的两个函数体精确对应。

---

## 附：自检清单（reviewer 用）

- [ ] §1.1 代码与 `lab_solved.py:21-36` **逐字一致**（含 docstring）。
- [ ] §3.1 代码与 `lab_solved.py:39-49` **逐字一致**（含 docstring）。
- [ ] §2.2 解释同时命中"prefill 步也 append 一个 token（`continue` 不触发）"与"由此步数 == completion token 数"两点。
- [ ] §2 所有行号引用（`llm_engine.py:51`、`:76-79`）与当前仓库一致。
- [ ] §3.2 引用的 `scheduler.py:86-88` 与当前仓库一致。
- [ ] §1.2 trace 形态描述覆盖：两条 prompt 同一 prefill step、decode 阶段 nst 全为 1、seq0 命中 eos 被 remove 后只剩 seq1。
- [ ] §3.3 TODO→答案映射两条一一对应，无遗漏。
