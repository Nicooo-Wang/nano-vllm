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

example.py 用同一份 `SamplingParams(max_tokens=64)` 提交两条 prompt（seq0、seq1）。两条 prompt 在 `add_request` 时就被加入 `scheduler.waiting`，下一步 `schedule()` 把它们**一起**送进 prefill（lesson 1 关闭 chunked prefill，单步 prefill 完成）。期望 trace 形态如下（`nst` = num_scheduled_tokens，`<promptN>` 表示 prompt N 的 token 数）：

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

## Task 2：`num_tokens` 正负号双重语义（explain）

### 2.1 源码出处

`nanovllm/engine/llm_engine.py:51`（`LLMEngine.step`）：

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

`nanovllm/engine/llm_engine.py:76-79`（`LLMEngine.generate`）：

```python
if num_tokens > 0:
    prefill_throughput = num_tokens / (perf_counter() - t)
else:
    decode_throughput = -num_tokens / (perf_counter() - t)
```

### 2.2 参考解释

`num_tokens` 的**符号**和**绝对值**各承担一种语义：

- **正数（`is_prefill=True`）**：`num_tokens = sum(seq.num_scheduled_tokens)`。绝对值 = 本步 prefill 实际处理的 prompt token 总数（多条 prompt 同一步 prefill 时是它们 nst 之和）。
- **负数（`is_prefill=False`，即 decode）**：`num_tokens = -len(seqs)`。负号编码"这一步是 decode"，绝对值 `len(seqs)` = 本步参与 decode 的 sequence 条数——因为 decode 阶段**每条 seq 每步只产 1 个 token**，所以 sequence 数就等于本步产出的 token 数。

`generate` 用**正负号分流**两条吞吐曲线（`llm_engine.py:76-79`）：
- `num_tokens > 0` → 走 `prefill_throughput = num_tokens / Δt`（token/s）。
- 否则 → 走 `decode_throughput = -num_tokens / Δt`（再取一次负号把长度还原成正数）。

这样进度条 `pbar.set_postfix({"Prefill": ..., "Decode": ...})` 的两条曲线**每一步只更新其中一条**——prefill 步只刷新 Prefill tok/s，decode 步只刷新 Decode tok/s，两条曲线互不污染。

### 2.3 判分要点（rubric）

学生只要**用自然语言讲清两点**即判通过，缺一不可：

| 必答点 | 通过标准 |
|--------|----------|
| **① 符号区分 prefill / decode** | 明确说出：正数表示 prefill，负数表示 decode（或等价表述"正负号用来区分两种阶段"）。 |
| **② 绝对值 = 本步 token 计数** | 明确说出：正数绝对值是 prefill token 总数（`sum(num_scheduled_tokens)`），负数绝对值是 decode 的 sequence 数（`len(seqs)`，因为每 seq 1 个 token）。 |

**加分项（非必需）**：能进一步指出"decode 阶段每 seq 每步固定 1 token，所以 sequence 数 == token 数"这一等价成立的**原因**；或能解释 `generate` 里两条曲线为何每步只更新一条。

**常见错误（扣分）**：只说"正数是 prefill、负数是 decode"却没说绝对值含义；或把 `-len(seqs)` 误解为"负的 token 数"。

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

（`run_checks` 中 `Task3 total_steps==num_completion_tokens` 与 `Task1 total_steps==num_completion_tokens` 校验的就是这一等式；`completion == len(completion_token_ids)` 校验的是 `num_completion_tokens` 确实在 `append_token` 时被同步自增。）

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
- [ ] §2.2 解释同时命中"符号区分 prefill/decode"与"绝对值 = 本步 token 计数"两点。
- [ ] §2 所有行号引用（`llm_engine.py:51`、`:76-79`）与当前仓库一致。
- [ ] §3.2 引用的 `scheduler.py:86-88` 与当前仓库一致。
- [ ] §1.2 trace 形态描述覆盖：两条 prompt 同一 prefill step、decode 阶段 nst 全为 1、seq0 命中 eos 被 remove 后只剩 seq1。
- [ ] §3.3 TODO→答案映射两条一一对应，无遗漏。
