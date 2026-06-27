# Lesson 2 参考答案（answer key）

> 面向工程师 / reviewer 的判分参考，**非学生可见文件**。
> 所有代码与行号均取自本仓库实际源码：
> - 参考实现：`course/lessons/lesson2/solution/lab_solved.py`
> - 学生作业：`course/lessons/lesson2/lab.py`（其中三处 `TODO(student)`）
> - 引擎源码（只读）：`nanovllm/layers/attention.py`、`nanovllm/engine/model_runner.py`、`nanovllm/utils/context.py`、`nanovllm/engine/scheduler.py`

---

## Task 1：`traced_attention_forward` 参考实现

### 1.1 参考代码（逐字摘自 `solution/lab_solved.py:41-62`）

```python
def traced_attention_forward(self, q, k, v):
    """TODO(student) — Task 1: record this step's flash-attention call into _trace,
    then call _orig_attention_forward(self, q, k, v).

    Attention.forward fires once per layer per step (num_hidden_layers times); every
    layer in a step shares the same context & shapes, so only LAYER 0 records.
    """
    if _layer_order.setdefault(id(self), len(_layer_order)) == 0:
        context = _get_context()
        is_prefill = context.is_prefill
        _trace.append({
            "is_prefill": is_prefill,
            "q_shape": tuple(q.shape), "k_shape": tuple(k.shape), "v_shape": tuple(v.shape),
            "cu_seqlens_q": context.cu_seqlens_q.tolist() if is_prefill else None,
            "cu_seqlens_k": context.cu_seqlens_k.tolist() if is_prefill else None,
            "max_seqlen_q": context.max_seqlen_q, "max_seqlen_k": context.max_seqlen_k,
            "context_lens": context.context_lens.tolist() if not is_prefill else None,
            "block_tables": (tuple(context.block_tables.shape)
                             if context.block_tables is not None else None),
            "slot_mapping": context.slot_mapping.tolist(),
        })
    return _orig_attention_forward(self, q, k, v)
```

**关键点（判分时检查）：**

1. **layer-0-only 机制——`_layer_order.setdefault(id(self), len(_layer_order)) == 0`**：`Attention.forward` 每步每层都触发一次（`num_hidden_layers` 次，Qwen3-0.6B 是 28 次）。`setdefault` 的语义是「如果 `id(self)` 不在 dict 里，就插进去、值设成当前 `len(_layer_order)`；返回插进去的值」。因此**每个 Attention 模块第一次被调到时**拿到一个递增的 layer index（第一个见到的得 0、第二个得 1 …），后续调用直接返回已存的值。整个 step 内第 0 层（`id(self)` 第一次出现）使该表达式 `== 0` 成立、`_trace.append` 执行；其它层 `!= 0`、跳过。这样每个 engine step 在 `_trace` 里只留一条记录，避免重复 28 份一模一样的形状。`id(self)` 是模块对象的 Python 标识（模块对象在模型构造后地址稳定，跨 step 不变），所以 layer 0 在所有 step 里都是同一个模块。
2. **`context = _get_context()`**：取 `Context` 全局单例（`context.py:18`）。`_get_context` 在 `main()` 里被赋成 `nanovllm.utils.context.get_context`。要 `get_context()` 而非 import 时的快照——因为 `set_context` 每步 `run` 前会**替换整个 `_CONTEXT` 对象**（`context.py:23`），必须取"当前这一步"的那个。
3. **字段按 prefill/decode 有条件记录**：prefill 记 `cu_seqlens_q/k` + `max_seqlen_q/k`、decode 记 `context_lens` + `block_tables` shape（对应 §3.2 Context 字段表里两路各自用到的字段）；`slot_mapping` 两路都记（`run_checks` Task 2 用 prefill 的那段）。用 `is_prefill if/else None` 区分是为了让 trace 记录的字段集稳定、便于学生看形态时对照。
4. **必须 `return _orig_attention_forward(...)`**：钩子不能吞掉原始返回值（`o = flash_attn_...(...)` 的注意力输出），否则模型前向断裂。

### 1.2 两条 prompt 的期望 trace 形态

`main()` 用同一份 `SamplingParams(max_tokens=4)` 提交两条 prompt（`_LONG_PROMPT`——约 600 token、跨 ≥3 个 block；以及 `"list all prime numbers within 100"`——约 30 token）。两条 prompt 在 `add_request` 时进 `scheduler.waiting`，下一步 `schedule()` 把它们**一起**送进同一个 prefill step（两条 prompt 总 token 数远小于 `max_num_batched_tokens=16384`，一次 prefill 算完整条 prompt、不触发 chunked 分支——同 L01 §1.2）。设第一条 prompt 长 `a`、第二条长 `b`（`a + b = total`）。

| step | 阶段 | `q_shape` | 关键 Context 字段 | 说明 |
|------|------|-----------|-------------------|------|
| 0 | PREFILL | `(total, 4, 64)`<br/>（`total = a + b`，packed） | `cu_seqlens_q = [0, a, a+b]`<br/>`cu_seqlens_k = [0, a, a+b]`<br/>`slot_mapping` 长度 = `a + b` | 两条 prompt 的 q/k/v **拼成一条 packed 张量**，`cu_seqlens_q` 是前缀和切段：`[0:a]` 属 seq0、`[a:a+b]` 属 seq1。`slot_mapping` 把这 `a+b` 个 token 的 K/V 散射进各自 block 的物理槽（见 Task 2）。`4 = num_heads`、`64 = head_dim`（Qwen3-0.6B）。 |
| 1..k | DECODE | `(2, 4, 64)`<br/>（`num_seqs = 2`） | `context_lens = [a+1, b+1]`<br/>`block_tables.shape = (2, ...)`<br/>`slot_mapping` 长度 = 2 | 每步每 seq 各产 1 个新 query，`context_lens` 是每条 seq 当前**有效 KV 长度**（prefill 那 1 步已存进 cache、每 decode 步 +1），首步 decode 是 `[a+1, b+1]`。`slot_mapping` 是**两个单点槽**（每 seq 的当前 block 末尾空位），对应 `model_runner.py:181` 的 decode 单点几何。 |
| k+1.. | DECODE | `(1, 4, 64)` | `context_lens = [..]`<br/>`slot_mapping` 长度 = 1 | 某步 seq0 命中 eos（或 `max_tokens=4`）→ `scheduler.py:90-92` 置 FINISHED 并 `running.remove(seq)`；之后 `num_seqs` 减到 1。 |

**核查点（`run_checks` 验证）：**
- `Task1 captured prefill varlen call`：`_trace` 里至少有一条 `is_prefill=True` 记录（lab 用 `enforce_eager=True`，两条 prompt 共享一个 prefill step）。
- `Task1 captured decode kvcache call`：至少一条 `is_prefill=False` 记录。
- `Task1 prefill q.shape is packed (total,H,D)`：`len(q_shape) == 3 and q_shape[0] == cu_seqlens_q[-1]`——packed 形态的判据是「q 第 0 维 == `cu_seqlens_q` 末项」（两条 prompt 的 token 总数）。
- `Task4 prefill cu_seqlens_q == [0,a,a+b]`：`cu_seqlens_q` 单调、首项 0、长度 ≥ 2（packed 切段前缀和的几何性质）。
- `Task4 decode q.shape is (num_seqs,H,D)`：`q_shape[0] == len(context_lens)`（每条 seq 一个 query）。

**注意：prefill 记录存 `cu_seqlens`、decode 记录存 `context_lens`**——这两套字段不能混。prefill 没有 `context_lens`（packed 靠 `cu_seqlens` 切段），decode 没有 `cu_seqlens`（每 seq 单 query、靠 `context_lens` 告诉 FA 每条 seq 在 cache 里有几个 KV）。`run_checks` 里 `cl = d["context_lens"] or []` 的 `or []` 就是为了防 decode 记录里 `context_lens` 为 `None`（理论上不会、但防御性写法）。

---

## Task 2：`prefill_slot_mapping` 参考实现

### 2.1 参考代码（逐字摘自 `solution/lab_solved.py:65-85`）

```python
def prefill_slot_mapping(block_table, block_size, start, num_tokens):
    """TODO(student) — Task 2: reproduce model_runner.py:151-161 prefill slot scatter.

    Each token maps to physical slot block_id*block_size + in-block offset. The first
    block's start is further offset by start%block_size; the last block is truncated to
    end - i*block_size.
    """
    end = start + num_tokens
    start_block = start // block_size
    end_block = (end + block_size - 1) // block_size
    slots = []
    for i in range(start_block, end_block):
        slot_start = block_table[i] * block_size
        if i == start_block:
            slot_start += start % block_size
        if i != end_block - 1:
            slot_end = block_table[i] * block_size + block_size
        else:
            slot_end = block_table[i] * block_size + end - i * block_size
        slots.extend(range(slot_start, slot_end))
    return slots
```

### 2.2 三段偏移几何解释

一条 prompt 的本次排片范围是 `[start, end)`（`end = start + num_tokens`），可能跨多个物理 block。每个 token 的物理槽 = `block_id * block_size + 块内偏移`。按遍历的 block index `i` 分三段：

| 区间 | `i` 取值 | `slot_start` | `slot_end` | 几何含义 |
|------|----------|--------------|------------|----------|
| 首区间 | `i == start_block` | `block_table[i] * block_size + start % block_size` | 整块（`+ block_size`）或末区间截断 | 起点 `start` 不在块边界时，加 `start % block_size` 偏移，跳到块内正确位置。`start=0` 时此偏移为 0。 |
| 中间整块 | `start_block < i < end_block-1` | `block_table[i] * block_size` | `block_table[i] * block_size + block_size` | 整块填满（full block），从块头到块尾。 |
| 末区间 | `i == end_block-1` | `block_table[i] * block_size` | `block_table[i] * block_size + end - i * block_size` | 终点 `end` 不在块边界时按 `end` 截断，只写到块内 `end - i*block_size` 这个位置。 |

**总槽数** = 各段 `slot_end - slot_start` 之和 = `(end - start)` = `num_tokens`（首块 `+= start%block_size` 与末块 `end - i*block_size` 恰好抵消块边界外的部分，链式求和正好等于 `num_tokens`）。这是 lab `run_checks` 里「重建 == 录到的 slot_mapping 前 `a` 个」的几何保证。

> **特殊情形——首末同一块**：若 `num_tokens` 很小、整个 `[start,end)` 都落在 `start_block == end_block-1` 这一个块里，此时 `slot_start` 既走首区间分支（`+= start%block_size`）、`slot_end` 又走末区间分支（`+ end - i*block_size`）。两个 `if` 不互斥，正确处理。

### 2.3 为何 live run 只覆盖 `start=0`

`run_checks` 里 Task 2 的重建调用是 `prefill_slot_mapping(_seq_blocks[seq_a_id], block_size, 0, a)`——**`start=0`**。原因：lab 用的是**全新 prefill**（两条 prompt 第一次进 `scheduler.waiting`、cache 里没有任何前缀），所以 `prepare_prefill`（`model_runner.py:139-141`）里 `start = seq.num_cached_tokens = 0`。于是首区间偏移 `start % block_size = 0` 退化为 0，三段几何里的"首区间偏移"那一项不显形——学生看真实 trace 时看不到 `start>0` 的样例。

`start > 0` 的两种真实场景（lab 不跑、但**单元测试覆盖**，见 `solution/test_checks.py`）：
1. **Chunked prefill**：一条长 prompt 一次 `max_num_batched_tokens` 预算算不完，分多次 prefill。第 2 次起的 `start = 已 cached 的 token 数`（不在块边界时 `start % block_size != 0`，首区间偏移才显形）。
2. **Prefix-cache 命中**（L4 内容）：prompt 的前缀已经在 cache 里，本次只排 suffix，`start = 命中的前缀长度`（`num_cached_tokens`），同样可能 `start % block_size != 0`。

GPU-free 单测会用 `start=300, block_size=256` 这类参数验证首区间偏移分支——这是为什么 `prefill_slot_mapping` 不能省掉 `if i == start_block: slot_start += start % block_size` 这一行。

---

## Task 3：`simulate_fa_calls` 参考实现

### 3.1 参考代码（逐字摘自 `solution/lab_solved.py:105-142`）

```python
def simulate_fa_calls(fa_varlen, fa_kvcache, device, dtype):
    """TODO(student) — Task 3: call the two flash-attention interfaces on toy inputs.
    fa_varlen/fa_kvcache are passed in (real flash_attn from main(); mocks in tests) —
    you only need to CALL them correctly. Focus: what args differ between prefill & decode.
    Returns (prefill_out, decode_out). Mirror attention.py:67-70 (prefill) / 72-74 (decode).
    """
    import torch
    num_heads, num_kv_heads, head_dim = TOY_NUM_HEADS, TOY_NUM_KV_HEADS, TOY_HEAD_DIM
    scale = head_dim ** -0.5
    # toy inputs (provided): 2 seqs of lengths [3, 2]
    q = torch.randn(5, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(5, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(5, num_kv_heads, head_dim, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    try:
        # TODO(student) ①: prefill — flash_attn_varlen_func contract (attention.py:67-70)
        prefill_out = fa_varlen(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                max_seqlen_q=3, max_seqlen_k=3, softmax_scale=scale, causal=True)
        # decode cache (provided): paged (num_blocks, block_size, kv_heads, head_dim)
        q_dec = torch.stack([q[2], q[4]])                                  # last-token q per seq
        k_cache = torch.zeros(2, 256, num_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros_like(k_cache)
        k_cache[0, :3], k_cache[1, :2] = k[:3], k[3:5]                     # simple slice prefill
        v_cache[0, :3], v_cache[1, :2] = v[:3], v[3:5]
        block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
        cache_seqlens = torch.tensor([3, 2], dtype=torch.int32, device=device)
        # TODO(student) ②: decode — flash_attn_with_kvcache contract (attention.py:72-74)
        decode_out = fa_kvcache(q_dec.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=cache_seqlens, block_table=block_table,
                                softmax_scale=scale, causal=True).squeeze(1)
    except FAContractError:
        raise
    except Exception as e:
        raise FAContractError(
            f"flash_attn call failed — likely wrong arguments. "
            f"Mirror attention.py:67-70 (prefill) / 72-74 (decode).\nOriginal: {e}") from e
    return prefill_out, decode_out
```

### 3.2 两个 FA 调用契约逐行对比

**① prefill——`flash_attn_varlen_func`**（对齐 `attention.py:67-70`）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `q, k, v` | `(5,4,64)` / `(5,1,64)` / `(5,1,64)` | **packed**——两条 seq 的 token 拼成第 0 维，无 padding。GQA：`num_heads=4, num_kv_heads=1`。 |
| `cu_seqlens_q` | `[0, 3, 5]` | packed 前缀和切段：token 0–2 属 seq0、token 3–4 属 seq1。 |
| `cu_seqlens_k` | `[0, 3, 5]` | 同上（普通 prefill 时 q/k 等长；prefix-cache 命中时 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`，见 `model_runner.py:162`）。 |
| `max_seqlen_q` / `max_seqlen_k` | `3` | packed 里最长那条的长度（`max(3,2)=3`）。FA 内部用来开 kernel。 |
| `softmax_scale` | `head_dim ** -0.5 = 1/8` | 注意力缩放。 |
| `causal` | `True` | 每条 prompt 内部做下三角 mask（token i 只看 token 0..i）。 |

**不传** `block_table`——普通 prefill 时本步新算的 K/V 就是注意力用的 K/V（prefix-cache 命中时才传，对齐 `attention.py:65-70`）。输出 `prefill_out` 形状 `(5, 4, 64)`——每个输入 token 一个输出。

**② decode——`flash_attn_with_kvcache`**（对齐 `attention.py:72-74`）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `q_dec.unsqueeze(1)` | `(2, 1, 4, 64)` | **必须 `unsqueeze(1)` 注入 `seq_len=1` 维**！kvcache 接口要求 query 是 `(batch, seqlen, heads, D)`，而 `q_dec` 只是 `(2, 4, 64)`。漏 `unsqueeze` 会 shape 报错（见 §3.4）。 |
| `k_cache, v_cache` | `(2, 256, 1, 64)` | **paged 缓存**（不传 k/v）——`(num_blocks, block_size, num_kv_heads, head_dim)`。2 个 block、`block_size=256`（`flash_attn_with_kvcache` 要求 block size 整除 256，对齐 `config.py:17`）。 |
| `cache_seqlens` | `[3, 2]` | 每条 seq 在 cache 里的**有效 KV 长度**（seq0 有 3 个、seq1 有 2 个）。 |
| `block_table` | `[[0], [1]]` | 逻辑 token → 物理 block 映射：seq0 的 token 在 block 0、seq1 的在 block 1。 |
| `softmax_scale` / `causal` | `1/8` / `True` | 同 prefill。 |

**不传** `cu_seqlens` / `max_seqlen`——decode 每条 seq 只 1 个 query，靠 `cache_seqlens` + `block_table` 定位历史 KV。输出 `(2, 1, 4, 64)`，`.squeeze(1)` 去掉 seq_len 维得 `(2, 4, 64)`。

> **`except FAContractError: raise` vs `except Exception as e: raise FAContractError(...)`**：前者让"已经包装过的"错误原样上抛（避免二次包装）；后者把 FA 内部抛的 `RuntimeError`（shape 不匹配、维度错等）**包装**成带诊断提示的 `FAContractError`，提示信息指向 `attention.py:67-70/72-74`——让学生一眼看出"是调参错了，去看源码对照"。

### 3.3 值交叉验证为何成立

`_verify_fa_calls`（`solution/lab_solved.py:88-102`）的核心断言：

```python
expected = torch.stack([prefill_out[2], prefill_out[4]])
ok = (decode_out.shape == expected.shape
      and torch.allclose(decode_out, expected, atol=1e-2))
```

即 **`decode_out ≈ torch.stack([prefill_out[2], prefill_out[4]])`**。为什么成立？

- `q_dec = torch.stack([q[2], q[4]])`——decode 取的是**每条 seq 最后一个 token 的 query**：seq0 末 token 是第 2 个（长度 3 → index 0,1,2）、seq1 末 token 是第 4 个（packed 偏移 3 + 长度 2 的末位 = index 4）。
- decode 的 cache 装的就是 prefill 算的那批 K/V：`k_cache[0,:3]=k[:3]`（seq0 的 3 个 KV）、`k_cache[1,:2]=k[3:5]`（seq1 的 2 个 KV）——**与 prefill 的输入 K/V 完全相同**。
- 于是 decode 的"末 token query 对全历史 KV 做 attention"，在数值上等价于 prefill 时"末 token 的输出"——因为 prefill 的 causal attention 让 token 2 恰好看 token 0,1,2、token 4 恰好看 token 3,4，与 decode 里 seq0 看 cache[0,:3]、seq1 看 cache[1,:2] 是**同一组 (q, K, V)**。
- 所以 `decode_out[0]`（seq0 末 token 输出）≈ `prefill_out[2]`、`decode_out[1]`（seq1）≈ `prefill_out[4]`，`stack` 后逐元素 `allclose`。

这是一个**强自检**：如果 decode 的 `cache_seqlens` 或 `block_table` 填错（比如 `cache_seqlens=[2,3]` 颠倒、或 `block_table=[[1],[0]]` 对调），decode 会用错 KV、结果立刻对不上。`atol=1e-2` 容忍 bfloat16 数值精度（lab 在 `cuda + bfloat16` 跑、精度 ~1e-2 量级）。

### 3.4 `FAContractError` 诊断：常见错误及提示

学生最可能犯的几类错，以及 `FAContractError` 包装后**原始异常信息**会指向什么：

| 错误 | 现象（FA 抛的原始异常） | 诊断要点 |
|------|------------------------|----------|
| **漏 `q_dec.unsqueeze(1)`** | `RuntimeError: q must have shape (batch, seqlen, nheads, headdim)` 之类的 shape 断言失败 | kvcache 接口要求 `(batch, seqlen, heads, D)` 4 维，`q_dec` 只有 3 维 → 必加 `.unsqueeze(1)` 注入 `seqlen=1`。 |
| **`cu_seqlens_q/k` 写反或漏传** | `RuntimeError`/`IndexError`——varlen 内部按 `cu_seqlens` 切段、索引越界或长度不匹配 | prefill 必须**同时**传 `cu_seqlens_q` 和 `cu_seqlens_k`（即便相等），且长度 = `num_seqs + 1 = 3`、首项 0、末项 = `total = 5`。 |
| **`cache_seqlens` / `block_table` 不对齐** | 不一定报错——可能"算完但结果错"（值交叉验证那步 `allclose` fail） | `cache_seqlens[i]` 必须等于 seq i 在 `k_cache[block_table[i]]` 里**实际写入**的 token 数；`block_table[i]` 必须指向 seq i 的 KV 所在物理 block。错位会让 decode 读错 KV。 |
| **漏 `.squeeze(1)`** | `decode_out` 多一维、`_verify_fa_calls` 的 shape 断言 fail（`(2,1,4,64) != (2,4,64)`） | kvcache 输出 `(num_seqs, 1, H, D)`，`.squeeze(1)` 去掉注入的 seq_len 维、与 prefill 输出对齐。 |
| **漏 try/except 或吞掉异常** | 没有 `FAContractError` 包装、裸 `RuntimeError` 直接上抛 | lab 的 `_verify_fa_calls` 和单测**期望** `FAContractError`（带诊断提示）；裸异常会让"调错参数"和"环境问题"混在一起难定位。 |

> 提示信息里的 `Mirror attention.py:67-70 (prefill) / 72-74 (decode)` 是让学生**对照引擎源码**而不是看文档——`attention.py` 的那两行就是权威调用形状。

---

## Task 4：explain 参考答案 + 判分要点（rubric）

Task 4 是 observe + explain，**不写代码**（`run_checks` 只验证观察项，解释项对照本节自检）。共 4 个问题，每题给判分要点。

### 4.1 为什么 prefill 用 packed varlen，而非 padded batched attention？

**参考答案（要点）：**

prefill 时一批里多条 prompt 长度不一（本课 lab 两条 prompt 一条 ~600 token、一条 ~30 token）。padded batched attention 要把所有 prompt **pad 到同一最大长度**（都 pad 到 600），短 prompt 多出来的 570 个位置**全是在算 padding 的废 attention**（softmax 对 padding 会被 mask、但 QK^T 和 softmax 的算力照样花掉）。算力浪费 = `(600-30) * 600 / (600*600)` 量级。

varlen 把所有 prompt 的 token **紧密排列**（`total = a + b`），靠 `cu_seqlens_q = [0, a, a+b]` 切段——**每个 token 都是真实 token、零 padding**。长 prompt 多算、短 prompt 少算，总算力 = 真实 attention 的下界。对 prefill 这种**算力密集**阶段（一次处理整条 prompt 的所有 token 对），省掉 padding 的算力收益巨大；decode 每步每 seq 只 1 个 query、算力本就小、padding 不疼（且 decode 用 kvcache 接口、不走 varlen）。

**判分要点（rubric）：**

| 必答点 | 通过标准 |
|--------|----------|
| **① padding 浪费算力** | 明确指出 padded batched 要把短 prompt pad 到最长、padding 位置仍在算 QK^T/softmax（被 mask 但算力已花）。 |
| **② varlen 无 padding、token 紧凑** | 明确指出 varlen 靠 `cu_seqlens` 切段、所有 token 真实、零浪费。 |

**加分项（非必需）：** 提到 prefill 是算力密集阶段、省 padding 收益大；decode 算力小/走 kvcache 接口所以不需要 varlen。

**常见错误（扣分）：** 答成"varlen 更快因为并行"（没说清是 padding 省）；或答成"varlen 处理变长"（只说现象没说动机）；或把原因归到 `cu_seqlens` 的实现细节而非算力经济性。

### 4.2 `flash_attn_with_kvcache` 的 `q.unsqueeze(1)` / `cache_seqlens` / `block_table` 各自的职责？

**参考答案（要点）：**

- **`q.unsqueeze(1)`**：kvcache 接口要求 query 形状是 `(batch, seqlen, heads, D)`，而 decode 时每条 seq 只 1 个新 query、`q_dec` 是 `(num_seqs, heads, D)` 3 维。`unsqueeze(1)` 在第 1 位**注入一个 `seq_len=1` 维**，凑成 `(num_seqs, 1, heads, D)`——告诉 FA"每条 seq 只查 1 个 token"。prefill 的 q 本来就是 `(total, heads, D)`（packed、无显式 seq_len 维，靠 `cu_seqlens` 切段），**所以 prefill 不需要 unsqueeze**——两路的 q 维度约定不同。

- **`cache_seqlens = context.context_lens`**：每条 seq 在 paged cache 里的**有效 KV 长度**（一条 list，`[len_seq0, len_seq1, ...]`）。decode 时 FA 要知道"这条 seq 的历史 KV 有多长"才能算 attention——但 KV 全在 paged cache 里、不是连续张量，`cache_seqlens` 就是**逻辑长度**信号。对应 `attention.py:73`（`cache_seqlens=context.context_lens`）。

- **`block_table = context.block_tables`**：逻辑 token → 物理 block 的映射矩阵 `(num_seqs, max_blocks_per_seq)`。`block_table[i, j]` 告诉 FA"seq i 的第 j 个逻辑 block 在物理 block 几"。paged cache 把每条 seq 的 KV **分散**存到不连续的物理 block 里（L3 会讲 block 分配/回收），FA 靠 `block_table` 才能把"逻辑上连续的历史 KV"拼回来读。对应 `attention.py:73`（`block_table=context.block_tables`）。

**两者职责区分**：`cache_seqlens` 说"有多长"（标量长度）、`block_table` 说"在哪几个物理块"（地址映射）。缺任何一个 FA 都不知道去哪读 KV。

**判分要点（rubric）：**

| 必答点 | 通过标准 |
|--------|----------|
| **① `unsqueeze(1)` 注入 seq_len=1 维** | 明确指出 kvcache 接口要求 `(batch, seqlen, heads, D)`、decode 每 seq 1 query 所以注入 `seqlen=1`；prefill 的 q 是 packed `(total, H, D)`、无 seq_len 维所以**不**需要 unsqueeze。 |
| **② `cache_seqlens` = 每条 seq 有效 KV 长度** | 明确指出它是逻辑长度信号、告诉 FA 每条 seq 历史有多长；值来自 `context.context_lens`。 |
| **③ `block_table` = 逻辑 token → 物理 block 映射** | 明确指出 paged cache 的 KV 分散在不连续物理 block、`block_table` 是地址映射让 FA 拼回逻辑连续；值来自 `context.block_tables`。 |
| **④ 两者职责区分** | 明确 `cache_seqlens` = 有多长、`block_table` = 在哪几块。 |

**常见错误（扣分）：** 把 `cache_seqlens` 和 `block_table` 混为一谈；或答 `unsqueeze` 是"加 batch 维"（其实是 seq_len 维，batch 维本来就有）。

### 4.3 prefill vs decode 在 FA 用法上的根本差异？

**参考答案（要点，对齐 TUTORIAL §3.3.3 四点）：**

| 维度 | prefill（`flash_attn_varlen_func`） | decode（`flash_attn_with_kvcache`） |
|------|-------------------------------------|-------------------------------------|
| **query 数量** | 多 query——一次处理**整条 prompt 的所有 token**（packed `total` 个） | 单 query/seq——每步每 seq 只产 **1 个新 query** |
| **K/V 来源** | **本步新算的** K/V（`attention.py:62-63` 先 `store_kvcache` 写进 cache，但 attention 用的还是这步 forward 算出来的 k, v）；prefix-cache 命中时换成 `k_cache/v_cache` | **paged 缓存** `k_cache/v_cache`——不重算、不转发 K/V，FA 直接从 cache 读历史 |
| **序列边界信号** | `cu_seqlens_q/k`（packed 前缀和切段） | `cache_seqlens`（每条有效长度）+ `block_table`（逻辑→物理块） |
| **算力/访存特征** | 算力密集（多 query × 多 KV） | 访存密集（单 query × 全历史 KV，瓶颈在读 cache） |

**根本差异一句话**：prefill 是"**多 query、KV 本步新算**"（既要算 K/V 又要做 attention，算力活）；decode 是"**单 query、KV 全来自缓存**"（只做 attention、K/V 不重算，访存活）。这就是为什么两路要用不同 FA 接口——varlen 优化 packed 多 query 的算力、kvcache 优化从 paged cache 读 KV 的访存。

**判分要点（rubric）：**

| 必答点 | 通过标准 |
|--------|----------|
| **① prefill 多 query、decode 单 query/seq** | 明确 prefill 一次算整条 prompt 的所有 token、decode 每步每 seq 只 1 个新 query。 |
| **② K/V 来源不同（最关键）** | 明确 prefill 的 KV 是**本步新算**的、decode 的 KV **全来自 paged 缓存**（不重算、不转发 K/V）。 |
| **③ 序列边界信号不同** | 提到 prefill 用 `cu_seqlens`、decode 用 `cache_seqlens` + `block_table`（不必背全四个差异，但至少点到边界信号这一项）。 |

**加分项：** 点出 prefill 算力密集 vs decode 访存密集；或提到 prefix-cache 命中时 prefill 也读 cache（但那是 L4 内容、本题不强求）。

**常见错误（扣分）：** 只答"prefill 算 prompt、decode 算 completion"（这是 L1 视角、没答到 FA 用法差异）；或答"prefill 用 varlen、decode 用 kvcache"（只列接口名、没说为什么不同）。

### 4.4 为什么 cudagraph 的 `-1` 哨兵必要？（额外 run，Q3b）

**参考答案（要点）：**

cudagraph（`capture_cudagraph`）要求输入张量**地址静态**——capture 后形状/指针不能变。nano-vllm 用**固定大小的静态缓冲** `graph_vars`（`model_runner.py:250-257`），按最大 batch 预分配（`max_bs = min(max_num_seqs, 512)`，`model_runner.py:226`）。每步 decode 把真实数据 copy 进去：

```python
# model_runner.py:206-207
graph_vars["slot_mapping"].fill_(-1)              # 206  先全填 -1
graph_vars["slot_mapping"][:bs] = context.slot_mapping   # 207  再覆盖前 bs 个真实槽
```

问题：真实 batch `bs` 往往 < `max_bs`（比如 `bs=2` 但缓冲开到 512），多出来的 `[bs:]` 位置怎么办？答案是**先用 `fill_(-1)` 全填哨兵、再用 `[:bs]` 覆盖真实槽**——于是 `[bs:]` 全是 `-1`。

`-1` 的作用在 `store_kvcache_kernel`（`attention.py:23`）：

```python
# attention.py:21-23
idx = tl.program_id(0)
slot = tl.load(slot_mapping_ptr + idx)
if slot == -1: return                        # 23  ← 哨兵！
```

每个 program 处理一个 token，`slot == -1` 就**直接 return、不写任何东西**。于是静态缓冲里 `[bs:]` 那些 `-1` 槽位的 program 全部 no-op。

**缺了哨兵会怎样**：如果不用 `-1` 填充、`[bs:]` 残留上一步的 stale slot（或初始化的 0），`store_kvcache` 会把这些**幽灵 token**的 K/V **散射进 slot 0**（或上一步的旧 slot）——直接污染缓存。这就是为什么 cudagraph 路径**必须**用 `-1` 哨兵占位。

> **对照 eager 路径**：eager 模式（`enforce_eager=True`，Task 1–4 全程）下 `slot_mapping` 是按真实 `bs` 动态构造的、长度恰好 = token 数，没有"多出来的位置"问题，所以不需要 `-1` 填充。`-1` 哨兵是 cudagraph 静态缓冲**独有**的需求。

**判分要点（rubric）：**

| 必答点 | 通过标准 |
|--------|----------|
| **① cudagraph 静态缓冲固定大小** | 明确指出 graph capture 要求地址/形状静态、缓冲按 `max_bs` 预分配、真实 batch 往往更小 → 多出位置。 |
| **② `fill_(-1)` + `[:bs]` 覆盖机制** | 明确先全填 `-1`、再覆盖前 `bs` 个真实槽，引用 `model_runner.py:206-207`。 |
| **③ `-1` 让 kernel no-op** | 明确 `store_kvcache_kernel`（`attention.py:23`）`if slot == -1: return` 跳过哨兵槽、不写任何东西。 |
| **④ 缺哨兵 → stale KV 污染 slot 0** | 明确不用 `-1` 的话幽灵 token 的 K/V 会散射进 slot 0 / 旧 slot、污染缓存。 |

**加分项：** 提到 eager 路径不需要哨兵（动态构造、长度恰好）；或点出 prefill 永远 eager（形状变化大、不 capture）所以哨兵只在 decode + cudagraph 路径出现。

**常见错误（扣分）：** 答成"`-1` 是 padding"（不是 padding，是 no-op 哨兵）；或答"`-1` 让 attention 跳过"（attention 不读 `slot_mapping`，是 `store_kvcache` kernel 跳过）；或只说"cudagraph 需要"没说清污染后果。

---

## Task 4 观察（observe）项 —— `run_checks` 验证

Task 4 的**观察**部分由 `run_checks` 自动验证（不判分、但学生能看到 PASS/FAIL），对应两条 check：

- `Task4 prefill cu_seqlens_q == [0,a,a+b]`：prefill 记录的 `cu_seqlens_q` 单调递增、首项 0、长度 ≥ 2（packed 切段前缀和的几何性质）。
- `Task4 decode q.shape is (num_seqs,H,D)`：decode 记录的 `q_shape[0] == len(context_lens)`（每条 seq 一个 query）。

学生在 trace 打印里能看到：step 0 PREFILL 的 `cu_q=[0, a, a+b]`、后续 DECODE 的 `q=(num_seqs,4,64)`、`ctx_lens=[..,..]`——这就是 §1.2 trace 形态的实测验证。

---

## TODO → 参考答案映射

`lab.py` 有三处 `TODO(student)`，**一一对应**到 `lab_solved.py` 的三个函数体：

| `lab.py` 位置 | TODO | `lab_solved.py` 对应实现 |
|---------------|------|--------------------------|
| `def traced_attention_forward(self, q, k, v)`（Task 1，`lab.py:41-60`，def 在第 41 行） | TODO(student) — Task 1 (trace)：把本步 FA 调用形状记进 `_trace`（只录 layer 0），再调用 `_orig_attention_forward` | `lab_solved.py:41-62` 整个函数体（见上 §1.1） |
| `def prefill_slot_mapping(block_table, block_size, start, num_tokens)`（Task 2，`lab.py:63-70`，def 在第 63 行） | TODO(student) — Task 2 (slot geometry)：复刻 `model_runner.py:151-161` 的 prefill slot 散射，返回长度 `num_tokens` 的 list | `lab_solved.py:65-85` 整个函数体（见上 §2.1） |
| `def simulate_fa_calls(fa_varlen, fa_kvcache, device, dtype)`（Task 3，`lab.py:90-112`，def 在第 90 行） | TODO(student) — Task 3 (FA calls)：在 toy 输入上调两个 FA 接口、返回 `(prefill_out, decode_out)`，错参抛 `FAContractError` | `lab_solved.py:105-142` 整个函数体（见上 §3.1） |

Task 4（observe + explain）**不写代码**：观察项由 `run_checks` 自动验证、解释项对照本文 §4 自检。`lab.py` 的 `traced_postprocess`（`lab.py:29-38`）和 `_verify_fa_calls`（`lab.py:73-87`）是 **PROVIDED**（非 TODO），直接出现在学生文件里。因此 `lab.py` 只有三个代码 TODO，与 `lab_solved.py` 的三个函数体**精确对应**——`lab_solved.py` 就是 `lab.py` 把三处 `raise NotImplementedError(...)` 换成实现后的版本（见 `lab_solved.py:1-7` 模块 docstring 的说明）。

---

## 附：自检清单（reviewer 用）

- [ ] §1.1 代码与 `lab_solved.py:41-62` **逐字一致**（含 docstring）。
- [ ] §2.1 代码与 `lab_solved.py:65-85` **逐字一致**（含 docstring）。
- [ ] §3.1 代码与 `lab_solved.py:105-142` **逐字一致**（含 docstring）。
- [ ] §1.2 trace 形态描述覆盖：两条 prompt 同一 prefill step（packed `(total,4,64)`）、`cu_seqlens_q=[0,a,a+b]`、`slot_mapping` 长 `a+b`；decode 步 `q=(2,4,64)`、`context_lens=[a+1,b+1]`、`slot_mapping` 两单点；prefill 记录存 `cu_seqlens`、decode 存 `context_lens`。
- [ ] §2.2 三段偏移表覆盖首/中/末三段 + 总槽数 = `num_tokens`。
- [ ] §2.3 解释 live run 为何只覆盖 `start=0`（fresh prefill）+ `start>0` 的两种场景（chunked prefill / prefix-cache 命中）。
- [ ] §3.2 两个 FA 调用的参数表覆盖 q 形状、K/V 来源、序列边界信号差异；§3.3 值交叉验证（`q_dec=stack([q[2],q[4]])` + cache 同批 KV）讲清为何 `decode_out ≈ stack([prefill_out[2],prefill_out[4]])`。
- [ ] §3.4 `FAContractError` 诊断覆盖：漏 `unsqueeze`、`cu_seqlens` 错、`cache_seqlens/block_table` 不对齐。
- [ ] §4.1–§4.4 每题有判分要点（必答点 + 通过标准），且点出常见错误。
- [ ] §4.4 引用 `model_runner.py:206-207` + `attention.py:23` 与当前仓库一致（已 grep 核对：`:206-207` 是 `fill_(-1)` + `[:bs]` 覆盖；`:23` 是 `if slot == -1: return`）。
- [ ] 所有行号引用（`attention.py:67-70` / `72-74` / `62-63` / `21-23`、`model_runner.py:139-141` / `151-161` / `162` / `181` / `206-207` / `226` / `250-257`、`context.py:18` / `23`、`scheduler.py:90-92`）与当前仓库一致。
- [ ] §"TODO → 参考答案映射"三条一一对应，无遗漏。
- [ ] 无残留 `TODO`/`FIXME`/debug print（除引用 `lab.py` 里的 `TODO(student)`）。
