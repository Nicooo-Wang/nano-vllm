# nano-vllm：从请求到输出的 LLM 推理全流程（系统视角）—— 课程大纲

> 产出方式：架构师 → 工程师 → Reviewer 多轮协作（**v2 已正式收敛**，详见文末）。所有内容锚定 nano-vllm 真实源码（`file:line`），无法确认者标 **待确认**。

## 课程定位

本课以 **nano-vllm**（~1,450 行、21 个 `.py` 文件、仅支持 Qwen3 的极简 vLLM 复刻）为教材，
面向具备 **系统 / 内核 / 推理优化** 背景的 OPERATOR 开发者。

课程目标是让学员 **理解整条 LLM 推理流程**（process-understanding，高粒度），而非重写每个内部函数。
沿 **『一个 request 从进入到输出 token』** 展开：请求全流程 → KV 缓存（从 flash-attention 接口出发）
→ Block/BlockManager 与 Sequence↔Block → 前缀缓存命中 → 连续批处理 → 模型前向与张量并行 → profiling 优化。

每节课 = **tutorial + lab**，lab 为高粒度编程作业：**跟踪 / 观察 / 解释 + 至多一个关键小实现**，
不逐函数留空。验证手段：trace-log / smoke / golden / shape / benchmark-sanity。**性能剖析在 L8 收口**，
要求 profile-before/after 的优化闭环。

**学员环境**：8× H200（用户确认；`nvidia-smi` 显示为 L20X，以用户说法为准）。⚠️ 开课前的前置依赖（torch/triton/flash_attn/nsys/模型）见文末。

## 教学形态（统一为高粒度）

- 所有 lesson 的 lab 均为 **高粒度**：以 *跟踪 / 观察 / 解释* 为主 + *至多一个关键小实现/扩展*，**不**逐函数留空、**不**从零搭项目。
- **L8** 为 profile-driven 优化 capstone：测量 → 定位 → 优化（CUDA graph 桶 / 前缀缓存命中 / torch.compile 融合，概念与配置层）→ 重测。

## Milestone 路线图

| Milestone | 主题 | Lessons |
|---|---|---|
| **M1** | 理解请求全流程与 KV 缓存主干（lifecycle + KV 管理 + 前缀缓存） | L1, L2, L3, L4 |
| **M2** | 理解批处理、模型结构与张量并行，并用 profiling 收口（连续批处理 + 模型/TP + profiling 优化） | L5, L6, L7, L8 |

## 课程总览（Lesson 索引）

| # | 主题 | Theme | M | 锚定源码（节选） |
|---|---|---|---|---|
| [L1](lessons/L01.md) | 一个 request 的完整旅程：从 llm.generate 到 token 输出 | ①请求全流程 | M1 | `nanovllm/engine/llm_engine.py:60-90`, `nanovllm/engine/llm_engine.py:43-55` |
| [L2](lessons/L02.md) | KV 缓存的接口与物理形态：从 flash-attention 调用到 paged 张量 | ②KV cache | M1 | `nanovllm/layers/attention.py:59-75`, `nanovllm/layers/attention.py:10-40` |
| [L3](lessons/L03.md) | Block、BlockManager 与 Sequence<->Block 的映射：KV 缓存的管理抽象 | ②KV cache | M1 | `nanovllm/engine/sequence.py:55-65`, `nanovllm/engine/block_manager.py:8-24` |
| [L4](lessons/L04.md) | 缓存命中 / 前缀缓存：内容寻址的 KV 复用 | ③缓存命中 | M1 | `nanovllm/engine/block_manager.py:35-41`, `nanovllm/engine/block_manager.py:58-73` |
| [L5](lessons/L05.md) | 连续批处理：调度、prefill/decode 交替、chunked prefill 与 preemption | ④连续批处理 | M2 | `nanovllm/engine/scheduler.py:25-73`, `nanovllm/engine/scheduler.py:42` |
| [L6](lessons/L06.md) | 模型前向结构：Qwen3 解码层、融合残差 RMSNorm 与 compute_logits 拆分点 | ⑤模型/TP | M2 | `nanovllm/models/qwen3.py:120-216`, `nanovllm/models/qwen3.py:146-159` |
| [L7](lessons/L07.md) | 张量并行切分：Attention/MLP 分片，以及 KV 缓存为何在 rank 间不传输 | ⑤模型/TP | M2 | `nanovllm/engine/model_runner.py:103-121`, `nanovllm/engine/model_runner.py:110` |
| [L8](lessons/L08.md) | Profiling 与 profile 驱动的优化：定位 decode 瓶颈并消除 launch 开销 | profiling | M2 | `nanovllm/engine/model_runner.py:195-212`, `nanovllm/engine/model_runner.py:222-257` |

## Profiling 与优化主线

Profiling 作为一条贯穿主线，从概念铺垫逐步收口到 capstone 的 profile 驱动优化。L1 的 num_tokens 正负号与吞吐 postfix 让学生第一次看到 prefill/decode 性能量级差异；L2 在观察 flash-attention 双路径与 slot_mapping 几何时让学生第一次接触"kernel 数量"概念（store_kvcache + flash-attn 两个独立 kernel），并在 Q3 提前让学生必须区分 eager 分支与 cudagraph 分支（enforce_eager=True 走 eager，enforce_eager=False 才走 graph.replay 与 slot_mapping.fill_(-1) 哨兵，model_runner.py:197-212）；L5 的吞吐曲线（prefill-tok/s vs decode-tok/s）让学生直观看到 decode 与 prefill 的性能量级差异；L6 讲清 compute_logits 为何是 cudagraph 的停止点、graph_bs 桶选择，为 L8 的 launch-overhead 优化铺垫；L7 的 TP 走查让学生理解 all_reduce/gather 的通信开销与 KV 分片无传输的事实。最终在 L8（也是 capstone）集中爆发：用 torch profiler / nsys / ncu 三件套，定位小 batch decode 的真实瓶颈是 kernel launch overhead 而非计算，然后做"测量->定位(cudagraph 桶/torch.compile 融合/前缀缓存命中)->优化->重测"的完整闭环。profiling 不是孤立的某一课，而是从 L1 到 L8 的概念递进，capstone 把它落地为可量化的优化决策。

## Capstone 设计（L8）

学生在 L8 已经定位到「decode 是 launch-overhead 受限」。Capstone 设计：要求学生用同一 bench.py（位于 repo 根，bench.py:15-28，其中 enforce_eager=False 在 line 15，generate 与 throughput 在 line 22-28）在 enforce_eager=True 与 enforce_eager=False 两种模式下测量 decode 吞吐与单步 GPU 活动时间占比，定位差距来源（cudagraph 开关：model_runner.py:36-37；大 batch 跳图阈值 model_runner.py:197）。然后引入一个 profile-driven 的概念优化选择：(a) 调整 capture_cudagraph 的 graph_bs 桶（model_runner.py:234 当前为 [1,2,4,8]+list(range(16,max_bs+1,16))）使小 batch 桶更密、减少 padding 浪费；(b) 提升前缀缓存命中率（构造共享 system-prompt 的请求集，对比 BlockManager ref_count 复用前后 prefill 时间，block_manager.py:83-88）；(c) 概念性理解 torch.compile 融合点（layernorm/rotary/silu/sampler 的 @torch.compile，见 layernorm.py:16,28 / rotary_embedding.py:37 / activation.py:8 / sampler.py:7）对 kernel 数量的影响。最后重测吞吐并解释哪一项带来最大收益、为什么——把「测量->定位->优化->重测」的闭环走完。优化停留在概念与配置层（CUDA graph 桶、缓存命中、融合），不做逐 kernel 改写。

---

## 各 Lesson 详细大纲

## L1 — 一个 request 的完整旅程：从 llm.generate 到 token 输出

_Milestone: M1_  |  _Theme: 1_  |  _置信度: confirmed_

**锚定源码**

- `nanovllm/engine/llm_engine.py:60-90`
- `nanovllm/engine/llm_engine.py:43-55`
- `nanovllm/engine/scheduler.py:81-92`
- `nanovllm/engine/scheduler.py:25-73`
- `nanovllm/engine/sequence.py:8-11`
- `nanovllm/engine/sequence.py:18-31`
- `nanovllm/engine/sequence.py:72-83`
- `nanovllm/layers/sampler.py:5-12`
- `nanovllm/llm.py:1-5`
- `example.py:6-29`

**Tutorial 涵盖**

- LLMEngine（被 nanovllm/llm.py 的 LLM 子类直接暴露为对外 LLM）的角色：generate()（llm_engine.py:60-90）是唯一对外入口，tokenize -> 逐 prompt add_request -> while not is_finished() 反复 step()
- add_request 把字符串 encode 成 token_ids，包成 Sequence 丢进 scheduler.waiting（llm_engine.py:43-48；tokenizer 在 llm_engine.py:32 加载，eos 在 llm_engine.py:33 回填 config）
- step() 三段式（llm_engine.py:49-55）：scheduler.schedule() 决定本步算什么 -> model_runner.call("run", seqs, is_prefill) 跑模型拿 token_ids -> postprocess 回填并判定结束
- Sequence 数据结构（sequence.py:18-31）：seq_id/status/token_ids/num_tokens/num_prompt_tokens/num_cached_tokens/num_scheduled_tokens/block_table/is_prefill/temperature/max_tokens
- 为什么 Sequence 是主进程与 GPU worker 之间唯一的 IPC 载体：__getstate__/__setstate__ 只序列化 6 个最小字段（sequence.py:72-83）；decode 时 last_state 是单个 last_token，prefill 时才是完整 token_ids 列表——精简跨进程传输
- SequenceStatus 三态机（sequence.py:8-11）：WAITING -> RUNNING -> FINISHED；postprocess 按 eos / max_tokens 翻转为 FINISHED（scheduler.py:89）
- postprocess 收尾（scheduler.py:81-92）：hash_blocks 登记新块 -> num_cached_tokens 累加 -> is_prefill 未算完则 continue -> 否则 append_token -> 命中 eos 或达 max_tokens 则 FINISHED + block_manager.deallocate 释放并从 running 移除
- 最终输出：generate 按 seq_id 排序、tokenizer.decode 还原文本，返回 {text, token_ids}（llm_engine.py:88-89）

**Lab 概览**

运行 example.py 端到端跑通 Qwen3-0.6B（enforce_eager=True, tp=1），在 llm_engine.step 与 scheduler.postprocess 处加轻量日志插桩，trace 一个 request 从 WAITING 到 FINISHED 的全部状态/数据流转；不重写任何 helper。

**Questions（高粒度：跟踪/观察/解释 + 至多一个关键小项；不逐函数留空）**

- Q1(trace)：在 llm_engine.py:49-55 的 step() 与 scheduler.py:81-92 的 postprocess() 加日志，对 example.py 的两个 prompt 记录每步的 (is_prefill, scheduled_seqs, num_scheduled_tokens, 新生成 token_id)，画出每个 Sequence 的状态机轨迹（WAITING->RUNNING->...->FINISHED）。验证：日志能解释每条 completion 的生成过程，状态流转与 SequenceStatus 枚举（sequence.py:8-11）一致，postprocess 里 is_prefill 且未算完会 continue 不 append token（scheduler.py:86-87）。
- Q2(observe+explain)：enforce_eager 保持 True 只放 1 条 prompt，观察 prefill 阶段 num_scheduled_tokens（可能因 max_num_batched_tokens=16384 一次吃完，config.py:9）与 decode 阶段每步=1 的差异；解释 step() 里 `num_tokens = ... if is_prefill else -len(seqs)`（llm_engine.py:51）如何用一个正负号同时编码『这是 decode』和『本步 token 计数』，并驱动 generate 里 Prefill/Decode 两条吞吐曲线（llm_engine.py:76-79）。
- Q3(key small item)：实现一个最小 wrapper，在 generate 返回前打印每个请求的 num_prompt_tokens / num_completion_tokens / 总 step 数，验证 num_completion_tokens 属性（sequence.py:43-45）与 SamplingParams.max_tokens（sampling_params.py:7）以及 postprocess 终止判断（scheduler.py:89）自洽。

**验证方式**

- Q1: smoke/trace-log（人工核对状态机轨迹与代码枚举一致）
- Q2: trace + 解释（rubric：正确指出 num_tokens 正负号的双重语义及其对吞吐曲线的驱动）
- Q3: unit/shape-log（打印值与 SamplingParams.max_tokens 自洽）

**涵盖内容（skills / concepts）**

- 端到端请求生命周期：llm.generate -> add_request -> step 循环 -> 采样 -> decode
- LLMEngine / Scheduler / Sequence / ModelRunner 四大组件边界与职责
- prompt -> token_ids -> Sequence 的数据流与 Sequence 作为主进程->GPU worker 唯一 IPC 载体
- step() 三段式：schedule -> run -> postprocess
- num_tokens 正/负号编码 prefill/decode 双语义（llm_engine.py:51）
- SequenceStatus 三态机与 eos/max_tokens 终止判定
- Sampler 的 Gumbel-max 近似采样（非 argmax/不取 greedy）
- 通过轻量插桩 trace 理解而非重写

**Profiling / 优化**

none（仅插桩 trace，为后续吞吐概念铺垫）

---

## L2 — KV 缓存的接口与物理形态：从 flash-attention 调用到 paged 张量

_Milestone: M1_  |  _Theme: 2_  |  _置信度: confirmed_

**锚定源码**

- `nanovllm/layers/attention.py:59-75`
- `nanovllm/layers/attention.py:10-40`
- `nanovllm/engine/model_runner.py:103-121`
- `nanovllm/engine/model_runner.py:129-170`
- `nanovllm/engine/model_runner.py:172-188`
- `nanovllm/engine/model_runner.py:195-212`
- `nanovllm/utils/context.py:6-27`

**Tutorial 涵盖**

- 第一步看接口——prefill 调 flash_attn_varlen_func：参数为 q,k,v + max_seqlen_q + cu_seqlens_q + max_seqlen_k + cu_seqlens_k + softmax_scale=self.scale + causal=True + 可选 block_table（attention.py:67-70）。注意：默认 k/v 是本步新算的；当前缀缓存命中（context.block_tables is not None）时，k/v 被换成 k_cache/v_cache（attention.py:65-66）并随 block_table 一起送进 varlen 让它读历史块
- 第二步看接口——decode 调 flash_attn_with_kvcache：参数为 q.unsqueeze(1)（注入一个 query 长度维）+ k_cache + v_cache + cache_seqlens=context.context_lens + block_table=context.block_tables + softmax_scale=self.scale + causal=True（attention.py:72-74）
- 第三步看物理形态：KV 缓存张量形状 (2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)（model_runner.py:115），其中 2 = K/V 两份；num_kv_heads = hf_config.num_key_value_heads // world_size（model_runner.py:110）——TP 下每个 rank 只存自己的 kv heads
- num_kvcache_blocks 的由来（model_runner.py:113）：由剩余显存反推（total * gpu_memory_utilization - used - peak + current) // block_bytes，gpu_memory_utilization 默认 0.9（config.py:12）；该值在 allocate_kv_cache 内被写回 config.num_kvcache_blocks（覆盖默认 -1，config.py:18），随后 Scheduler 在构造时读取（scheduler.py:15）——注意它不是构造期可手动设置的旋钮
- block_bytes 计算（model_runner.py:112）：2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
- kv_cache 如何挂到每层 Attention：遍历 model.modules()，给有 k_cache/v_cache 属性的模块赋值 self.k_cache = kv_cache[0, layer_id]（model_runner.py:116-121）
- Triton store_kvcache kernel（attention.py:10-30）：每个 program 处理一个 token，按 slot_mapping[idx] 把刚算出的 K/V 一行散射进 paged 缓存；slot==-1 哨兵直接 return（attention.py:23），用于 CUDA-graph 静态缓冲保护
- store_kvcache launcher 的 stride/numel 断言（attention.py:36-39）：保证 head_dim 维度连续（stride(-1)==1）、num_heads 维度步长==head_dim、cache 第 1 维步长==D、slot_mapping 长度==token 数 N
- slot_mapping 的几何含义：prepare_prefill 把每个 token 映射到 block_id*block_size + 块内偏移的物理槽（model_runner.py:151-161），首区间起点再 += start%block_size、末区间终点按 end-i*block_size 截断（model_runner.py:154-160）；prepare_decode 用 block_table[-1]*block_size + last_block_num_tokens - 1 单点（model_runner.py:181）
- Context（context.py:6-27：@dataclass 在 line 5、class Context 在 line 6、字段 7-14、set_context/reset_context 辅助函数到 line 27）把这些 per-step 元数据（is_prefill/cu_seqlens_q/k/max_seqlen_q/k/slot_mapping/context_lens/block_tables）放进全局单例 _CONTEXT，实现调度（主进程）与计算（Attention 层）的解耦

**Lab 概览**

本课以 flash-attention 在 prefill/decode 两阶段的**用法**为主线（Q1/Q2 两个 dedicated question），辅以 paged 张量物理形态的观察（Q3）。不重写任何 helper：在 Attention.forward 处插桩，观察两个 FA 接口的真实调用与参数，并构造/解释 varlen 与 paged-kvcache 两种调用契约。

**Questions（高粒度：跟踪/观察/解释 + 至多一个关键小项；不逐函数留空）**

- Q1（prefill 用法 · observe + construct + explain）：聚焦 flash_attn_varlen_func 的 prefill 调用（attention.py:67-70）。(a) 在 attention.py:60 插桩，打印一次 prefill 的 context.cu_seqlens_q、context.cu_seqlens_k、context.max_seqlen_q/k，以及 q/k/v 的 shape（应为 (total_tokens, num_heads, head_dim) 的 packed 形式，非按 seq padding）；(b) 构造验证：对两条 prompt（长度 a、b）手算 cu_seqlens_q=[0, a, a+b]，与打印值核对；(c) 解释两点——为什么 prefill 用 varlen（packed、无 padding）而非 padded batched attention？以及前缀缓存命中分支（context.block_tables is not None 时把 k/v 换成 k_cache/v_cache 并传 block_table，attention.py:65-66,70）如何让 FA 直接从 paged cache 读历史 KV、跳过对已缓存 token 的重算。验证：打印的 cu_seqlens 与手算一致；能正确陈述 varlen packed 的动机与 prefix-cache 分支的作用。
- Q2（decode 用法 · observe + explain）：聚焦 flash_attn_with_kvcache 的 decode 调用（attention.py:72-74）。(a) decode 步打印 q.shape（(num_seqs, num_heads, head_dim)）、q.unsqueeze(1) 后的 shape（注入 query 长度维 1）、context.context_lens、context.block_tables 的 shape；(b) 解释四个用法语义：① 为什么 decode 要 q.unsqueeze(1)（注入 seq_len=1 的 query 维）而 prefill 的 q 不需要？② cache_seqlens=context.context_lens 告诉 FA 每条序列在 paged cache 里有几个有效 KV token；③ block_table=context.block_tables 把逻辑 token 映射到物理 block，让 FA 从 paged cache 读全部历史 KV（decode 不重算也不转发 K/V）；④ 与 prefill 的根本差异（decode 每步每序列只产 1 个 query、KV 全来自缓存；prefill 产多个 query、KV 本步新算 + 可选读缓存前缀）。验证：能正确陈述 unsqueeze 动机、cache_seqlens 与 block_table 各自职责、以及 prefill vs decode 在 FA 用法上的差异。
- Q3（paged 物理形态 · observe + explain）：合并 slot_mapping 几何与 slot==-1 哨兵。(a) 打印长度约 600 的 prompt（block_size=256）prefill 的真实 slot_mapping（prepare_prefill 的 start_block/end_block 循环 model_runner.py:151-161），解释每个区间为何是 block_id*block_size + 块内偏移、首区间为何再 += start%block_size、末区间为何按 token 数截断；(b) 关闭 enforce_eager（用 LLM(..., enforce_eager=False) 或 bench.py）观察 decode 走 graph.replay 分支（model_runner.py:199-212）时 slot_mapping 先 fill_(-1) 再覆盖前 bs 个（model_runner.py:206-207），结合 store_kvcache_kernel 的 if slot == -1: return（attention.py:23）解释为何 CUDA-graph 静态缓冲必须用 -1 哨兵占位、去掉会写到哪里。验证：能正确陈述 slot_mapping 三处偏移几何 + 哨兵作用与越界后果。

**验证方式**

- Q1: trace-log + construct（cu_seqlens 手算与打印一致）+ explain（varlen 动机 + prefix-cache 分支作用）
- Q2: trace-log + explain（unsqueeze 动机 / cache_seqlens 与 block_table 职责 / prefill vs decode 的 FA 用法差异）
- Q3: observe + explain（slot_mapping 三处偏移几何 + 哨兵作用与越界后果）

**涵盖内容（skills / concepts）**

- flash_attn_varlen_func（prefill）与 flash_attn_with_kvcache（decode）两个接口的调用契约
- PagedAttention 的物理张量布局与 slot_mapping 散射原理
- Triton store_kvcache kernel 的程序模型与 slot==-1 哨兵保护
- CUDA graph 静态缓冲与 slot==-1 哨兵的协作（及其仅在 enforce_eager=False 的 decode 路径上可达的前提）
- prefix-cache 命中时 prefill 走 block_table 路径（k/v 换成 k_cache/v_cache）
- prefill vs decode 在 attention 接口与 Context metadata 上的差异
- slot 偏移的几何含义（block_id*block_size + 块内 offset，首区间再 += start%block_size）

**Profiling / 优化**

none（观察 + 解释，为 L8 profiling 铺垫 kernel 数量概念）

---

## L3 — Block、BlockManager 与 Sequence<->Block 的映射：KV 缓存的管理抽象

_Milestone: M1_  |  _Theme: 2_  |  _置信度: confirmed_

**锚定源码**

- `nanovllm/engine/sequence.py:55-65`
- `nanovllm/engine/block_manager.py:8-24`
- `nanovllm/engine/block_manager.py:26-33`
- `nanovllm/engine/block_manager.py:43-121`
- `nanovllm/engine/llm_engine.py:21`
- `nanovllm/config.py:17-22`

**Tutorial 涵盖**

- 三个索引（block_manager.py:31-33）：hash_to_block_id（dict，内容寻址，前缀缓存用）、free_block_ids（deque，空闲块）、used_block_ids（set，在用块）
- Block 载体（block_manager.py:8-24）：block_id / ref_count / hash / token_ids；update(hash, token_ids) 写入内容、reset() 复位为 ref_count=1/hash=-1/空列表
- Sequence<->Block 关系：Sequence.block_table 是物理 block_id 列表（sequence.py:28）；num_blocks = ceil(num_tokens / block_size)（sequence.py:56-57）、last_block_num_tokens（sequence.py:60-61）、block(i) 取第 i 块的 token 切片（sequence.py:63-65）——给出 token 流到物理块的切分
- block_size=256 是 Sequence 类属性默认值（sequence.py:15），在 LLMEngine 构造时按 config.kvcache_block_size 覆盖（llm_engine.py:21），Config 里 assert %256==0（config.py:22）
- append 路径：can_append/may_append 的 `len(seq) % block_size == 1` gate（block_manager.py:103-108）——只有恰好『首 token 落入新块』时才需要分配新块（布尔转 int: True->1, False->0），否则复用尾块剩余槽位
- allocate（block_manager.py:75-92）：命中 used 块 ref_count+=1 复用，命中 free 块取走并置 ref_count=1，未命中部分 _allocate_block 取新块；并设置 seq.num_cached_tokens
- deallocate（block_manager.py:94-101）：逆序遍历 block_table，每块 ref_count-=1，到 0 才 _deallocate_block 真正回收到 free 队列——共享块不会被提前释放
- _allocate_block 的细节（block_manager.py:43-51）：取 free 队首，若该块旧 hash 仍登记则从 hash_to_block_id 删除（避免脏命中），reset 后加入 used
- block_size=256 较大粒度为何对 flash-attention 的 block_table 访问友好：每块连续 256 行 KV，减少 block_table 条目数与间接寻址开销

**Lab 概览**

追踪一个 Sequence 的 block_table 在 prefill 与多次 decode 中如何增长，验证块边界分配规则与物理映射；对比 can_append 与 may_append 的职责分离。

**Questions（高粒度：跟踪/观察/解释 + 至多一个关键小项；不逐函数留空）**

- Q1(trace)：对长度约 600 的 prompt 跑 1 次 prefill + 若干 decode，在每次 step 后打印 seq.block_table 与 len(seq)。验证：prefill 后 block_table 已含 ceil(600/256)=3 块（由 allocate 写入 block_manager.py:89-91）；decode 每跨过块边界（len(seq)%block_size==1）时 block_table 增 1，对应 may_append 的 gate（block_manager.py:107-108）。
- Q2(key item, small extension)：实现一个 helper `seq_to_slots(seq)` 返回当前 seq 全部 token 的物理 slot 列表（用 block_table + block_size + last_block_num_tokens 计算），与 prepare_decode 里单点 slot 公式（model_runner.py:181）对齐——验证最后一个 token 的 slot = seq_to_slots(seq)[-1]。注意末块只填到 last_block_num_tokens 个槽。
- Q3(observe+explain)：对比 can_append（block_manager.py:103-104，返回 bool，判定 free 块是否够）与 may_append（106-108，按需分配新块）的职责分离：为什么『能否 append』和『真正 append』要拆成两步？结合 scheduler 的 preemption 循环（scheduler.py:60-65）解释。验证：学生能正确指出 can_append 是 preemption 决策的前置条件（不够就 preempt 别人腾块），may_append 是分配副作用。

**验证方式**

- Q1: trace-log（块数与长度、append 时机的数值匹配）
- Q2: unit/golden（seq_to_slots 端点与 decode 单点公式一致）
- Q3: explain（rubric：正确解释 can/may 二步与 preemption 决策的耦合）

**涵盖内容（skills / concepts）**

- PagedAttention 的分页抽象：free deque / used set / hash_to_block_id 内容寻址表
- Block 的 ref_count / hash / token_ids 与 update/reset 语义
- Sequence token 流到物理 block 的映射（block_table + num_blocks/last_block_num_tokens/block(i)）
- append 时按块边界分配的惰性策略（len % block_size == 1 gate）
- allocate/deallocate 的引用计数（共享块 ref_count+=1，回收到 0 才 free）
- config -> Sequence.block_size -> prepare_prefill/decode 的纵向打通
- block_size=256 较大粒度对 flash-attention block_table 访问友好的工程理由

**Profiling / 优化**

none

---

## L4 — 缓存命中 / 前缀缓存：内容寻址的 KV 复用

_Milestone: M1_  |  _Theme: 3_  |  _置信度: confirmed_

**锚定源码**

- `nanovllm/engine/block_manager.py:35-41`
- `nanovllm/engine/block_manager.py:58-73`
- `nanovllm/engine/block_manager.py:75-92`
- `nanovllm/engine/block_manager.py:110-120`
- `nanovllm/engine/scheduler.py:29-52`
- `nanovllm/layers/attention.py:64-70`
- `nanovllm/engine/model_runner.py:162-163`

**Tutorial 涵盖**

- 内容寻址的来源：xxhash 链式哈希 compute_hash（block_manager.py:35-41）——prefix != -1 时先把前一块 hash 的 8 字节 little-endian 作 salt（block_manager.py:38-39），再喂本块 token_ids，使前缀哈希天然可链接
- can_allocate 三态（block_manager.py:58-73）：逐块链接哈希查 hash_to_block_id，命中且 token_ids 一致则 num_cached_blocks++（并视情况递减 num_new_blocks）；最后若 free 块不够装新块则返回 -1（KV 压力拒绝），否则返回 num_cached_blocks（可命中块数）
- 跳过末块：循环 `for i in range(seq.num_blocks - 1)`（block_manager.py:62）——最后一块可能未满，内容不稳定不参与寻址
- allocate 的复用分支（block_manager.py:83-88）：命中已 used 块 ref_count+=1（多 Sequence 共享），命中 free 块则从 free 移除置 used 并 ref_count=1；只有 num_cached_blocks 之后的新块才 _allocate_block
- 共享系统 prompt 如何跳过重算：相同前缀只算一次，后续请求复用同一批物理块（ref_count 多个引用），对应 token 不进 prefill 输入（num_tokens = seq.num_tokens - num_cached_blocks * block_size，scheduler.py:39）
- can_allocate 在 scheduler.schedule prefill 分支被消费（scheduler.py:35-45）：== -1 直接 break（KV 不够），否则 allocate 并按 num_cached_blocks 预扣 token 数
- hash_blocks 事后登记（block_manager.py:110-120）：每步 postprocess 把刚算完的整块哈希写回 hash_to_block_id，使后续请求能命中；这是『先算后登记』的设计——本步算完的块下一步才对别人可见

**Lab 概览**

构造共享长前缀的请求集，观察 BlockManager 复用已缓存块（ref_count+=1）从而跳过对前缀的重复计算；trace hash_blocks 的事后登记。

**Questions（高粒度：跟踪/观察/解释 + 至多一个关键小项；不逐函数留空）**

- Q1(key item, small extension)：写一个驱动，先发一条很长的 system-prompt+question 请求让它 prefill 完成，再发 2 条『相同 system prompt + 不同 question』请求，在每条 add_request 前后打印 block_manager.hash_to_block_id 的长度与 can_allocate 返回值。验证：后续请求的 can_allocate 返回正数 = 共享前缀块数（block_manager.py:58-73），且对应块的 ref_count 递增（block_manager.py:83-84 used 分支）。
- Q2(observe+explain)：解释为什么 can_allocate 只对『完整块』做哈希（循环到 seq.num_blocks-1，block_manager.py:62，跳过最后不满块）——末块还会增长无法稳定寻址；并解释为什么 prepare_prefill 里 `if cu_seqlens_k[-1] > cu_seqlens_q[-1]`（model_runner.py:162-163）是『本步命中了前缀缓存』的信号——seqlen_k 含历史已缓存 token 而 seqlen_q 只含本步要算的 token，命中时 k>q 触发建 block_tables 并让 attention 走 paged 路径（attention.py:65-66）。
- Q3(trace)：观察 hash_blocks（postprocess 每步调用，scheduler.py:83）如何把新算完的块事后登记进 hash_to_block_id（block_manager.py:110-120）。验证：登记的块数 = floor((num_cached_tokens + num_scheduled_tokens)/256) - floor(num_cached_tokens/256)；当 start==end（本步没跨过块边界）直接 return（block_manager.py:113）。

**验证方式**

- Q1: smoke + shape-log（ref_count 递增、can_allocate 返回共享块数与预期一致）
- Q2: explain（rubric：正确解释跳过末块 + cu_seqlens_k>q 信号 + attention 路径切换）
- Q3: trace-log（哈希表增量与公式匹配）

**涵盖内容（skills / concepts）**

- 内容寻址前缀缓存：xxhash 链式哈希（前一块 hash 作下一块 salt）
- can_allocate 三态返回（-1=KV 压力拒绝 / >=0=可命中块数）与跳过末块的工程理由
- allocate 的 ref_count 共享复用分支（used 块 +=1，free 块置 1）
- hash_blocks 事后哈希注册（start/end 块区间，跳过未满块）
- 前缀命中如何改写 attention 输入路径（k/v 换成 k_cache/v_cache + block_table）
- 共享系统 prompt 复用的工程价值（跳过重复 prefill 计算）

**Profiling / 优化**

none

---

## L5 — 连续批处理：调度、prefill/decode 交替、chunked prefill 与 preemption

_Milestone: M2_  |  _Theme: 4_  |  _置信度: confirmed_

**锚定源码**

- `nanovllm/engine/scheduler.py:25-73`
- `nanovllm/engine/scheduler.py:42`
- `nanovllm/engine/scheduler.py:54-73`
- `nanovllm/engine/scheduler.py:75-92`
- `nanovllm/config.py:9-12`
- `nanovllm/config.py:18`
- `nanovllm/engine/model_runner.py:103-114`
- `nanovllm/engine/sequence.py:8-11`

**Tutorial 涵盖**

- schedule() 两段式（scheduler.py:25-73）：先尽量 prefill（while waiting 且未超 max_num_seqs），prefill 为空才 decode（while running）
- 双预算：max_num_batched_tokens（token 预算）与 max_num_seqs（seq 预算），均在构造时从 Config 缓存（scheduler.py:11-12，config.py:9-10）
- remaining = max_num_batched_tokens - num_batched_tokens（scheduler.py:32）；==0 直接 break
- chunked prefill 只允许第一条：`if remaining < num_tokens and scheduled_seqs: break`（scheduler.py:42）——首条可被切成多步 prefill（seq.num_scheduled_tokens = min(num_tokens, remaining)，scheduler.py:46），其余条要么整条进要么推迟
- prefill 完成判定：num_cached_tokens + num_scheduled_tokens == num_tokens 才转 RUNNING 并从 waiting 移到 running（scheduler.py:48-51）；否则 seq 仍在 waiting 等下一段 chunk
- 多请求同批共存：同一步内多条 prefill seq 按各 num_scheduled_tokens 共享 token 预算；decode 步每条贡献 1 token（num_scheduled_tokens=1，scheduler.py:67）
- prefill/decode 交替：有 prefill 待算就先算 prefill（return scheduled_seqs, True，scheduler.py:54-55），否则 decode（return ..., False，scheduler.py:73）
- num_kvcache_blocks 的真实来源（model_runner.py:103-114）：它不是构造期可手动设置的旋钮，而是在 allocate_kv_cache 内由剩余显存与 config.gpu_memory_utilization（config.py:12 默认 0.9）反推得到（model_runner.py:113），随后 assert > 0（model_runner.py:114），写回 config.num_kvcache_blocks（覆盖默认 -1，config.py:18），Scheduler 在构造时才读取该已计算值（scheduler.py:15）。因此调整 KV 容量的真实杠杆是 gpu_memory_utilization 或并发请求量，而非直接设 num_kvcache_blocks
- KV 压力下的 preemption：decode 时若 can_append 失败（free 块不够），从 running 尾部 preempt（scheduler.py:60-65）——preempt 把 seq 置 WAITING + is_prefill=True + deallocate 全部块 + appendleft 回 waiting（scheduler.py:75-79），本质是还回 KV、代价是该 seq 重新 prefill 重算
- 顺序保持：decode 末尾 extendleft(reversed(scheduled_seqs))（scheduler.py:72）还原 running 顺序

**Lab 概览**

用 bench.py 风格的多请求负载，插桩 scheduler.schedule 观察一步内如何混合多个不同阶段/长度的请求，并构造 KV 压力场景触发 preemption。

**Questions（高粒度：跟踪/观察/解释 + 至多一个关键小项；不逐函数留空）**

- Q1(trace)：把 max_num_batched_tokens 临时调小（如 512）并发若干长 prompt，打印每步 schedule 返回的 (seqs, is_prefill) 以及每条 seq 的 num_scheduled_tokens，观察『同一步内多条 prefill 共享 token 预算』和『只有首条允许 chunked』。验证：除首条外，其余 prefill seq 的 num_scheduled_tokens 要么是全量、要么因 remaining < num_tokens 且 scheduled_seqs 非空而 break（scheduler.py:42-43）；首条可被 min(num_tokens, remaining) 切片（scheduler.py:46）。
- Q2(observe+explain)：num_kvcache_blocks 不是构造期可手动设置的旋钮——它在 allocate_kv_cache 内由剩余显存与 gpu_memory_utilization（config.py:12 默认 0.9）运行期反推（model_runner.py:113），写回 config.num_kvcache_blocks 后才被 Scheduler 构造读取（scheduler.py:15，默认 -1 见 config.py:18）。因此触发 KV 压力的可行杠杆是：(a) 降低 gpu_memory_utilization（如降到 0.1）让分配出的块数变少，从而自然耗尽 free_block_ids；或 (b) 直接并发多条超长 prompt 把 free 队列占满。任选其一，在 decode 阶段触发 can_append 失败 -> preempt（scheduler.py:60-65）。打印被 preempt 的 seq 重新进入 waiting 左端（scheduler.py:79）并变为 is_prefill=True 的过程，解释 preemption 为何要 deallocate 全部块并重算（block_manager.py:94-101 + is_prefill 翻转 scheduler.py:77）——代价是释放的 KV 下次要重新 prefill。
- Q3(key item, small extension)：generate 的 step 循环已有 Prefill/Decode throughput postfix（llm_engine.py:80-83），让学生据此画一条随时间变化的 prefill-tok/s 与 decode-tok/s 曲线，标注每段对应『多 prefill 混批 / 纯 decode』阶段，并解释 num_tokens 正/负号如何驱动这两条曲线（llm_engine.py:51,76-79）。

**验证方式**

- Q1: trace-log（每步 seq 数/token 数与 chunked-gate 行为匹配代码）
- Q2: trace + explain（rubric：正确描述 preempt 流程 + deallocate/is_prefill 翻转 + 重算代价；正确识别 num_kvcache_blocks 非构造旋钮、gpu_memory_utilization/并发长请求才是触发 KV 压力的真实杠杆）
- Q3: shape-log/demo（吞吐曲线 + 阶段标注 + 正负号语义解释）

**涵盖内容（skills / concepts）**

- 连续批处理与多阶段请求共存（waiting/running 双 deque）
- max_num_batched_tokens / max_num_seqs 双预算（构造时缓存，scheduler.py:11-12）
- chunked prefill 只允许首条的工程权衡（scheduler.py:42-43 的 break）
- num_kvcache_blocks 的真实来源：运行期由 gpu_memory_utilization 反推（非构造期旋钮）
- KV 压力下的 preemption：deallocate 全部块、回 waiting、is_prefill 翻转、重算 trade-off
- prefill/decode 优先级交替（有 prefill 先 prefill）
- prefill 完成才转 RUNNING（scheduler.py:48-51）
- throughput postfix 的 prefill/decode 双曲线语义

**Profiling / 优化**

none（为 L8 的 decode 瓶颈观察埋下吞吐曲线概念）

---

## L6 — 模型前向结构：Qwen3 解码层、融合残差 RMSNorm 与 compute_logits 拆分点

_Milestone: M2_  |  _Theme: 5_  |  _置信度: confirmed_

**锚定源码**

- `nanovllm/models/qwen3.py:120-216`
- `nanovllm/models/qwen3.py:146-159`
- `nanovllm/models/qwen3.py:14-118`
- `nanovllm/layers/layernorm.py:16-50`
- `nanovllm/layers/embed_head.py:45-66`
- `nanovllm/engine/model_runner.py:195-212`
- `nanovllm/engine/model_runner.py:222-257`

**Tutorial 涵盖**

- Qwen3DecoderLayer.forward 双残差结构（qwen3.py:146-159）：input_layernorm(+residual) -> self_attn -> post_attention_layernorm(+residual) -> mlp，hidden_states 与 residual 双线全程传递
- RMSNorm 两种模式（layernorm.py:42-50）：residual is None 走 rms_forward（首层）；否则走 add_rms_forward『先 add 残差再归一化』的融合（layernorm.py:28-40），均 @torch.compile 融合
- Qwen3Attention（qwen3.py:14-88）：qkv_proj(QKVParallelLinear 融合 q/k/v) -> split -> view -> q_norm/k_norm（Qwen3 特有 per-head QK 归一化，qwen3.py:69-70,82-84）-> rotary -> attn -> o_proj(RowParallelLinear)
- Qwen3MLP（qwen3.py:91-117）：gate_up_proj(MergedColumnParallelLinear, gate+up 各切) -> SiluAndMul(@torch.compile, activation.py:8) -> down_proj(RowParallelLinear)
- Qwen3Model.forward（qwen3.py:173-183）：embed_tokens -> 逐层 (hidden, residual) -> 最终 norm(hidden, residual) 收尾
- compute_logits 作为 cudagraph 停止点（qwen3.py:205-216：forward 在 205-210 只返回 hidden_states 进图；compute_logits 在 212-216 单独调用）：model(input_ids, positions) 只返回 hidden_states（hidden_size 固定，进图），lm_head/compute_logits 在 graph 外单独调用（model_runner.py:198, 212）
- 为何 logits 不进图：ParallelLMHead prefill 用 cu_seqlens_q[1:]-1 取每序列末 token（embed_head.py:59-60，索引动态）；logits 维度=vocab 且 tp>1 时需 dist.gather 到 rank0（embed_head.py:62-65，集合通信不宜进静态图）
- tie_word_embeddings（qwen3.py:202-203）：lm_head.weight 直接共享 embed_tokens.weight，Qwen3-0.6B 适用，省一份参数
- capture_cudagraph 桶式捕获（model_runner.py:222-257）：max_bs=min(max_num_seqs,512)；graph_bs=[1,2,4,8]+list(range(16,max_bs+1,16))；reversed 遍历捕获（大桶先建 pool 共享）；graph_vars 复用静态 input_ids/positions/slot_mapping/context_lens/block_tables/outputs；捕获时只跑 model() 不跑 compute_logits

**Lab 概览**

trace 一个 decoder 层的双残差流，定位 compute_logits 为何被排除在 cudagraph 捕获之外（变长 vocab logits + 末 token 索引动态），并观察 graph_bs 桶选择逻辑。

**Questions（高粒度：跟踪/观察/解释 + 至多一个关键小项；不逐函数留空）**

- Q1(observe+explain)：解释 DecoderLayer 的残差隧道（qwen3.py:152-159）：residual is None 时 input_layernorm(hidden_states) 而 residual=原 hidden_states；否则 input_layernorm(hidden_states, residual) 先 add 残差再归一化（add_rms_forward 的 `x.float().add_(residual.float())`，layernorm.py:35）。验证：学生能画出 hidden_states 与 residual 两条线在各层的演化，并指出归一化基于『加完残差后的值』，residual 被更新为 add 后的值（layernorm.py:36）。
- Q2(key item, explain)：定位 compute_logits（qwen3.py:212-216 -> ParallelLMHead）为何是 cudagraph 停止点——model() 只返回 hidden_states（hidden_size 固定，进图），而 compute_logits 在 run_model 里 graph 外调用（model_runner.py:198 eager 分支 / 212 graph 分支都是 model 之后单独调）。prefill 时 ParallelLMHead 用 cu_seqlens_q[1:]-1 取每 seq 末 token（embed_head.py:59-60）。验证：学生能正确指出两个原因——(a) logits 维度=vocab_size 随策略/scale 变且 gather 是集合通信不宜进静态图；(b) prefill 末 token 索引 cu_seqlens_q[1:]-1 动态变化。
- Q3(trace)：打印 capture_cudagraph 的 graph_bs（model_runner.py:234）与 run_model 选桶逻辑 `next(x for x in self.graph_bs if x >= bs)`（model_runner.py:202），对 bs=3/5/20 各选中哪个桶、padding 浪费多少槽位。验证：桶是 [1,2,4,8]+list(range(16, max_bs+1, 16))，bs=3->桶4（浪费1），bs=5->桶8（浪费3），bs=20->桶32（浪费12）；并注意 bs>512 走 eager（model_runner.py:197）。

**验证方式**

- Q1: explain + trace-log（rubric：双残差隧道叙述正确，含 add_rms_forward 更新 residual）
- Q2: explain（rubric：正确指出 cudagraph 停止点的两个原因——vocab 维度/gather + 末 token 索引动态）
- Q3: trace-log（桶选择数值匹配，padding 浪费计算正确，含 >512 走 eager）

**涵盖内容（skills / concepts）**

- Qwen3 decoder 层前向结构（双残差隧道：input_layernorm/attn/post_attention_layernorm/mlp）
- RMSNorm 两种模式：rms_forward（首层/无残差）与 add_rms_forward（融合 add 残差），均 @torch.compile
- Qwen3 特有的 per-head q_norm/k_norm
- compute_logits 作为 cudagraph 停止点的原因（vocab 维度 + prefill 末 token 索引动态）
- VocabParallelEmbedding mask + all_reduce
- ParallelLMHead 取 last-token（prefill cu_seqlens_q[1:]-1）+ gather 到 rank0
- capture_cudagraph 桶式捕获与 graph_vars 覆盖（slot_mapping fill_(-1) 哨兵）

**Profiling / 优化**

none（cudagraph 概念，为 L8 launch-overhead 优化铺垫）

---

## L7 — 张量并行切分：Attention/MLP 分片，以及 KV 缓存为何在 rank 间不传输

_Milestone: M2_  |  _Theme: 5_  |  _置信度: 待确认: 多卡(tp=2..8)端到端跑通 + NCCL tcp://localhost:2333 进程组可用 + nsys trace 需 H200 集群环境，待 prereq 确认；单卡(tp=1)部分已确认可用（example.py）_

**锚定源码**

- `nanovllm/engine/model_runner.py:103-121`
- `nanovllm/engine/model_runner.py:110`
- `nanovllm/layers/attention.py:59-75`
- `nanovllm/models/qwen3.py:14-67`
- `nanovllm/layers/linear.py:96-128`
- `nanovllm/layers/linear.py:131-156`
- `nanovllm/layers/embed_head.py:9-66`
- `nanovllm/engine/model_runner.py:214-220`
- `nanovllm/engine/llm_engine.py:24-31`
- `nanovllm/utils/loader.py:12-28`

**Tutorial 涵盖**

- 进程组与 rank 拓扑：dist.init_process_group("nccl", "tcp://localhost:2333", world_size, rank)（model_runner.py:26）；LLMEngine 在主进程为 rank 1..tp-1 spawn 子进程（llm_engine.py:24-31），rank0 留在主进程
- rank0 主控 + rank>=1 worker 的 IPC：rank0 持 SharedMemory(create=True) + 一组 Event；rank>=1 进 loop() 阻塞 event.wait -> read_shm -> call -> 循环（model_runner.py:41-48,61-74）；call 通过 write_shm 广播 method_name+args（model_runner.py:76-89）
- 切分方向总览：ColumnParallelLinear 按输出维切（output_size//tp，linear.py:54-73），RowParallelLinear 按输入维切（input_size//tp + all_reduce，linear.py:131-156）
- Attention：qkv_proj 用 QKVParallelLinear 融合 q/k/v（num_heads 与 num_kv_heads 都已 divide(total, tp_size) per-rank，linear.py:109-110），o_proj 用 RowParallelLinear（all_reduce，linear.py:154-155）
- MLP：gate_up_proj 用 MergedColumnParallelLinear（gate+up 各切，linear.py:76-93），down_proj 用 RowParallelLinear（all_reduce）
- 关键澄清——KV 缓存按 kv_head 分片：allocate_kv_cache 里 num_kv_heads = hf_config.num_key_value_heads // world_size（model_runner.py:110），kv_cache 第 4 维就是 per-rank kv_heads（model_runner.py:115）；每个 rank 只分配/存储/计算自己的 kv_heads，KV 在 rank 间从不传输也从不广播
- 为何分片可行（核心纠错）：GQA + 头切分使每个 rank 独立对自己那份 kv_heads 做 attention 即可得到本 rank q_heads 的完整结果；q heads 与 kv heads 按同一 tp 边界切分（Qwen3Attention 里 num_heads/num_kv_heads 都 //tp，qwen3.py:32,35），与 kv_cache 分片边界一致——无需其它 rank 的 K/V
- 真实 inter-rank 通信只有三类：(1) RowParallelLinear 的 all_reduce（attn o_proj + mlp down_proj 激活，linear.py:154-155）；(2) VocabParallelEmbedding/ParallelLMHead 的 mask+all_reduce / dist.gather 把 logits 汇到 rank0（embed_head.py:41,62-65）；(3) Sampler 只在 rank0 跑（model_runner.py:216,218），rank>=1 返回 None
- VocabParallelEmbedding 也是 mask + all_reduce（embed_head.py:34-42）：每个 rank 只持有 vocab 的一段，查询时 mask 掉非本段 id 再 all_reduce 合并
- 权重加载如何按 rank 切分：weight_loader 用 narrow/chunk 取本 rank 片段（ColumnParallel linear.py:65-70，RowParallel linear.py:142-150，QKV linear.py:114-128），packed_modules_mapping（qwen3.py:187-193）把 HF 的 q/k/v_proj 拆进融合 qkv_proj、gate/up_proj 拆进 gate_up_proj（loader.py:12-28）

**Lab 概览**

在单卡(tp=1)与多卡(tp=2)各跑 decode，trace 每个通信点，验证 KV 分片、无跨 rank KV 传输、rank0-only 采样；走查单步 decode 每个 rank 各自做什么。

**Questions（高粒度：跟踪/观察/解释 + 至多一个关键小项；不逐函数留空）**

- Q1(key item, trace)：以 tensor_parallel_size=2 运行，在 RowParallelLinear.forward 加日志（linear.py:152-155）确认每层 attention 的 o_proj 与 mlp 的 down_proj 各做一次 all_reduce；在 ParallelLMHead.forward 加日志（embed_head.py:62-65）确认 logits gather 到 rank0；在 run 里确认 sampler 只在 rank0 执行（model_runner.py:216,218）。验证：一次 decode 的 all_reduce 次数 = 2 * num_hidden_layers（attn o_proj + mlp down_proj 各一）+ 1（VocabParallelEmbedding，embed_head.py:41，tp>1 时），gather 次数=1，sampler 输出只在 rank0 非 None。
- Q2(observe+explain，纠正误区)：打印各 rank 的 self.k_cache/self.v_cache shape（Attention 里，来自 model_runner.py:115-121，num_kv_heads = num_key_value_heads // world_size，model_runner.py:110）。明确每个 rank 只存/算自己的 kv_heads。验证：学生能正确陈述『KV 在 rank 间从不传输也从不广播，因为 GQA 下每个 rank 独立对自己的 kv_heads 做 attention，q heads 与 kv heads 按同一 tp 边界切分对齐，无需其它 rank 的 K/V』，并指出唯一跨 rank 通信是 Q1 的三类（all_reduce / gather / rank0 sampler）。
- Q3(trace, 单步走查)：对一次 decode，按顺序列出 rank0 与 rank1 各自动作——(a) rank0 write_shm 广播 (method_name, seqs)（model_runner.py:76-83），rank1 read_shm + event.wait 反序列化得到相同 seqs；(b) 各自 prepare_decode 得到相同 input_ids/positions（基于相同 seq.last_token/len，model_runner.py:172-188）；(c) 各自 VocabParallelEmbedding mask+all_reduce（embed_head.py:35-42）；(d) 各 decoder 层：各自算自己 num_heads/num_kv_heads 的 QKV + 对自己 k_cache/v_cache 做 attention（kv 只读本 rank 缓存）+ o_proj/down_proj all_reduce；(e) model() 返回 hidden 后，rank0 gather logits + 采样，rank1 sampler 返回 None（model_runner.py:216-218）。验证：学生叙事与代码 1:1 对应，无任何『KV 跨 rank』错误叙述。

**验证方式**

- Q1: trace-log（all_reduce/gather/sampler 计数与代码点匹配）
- Q2: shape-log + explain（k_cache/v_cache shape 按 kv_heads 分片确认 + 纠错叙述 rubric）
- Q3: trace + explain（单步走查与代码 1:1，rubric 强调无 KV 跨 rank）

**涵盖内容（skills / concepts）**

- TP 切分方向：ColumnParallel(输出维) / RowParallel(输入维+all_reduce) / QKVParallel(融合 q/k/v，heads 已 per-rank) / MergedColumnParallel(gate+up)
- KV cache 按 kv_head 跨 rank 分片、无跨 rank KV 传输/广播的正确认知（核心纠错点）
- 真实 inter-rank 通信三类：RowParallelLinear all_reduce / VocabParallelEmbedding+ParallelLMHead(all_reduce/gather) / Sampler rank0-only
- GQA + 头切分使每个 rank 独立对自己的 kv_heads 做 attention 的等价性证明
- rank0 主控 + rank>=1 worker 的 shm IPC 与状态机（loop/read_shm/write_shm）
- safetensors 切分加载（weight_loader narrow/chunk + packed_modules_mapping 拆 q/k/v、gate/up）

**Profiling / 优化**

none（结构理解课，为 L8 profiling 的通信/计算重叠铺垫）

---

## L8 — Profiling 与 profile 驱动的优化：定位 decode 瓶颈并消除 launch 开销

_Milestone: M2_  |  _Theme: profiling_  |  _置信度: 待确认: nsys/ncu 安装、torch profiler 可用性、8xH200 多卡环境与 Qwen3-0.6B 下载均为真实 prereq，待确认就绪；torch profiler 部分单卡即可完成_

**锚定源码**

- `nanovllm/engine/model_runner.py:195-212`
- `nanovllm/engine/model_runner.py:222-257`
- `nanovllm/engine/model_runner.py:36-37`
- `nanovllm/engine/model_runner.py:197`
- `nanovllm/engine/llm_engine.py:72-83`
- `bench.py:15-28`
- `nanovllm/layers/layernorm.py:16-40`
- `nanovllm/layers/rotary_embedding.py:37`
- `nanovllm/layers/activation.py:8`
- `nanovllm/layers/sampler.py:7`
- `nanovllm/engine/scheduler.py:36-45`

**Tutorial 涵盖**

- 三类 profiler 分工（待确认环境）：torch profiler（python/cpu/op 级 timeline）、nsys（系统级 GPU/CPU overlap）、ncu（单 kernel 寄存器/occupancy）
- decode 为何 launch-overhead 受限：每步 batch 小、kernel 多（embed + 每层 norm/attn/mlp + sampler），host launch 成本占比高
- cudagraph 如何消除 launch overhead：run_model 在 decode 走 graph.replay()（model_runner.py:199-212），一次 replay 替代上百次 host launch；eager 或 bs>512 走 model_runner.py:197-198 常规路径
- graph 捕获代价：静态 buffer（graph_vars，model_runner.py:250-257）+ 桶式 bs（graph_bs）+ slot_mapping.fill_(-1) 哨兵保证越界安全（model_runner.py:206）
- torch.compile 融合点：RMSNorm(rms_forward/add_rms_forward)、Rotary、SiluAndMul、Sampler 都 @torch.compile（layernorm.py:16/28, rotary_embedding.py:37, activation.py:8, sampler.py:7）——融合多个 elementwise op 减少 kernel 数
- prefix-cache 命中率作为另一条优化杠杆：减少 prefill 计算量（theme 3 的直接收益，attention.py:65-66 命中时 k/v 用缓存）
- 测量 -> 定位 -> 优化 -> 重测 的闭环方法论

**Lab 概览**

用 torch profiler + 可选 nsys/ncu 对比 enforce_eager 与 cudagraph 模式下 decode 单步的 kernel launch 与 GPU 活动占比，定位小 batch decode 的 launch-overhead 瓶颈，再做一个 profile-driven 的概念优化（调 cudagraph 桶 / 提升前缀缓存命中率）。

**Questions（高粒度：跟踪/观察/解释 + 至多一个关键小项；不逐函数留空）**

- Q1(trace/profile)：对 bs=1 的纯 decode 步，分别在 enforce_eager=True 与 enforce_eager=False 下用 torch profiler 抓 trace，比较单步总耗时、kernel 数量、CPU launch 与 GPU 执行的重叠/间隙。验证：eager 模式单步 kernel 数远多于 graph 模式（embed + 每层 norm/qkv/rotary/attn/mlp/silu + sampler 多个独立 kernel + host launch），graph 模式把整步压成一次 replay（model_runner.py:211），CPU-GPU gap 显著缩小。
- Q2(key item, profile-driven optimization)：测量后定位到 decode 是 launch-bound，选择一个优化杠杆并重测——选项 A：调整 capture_cudagraph 的 graph_bs 桶密度（model_runner.py:234，让小 bs 桶更密减少 padding）；选项 B：构造共享 system prompt 的请求集提升前缀缓存命中率，对比 prefill 时间（命中前后，scheduler.py:36-45 的 num_cached_blocks 影响 + attention.py:65-66 走 paged 路径）；选项 C：概念性解释 torch.compile 融合点（layernorm/rotary/silu/sampler 的 @torch.compile）对 kernel 数量的影响。验证：优化前后吞吐曲线（llm_engine.py:80-83 的 Decode tok/s）有可量化提升，且学生能解释该提升来自减少 launch overhead / 减少 prefill 计算 / 减少 kernel 数 中的哪一项。
- Q3(observe+explain)：用 profiler 的 GPU 利用率/时间线，解释为什么 prefill（大 token 量、计算密集）几乎不受 launch overhead 影响，而 decode（每步 1 token/batch、kernel 多而小）是典型 launch-bound 场景——这正是 cudagraph 与连续批处理(max_num_seqs 让多请求同批 decode 放大 batch)存在的根本理由。验证：学生叙事正确引用 model_runner.py:197 的 `input_ids.size(0) > 512` 大 batch 跳 graph 的设计意图（大 batch 本身摊薄了 launch overhead，捕获/回放反成负担）。

**验证方式**

- Q1: benchmark-sanity + profile-trace（eager vs graph 的 kernel 数 / gap 量化对比）
- Q2: benchmark-sanity（优化前后 Decode tok/s 量化提升 + 正确归因）
- Q3: explain + profile（rubric：prefill 计算密集 vs decode launch-bound 对比叙述 + 引用 model_runner.py:197 大 batch 跳图设计意图）

**涵盖内容（skills / concepts）**

- torch profiler / nsys / ncu 三件套分工与定位瓶颈
- 小 batch decode 的 launch-overhead 瓶颈识别（kernel 多而小、host launch 占比高）
- CUDA graph 静态捕获消除 launch 开销的机制（一次 replay 替代上百次 launch）
- 大 batch(>512) 跳图的设计意图（model_runner.py:197，大 batch 本身摊薄 launch overhead）
- prefix-cache 命中率作为吞吐杠杆
- torch.compile 融合点（RMSNorm/Rotary/SiluAndMul/Sampler）对 kernel 数量的影响
- 闭环优化方法论（测量 -> 定位 -> 优化 -> 重测）

**Profiling / 优化**

本课即 profiling 主线 + 最终 capstone 的 profile-driven 优化步骤（测量 -> 定位 -> 优化 -> 重测）

---

## 评审结论与前置准备（必读）

### 评审循环结果

- Architecture → Engineer → Reviewer 循环跑了 **5 轮，第 5 轮通过 ✅（**正式收敛**）。
- 末轮 Reviewer 判定 **verdict = pass**，且 `blocking / accuracy / altitude / coverage / pedagogy / feasibility` **全部为 0**。
- Reviewer 通读全部 21 个源文件，确认每个 `file:line` 锚点准确、无编造；粒度为高粒度（process-understanding）；5 主题全覆盖；KV「按 kv_head 分片、不在 rank 间传输」表述正确。
- （v1 的 14 课粒度版已被本 v2 高粒度版取代。）

### 环境认定

- 硬件：**8× H200**（用户确认；`nvidia-smi` 显示为 L20X，以用户说法为准）。本轮 Reviewer 已按要求不再以 GPU 型号 / 工具链缺失为问题项。
- 无论 H200 还是 L20X（96GB CUDA 卡），各 Lab 均可运行；型号只影响峰值性能数字。

### 开课前必须准备的前置依赖（实机探测于本服务器 base 环境）

- ⚠️ **torch / triton / flash_attn 未安装**（`import` 失败）—— 需先建好含 `torch>=2.4` / `triton>=3.0` / `flash-attn` 的环境。
- ⚠️ **nsys 未安装**（`ncu` 已装于 `/usr/local/bin/ncu`）—— L8 的 nsys timeline 部分需先装 Nsight Systems。
- ⚠️ **Qwen3-0.6B 未下载**（`~/huggingface/Qwen3-0.6B/` 不存在）—— 各 Lab 默认模型，需先下载。

### 仍需课程方确认的 待确认 项（来自课程 open_questions）

- 待确认 prereq：8x H200 集群环境与 NCCL 进程组（tcp://localhost:2333, model_runner.py:26）是否就绪——L7 的多卡 trace 与 L8 的多卡 profiling 依赖它；单卡(tp=1)部分已确认可跑（example.py enforce_eager=True）。注意 GPU 型号以用户确认为准（nvidia-smi 可能误显 L20X，不影响可行性）。
- 待确认 prereq：torch / triton / flash_attn 工具链是否已安装在目标环境（attention.py 依赖 flash_attn_varlen_func / flash_attn_with_kvcache 与 Triton store_kvcache_kernel）。
- 待确认 prereq：Qwen3-0.6B 权重是否已下载到 ~/huggingface/Qwen3-0.6B/（example.py:7 与 bench.py:14 的默认路径）。
- 待确认 prereq：L8 的 nsys / ncu 是否已安装；若缺失，L8 Q1 可退化为纯 torch profiler（profiler 部分单卡即可完成，nsys/ncu 为增强项）。
- 待确认：L4 共享系统 prompt 的跨请求复用 demo（ref_count+=1）在 tp=1 单卡下即可验证，多卡不是硬要求。需向学生说明：BlockManager 在主进程（Scheduler 内）是单实例、内容寻址；model_runner 侧的 KV 缓存是按 rank 分片存储的，二者职责分离——前缀命中的『块复用』决策在主进程完成，物理 KV 写入由各 rank 各自对自己的 kv_heads 执行。
- 设计决策待确认：是否将 L6（模型前向）与 L7（TP）合并为一节超长课——当前拆为两节以保 high-altitude 节奏；若希望压到 7 课以内可合并，但 L7 的『KV 不跨 rank』纠正专题建议保留独立篇幅。
- 待确认：bench.py 默认 enforce_eager=False（bench.py:15），需 cudagraph 可用；L8 若用 bench.py 做 capstone 测量需确认 capture_cudagraph 在目标卡上能成功捕获（graph_bs 最大桶取决于 max_num_seqs 与 512 的较小值，model_runner.py:226）。
- 待确认：L5 Q2 触发 KV 压力的两条杠杆中——降低 gpu_memory_utilization（config.py:12）是构造 LLM 时可传入的 Config 字段，但 ModelRunner 在构造后才回写 config.num_kvcache_blocks（model_runner.py:113），学生在引擎构造前后修改该字段的时序需在 lab 说明里点明；并发多条超长 prompt 耗尽 free 队列则不依赖任何时序，是更稳健的触发方式。

### Reviewer 肯定的 strengths（摘录）

- ACCURACY: Audited all 21 .py files against the cited anchors. Every file:line is correct: model_runner kv_cache shape + num_kv_heads//world_size (110,115), prefix-cache signal cu_seqlens_k>q (162), eager/graph split + bs>512 gate (197), slot fill_(-1) guard (206-207), graph_bs buckets (234), bucket-select (202), sampler rank0-only (216,218); attention store_kvcache kernel + slot==-1 guard (23), dual flash path + prefix k/v swap (64-70); block_manager three indices (31-33), xxhash chain (35-41), can_allocate 3-state (58-73), allocate ref_count reuse (83-88), can/may_append gate (103-108), hash_blocks (110-120); linear/embed_head TP directions, all_reduce points, gather-to-rank0, last-token index (59-60); config defaults + asserts (9,12,18,22,23); qwen3 decoder/MLP/module-mapping + compute_logits split (146-159,205-216,187-193); no MoE (correctly excluded). No fabricated internals.
- TP CORRECTION (L7) is the keystone and it is technically exact: KV cache is allocated/stored per kv_head across ranks (allocate_kv_cache num_kv_heads//world_size, model_runner.py:110,115), there is NO KV transfer/broadcast between ranks, and the ONLY inter-rank comms are RowParallelLinear all_reduce (linear.py:154-155), VocabParallelEmbedding all_reduce (embed_head.py:41), ParallelLMHead dist.gather to rank0 (embed_head.py:62-65), and sampler rank0-only (model_runner.py:216,218). The Q1 comm-count formula (all_reduce = 2*num_hidden_layers + 1 [embedding], gather = 1) is correct. Q3 single-step walk-through maps 1:1 to source. This directly and correctly refutes the 'KV sent between ranks' misconception.
- ALTITUDE: Genuinely high-level and process-focused, honoring the rejected-14-lesson corrective. Each lab is exactly 3 questions mixing instrument/trace/observe-and-explain with at most ONE crystallizing small implementation (L1 throughput wrapper, L3 seq_to_slots, L4 shared-prefix driver). No lab blanks out internal helpers; no function-level over-decomposition. Matches 'understand the whole pipeline, do not reimplement' intent.
- KV SPINE (L2->L3->L4) correctly follows the mandated teaching order: L2 STARTS from the two flash-attention call contracts (flash_attn_varlen_func with cu_seqlens_q/k + optional block_table; flash_attn_with_kvcache with cache_seqlens/context_lens + block_table), then descends to paged tensor shape + Triton store_kvcache scatter + slot==-1 guard; L3 explains Block/BlockManager + Sequence<->Block mapping (block_table, num_blocks, last_block_num_tokens, block(i)); L4 covers content-addressed prefix caching. The subtle point that prefix-hit prefill swaps k/v for k_cache/v_cache AND feeds block_table into varlen (attention.py:65-70) is captured correctly.
- PEDAGOGY: clean request->output progression (L1 end-to-end -> KV spine L2-L4 -> continuous batching L5 -> model/TP L6-L7 -> profiling capstone L8). Validation variety is genuine (trace-log, observe+explain rubrics, golden/shape-log, benchmark-sanity). Profiling is a real thread from L1 (num_tokens sign + throughput postfix) through L2/L5/L6 (kernel-count, throughput curves, cudagraph stop-point) to the L8 capstone measure->locate->optimize->re-measure loop.
- capstone_design is coherent and source-grounded: enforce_eager toggle (bench.py:15), cudagraph bucket density (model_runner.py:234), prefix-cache hit via ref_count reuse (block_manager.py:83-88), torch.compile fusion points (layernorm.py:16,28 / rotary_embedding.py:37 / activation.py:8 / sampler.py:7). Stays at concept/config level, no per-kernel rewrites.
- 待确认 is used honestly and ONLY for environment prereqs (8xH200 cluster + NCCL tcp://localhost:2333, torch/triton/flash_attn toolchain, Qwen3-0.6B weights, nsys/ncu) and legitimate design open-questions (merge L6+L7?), never to mask a content gap or fabrication. The GPU-model / toolchain items are correctly NOT flagged as content defects per the authoritative environment.

