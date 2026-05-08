"""
模型执行器（ModelRunner）：负责实际的 GPU 计算。

职责：
1. 加载模型权重到 GPU
2. 分配 KV Cache 显存
3. 准备每一步的输入张量（input_ids, positions, slot_mapping 等）
4. 执行模型前向传播（支持 eager 模式和 CUDA Graph 模式）
5. 多 GPU 通信（通过共享内存广播指令给 worker）

初始化流程：
  init_process_group → 加载模型 → warmup → 分配 KV Cache → 捕获 CUDA Graph

CUDA Graph 的作用：
  decode 阶段每次只处理 1 个 token/序列，计算量小但 CPU 开销（kernel launch）相对大。
  CUDA Graph 把一次完整的 forward 录制下来，之后直接 replay，跳过所有 CPU 调度开销。
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # ===== 初始化分布式通信 + 模型加载 =====
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 创建模型并加载权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # warmup：跑一次最大输入，让 PyTorch 完成所有懒初始化
        self.warmup_model()
        # 根据剩余显存计算能分配多少 KV Cache block
        self.allocate_kv_cache()
        # 捕获 CUDA Graph（仅 decode 阶段使用）
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ===== 多 GPU 共享内存通信 =====
        # rank 0 创建共享内存，worker 通过它接收指令
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # worker 进入事件循环，等待 rank 0 的指令
                self.loop()

    def exit(self):
        """清理：关闭共享内存、释放 CUDA Graph、销毁进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # ===== 多 GPU 通信机制 =====
    # rank 0 通过 write_shm 把方法名和参数写入共享内存，
    # worker 通过 read_shm 读取并执行相同的方法，保证所有 GPU 同步执行。

    def loop(self):
        """Worker 事件循环：等待指令 → 执行 → 等待下一个"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """Worker 从共享内存读取指令"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()       # 阻塞等待 rank 0 发信号
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """Rank 0 把指令写入共享内存并通知所有 worker"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()         # 唤醒所有 worker

    def call(self, method_name, *args):
        """统一调用入口：rank 0 会先广播给 worker，然后自己也执行"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ===== 初始化阶段 =====

    def warmup_model(self):
        """
        Warmup：用最大可能的输入跑一次 forward。
        目的：
        1. 触发 PyTorch 的懒初始化（CUDA context、cuDNN 等）
        2. 记录峰值显存，用于后续计算 KV Cache 可用空间
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        根据剩余 GPU 显存分配 KV Cache。

        计算公式：
        可用显存 = 总显存 × gpu_memory_utilization - 已用显存 - (峰值 - 当前)
        block 数 = 可用显存 / 每个 block 的字节数

        每个 block 的大小：
        2(K+V) × num_layers × block_size × num_kv_heads × head_dim × dtype_size
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 分配一个大张量作为所有层的 KV Cache
        # 形状: [2(K/V), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 把 KV Cache 的切片分配给每个 Attention 层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # ===== 输入准备 =====

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把所有序列的 block_table 拼成一个 2D 张量（padding 到相同长度）"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 prefill 阶段的输入。

        Prefill 的特点：多个序列的 token 拼接成一个长序列（varlen format），
        用 cu_seqlens_q/k 记录每个序列的边界。

        slot_mapping：每个 token 对应的 KV Cache 物理位置
        （物理位置 = block_id × block_size + block 内偏移）
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]      # query 的累积序列长度（本次实际计算的 token）
        cu_seqlens_k = [0]      # key 的累积序列长度（含 cached 部分）
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens           # 从 cached 之后开始
            seqlen_q = seq.num_scheduled_tokens     # 本次要计算的 token 数
            end = start + seqlen_q
            seqlen_k = end                          # attention 需要看到的总长度
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup 时没有 block_table
                continue
            # 计算 slot_mapping：把逻辑位置映射到物理 KV Cache 位置
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # 如果有 prefix cache 命中（k 比 q 长），需要传 block_tables 给 flash_attn
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        # 转为 GPU 张量（pin_memory + non_blocking 实现异步传输）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 decode 阶段的输入。

        Decode 的特点：每个序列只输入最后一个 token，
        但 attention 需要看到所有历史 token 的 KV Cache。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)                # 只输入最后一个 token
            positions.append(len(seq) - 1)                  # 位置 = 当前总长度 - 1
            context_lens.append(len(seq))                   # attention 需要看到的总长度
            # 新 token 的 KV 要写入的物理位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数（只有 rank 0 需要做采样）"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ===== 模型执行 =====

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向传播。

        - Prefill / eager 模式 / batch > 512：直接跑 PyTorch eager forward
        - Decode 且 batch <= 512：使用预录制的 CUDA Graph（更快）
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph 模式：把输入拷贝到预分配的缓冲区，然后 replay
            bs = input_ids.size(0)
            context = get_context()
            # 找到 >= bs 的最小预录制 graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 把实际数据拷贝到 graph 的输入缓冲区
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 直接 replay 录制好的 GPU 操作序列
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """完整的一步推理：准备输入 → 跑模型 → 采样"""
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # 只有 rank 0 做采样（其他 rank 不需要 logits）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # ===== CUDA Graph 捕获 =====

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        预录制多个 batch size 的 CUDA Graph。

        为什么需要多个：CUDA Graph 录制时 tensor shape 是固定的，
        所以需要为不同的 batch size 各录制一个 graph。
        运行时选择 >= 实际 bs 的最小 graph。

        录制的 batch size：[1, 2, 4, 8, 16, 32, ..., max_bs]
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 预分配固定大小的输入/输出缓冲区
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小录制（共享 memory pool 减少显存碎片）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # 录制
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存缓冲区引用，运行时通过修改这些缓冲区的内容来传入不同的输入
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
