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
    """
    模型运行器，负责实际的模型推理执行。

    核心功能：
    1. 加载和运行模型（前向推理）
    2. 管理张量并行和多进程通信
    3. 分配和管理KV-cache
    4. 支持CUDA Graph优化
    5. 处理prefill和decode两种模式
    6. 管理分布式训练/推理环境

    架构设计：
    - rank 0（主进程）：负责任务分发、结果聚合和采样
    - rank > 0（子进程）：仅负责模型分片的前向计算
    - 使用SharedMemory和Event进行进程间通信
    - 使用NCCL进行GPU间的张量通信

    两种执行模式：
    1. Eager模式：直接执行每次推理，灵活但有开销
    2. CUDA Graph模式：预捕获图并重用，高效但需固定batch size
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型运行器。

        使用方法：
        # 主进程（rank 0）
        runner = ModelRunner(config, 0, events)

        # 子进程（rank > 0）
        runner = ModelRunner(config, rank, event)

        参数：
        - config: 配置对象，包含模型路径和推理参数
        - rank: 进程排名，0表示主进程
        - event: 进程间同步事件，用于子进程等待指令

        初始化流程：
        1. 初始化分布式环境（NCCL）
        2. 设置CUDA设备
        3. 加载模型并设置数据类型
        4. 模型预热（warmup）
        5. 分配KV-cache
        6. 捕获CUDA Graph（如果启用）
        7. 初始化进程间通信
        8. 子进程进入消息循环

        使用场景：
        1. 初始化主推理进程
        2. 初始化张量并行子进程
        3. 模型热启动和预热
        4. 分布式环境设置

        注意：
        - rank 0负责主要的控制逻辑
        - rank > 0的进程会进入loop()等待指令
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager  # 是否强制使用eager模式
        self.world_size = config.tensor_parallel_size  # 并行进程数
        self.rank = rank  # 当前进程排名
        self.event = event  # 同步事件

        # ===== 1. 初始化分布式环境 =====
        # 使用NCCL后端进行GPU间通信
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        # ===== 2. 设置CUDA设备 =====
        torch.cuda.set_device(rank)  # 将当前GPU设备设置为rank对应的GPU

        # ===== 3. 加载模型 =====
        # 保存默认数据类型，稍后恢复
        default_dtype = torch.get_default_dtype()
        # 设置模型的数据类型（如float16）
        torch.set_default_dtype(hf_config.torch_dtype)
        # 设置默认设备为CUDA
        torch.set_default_device("cuda")

        # 创建模型实例
        self.model = Qwen3ForCausalLM(hf_config)
        # 加载模型权重
        load_model(self.model, config.model)

        # 创建采样器（用于从logits采样token）
        self.sampler = Sampler()

        # ===== 4. 模型预热 =====
        # 执行一次推理以初始化CUDA内核和缓存
        self.warmup_model()

        # ===== 5. 分配KV-cache =====
        # 根据可用显存计算KV-cache大小并分配
        self.allocate_kv_cache()

        # ===== 6. 捕获CUDA Graph（可选） =====
        # CUDA Graph可以减少开销，提高推理速度
        # 但需要固定batch size，且不支持非常小的batch
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ===== 7. 初始化进程间通信（多进程模式） =====
        if self.world_size > 1:
            if rank == 0:
                # 主进程：创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 等待所有进程就绪
            else:
                # 子进程：连接到共享内存
                dist.barrier()  # 等待主进程创建
                self.shm = SharedMemory(name="nanovllm")
                # 进入消息循环，等待主进程指令
                self.loop()

    def exit(self):
        """
        退出并清理资源。

        使用方法：
        runner.call("exit")

        使用场景：
        1. 程序正常退出时的清理
        2. 进程池的优雅关闭
        3. 释放CUDA显存和分布式进程组

        清理内容：
        1. 关闭共享内存连接
        2. 删除CUDA Graph
        3. 同步GPU
        4. 销毁分布式进程组
        """
        # 多进程模式：清理共享内存
        if self.world_size > 1:
            self.shm.close()  # 关闭共享内存连接
            dist.barrier()  # 等待所有进程

            # 只有主进程可以unlink共享内存
            if self.rank == 0:
                self.shm.unlink()  # 删除共享内存对象

        # 清理CUDA Graph
        if not self.enforce_eager:
            del self.graphs, self.graph_pool

        # 同步GPU，确保所有操作完成
        torch.cuda.synchronize()

        # 销毁分布式进程组
        dist.destroy_process_group()

    def loop(self):
        """
        子进程的消息循环，等待主进程指令。

        使用方法：
        子进程自动调用，无需手动调用

        工作原理：
        1. 持续等待主进程发送指令（通过SharedMemory）
        2. 接收并执行指令（method_name和参数）
        3. 直到收到"exit"指令才退出循环

        使用场景：
        1. 子进程的常驻消息处理
        2. 分布式推理时的任务分发
        3. 进程间同步和通信

        指令格式：
        - method_name: 要调用的方法名（如"run", "exit"）
        - args: 方法参数
        """
        while True:
            # 1. 读取共享内存中的指令
            method_name, args = self.read_shm()

            # 2. 执行指令
            self.call(method_name, *args)

            # 3. 如果是退出指令，结束循环
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取指令（子进程调用）。

        使用方法：
        子进程通过Event等待，主进程写入后自动唤醒
        method_name, args = self.read_shm()

        协议格式：
        - 前4字节：数据长度（little-endian）
        - 后续字节：pickle序列化的[method_name, *args]

        返回值：
        - method_name: 方法名
        - args: 参数列表

        使用场景：
        1. 子进程等待主进程指令
        2. 进程间通信的接收端
        3. 分布式指令分发

        注意：
        仅供rank > 0的子进程使用
        """
        assert self.world_size > 1 and self.rank > 0

        # 等待主进程发送指令
        self.event.wait()

        # 读取数据长度（前4字节）
        n = int.from_bytes(self.shm.buf[0:4], "little")

        # 读取并反序列化数据
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])

        # 清除事件标记
        self.event.clear()

        return method_name, args

    def write_shm(self, method_name, *args):
        """
        向共享内存写入指令（主进程调用）。

        使用方法：
        主进程写入指令并通知所有子进程
        self.write_shm("run", seqs, is_prefill)

        参数：
        - method_name: 要调用的方法名
        - *args: 方法参数

        协议格式：
        - 前4字节：数据长度（little-endian）
        - 后续字节：pickle序列化的[method_name, *args]

        通知机制：
        - 写入后设置所有Event，唤醒等待的子进程

        使用场景：
        1. 主进程分发任务给子进程
        2. 启动子进程执行特定方法
        3. 进程间指令传递

        注意：
        仅供rank 0的主进程使用
        """
        assert self.world_size > 1 and self.rank == 0

        # 序列化数据
        data = pickle.dumps([method_name, *args])
        n = len(data)

        # 写入数据长度
        self.shm.buf[0:4] = n.to_bytes(4, "little")

        # 写入数据
        self.shm.buf[4:n+4] = data

        # 通知所有子进程
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        调用指定方法，支持多进程分发。

        使用方法：
        # 单进程模式：直接调用
        result = self.call("run", seqs, is_prefill)

        # 多进程模式：主进程分发，子进程执行
        result = self.call("run", seqs, is_prefill)

        参数：
        - method_name: 方法名
        - *args: 方法参数

        行为差异：
        - rank 0：先写入共享内存，再本地调用
        - rank > 0：通过loop()接收指令后调用

        返回值：
        - 方法的执行结果（仅rank 0收到返回值）

        使用场景：
        1. 统一的接口调用方式
        2. 隐藏多进程复杂性
        3. 远程过程调用（RPC）
        """
        # 多进程模式：主进程先写入共享内存
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)

        # 获取方法并调用
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        模型预热，初始化CUDA内核和缓存。

        使用方法：
        在模型初始化后自动调用
        self.warmup_model()

        预热策略：
        1. 创建最大长度的虚拟序列
        2. 执行一次完整的prefill推理
        3. 触发CUDA内核的JIT编译和缓存

        目的：
        1. 预编译CUDA内核，避免首次推理延迟
        2. 初始化GPU内存分配器
        3. 填充CPU/GPU页缓存
        4. 减少内存分配开销

        使用场景：
        1. 生产环境部署（避免冷启动）
        2. 性能基准测试
        3. 确保模型能正常运行

        注意：
        - 预热会消耗时间和显存
        - 仅在初始化时执行一次
        - 使用全零token模拟真实推理
        """
        # 清空缓存并重置统计
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 计算预热时的序列数量
        # 确保能填满整个batch，充分利用GPU
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)

        # 创建虚拟序列（用0填充，模拟真实token）
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]

        # 执行一次完整的prefill推理
        self.run(seqs, True)

        # 再次清空缓存
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配KV-cache显存。

        使用方法：
        在模型初始化时自动调用
        self.allocate_kv_cache()

        计算策略：
        1. 获取GPU总显存和已使用显存
        2. 加上峰值显存和当前显存使用
        3. 根据利用率计算可用显存
        4. 除以每个block的字节数得到block数量

        公式：
        num_blocks = (total * gpu_memory_utilization - used - peak + current) / block_size_bytes

        使用场景：
        1. 推理前的显存分配
        2. KV-cache的预分配
        3. 内存使用优化

        内存布局：
        - 维度0: 2 (K和V)
        - 维度1: num_hidden_layers (层数)
        - 维度2: num_blocks (块数)
        - 维度3: block_size (块大小，默认256)
        - 维度4: num_kv_heads (KV头数，已除以tensor_parallel_size)
        - 维度5: head_dim (头维度)

        注意：
        - 在所有进程上分配相同的KV-cache
        - 每个attention层共享同一个KV-cache实例
        - 只分配空间，不初始化（避免开销）
        """
        config = self.config
        hf_config = config.hf_config

        # 获取GPU显存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free

        # 获取显存使用统计
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 计算每个KV头的数量（已除以tensor parallel size）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size

        # 计算每个block的字节数
        # 2表示K和V两个tensor
        block_bytes = (
            2  # K and V
            * hf_config.num_hidden_layers  # 层数
            * self.block_size  # block大小
            * num_kv_heads  # KV头数
            * hf_config.head_dim  # 头维度
            * hf_config.torch_dtype.itemsize  # 数据类型字节数
        )

        # 计算可分配的block数量
        config.num_kvcache_blocks = int(
            (total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0, "KV-cache分配失败：显存不足"

        # 分配KV-cache tensor
        # 形状：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.empty(
            2,  # K and V
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
        )

        # 将KV-cache分配给模型中的attention层
        layer_id = 0
        for module in self.model.modules():
            # 找到所有有k_cache和v_cache属性的模块（即Attention层）
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # K cache
                module.v_cache = self.kv_cache[1, layer_id]  # V cache
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备block table张量，用于flash-attn的blocked KV-cache。

        使用方法：
        block_tables = self.prepare_block_tables(seqs)

        参数：
        - seqs: 序列列表

        返回值：
        - block_tables: [num_seqs, max_num_blocks]的int32张量
          * -1表示无效block
          * 非负数表示block ID

        用途：
        1. 告诉flash-attn每个序列的KV在哪些block中
        2. 支持非连续内存访问
        3. 实现高效的KV-cache管理

        使用场景：
        1. prefill阶段的批量处理
        2. decode阶段的单个token生成
        3. prefix caching的块索引
        """
        # 找出最大block table长度
        max_len = max(len(seq.block_table) for seq in seqs)

        # 将所有block table填充到相同长度（用-1填充）
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]

        # 转换为tensor，启用pin_memory并异步传输到GPU
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备prefill阶段的数据。

        使用方法：
        input_ids, positions = self.prepare_prefill(seqs)

        参数：
        - seqs: 要处理的序列列表

        返回值：
        - input_ids: [num_tokens]的int64张量，所有序列的输入token
        - positions: [num_tokens]的int64张量，对应的位置编码

        Prefill特点：
        1. 批量处理多个序列的所有输入token
        2. 高吞吐量，优先处理能力
        3. 支持prefix caching（如果可用）

        准备工作：
        1. 提取每个序列的新token（跳过缓存部分）
        2. 计算累积序列长度（cu_seqlens）
        3. 计算最大序列长度
        4. 建立slot mapping（KV-cache位置映射）
        5. 设置全局context

        注意：
        - 只处理num_cached_tokens之后的token（缓存部分跳过）
        - 支持不规则的序列长度（varlen）
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # 累积序列长度（query）
        cu_seqlens_k = [0]  # 累积序列长度（key）
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []  # KV-cache槽位映射
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)

            # 新token（排除缓存部分）
            new_tokens = seq[seq.num_cached_tokens:]
            input_ids.extend(new_tokens)

            # 位置编码（从缓存长度开始）
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            # 计算序列长度（query和key）
            seqlen_q = seqlen - seq.num_cached_tokens  # 新增token数
            seqlen_k = seqlen  # 总token数（包含缓存）

            # 更新累积长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            # 更新最大长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # 预热阶段跳过（warmup时block_table为空）
            if not seq.block_table:
                continue

            # 建立slot mapping（仅对新分配的block）
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                block_id = seq.block_table[i]
                start = block_id * self.block_size

                # 计算block内的有效token数
                if i != seq.num_blocks - 1:
                    end = start + self.block_size  # 完整block
                else:
                    end = start + seq.last_block_num_tokens  # 最后block可能不满

                # 添加槽位映射
                slot_mapping.extend(list(range(start, end)))

        # 如果有prefix cache（总key长度 > query长度），准备block tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 转换为tensor并传输到GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置全局context
        set_context(
            True,  # is_prefill
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables
        )

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备decode阶段的数据。

        使用方法：
        input_ids, positions = self.prepare_decode(seqs)

        参数：
        - seqs: 要处理的序列列表

        返回值：
        - input_ids: [num_seqs]的int64张量，每个序列的最后一个token
        - positions: [num_seqs]的int64张量，每个序列的位置

        Decode特点：
        1. 逐个生成新token
        2. 低延迟，快速响应
        3. 每个序列独立处理

        准备工作：
        1. 提取每个序列的最后一个token
        2. 计算每个序列的当前位置
        3. 计算每个序列的上下文长度
        4. 建立slot mapping（KV-cache写入位置）
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            # 当前token（要生成下一个token的输入）
            input_ids.append(seq.last_token)

            # 当前位置
            positions.append(len(seq) - 1)

            # 上下文长度（总token数）
            context_lens.append(len(seq))

            # KV-cache写入位置
            # 最后一个block的最后一个有效位置
            last_block_idx = seq.block_table[-1]
            slot = last_block_idx * self.block_size + seq.last_block_num_tokens - 1
            slot_mapping.append(slot)

        # 转换为tensor并传输到GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 准备block tables
        block_tables = self.prepare_block_tables(seqs)

        # 设置全局context
        set_context(
            False,  # is_prefill
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables
        )

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数。

        使用方法：
        temperatures = self.prepare_sample(seqs)

        参数：
        - seqs: 序列列表

        返回值：
        - temperatures: [num_seqs]的float32张量，每个序列的温度参数

        用途：
        1. 为每个序列提取温度参数
        2. 用于logits缩放
        3. 支持不同序列使用不同温度

        注意：
        仅rank 0（主进程）需要此方法
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)

        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
