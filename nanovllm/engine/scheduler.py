"""
调度器（Scheduler）：决定每一步（step）哪些序列参与计算。

调度策略：
1. 优先 prefill：如果 waiting 队列有序列，先做 prefill
2. 没有 waiting 时做 decode：让所有 running 序列各生成一个 token
3. 内存不足时 preempt：抢占最后加入的序列，释放其 KV Cache

关键设计：
- Prefill 优先保证新请求尽快开始生成（降低首 token 延迟）
- Chunked Prefill：长 prompt 可以分多次处理，避免一次占满 GPU
- Preemption：类似操作系统的 swap，被抢占的序列回到 waiting 重新 prefill
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # waiting: 等待 prefill 的序列队列
        self.waiting: deque[Sequence] = deque()
        # running: 正在 decode 的序列队列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """所有序列都处理完毕"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加新请求到 waiting 队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度逻辑。返回 (本轮参与计算的序列列表, 是否是 prefill)。

        调度优先级：prefill > decode
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ===== 阶段 1: 尝试 Prefill =====
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # 首次调度：检查 prefix cache 命中情况并分配 block
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # 内存不足，停止调度更多 prefill
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已经分配过 block（chunked prefill 的后续 chunk）
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Chunked Prefill：只允许第一个序列被截断，后续序列必须完整放入
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # 计算本轮实际处理的 token 数（可能被 remaining 截断）
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 如果这个序列的所有 token 都处理完了，转入 running
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True  # 有 prefill 就返回

        # ===== 阶段 2: Decode =====
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查是否有空间追加一个 token
            while not self.block_manager.can_append(seq):
                # 内存不足：抢占其他序列释放空间
                if self.running:
                    self.preempt(self.running.pop())  # 抢占最后加入的（LIFO）
                else:
                    # 没有其他序列可抢占，只能抢占自己
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1  # decode 每次只处理 1 个 token
                seq.is_prefill = False
                self.block_manager.may_append(seq)  # 必要时分配新 block
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把调度的序列放回 running 队列头部
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占一个序列：释放其 KV Cache，回到 waiting 队列重新排队。
        下次被调度时需要重新 prefill（但可能命中 prefix cache）。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        每步计算完成后的后处理：
        1. 对完成的 block 计算 hash（注册到 prefix cache）
        2. 更新缓存计数
        3. 追加新 token
        4. 检查是否结束
        """
        for seq, token_id in zip(seqs, token_ids):
            # 注册刚计算完的 block 到 prefix cache
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # Chunked prefill：如果还没处理完所有 prompt token，跳过 token 追加
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # 追加新生成的 token
            seq.append_token(token_id)
            # 检查终止条件
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
