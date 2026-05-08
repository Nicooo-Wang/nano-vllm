"""
序列（Sequence）：代表一个推理请求的完整生命周期。

一个 Sequence 从 WAITING 状态开始，经过 prefill 进入 RUNNING 状态（逐 token 生成），
最终到达 FINISHED 状态。它记录了：
- 所有 token（prompt + 已生成的 completion）
- KV Cache 的 block 分配情况（block_table）
- 缓存状态（哪些 token 的 KV 已经算过了）
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()    # 等待 prefill
    RUNNING = auto()    # 正在 decode（逐 token 生成中）
    FINISHED = auto()   # 生成完毕


class Sequence:
    block_size = 256            # KV Cache block 大小，由 Config 在启动时设置
    counter = count()           # 全局递增 ID 生成器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)        # 唯一标识
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)            # 完整 token 序列（prompt + completion）
        self.last_token = token_ids[-1]             # 最后一个 token（decode 时只需要这一个作为输入）
        self.num_tokens = len(self.token_ids)       # 当前总 token 数
        self.num_prompt_tokens = len(token_ids)     # prompt 的 token 数（固定不变）
        self.num_cached_tokens = 0                  # 已经计算过 KV Cache 的 token 数
        self.num_scheduled_tokens = 0               # 本轮被调度计算的 token 数
        self.is_prefill = True                      # 是否还在 prefill 阶段
        self.block_table = []                       # 逻辑 block → 物理 block ID 的映射
        # 采样参数直接存在 Sequence 上，方便跨进程传输时减少序列化开销
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """当前序列需要多少个 KV Cache block（向上取整）"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个 block 中已填充的 token 数"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第 i 个 block 对应的 token_ids（用于计算 prefix cache hash）"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """decode 阶段：追加一个新生成的 token"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        自定义序列化（用于多 GPU 间通过共享内存传输）。
        - prefill 时需要传完整 token_ids（因为 worker 需要做 embedding lookup）
        - decode 时只需要传 last_token（节省传输量）
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """反序列化：在 worker 进程中恢复 Sequence 的关键字段"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
