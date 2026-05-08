"""
词表并行的 Embedding 和 LM Head。

在张量并行中，词表（vocab）也需要切分到多个 GPU：
- VocabParallelEmbedding：每个 GPU 只存储词表的一部分 embedding
- ParallelLMHead：每个 GPU 只计算词表一部分的 logits

通信模式：
- Embedding：每个 GPU 查自己那部分词表，不在范围内的 token 输出 0，
  然后 all_reduce 合并（因为每个 token 只会命中一个 GPU 的词表范围）
- LM Head：每个 GPU 算出部分 logits，然后 gather 到 rank 0 拼接
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词表并行 Embedding。

    词表被均匀切分到 tp_size 个 GPU 上。
    每个 GPU 只存储 [vocab_start_idx, vocab_end_idx) 范围的 embedding。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整词表权重中取出自己负责的那部分"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 只处理落在自己词表范围内的 token，其他的 mask 为 0
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 不在范围内的 token 输出置 0，然后 all_reduce 合并
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    词表并行 LM Head（语言模型输出头）。

    功能：hidden_states → logits（词表大小的概率分布）
    实现：和 Embedding 共享权重结构，但做的是 linear 而非 embedding lookup。

    Prefill 时只取每个序列最后一个 token 的 hidden_state 来计算 logits
    （因为只有最后一个 token 需要预测下一个 token）。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # Prefill 时多个序列拼接在一起，只取每个序列的最后一个 token
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # x @ weight.T → logits
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 每个 GPU 有部分词表的 logits，gather 到 rank 0 拼接成完整 logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
