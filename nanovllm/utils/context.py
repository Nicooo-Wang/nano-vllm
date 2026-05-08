"""
全局上下文管理。

在 vLLM 架构中，模型的 forward 函数签名是固定的 (input_ids, positions)，
但 Attention 层需要额外的元数据（如 KV Cache 位置、序列长度等）。
这里用全局变量传递这些信息，避免在模型每一层都传递大量参数。

这是一个工程上的 trade-off：牺牲了一点代码纯净性，换来了模型定义的简洁。
"""

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # 当前是 prefill 还是 decode 阶段
    is_prefill: bool = False
    # prefill 时：每个序列的 query 累积长度（用于 flash_attn_varlen_func）
    cu_seqlens_q: torch.Tensor | None = None
    # prefill 时：每个序列的 key 累积长度（含 cached 部分）
    cu_seqlens_k: torch.Tensor | None = None
    # prefill 时：batch 中最长的 query 长度
    max_seqlen_q: int = 0
    # prefill 时：batch 中最长的 key 长度
    max_seqlen_k: int = 0
    # KV Cache 写入位置的映射（token → 物理 slot）
    slot_mapping: torch.Tensor | None = None
    # decode 时：每个序列的上下文总长度
    context_lens: torch.Tensor | None = None
    # block table：逻辑 block → 物理 block 的映射表
    block_tables: torch.Tensor | None = None


# 全局单例，每次 step 开始时 set，结束时 reset
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
