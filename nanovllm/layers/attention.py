import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton内核：将K和V存储到KV-cache。

    功能：
    1. 从输入tensor中读取K和V
    2. 根据slot_mapping将数据写入KV-cache
    3. 支持批量并行处理

    内存布局要求：
    - K和V的stride(-1)必须为1（连续存储）
    - K和V的stride(1)必须等于head_dim
    - KV-cache的stride(1)必须等于D (num_heads * head_dim)
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    将K和V存储到KV-cache。

    使用方法：
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)

    参数：
    - key: 当前的K tensor [N, num_heads, head_dim]
    - value: 当前的V tensor [N, num_heads, head_dim]
    - k_cache: K-cache [num_blocks, block_size, num_heads, head_dim]
    - v_cache: V-cache [num_blocks, block_size, num_heads, head_dim]
    - slot_mapping: 槽位映射 [N]

    功能：
    调用Triton内核高效地将K和V写入KV-cache
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    注意力层，支持KV-cache和flash-attention。

    核心功能：
    1. 标准的多头注意力计算
    2. KV-cache存储和管理
    3. 支持prefix caching
    4. 自动选择prefill或decode模式

    两种模式：
    - Prefill模式：使用flash_attn_varlen_func，支持变长序列和prefix caching
    - Decode模式：使用flash_attn_with_kvcache，利用已缓存的KV
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
