"""
Attention 层：Flash Attention + Paged KV Cache。

这是整个推理引擎最核心的计算单元，负责：
1. 把新计算的 K/V 写入 KV Cache（通过 Triton kernel）
2. 执行 attention 计算（通过 Flash Attention）

两种模式：
- Prefill：使用 flash_attn_varlen_func，支持变长序列拼接
  - 无 prefix cache 时：Q/K/V 都是新计算的
  - 有 prefix cache 时：Q 是新的，K/V 从 cache 中读取
- Decode：使用 flash_attn_with_kvcache，每个序列只有 1 个新 Q token

store_kvcache_kernel 是一个 Triton kernel，负责把 K/V 写入到
物理 block 的正确位置（由 slot_mapping 指定）。
"""

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
    D: tl.constexpr,    # num_kv_heads × head_dim（每个 token 的 KV 总维度）
):
    """
    Triton kernel：把一个 token 的 K/V 写入 KV Cache 的指定 slot。

    每个 program（线程块）处理一个 token。
    slot_mapping[idx] 指定了这个 token 应该写入 cache 的哪个位置。
    slot == -1 表示跳过（CUDA Graph padding 用）。
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # 从 key/value 张量中读取当前 token 的数据
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 写入 KV Cache 的对应 slot
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """启动 Triton kernel，把 batch 中所有 token 的 K/V 写入 cache"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    Paged Attention 层。

    k_cache/v_cache 在初始化时是空张量，由 ModelRunner.allocate_kv_cache()
    在启动时分配并赋值（指向全局 KV Cache 大张量的切片）。
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
        # 占位符，会被 ModelRunner 替换为实际的 cache 切片
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 把新计算的 K/V 写入 cache（warmup 时 cache 为空，跳过）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                # Prefix cache 命中：K/V 从 cache 中读取（而非用刚计算的）
                k, v = k_cache, v_cache
            # Prefill attention：变长序列格式，支持 batch 内不同长度的序列
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # Decode attention：每个序列只有 1 个新 Q，从 cache 中读取所有历史 K/V
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
