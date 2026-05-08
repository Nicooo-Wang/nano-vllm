"""
旋转位置编码（RoPE - Rotary Position Embedding）。

RoPE 是现代 LLM 的标准位置编码方式（替代了原始 Transformer 的正弦位置编码）。
核心思想：通过旋转向量来编码位置信息，使得 attention score 只依赖于相对位置。

数学原理：
  对于位置 p 的向量 x = [x1, x2, x3, x4, ...]，两两配对旋转：
  [x1, x2] → [x1*cos(pθ) - x2*sin(pθ), x2*cos(pθ) + x1*sin(pθ)]
  其中 θ_i = base^(-2i/d)，不同维度有不同的旋转频率。

get_rope 使用 lru_cache 确保全局只创建一个 RotaryEmbedding 实例。
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对输入向量应用旋转：把 x 拆成前后两半，做 2D 旋转"""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 计算每个维度对的旋转频率：θ_i = 1 / (base^(2i/d))
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 预计算所有位置的 cos/sin 值（避免运行时重复计算）
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rotary_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        # 缓存形状: [max_pos, 1, rotary_dim]（1 是为了广播到多个 head）
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据位置索引查表，对 Q 和 K 应用旋转"""
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """全局单例：所有 Attention 层共享同一个 RotaryEmbedding"""
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
