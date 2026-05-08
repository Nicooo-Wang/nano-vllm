"""
Qwen3 模型定义。

标准的 Decoder-only Transformer 架构：
  Embedding → N × DecoderLayer → RMSNorm → LM Head

每个 DecoderLayer 包含：
  RMSNorm → Self-Attention → RMSNorm → MLP（SwiGLU）

特点：
- 使用 GQA（Grouped Query Attention）：K/V head 数少于 Q head 数
- 使用 RoPE 位置编码
- 使用 SwiGLU 激活函数
- Q/K 有可选的 RMSNorm（qkv_bias=False 时启用，Qwen3 特有）
- 支持 tie_word_embeddings（Embedding 和 LM Head 共享权重）

packed_modules_mapping：定义了 HuggingFace 权重名到本项目合并参数名的映射。
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 的 Multi-Head Attention。

    数据流：
    hidden_states → QKV_proj → split(Q,K,V) → RoPE → Attention → O_proj → output

    张量并行：QKV_proj 是 ColumnParallel，O_proj 是 RowParallel。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size        # 当前 GPU 负责的 Q head 数
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 当前 GPU 负责的 KV head 数
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5    # attention score 的缩放因子
        self.qkv_bias = qkv_bias

        # Q/K/V 合并为一个矩阵乘法
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影（RowParallel：需要 all_reduce）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # 旋转位置编码（全局共享单例）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # Paged Attention（含 KV Cache 读写）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen3 特有：无 bias 时对 Q/K 做 RMSNorm
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. QKV 投影
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 2. Q/K Norm（Qwen3 特有）
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 3. 旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 4. Attention（含 KV Cache 写入和读取）
        o = self.attn(q, k, v)
        # 5. 输出投影
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 的 MLP（SwiGLU 变体）。

    数据流：x → gate_up_proj → SiLU(gate) * up → down_proj → output

    gate_proj 和 up_proj 合并为一个矩阵（MergedColumnParallel），
    down_proj 是 RowParallel。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,    # gate 和 up 大小相同
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)    # SiLU(gate) * up
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    一个 Transformer Decoder 层。

    结构（Pre-Norm）：
      residual ─────────────────────────────────────────→ add ──→ add ──→
         │                                                 ↑        ↑
         └→ RMSNorm → Self-Attention ──────────────────────┘        │
                                        └→ RMSNorm → MLP ──────────┘

    使用 fused add+RMSNorm 减少显存读写。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 第一个子层：Attention
        if residual is None:
            # 第一层：还没有 residual，直接 norm
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：fused add + norm（hidden_states + residual → norm）
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # 第二个子层：MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 主体：Embedding + N 层 DecoderLayer + 最终 RMSNorm"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层的 residual 加回来并做最终 norm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    完整的 Qwen3 因果语言模型。

    packed_modules_mapping：HuggingFace 权重名 → 本项目合并参数名的映射。
    例如 HF 的 q_proj 权重要加载到本项目的 qkv_proj 参数的 "q" 分片。
    """
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 权重共享：LM Head 和 Embedding 使用同一份权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播：返回最后一层的 hidden_states"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """把 hidden_states 映射到词表空间得到 logits"""
        return self.lm_head(hidden_states)
