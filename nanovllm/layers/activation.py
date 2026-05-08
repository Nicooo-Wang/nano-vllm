"""
SwiGLU 激活函数。

现代 LLM（如 LLaMA、Qwen）的 MLP 使用 SwiGLU 替代传统的 ReLU/GELU。
gate_up_proj 输出的维度是 2×intermediate_size，前半部分是 gate，后半部分是 up。
SwiGLU(x) = SiLU(gate) * up

@torch.compile 让 PyTorch 把这个函数编译成一个融合的 CUDA kernel，
避免多次 kernel launch 和中间张量的显存分配。
"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的最后一维是 2×intermediate_size，拆成 gate 和 up 两部分
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
