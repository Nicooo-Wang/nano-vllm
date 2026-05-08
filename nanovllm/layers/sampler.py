"""
采样器：从 logits 中采样下一个 token。

使用 Gumbel-Max 技巧实现 temperature sampling：
1. logits / temperature → 调整分布的尖锐程度
2. softmax → 概率分布
3. probs / Exponential(1) → 等价于从 Categorical 分布采样
4. argmax → 得到采样结果

Gumbel-Max 的好处：避免了 torch.multinomial 的性能问题，
且整个过程可以被 torch.compile 编译成一个高效的融合 kernel。
"""

import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 1. Temperature scaling：temperature 越高分布越平坦（更随机）
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 2. Softmax 得到概率分布
        probs = torch.softmax(logits, dim=-1)
        # 3. Gumbel-Max 采样：等价于 torch.multinomial 但更快
        #    probs / Exp(1) 的 argmax 等价于从 Categorical(probs) 采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
