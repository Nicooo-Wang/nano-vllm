import torch
from torch import nn


class Sampler(nn.Module):
    """
    采样器，从logits中采样token。

    采样方法：
    使用Gumbel-Max技巧实现温度采样：
    1. 将logits除以温度进行缩放
    2. 计算softmax概率
    3. 添加Gumbel噪声
    4. 选择最大概率的token

    优点：
    - 可微（支持梯度流）
    - 高效（无需排序）
    - 支持不同温度

    使用场景：
    - decode阶段从logits采样下一个token
    - 支持批量采样
    - 每个序列可以有不同的温度
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        前向传播，执行采样。

        使用方法：
        tokens = sampler(logits, temperatures)

        参数：
        - logits: 模型输出的logits [batch_size, vocab_size]
        - temperatures: 每个序列的温度 [batch_size]

        返回值：
        - 采样的token IDs [batch_size]

        算法：
        1. 温度缩放：logits = logits / temperature
        2. Softmax：probs = softmax(logits)
        3. Gumbel噪声：noise = -log(-log(U)), U ~ Uniform(0,1)
        4. 采样：token = argmax(log(probs) + noise)
        """
        # 温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 计算概率
        probs = torch.softmax(logits, dim=-1)

        # Gumbel-Max采样
        sample_tokens = (
            probs
            .div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10))  # 添加Gumbel噪声
            .argmax(dim=-1)  # 选择最大概率的token
        )

        return sample_tokens
