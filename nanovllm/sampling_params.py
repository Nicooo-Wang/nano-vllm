"""采样参数配置。控制 LLM 生成文本时的随机性和长度。"""

from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    # temperature 控制生成的随机性：越高越随机，越低越确定
    temperature: float = 1.0
    # 最多生成多少个 token
    max_tokens: int = 64
    # 是否忽略 EOS（结束符），为 True 时会强制生成到 max_tokens
    ignore_eos: bool = False

    def __post_init__(self):
        # 不允许 temperature=0（贪心解码），因为本项目只实现了随机采样
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
