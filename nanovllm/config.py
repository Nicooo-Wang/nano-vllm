"""全局配置。定义推理引擎的所有可调参数。"""

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    # 模型路径（本地 HuggingFace 格式目录）
    model: str
    # 一次 prefill 最多处理多少个 token（控制 GPU 显存峰值）
    max_num_batched_tokens: int = 16384
    # 同时处理的最大序列数（batch size 上限）
    max_num_seqs: int = 512
    # 模型支持的最大上下文长度
    max_model_len: int = 4096
    # GPU 显存利用率，剩余部分留给 KV Cache
    gpu_memory_utilization: float = 0.9
    # 张量并行的 GPU 数量
    tensor_parallel_size: int = 1
    # 为 True 时禁用 CUDA Graph，使用 eager 模式（方便调试）
    enforce_eager: bool = False
    # HuggingFace 模型配置（自动从模型目录加载）
    hf_config: AutoConfig | None = None
    # EOS token id（由 tokenizer 填充）
    eos: int = -1
    # KV Cache 每个 block 存储多少个 token（类似内存页大小）
    kvcache_block_size: int = 256
    # KV Cache 总 block 数（运行时根据剩余显存计算）
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 从模型目录加载 HuggingFace config（包含层数、head 数等信息）
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 实际最大长度不能超过模型训练时的位置编码上限
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
