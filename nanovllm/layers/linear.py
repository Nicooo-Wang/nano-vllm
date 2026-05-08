"""
张量并行（Tensor Parallelism）的 Linear 层。

在多 GPU 推理中，模型权重被切分到不同 GPU 上。切分方式有两种：
- Column Parallel：按输出维度切分（每个 GPU 算一部分输出）
- Row Parallel：按输入维度切分（每个 GPU 算部分结果，最后 all_reduce 求和）

典型用法（以 Attention 为例）：
  QKV_proj: ColumnParallel（每个 GPU 算自己负责的 head）
  O_proj: RowParallel（每个 GPU 的部分结果 all_reduce 合并）

weight_loader 方法：每个 Linear 子类定义了如何从完整权重中提取自己那份切片。
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """所有 Linear 层的基类"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,  # 张量并行的切分维度（0=行, 1=列）
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # weight_loader 挂在参数上，load_model 时会调用它来加载权重
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """复制式 Linear：每个 GPU 持有完整权重（不切分）"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接复制完整权重"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行 Linear：按输出维度切分。

    完整权重 shape: [output_size, input_size]
    每个 GPU 持有: [output_size / tp_size, input_size]

    每个 GPU 独立计算自己那部分输出，无需通信。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中取出自己 rank 对应的切片"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行 Linear：多个 ColumnParallel 合并成一个矩阵。

    用于 MLP 的 gate_up_proj：把 gate_proj 和 up_proj 合并成一个矩阵，
    一次矩阵乘法算出两个结果，减少 kernel launch 开销。

    output_sizes = [intermediate_size, intermediate_size]
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """加载合并矩阵中的某个分片（如 gate_proj 是 shard 0，up_proj 是 shard 1）"""
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 并行 Linear：把 Q/K/V 三个投影合并成一个矩阵。

    输出布局: [Q_heads × head_dim | K_heads × head_dim | V_heads × head_dim]
    每个 GPU 只持有自己负责的 head 对应的部分。

    支持 GQA（Grouped Query Attention）：num_kv_heads < num_heads
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """加载 Q/K/V 中的某一个到合并矩阵的对应位置"""
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行 Linear：按输入维度切分。

    完整权重 shape: [output_size, input_size]
    每个 GPU 持有: [output_size, input_size / tp_size]

    每个 GPU 算出部分结果后，通过 all_reduce 求和得到最终输出。
    通常与 ColumnParallel 配对使用（Column 的输出直接作为 Row 的输入，无需通信）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中取出自己 rank 对应的输入维度切片"""
        param_data = param.data
        if param_data.ndim == 1:
            # bias 不切分，每个 GPU 持有完整 bias
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只有 rank 0 加 bias（避免 all_reduce 后 bias 被重复加）
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)  # 所有 GPU 的部分结果求和
        return y
