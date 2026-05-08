"""
模型权重加载器。

HuggingFace 模型的权重名称（如 q_proj, k_proj, v_proj）和本项目的参数名称
（如 qkv_proj）不完全一致，因为本项目把 Q/K/V 合并成了一个矩阵以提高效率。
这个 loader 负责处理这种映射关系，把 HF 权重正确加载到合并后的参数中。
"""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认加载方式：直接复制"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    从 safetensors 文件加载模型权重。

    packed_modules_mapping 定义了 HF 权重名 → 本项目参数名的映射：
    - "q_proj" → ("qkv_proj", "q")  表示 q_proj 权重要加载到 qkv_proj 的 q 分片
    - "gate_proj" → ("gate_up_proj", 0)  表示 gate_proj 加载到 gate_up_proj 的第 0 个分片

    每个参数上挂载的 weight_loader 方法知道如何处理张量并行的切分。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否是需要合并的权重（如 q_proj → qkv_proj）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重，直接按名称加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
