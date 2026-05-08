"""
LLM 推理引擎主循环。

这是整个系统的入口和协调者，负责：
1. 初始化：创建 ModelRunner（加载模型）、Scheduler、Tokenizer
2. 多 GPU 支持：用 multiprocessing 启动 worker 进程
3. 主循环：反复调用 schedule() → run() → postprocess() 直到所有请求完成

调用链：
  用户 → generate() → [add_request() × N] → while not finished: step()
  step() = schedule() + model_runner.run() + postprocess()
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 从 kwargs 中提取属于 Config 的参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 设置全局 block size（Sequence 类需要用到）
        Sequence.block_size = config.kvcache_block_size

        # ===== 多 GPU 初始化 =====
        # rank 0 在主进程运行，rank 1~N 各启动一个子进程
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 用于通知 worker 有新任务
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # rank 0 的 ModelRunner 在主进程中运行
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 创建调度器（此时 config.num_kvcache_blocks 已由 ModelRunner 计算好）
        self.scheduler = Scheduler(config)

        # 注册退出清理函数
        atexit.register(self.exit)

    def exit(self):
        """清理资源：通知所有 worker 退出，等待进程结束"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """添加一个推理请求。支持文本或 token_ids 输入。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一步推理：
        1. 调度器决定本轮处理哪些序列
        2. ModelRunner 执行前向传播 + 采样
        3. 后处理：追加 token、检查终止条件

        返回：(已完成的序列列表, 本轮处理的 token 数)
        num_tokens > 0 表示 prefill，< 0 表示 decode（绝对值 = batch size）
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成接口（离线推理）。

        流程：
        1. 把所有 prompt 加入调度队列
        2. 循环调用 step() 直到全部完成
        3. 按原始顺序返回结果
        """
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # 统计吞吐量
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 按 seq_id 排序，保证输出顺序与输入一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
