"""第 1 课 lab —— 追踪一个 request 穿过 nano-vllm 的完整旅程。

填好下面两个 TODO 函数：
  - Task 1: traced_postprocess  （把每个 engine step 记进 _trace）
  - Task 3: summarize_request   （返回 prompt / completion / 步数计数）
Task 2 是观察 + 解释（不写代码）—— 在下面标记好的注释块里作答。
然后运行：uv run python course/lessons/lesson1/lab.py
若记录正确，你会看到 "All checks passed ✓"。
"""
import os

_trace = []            # 每个 engine step 一条记录
_seqs = {}             # seq_id -> Sequence
_orig_postprocess = None
_orig_add = None


def traced_add(self, seq):
    """Scheduler.add 的钩子：按 id 记下每个 Sequence（已实现）。"""
    _seqs[seq.seq_id] = seq
    return _orig_add(self, seq)


# === Task 2 (observe + explain) — 无需写代码 ===
# 跑完后阅读本 lab 打印的 `--- trace ---` 表格。
#
# Observe（已由 run_checks 自动验证，不必再答）：
#   * 你的 trace 里两条 prompt 在同一个 step 0 PREFILL 中一起被算
#     （seqs=[...] 里含两个 id）。
#   * 在那个 prefill step 里，每条 seq 的 num_scheduled_tokens == 它各自的 prompt
#     长度；而在每个 DECODE step 里每条 seq 的 num_scheduled_tokens == 1。
#   （run_checks 对应的 key："Task2 prefill nst==num_prompt_tokens"、
#    "Task2 decode nst all==1"。）
#
# Explain（对照 ANSWERS.md §2 自检 —— 不自动判分）：
#   用你自己的话讲：为什么每条 request 的
#   `total_steps == num_completion_tokens`？
#   提示：prefill 步本身就会吐出第 1 个 completion token —— 见
#   scheduler.py:86-88（`continue` 只在 chunked prefill 时触发；当整条
#   prompt 一次算完时 `append_token` 会照跑），所以一个 seq 参与的每一步
#   都恰好 append 一个 token。
#
# 重要：Explain 不由 run_checks 判定。即便 Explain 留空，也可能打出
#   "All checks passed ✓"，但空着它意味着跳过了本课的核心收获。请把答案写在下面。
#
# 你的答案：
#   （在此作答）


def traced_postprocess(self, seqs, token_ids, is_prefill):
    """TODO(student) — Task 1 (trace)：把本步记进 _trace，再调用
    原始实现 _orig_postprocess(self, seqs, token_ids, is_prefill)。

    必须在调用原始实现之前抓取每条 seq 的 num_scheduled_tokens
    （postprocess 会在 scheduler.py:85 把它清零）。建议的记录结构：
        {"is_prefill": is_prefill,
         "before": [(s.seq_id, s.num_scheduled_tokens, s.status.name), ...],
         "token_ids": list(token_ids),
         "after":   [(s.seq_id, s.status.name, s.is_finished), ...]}
    """
    raise NotImplementedError("Task 1: fill in the trace recording")


def summarize_request(seq):
    """TODO(student) — Task 3 (小封装)：返回
    (num_prompt_tokens, num_completion_tokens, total_steps)。

    total_steps = 这个 seq 参与过的 engine step 数（在 _trace 里数）。
    """
    raise NotImplementedError("Task 3: fill in summarize_request")


def run_checks(max_tokens):
    """把 _trace 对照 request 生命周期的若干不变式进行校验。

    每条不变式都横跨所有 sequence 一起评估，只报告一次（作为合取——
    任一 sequence 不满足即判该 check 失败）。这样结果 key 保持稳定，
    便于在测试里按名字查找。
    """
    # 收集每条 seq 的中间量。
    per_seq = []
    for seq_id, seq in _seqs.items():
        steps = [r for r in _trace if any(b[0] == seq_id for b in r["before"])]
        prefill_steps = [r for r in steps if r["is_prefill"]]
        finished = any(any(a[0] == seq_id and a[2] for a in r["after"]) for r in steps)
        prefill_nst = [b[1] for r in prefill_steps for b in r["before"] if b[0] == seq_id]
        decode_nst = [b[1] for r in steps if not r["is_prefill"]
                      for b in r["before"] if b[0] == seq_id]
        _p, c, t = summarize_request(seq)
        per_seq.append({
            "seq": seq,
            "n_prefill_steps": len(prefill_steps),
            "finished": finished,
            "n_steps": len(steps),
            "prefill_nst": prefill_nst,
            "decode_nst": decode_nst,
            "completion": c,
            "total_steps": t,
        })

    results = []
    # Task 1：状态机轨迹（恰好一个 prefill step、到达 FINISHED、
    # total_steps==num_completion_tokens）。
    results.append(("Task1 one prefill step",
                    all(p["n_prefill_steps"] == 1 for p in per_seq)))
    results.append(("Task1 reached FINISHED",
                    all(p["finished"] for p in per_seq)))
    results.append(("Task1 trace recorded all steps (count==num_completion_tokens)",
                    all(p["n_steps"] == p["seq"].num_completion_tokens for p in per_seq)))
    # Task 2：num_scheduled_tokens（prefill 的 nst==num_prompt_tokens、
    # decode 的 nst 全为 1）。
    results.append(("Task2 prefill nst==num_prompt_tokens",
                    all(p["prefill_nst"] == [p["seq"].num_prompt_tokens] for p in per_seq)))
    results.append(("Task2 decode nst all==1",
                    all(len(p["decode_nst"]) > 0 and all(n == 1 for n in p["decode_nst"])
                        for p in per_seq)))
    # Task 3：summarize_request（completion==len(completion_token_ids)、
    # completion<=max_tokens、total_steps==num_completion_tokens）。
    results.append(("Task3 completion==len(completion_token_ids)",
                    all(p["completion"] == len(p["seq"].completion_token_ids) for p in per_seq)))
    results.append(("Task3 completion<=max_tokens",
                    all(p["completion"] <= max_tokens for p in per_seq)))
    results.append(("Task3 summarize_request total_steps==num_completion_tokens",
                    all(p["total_steps"] == p["seq"].num_completion_tokens for p in per_seq)))
    return results


def _check_env(model_path):
    missing = []
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    try:
        import flash_attn  # noqa: F401
    except Exception:
        missing.append("flash-attn")
    if not os.path.isdir(model_path):
        missing.append(f"model at {model_path}")
    if missing:
        print("Missing prerequisites: " + ", ".join(missing))
        print("Run:\n  uv sync\n  huggingface-cli download Qwen/Qwen3-0.6B "
              "--local-dir " + model_path)
        raise SystemExit(1)


def _print_trace():
    print("\n--- trace (one row per engine step) ---")
    for i, r in enumerate(_trace):
        flag = "PREFILL" if r["is_prefill"] else "DECODE "
        nst = {b[0]: b[1] for b in r["before"]}
        print(f"step {i:2d} {flag} seqs={[b[0] for b in r['before']]} "
              f"nst={nst} new_tokens={r['token_ids']}")


def main():
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.scheduler import Scheduler
    from transformers import AutoTokenizer

    global _orig_postprocess, _orig_add
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    max_tokens = 64

    _check_env(model_path)

    # 装钩子 —— 不改 nanovllm 源码
    _orig_postprocess = Scheduler.postprocess
    _orig_add = Scheduler.add
    Scheduler.postprocess = traced_postprocess
    Scheduler.add = traced_add

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sp = SamplingParams(temperature=0.6, max_tokens=max_tokens)
    raw = ["introduce yourself", "list all prime numbers within 100"]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
               for p in raw]
    llm.generate(prompts, sp)

    _print_trace()
    print("\n--- per-request summary (Task 3) ---")
    for seq_id in sorted(_seqs):
        p, c, t = summarize_request(_seqs[seq_id])
        print(f"seq {seq_id}: prompt_tokens={p} completion_tokens={c} steps={t}")

    print("\n--- checks ---")
    results = run_checks(max_tokens)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if all(ok for _, ok in results):
        print("\nAll checks passed ✓")
    else:
        print("\nSome checks FAILED — review your TODO implementations.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
