"""第 1 课 lab（参考答案）—— 追踪一个 request 的旅程。

学生文件 `../lab.py` 就是本文件挖空两个 TODO 后的版本。
导入本模块不会 import nanovllm/torch（它们只在 main() 里用到），
因此 test_checks.py 里的合成测试无需 GPU 即可运行。
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


# === Task 2 (observe + explain) — 参考答案 ===
# Observe（已由 run_checks 自动验证）：两条 prompt 在同一个 step 0 里一起
#   prefill；每条 seq 的 prefill num_scheduled_tokens == 它的 prompt 长度；
#   每个 DECODE step 的 num_scheduled_tokens == 1。
#
# Explain（参考答案 —— 见 ANSWERS.md §2）：为什么每条 request 的
#   `total_steps == num_completion_tokens`？
# 参考答案：
#   因为一个 seq 参与的每个 engine step 都恰好为它 append 一个 token。
#   prefill 步也不例外：scheduler.py:86-88 的 `continue` 只在 chunked
#   prefill（num_cached_tokens < num_tokens）时触发。本课的 smoke prompt
#   足够短，一次 prefill 就能塞进 max_num_batched_tokens，所以
#   num_cached_tokens == num_tokens，`continue` 不触发，于是
#   `seq.append_token(token_id)` 照跑 —— prefill 步本身就吐出第 1 个
#   completion token。之后的每个 DECODE step 同样各 append 一个。所以
#   该 seq 参与的步数就等于 append 到它身上的 token 数，即
#   num_completion_tokens。（即便多条 seq 共享一个 prefill step，也是
#   按各自计的 —— 数该 seq 在 `before` 里出现的次数。）


def traced_postprocess(self, seqs, token_ids, is_prefill):
    """TODO(student)：把本步记进 _trace，再调用原始实现。

    每个 seq 进入时仍带着它的 num_scheduled_tokens（原始 postprocess
    会在 scheduler.py:85 把它清零），所以要在 `before` 里抓取。
    """
    before = [(s.seq_id, s.num_scheduled_tokens, s.status.name) for s in seqs]
    result = _orig_postprocess(self, seqs, token_ids, is_prefill)
    after = [(s.seq_id, s.status.name, s.is_finished) for s in seqs]
    _trace.append({
        "is_prefill": is_prefill,
        "before": before,
        "token_ids": list(token_ids),
        "after": after,
    })
    return result


def summarize_request(seq):
    """TODO(student)：返回 (num_prompt_tokens, num_completion_tokens, total_steps)。

    total_steps = 这个 seq 参与过的 engine step 数（在 _trace 里数）。
    它应当等于 num_completion_tokens —— prefill 步本身就会吐出第 1 个
    token（scheduler.py:86-88），所以参与的每一步都 append 一个。
    """
    total_steps = sum(
        1 for r in _trace if any(b[0] == seq.seq_id for b in r["before"])
    )
    return (seq.num_prompt_tokens, seq.num_completion_tokens, total_steps)


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
