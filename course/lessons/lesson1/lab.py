"""Lesson 1 lab (reference solution) — trace the journey of a request.

The student file `../lab.py` is this file with the two TODOs blanked.
Importing this module does NOT import nanovllm/torch (those live in main()),
so the synthetic tests in test_checks.py run without a GPU.
"""
import os

_trace = []            # one record per engine step
_seqs = {}             # seq_id -> Sequence
_orig_postprocess = None
_orig_add = None


def traced_add(self, seq):
    """Hook for Scheduler.add: remember every Sequence by id (provided)."""
    _seqs[seq.seq_id] = seq
    return _orig_add(self, seq)


def traced_postprocess(self, seqs, token_ids, is_prefill):
    """TODO(student) — Task 1 (trace): record this step into _trace, then call
    the original via _orig_postprocess(self, seqs, token_ids, is_prefill).

    Capture each seq's num_scheduled_tokens BEFORE calling the original
    (postprocess resets it to 0 at scheduler.py:85). Suggested record:
        {"is_prefill": is_prefill,
         "before": [(s.seq_id, s.num_scheduled_tokens, s.status.name), ...],
         "token_ids": list(token_ids),
         "after":   [(s.seq_id, s.status.name, s.is_finished), ...]}
    """
    raise NotImplementedError("Task 1: fill in the trace recording")


def summarize_request(seq):
    """TODO(student) — Task 3 (small wrapper): return
    (num_prompt_tokens, num_completion_tokens, total_steps).

    total_steps = number of engine steps this seq participated in (count _trace).
    """
    raise NotImplementedError("Task 3: fill in summarize_request")


def run_checks(max_tokens):
    """Verify _trace against the request-lifecycle invariants.

    Each invariant is evaluated across all sequences and reported once as a
    conjunction (any sequence failing the invariant makes the check fail).
    This keeps the result keys stable for dict-based lookups in tests.
    """
    # Gather per-sequence intermediate values.
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
    # Task 1: state-machine trajectory (one prefill step, reached FINISHED,
    # total_steps==num_completion_tokens).
    results.append(("Task1 one prefill step",
                    all(p["n_prefill_steps"] == 1 for p in per_seq)))
    results.append(("Task1 reached FINISHED",
                    all(p["finished"] for p in per_seq)))
    results.append(("Task1 total_steps==num_completion_tokens",
                    all(p["n_steps"] == p["seq"].num_completion_tokens for p in per_seq)))
    # Task 2: num_scheduled_tokens (prefill nst==num_prompt_tokens,
    # decode nst all==1).
    results.append(("Task2 prefill nst==num_prompt_tokens",
                    all(p["prefill_nst"] == [p["seq"].num_prompt_tokens] for p in per_seq)))
    results.append(("Task2 decode nst all==1",
                    all(len(p["decode_nst"]) > 0 and all(n == 1 for n in p["decode_nst"])
                        for p in per_seq)))
    # Task 3: summarize_request (completion==len(completion_token_ids),
    # completion<=max_tokens, total_steps==num_completion_tokens).
    results.append(("Task3 completion==len(completion_token_ids)",
                    all(p["completion"] == len(p["seq"].completion_token_ids) for p in per_seq)))
    results.append(("Task3 completion<=max_tokens",
                    all(p["completion"] <= max_tokens for p in per_seq)))
    results.append(("Task3 total_steps==num_completion_tokens",
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
