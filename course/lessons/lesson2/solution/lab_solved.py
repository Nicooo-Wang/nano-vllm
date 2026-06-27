"""Lesson 2 lab (reference solution) — flash-attention calling contracts + paged form.

The student file `../lab.py` is this file with the three TODOs blanked.
Importing this module does NOT import nanovllm/torch/flash_attn (those live in
main()/main_cudagraph(); simulate_fa_calls does a function-local `import torch`),
so the synthetic tests in test_checks.py run without a GPU.
"""
import sys

# ---- toy constants for Task 3 (simulate_fa_calls) ----
TOY_NUM_HEADS = 4
TOY_NUM_KV_HEADS = 1
TOY_HEAD_DIM = 64
BLOCK_SIZE = 256   # config.kvcache_block_size default (config.py:17)

_trace = []                 # one record per engine step (layer 0's Attention.forward call)
_seq_blocks = {}            # seq_id -> list[block_id]  (snapshot of block_table per step)
_orig_attention_forward = None
_orig_postprocess = None
_get_context = None         # set in main() to nanovllm.utils.context.get_context
_layer_order = {}           # id(Attention module) -> layer index (only idx 0 records)


class FAContractError(Exception):
    """Raised when simulate_fa_calls mis-calls flash-attention (wrong args / wrong semantics).
    Carries a diagnostic hint pointing at the likely fix + the attention.py line to mirror."""


def traced_postprocess(self, seqs, token_ids, is_prefill):
    """Hook for Scheduler.postprocess (PROVIDED, not a student TODO).

    Snapshot each seq's block_table at ENTRY, before the original postprocess runs:
    finished seqs get deallocate()'d inside postprocess (scheduler.py:91) which calls
    block_table.clear() (block_manager.py:101), so reading it post-run would be empty.
    """
    for s in seqs:
        _seq_blocks[s.seq_id] = list(s.block_table)
    return _orig_postprocess(self, seqs, token_ids, is_prefill)


def traced_attention_forward(self, q, k, v):
    """TODO(student) — Task 1: record this step's flash-attention call into _trace,
    then call _orig_attention_forward(self, q, k, v).

    Attention.forward fires once per layer per step (num_hidden_layers times); every
    layer in a step shares the same context & shapes, so only LAYER 0 records.
    """
    if _layer_order.setdefault(id(self), len(_layer_order)) == 0:
        context = _get_context()
        is_prefill = context.is_prefill
        _trace.append({
            "is_prefill": is_prefill,
            "q_shape": tuple(q.shape), "k_shape": tuple(k.shape), "v_shape": tuple(v.shape),
            "cu_seqlens_q": context.cu_seqlens_q.tolist() if is_prefill else None,
            "cu_seqlens_k": context.cu_seqlens_k.tolist() if is_prefill else None,
            "max_seqlen_q": context.max_seqlen_q, "max_seqlen_k": context.max_seqlen_k,
            "context_lens": context.context_lens.tolist() if not is_prefill else None,
            "block_tables": (tuple(context.block_tables.shape)
                             if context.block_tables is not None else None),
            "slot_mapping": context.slot_mapping.tolist(),
        })
    return _orig_attention_forward(self, q, k, v)


def prefill_slot_mapping(block_table, block_size, start, num_tokens):
    """TODO(student) — Task 2: reproduce model_runner.py:151-161 prefill slot scatter.

    Each token maps to physical slot block_id*block_size + in-block offset. The first
    block's start is further offset by start%block_size; the last block is truncated to
    end - i*block_size.
    """
    end = start + num_tokens
    start_block = start // block_size
    end_block = (end + block_size - 1) // block_size
    slots = []
    for i in range(start_block, end_block):
        slot_start = block_table[i] * block_size
        if i == start_block:
            slot_start += start % block_size
        if i != end_block - 1:
            slot_end = block_table[i] * block_size + block_size
        else:
            slot_end = block_table[i] * block_size + end - i * block_size
        slots.extend(range(slot_start, slot_end))
    return slots


def _verify_fa_calls(prefill_out, decode_out):
    """Check simulate_fa_calls outputs (PROVIDED). Returns list[(name, ok)]."""
    import torch
    total, num_seqs, num_heads, head_dim = 5, 2, TOY_NUM_HEADS, TOY_HEAD_DIM
    results = []
    results.append(("Task3 simulate prefill out shape (total,H,D)",
                    tuple(prefill_out.shape) == (total, num_heads, head_dim)))
    results.append(("Task3 simulate decode out shape (num_seqs,H,D)",
                    tuple(decode_out.shape) == (num_seqs, num_heads, head_dim)))
    expected = torch.stack([prefill_out[2], prefill_out[4]])
    ok = (decode_out.shape == expected.shape
          and torch.allclose(decode_out, expected, atol=1e-2))
    results.append(("Task3 decode out == prefill last-token (allclose) "
                    "— check cache_seqlens/block_table vs cache prefill", ok))
    return results


def simulate_fa_calls(fa_varlen, fa_kvcache, device, dtype):
    """TODO(student) — Task 3: call the two flash-attention interfaces on toy inputs.
    fa_varlen/fa_kvcache are passed in (real flash_attn from main(); mocks in tests) —
    you only need to CALL them correctly. Focus: what args differ between prefill & decode.
    Returns (prefill_out, decode_out). Mirror attention.py:67-70 (prefill) / 72-74 (decode).
    """
    import torch
    num_heads, num_kv_heads, head_dim = TOY_NUM_HEADS, TOY_NUM_KV_HEADS, TOY_HEAD_DIM
    scale = head_dim ** -0.5
    # toy inputs (provided): 2 seqs of lengths [3, 2]
    q = torch.randn(5, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(5, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(5, num_kv_heads, head_dim, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    try:
        # TODO(student) ①: prefill — flash_attn_varlen_func contract (attention.py:67-70)
        prefill_out = fa_varlen(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                max_seqlen_q=3, max_seqlen_k=3, softmax_scale=scale, causal=True)
        # decode cache (provided): paged (num_blocks, block_size, kv_heads, head_dim)
        q_dec = torch.stack([q[2], q[4]])                                  # last-token q per seq
        k_cache = torch.zeros(2, 256, num_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros_like(k_cache)
        k_cache[0, :3], k_cache[1, :2] = k[:3], k[3:5]                     # simple slice prefill
        v_cache[0, :3], v_cache[1, :2] = v[:3], v[3:5]
        block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
        cache_seqlens = torch.tensor([3, 2], dtype=torch.int32, device=device)
        # TODO(student) ②: decode — flash_attn_with_kvcache contract (attention.py:72-74)
        decode_out = fa_kvcache(q_dec.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=cache_seqlens, block_table=block_table,
                                softmax_scale=scale, causal=True).squeeze(1)
    except FAContractError:
        raise
    except Exception as e:
        raise FAContractError(
            f"flash_attn call failed — likely wrong arguments. "
            f"Mirror attention.py:67-70 (prefill) / 72-74 (decode).\nOriginal: {e}") from e
    return prefill_out, decode_out


def run_checks(max_tokens, fa_results=None, block_size=BLOCK_SIZE):
    """Verify _trace + slot geometry (+ optional FA simulation) against invariants."""
    results = []
    prefill_recs = [r for r in _trace if r["is_prefill"]]
    decode_recs = [r for r in _trace if not r["is_prefill"]]
    results.append(("Task1 captured prefill varlen call", len(prefill_recs) >= 1))
    results.append(("Task1 captured decode kvcache call", len(decode_recs) >= 1))
    if prefill_recs:
        p = prefill_recs[0]
        q_shape = p["q_shape"]
        cu = p["cu_seqlens_q"]
        packed = len(q_shape) == 3 and cu is not None and q_shape[0] == cu[-1]
        results.append(("Task1 prefill q.shape is packed (total,H,D)", packed))
        results.append(("Task4 prefill cu_seqlens_q == [0,a,a+b]",
                        cu == sorted(cu) and cu[0] == 0 and len(cu) >= 2))
        # Task 2: reconstruct seq_A's slot slice and compare to captured
        if _seq_blocks:
            seq_a_id = min(_seq_blocks)
            a = cu[1]
            recon = prefill_slot_mapping(_seq_blocks[seq_a_id], block_size, 0, a)
            results.append(("Task2 prefill_slot_mapping matches captured slot_mapping",
                            recon == list(p["slot_mapping"])[:a]))
        else:
            results.append(("Task2 prefill_slot_mapping matches captured slot_mapping", False))
    if decode_recs:
        d = decode_recs[0]
        cl = d["context_lens"] or []
        results.append(("Task4 decode q.shape is (num_seqs,H,D)",
                        len(d["q_shape"]) == 3 and d["q_shape"][0] == len(cl)))
    if fa_results is not None:
        prefill_out, decode_out = fa_results
        results.extend(_verify_fa_calls(prefill_out, decode_out))
    return results


def _check_env(model_path):
    import os
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
        print("See course/lessons/lesson1/TUTORIAL.md §0 for setup.")
        raise SystemExit(1)


def _print_trace():
    print("\n--- trace (one row per engine step, layer 0's Attention.forward) ---")
    for i, r in enumerate(_trace):
        flag = "PREFILL" if r["is_prefill"] else "DECODE "
        if r["is_prefill"]:
            print(f"step {i:2d} {flag} q={r['q_shape']} cu_q={r['cu_seqlens_q']} "
                  f"cu_k={r['cu_seqlens_k']} slot[:6]={r['slot_mapping'][:6]}...")
        else:
            print(f"step {i:2d} {flag} q={r['q_shape']} ctx_lens={r['context_lens']} "
                  f"block_tables={r['block_tables']} slot={r['slot_mapping']}")


# a long English prompt that tokenizes to > 512 tokens (spans >= 3 blocks of 256)
_LONG_PROMPT = (
    "Explain in detail how paged attention works. " * 60
)


def main():
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.layers.attention import Attention
    from nanovllm.utils.context import get_context
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    from transformers import AutoTokenizer
    import os
    import torch

    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    max_tokens = 4
    _check_env(model_path)

    # Task 3: manual FA call on real flash_attn (GPU). Raises FAContractError if mis-called.
    fa_prefill, fa_decode = simulate_fa_calls(
        flash_attn_varlen_func, flash_attn_with_kvcache, "cuda", torch.bfloat16)

    # LLM() runs warmup_model() inside __init__, which fires Attention.forward for dummy
    # seqs — install hooks AFTER so warmup stays hook-free and _trace holds only the real
    # generate() calls (else _trace[0] would be the warmup prefill and checks would mis-index).
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

    # install hooks — do NOT modify nanovllm source
    global _orig_attention_forward, _orig_postprocess, _get_context
    _orig_attention_forward = Attention.forward
    _orig_postprocess = Scheduler.postprocess
    _get_context = get_context
    Attention.forward = traced_attention_forward
    Scheduler.postprocess = traced_postprocess

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sp = SamplingParams(temperature=0.6, max_tokens=max_tokens)
    raw = [_LONG_PROMPT, "list all prime numbers within 100"]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
               for p in raw]
    llm.generate(prompts, sp)

    _print_trace()
    print("\n--- checks ---")
    results = run_checks(max_tokens, fa_results=(fa_prefill, fa_decode))
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if all(ok for _, ok in results):
        print("\nAll checks passed ✓")
    else:
        print("\nSome checks FAILED — review your TODO implementations.")
        raise SystemExit(1)


def main_cudagraph():
    """Run 2 (Q3b): observe the -1 sentinel in graph_vars['slot_mapping'].
    Separate process — a second LLM() in-process would re-init the NCCL group."""
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import os

    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    _check_env(model_path)
    try:
        llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    except Exception as e:
        print(f"[Q3b] capture_cudagraph unavailable on this GPU: {e}")
        print("[Q3b] Fallback (explain-from-code): model_runner.py:206-207 does")
        print("       slot_mapping.fill_(-1) then overwrites [:bs]; attention.py:23 has")
        print("       `if slot == -1: return` — the -1 sentinel lets the CUDA-graph static")
        print("       buffer ignore slots beyond the real batch (else it would scatter into slot 0).")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sp = SamplingParams(temperature=0.6, max_tokens=4)
    raw = ["introduce yourself", "hi"]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
               for p in raw]
    llm.generate(prompts, sp)

    sm = llm.model_runner.graph_vars["slot_mapping"]
    bs = 2
    n_sentinel = int((sm[bs:] == -1).sum().item())
    print("\n--- Q3b: cudagraph slot_mapping -1 sentinel (model_runner.py:206-207) ---")
    print(f"graph_vars['slot_mapping'].shape = {tuple(sm.shape)} (= min(max_num_seqs, 512))")
    print(f"  [:{bs}]  real slots : {sm[:bs].tolist()}")
    print(f"  [{bs}:]  -1 sentinel : {n_sentinel} of {sm.numel() - bs} are -1")
    print("  → store_kvcache_kernel skips them (attention.py:23 `if slot == -1: return`);")
    print("    without the sentinel, the static graph buffer would scatter stale KV into slot 0.")


if __name__ == "__main__":
    if "--cudagraph" in sys.argv:
        main_cudagraph()
    else:
        main()
