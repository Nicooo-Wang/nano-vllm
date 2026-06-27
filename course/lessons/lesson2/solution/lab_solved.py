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
        k_cache = torch.zeros(2, 4, num_kv_heads, head_dim, device=device, dtype=dtype)
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
    raise NotImplementedError("Task 5: fill in run_checks")


def _check_env(model_path):
    raise NotImplementedError("Task 5")


def _print_trace():
    raise NotImplementedError("Task 5")


def main():
    raise NotImplementedError("Task 5")


def main_cudagraph():
    raise NotImplementedError("Task 5")


if __name__ == "__main__":
    if "--cudagraph" in sys.argv:
        main_cudagraph()
    else:
        main()
