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


def simulate_fa_calls(fa_varlen, fa_kvcache, device, dtype):
    raise NotImplementedError("Task 3: fill in the two flash-attention calls")


def _verify_fa_calls(prefill_out, decode_out):
    raise NotImplementedError("Task 4b: fill in _verify_fa_calls")


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
