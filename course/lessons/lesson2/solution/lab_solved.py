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
    raise NotImplementedError("Task 1: fill in the FA-call trace recording")


def prefill_slot_mapping(block_table, block_size, start, num_tokens):
    raise NotImplementedError("Task 2: fill in prefill_slot_mapping")


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
