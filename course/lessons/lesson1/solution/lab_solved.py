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
    """TODO(student): record this step into _trace, then call the original.

    Each seq still carries its num_scheduled_tokens at entry (postprocess
    resets it to 0 internally at scheduler.py:85), so capture it in `before`.
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
    raise NotImplementedError("Task 3")


def run_checks(max_tokens):
    raise NotImplementedError("Task 4")


def main():
    raise NotImplementedError("Task 4b")


if __name__ == "__main__":
    main()
