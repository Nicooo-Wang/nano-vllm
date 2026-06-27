"""Synthetic unit tests for lab_solved (no GPU, no nano-vllm, no flash-attn).
Run: python solution/test_checks.py
"""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
import lab_solved as lab


def test_traced_postprocess_snapshots_block_table_before_original():
    lab._seq_blocks = {}
    seq = types.SimpleNamespace(seq_id=7, block_table=[3, 8])
    cleared = []  # original postprocess will clear block_table (block_manager.py:101)

    def fake_orig(self, seqs, token_ids, is_prefill):
        for s in seqs:
            s.block_table.clear()   # mimic deallocate clearing it
            cleared.append(s.seq_id)
        return "done"

    lab._orig_postprocess = fake_orig
    out = lab.traced_postprocess("SELF", [seq], [42], True)

    assert out == "done", "must call original and pass through its return value"
    assert cleared == [7], "original must run exactly once"
    assert lab._seq_blocks[7] == [3, 8], "must snapshot block_table BEFORE original clears it"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for t in TESTS:
        try:
            t(); print(f"[PASS] {t.__name__}")
        except (AssertionError, NotImplementedError) as e:  # NotImplementedError → FAIL, not crash
            failures += 1; print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(TESTS)-failures}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)
