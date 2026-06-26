"""Synthetic-trace unit tests for lab_solved (no GPU).
Run: python solution/test_checks.py
"""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
import lab_solved as lab


def test_traced_add_captures_and_calls_original():
    lab._seqs = {}
    seq = types.SimpleNamespace(seq_id=99)
    lab._orig_add = lambda self, s: "added"
    out = lab.traced_add("SELF", seq)
    assert out == "added"
    assert lab._seqs[99] is seq


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for t in TESTS:
        try:
            t(); print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            failures += 1; print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(TESTS)-failures}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)
