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


def test_traced_postprocess_records_before_calls_then_after():
    called = []
    seqs = [types.SimpleNamespace(
        seq_id=7, num_scheduled_tokens=3,
        status=types.SimpleNamespace(name="RUNNING"), is_finished=False)]
    lab._trace = []

    def fake_orig(self, seqs, token_ids, is_prefill):
        called.append((self, is_prefill))
        seqs[0].status.name = "FINISHED"
        seqs[0].is_finished = True

    lab._orig_postprocess = fake_orig
    lab.traced_postprocess("SELF", seqs, [42], True)

    assert called == [("SELF", True)], "must call original exactly once"
    assert len(lab._trace) == 1
    rec = lab._trace[0]
    assert rec["is_prefill"] is True
    assert rec["before"] == [(7, 3, "RUNNING")], "capture nst BEFORE original"
    assert rec["token_ids"] == [42]
    assert rec["after"] == [(7, "FINISHED", True)], "capture status AFTER original"


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
