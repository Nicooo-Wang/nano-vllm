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


def _seq(seq_id, prompt, completion, finished=False):
    s = types.SimpleNamespace(
        seq_id=seq_id,
        num_prompt_tokens=prompt,
        num_tokens=prompt + completion,
        token_ids=list(range(prompt + completion)),
        status=types.SimpleNamespace(name="FINISHED" if finished else "RUNNING"),
        is_finished=finished,
    )
    s.completion_token_ids = s.token_ids[s.num_prompt_tokens:]
    s.num_completion_tokens = s.num_tokens - s.num_prompt_tokens
    return s


def _rec(is_prefill, entries, token_ids):
    """entries: list of (seq_id, num_scheduled_tokens)."""
    return {
        "is_prefill": is_prefill,
        "before": [(sid, nst, "RUNNING") for sid, nst in entries],
        "token_ids": list(token_ids),
        "after": [(sid, "RUNNING", False) for sid, _ in entries],
    }


def _mark_finished(record, seq_id):
    record["after"] = [(sid, "FINISHED" if sid == seq_id else nm,
                        sid == seq_id or fin)
                       for (sid, nm, fin) in record["after"]]


def _build_valid_scenario():
    """seq0 在 4 个 completion token 时命中 eos；seq1 在 5 个时到 max。"""
    s0 = _seq(0, prompt=10, completion=4, finished=True)
    s1 = _seq(1, prompt=12, completion=5, finished=True)
    lab._seqs = {0: s0, 1: s1}
    lab._trace = [
        _rec(True, [(0, 10), (1, 12)], [100, 200]),   # step 0 prefill 两条
        _rec(False, [(0, 1), (1, 1)], [101, 201]),    # step 1 decode 两条
        _rec(False, [(0, 1), (1, 1)], [102, 202]),    # step 2 decode 两条
        _rec(False, [(0, 1), (1, 1)], [103, 203]),    # step 3 decode 两条，seq0 结束
        _rec(False, [(1, 1)], [204]),                 # step 4 只 decode seq1，结束
    ]
    _mark_finished(lab._trace[3], 0)
    _mark_finished(lab._trace[4], 1)
    return s0, s1


def test_summarize_request_counts_steps():
    s0, s1 = _build_valid_scenario()
    assert lab.summarize_request(s0) == (10, 4, 4)
    assert lab.summarize_request(s1) == (12, 5, 5)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for t in TESTS:
        try:
            t(); print(f"[PASS] {t.__name__}")
        except (AssertionError, NotImplementedError) as e:
            failures += 1; print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(TESTS)-failures}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)
