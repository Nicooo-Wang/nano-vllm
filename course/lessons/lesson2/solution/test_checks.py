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


def _fake_context(is_prefill):
    return types.SimpleNamespace(
        is_prefill=is_prefill,
        cu_seqlens_q=types.SimpleNamespace(tolist=lambda: [0, 3, 5]),
        cu_seqlens_k=types.SimpleNamespace(tolist=lambda: [0, 3, 5]),
        max_seqlen_q=3, max_seqlen_k=3,
        context_lens=types.SimpleNamespace(tolist=lambda: [3, 2]),
        block_tables=types.SimpleNamespace(shape=(2, 1)),
        slot_mapping=types.SimpleNamespace(tolist=lambda: [10, 11, 12, 13, 14]),
    )


def _fake_qkv(shape):
    return types.SimpleNamespace(shape=shape)


def test_traced_attention_forward_records_only_layer0_and_calls_original():
    lab._trace = []
    lab._layer_order = {}
    lab._get_context = lambda: _fake_context(is_prefill=True)
    returned = []
    lab._orig_attention_forward = lambda self, q, k, v: returned.append("orig") or "OUT"

    layer0 = object()                       # first Attention module
    layer1 = object()                       # second Attention module
    out = lab.traced_attention_forward(layer0, _fake_qkv((5, 4, 64)), _fake_qkv((5, 1, 64)), _fake_qkv((5, 1, 64)))
    lab.traced_attention_forward(layer1, _fake_qkv((5, 4, 64)), _fake_qkv((5, 1, 64)), _fake_qkv((5, 1, 64)))

    assert out == "OUT" and returned == ["orig", "orig"], "original runs once per layer (only recording is layer-0-gated) and passes through"
    assert len(lab._trace) == 1, "only layer 0 records; layer 1 must be skipped"
    rec = lab._trace[0]
    assert rec["is_prefill"] is True
    assert rec["q_shape"] == (5, 4, 64) and rec["k_shape"] == (5, 1, 64)
    assert rec["cu_seqlens_q"] == [0, 3, 5]
    assert rec["context_lens"] is None      # prefill record does not store context_lens
    assert rec["block_tables"] == (2, 1)
    assert rec["slot_mapping"] == [10, 11, 12, 13, 14]
    # second call from the SAME layer0 must also record (one trace row per step)
    lab.traced_attention_forward(layer0, _fake_qkv((5, 4, 64)), _fake_qkv((5, 1, 64)), _fake_qkv((5, 1, 64)))
    assert len(lab._trace) == 2


def test_prefill_slot_mapping_fresh_noncontiguous():
    # 3 blocks of 256, non-contiguous block ids so scatter is visible; start=0
    got = lab.prefill_slot_mapping([7, 3, 9], 256, start=0, num_tokens=600)
    expected = (list(range(7 * 256, 7 * 256 + 256))      # block 7, full (256)
                + list(range(3 * 256, 3 * 256 + 256))    # block 3, full (256)
                + list(range(9 * 256, 9 * 256 + 88)))    # block 9, last truncated (600-512=88)
    assert got == expected
    assert len(got) == 600


def test_prefill_slot_mapping_chunked_first_block_offset():
    # start=300 → first block offset += 300%256=44; last block truncated to end - i*block_size
    got = lab.prefill_slot_mapping([7, 3, 9], 256, start=300, num_tokens=256)   # end=556
    expected = (list(range(3 * 256 + 44, 3 * 256 + 256))    # block 3 (start_block=1), 212 slots
                + list(range(9 * 256, 9 * 256 + 44)))       # block 9 (last), 556-512=44 slots
    assert got == expected
    assert len(got) == 256


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
