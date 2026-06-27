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


def _make_mock_fa(device="cpu", dtype=None):
    """Mock fa_varlen/fa_kvcache that (a) validate the calling contract at call time and
    (b) return cross-check-consistent stubs. Uses function-local torch (no GPU needed)."""
    import torch
    if dtype is None:
        dtype = torch.float32
    state = {}
    total, num_heads, head_dim = 5, lab.TOY_NUM_HEADS, lab.TOY_HEAD_DIM

    def mock_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    softmax_scale, causal):
        assert q.ndim == 3, "varlen q must be packed (total, nheads, head_dim), not per-seq padded"
        assert tuple(q.shape)[0] == cu_seqlens_q[-1], "cu_seqlens_q[-1] must equal total tokens"
        assert causal is True
        state["prefill"] = torch.randn(total, num_heads, head_dim, device=device, dtype=dtype)
        return state["prefill"]

    def mock_kvcache(q, k_cache, v_cache, *, cache_seqlens, block_table,
                     softmax_scale, causal):
        assert q.ndim == 4 and tuple(q.shape)[1] == 1, \
            "kvcache q must be (batch, seqlen=1, nheads, head_dim) — did you q_dec.unsqueeze(1)?"
        assert block_table.ndim == 2, "block_table must be 2-D (seq -> physical blocks)"
        assert cache_seqlens.ndim == 1, "cache_seqlens must be 1-D (valid KV count per seq)"
        pf = state["prefill"]
        # cross-check-consistent stub: decode reads the same KV as prefill's last token
        return torch.stack([pf[2], pf[4]]).unsqueeze(1)   # (2,1,num_heads,head_dim)

    return mock_varlen, mock_kvcache


def test_simulate_fa_calls_correct_contract():
    import torch
    mv, mk = _make_mock_fa()
    prefill_out, decode_out = lab.simulate_fa_calls(mv, mk, "cpu", torch.float32)
    # _verify_fa_calls should pass on a correct call
    failed = [n for n, ok in lab._verify_fa_calls(prefill_out, decode_out) if not ok]
    assert not failed, f"correct call should verify clean, failed: {failed}"


def test_simulate_fa_calls_wraps_fa_errors():
    import torch

    def raising(*a, **k):
        raise RuntimeError("fake flash_attn shape error")

    try:
        lab.simulate_fa_calls(raising, raising, "cpu", torch.float32)
        assert False, "should have raised FAContractError"
    except lab.FAContractError as e:
        assert "attention.py" in str(e), "diagnostic must point at attention.py to mirror"


def test_verify_fa_calls_rejects_value_mismatch():
    import torch
    prefill_out = torch.randn(5, 4, 64)
    decode_out = torch.randn(2, 4, 64)   # NOT equal to stack([prefill_out[2], prefill_out[4]])
    failed = [n for n, ok in lab._verify_fa_calls(prefill_out, decode_out) if not ok]
    assert any("cache_seqlens" in n or "allclose" in n for n in failed), \
        "value mismatch must flag the cross-check naming cache_seqlens/block_table"


def test_verify_fa_calls_rejects_bad_prefill_shape():
    import torch
    bad_prefill = torch.randn(7, 4, 64)   # wrong total dim
    decode_out = torch.randn(2, 4, 64)
    failed = [n for n, ok in lab._verify_fa_calls(bad_prefill, decode_out) if not ok]
    assert any("prefill out shape" in n for n in failed)


def _build_trace_scenario():
    """Synthetic _trace + _seq_blocks for run_checks (block_size=4 for tiny numbers)."""
    lab._trace = [
        {"is_prefill": True,
         "q_shape": (9, 4, 64), "k_shape": (9, 1, 64), "v_shape": (9, 1, 64),
         "cu_seqlens_q": [0, 6, 9], "cu_seqlens_k": [0, 6, 9],
         "max_seqlen_q": 6, "max_seqlen_k": 6, "context_lens": None,
         "block_tables": None,
         "slot_mapping": [8, 9, 10, 11, 20, 21,   # seq_A (block_table [2,5]): 4 + 2
                          30, 31, 32]},            # seq_B (block 7): 3
        {"is_prefill": False,
         "q_shape": (2, 4, 64), "k_shape": (2, 1, 64), "v_shape": (2, 1, 64),
         "cu_seqlens_q": None, "cu_seqlens_k": None,
         "max_seqlen_q": 0, "max_seqlen_k": 0, "context_lens": [7, 4],
         "block_tables": (2, 2), "slot_mapping": [99, 100]},
    ]
    lab._seq_blocks = {10: [2, 5], 11: [7]}   # seq_A (id 10) = long prompt; seq_B (id 11)


def test_run_checks_all_pass_on_valid_scenario():
    _build_trace_scenario()
    failed = [n for n, ok in lab.run_checks(max_tokens=4, fa_results=None, block_size=4)
              if not ok and "Task3" not in n]
    assert not failed, f"unexpected failures: {failed}"


def test_run_checks_catches_wrong_slot_geometry():
    _build_trace_scenario()
    lab._trace[0]["slot_mapping"][0] = 999   # corrupt seq_A's first slot
    results = dict(lab.run_checks(max_tokens=4, fa_results=None, block_size=4))
    assert results["Task2 prefill_slot_mapping matches captured slot_mapping"] is False


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
