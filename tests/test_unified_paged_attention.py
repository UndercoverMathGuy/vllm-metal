# SPDX-License-Identifier: Apache-2.0
"""Numerical contract of the unified packed-GQA paged attention kernel.

Two grids serve decode, verification windows and prefill, so every case is
checked against the same independent oracle across the shapes that pick
different grids (direct for short sequences, staged for long), split-KV on and
off (`partition` is the number of canonical partitions per split group), and
the score modifiers (causal, sliding window, softcap, sinks).  FP32 runs are
held to fp32 rounding against an FP64 oracle, so a masking, merge or rescale
error cannot hide inside half-precision tolerances.  How a launch spreads the
work must never change a bit: that is checked bitwise.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tests.gqa_test_utils import _assert_close, _grouped_paged_reference
from vllm_metal.metal import get_ops

BLOCK_SIZE = 16
FP32_TOLERANCE = 1e-5


def _oracle_dtype(dtype):
    return mx.float64 if dtype == mx.float32 else mx.float32


def _assert_matches(out, ref, dtype):
    if dtype != mx.float32:
        _assert_close(out, ref, dtype)
        return
    out_np, ref_np = np.array(out), np.array(ref)
    np.testing.assert_allclose(out_np, ref_np, atol=FP32_TOLERANCE, rtol=FP32_TOLERANCE)
    assert np.linalg.norm(out_np - ref_np) / np.linalg.norm(ref_np) < 2e-6


def _inputs(
    query_lens,
    kv_lens,
    num_q_heads,
    num_kv_heads,
    head_size,
    dtype,
    seed,
    block_size=BLOCK_SIZE,
):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    blocks_per_seq = [(n + block_size - 1) // block_size for n in kv_lens]
    width = max(blocks_per_seq)
    # Non-contiguous pages: a shuffled pool shared by every sequence.
    pool = rng.permutation(sum(blocks_per_seq) + 3).tolist()
    tables = np.zeros((len(kv_lens), width), dtype=np.int32)
    taken = 0
    for i, n in enumerate(blocks_per_seq):
        tables[i, :n] = pool[taken : taken + n]
        taken += n
    shape = (len(pool), block_size, num_kv_heads, head_size)
    key_cache = mx.random.normal(shape).astype(dtype)
    value_cache = mx.random.normal(shape).astype(dtype)
    query = mx.random.normal((sum(query_lens), num_q_heads, head_size)).astype(dtype)
    cu = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    return (
        query,
        key_cache,
        value_cache,
        tables,
        mx.array(tables),
        mx.array(kv_lens, dtype=mx.int32),
        mx.array(cu),
    )


def _sink_reference(query, key_cache, value_cache, kv_len, table, scale, sinks):
    """Single-sequence decode oracle with one sink logit per query head."""
    dtype = _oracle_dtype(query.dtype)
    with mx.stream(mx.cpu):
        _, _, kv_heads, head_size = key_cache.shape
        heads = query.shape[1]
        pages = mx.array(table[: (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE])
        keys = key_cache[pages].reshape(-1, kv_heads, head_size)[:kv_len]
        values = value_cache[pages].reshape(-1, kv_heads, head_size)[:kv_len]
        keys = mx.repeat(keys.astype(dtype), heads // kv_heads, axis=1)
        values = mx.repeat(values.astype(dtype), heads // kv_heads, axis=1)
        scores = mx.einsum("hd,nhd->hn", query[0].astype(dtype) * scale, keys)
        scores = mx.concatenate([scores, sinks[:, None].astype(dtype)], axis=1)
        probs = mx.softmax(scores, axis=-1)[:, :kv_len]
        return mx.einsum("hn,nhd->hd", probs, values)[None]


def _run(
    query_lens,
    kv_lens,
    *,
    num_q_heads=32,
    num_kv_heads=8,
    head_size=128,
    dtype=mx.float16,
    sliding_window=-1,
    softcap=0.0,
    partition=-1,
    config=None,
    seed=0,
    block_size=BLOCK_SIZE,
    num_decode_tokens=-1,
):
    query, key_cache, value_cache, tables_np, tables, lengths, cu = _inputs(
        query_lens,
        kv_lens,
        num_q_heads,
        num_kv_heads,
        head_size,
        dtype,
        seed,
        block_size,
    )
    scale = head_size**-0.5
    out = get_ops().unified_paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        softcap,
        tables,
        lengths,
        cu,
        block_size,
        max(kv_lens),
        sliding_window,
        max(query_lens),
        partition=partition,
        config=config,
        num_decode_tokens=num_decode_tokens,
    )
    ref = _grouped_paged_reference(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=list(query_lens),
        kv_lens=list(kv_lens),
        block_tables=tables_np,
        scale=scale,
        sliding_window=None if sliding_window < 0 else sliding_window,
        soft_cap=softcap,
        compute_dtype=_oracle_dtype(dtype),
    )
    # One eval over GPU and CPU-float64 arrays is rejected by MLX.
    mx.eval(out)
    mx.eval(ref)
    return out, ref


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("partition", [0, 1, 3])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads", [(8, 8), (16, 8), (32, 8), (64, 8)]
)
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_decode_matches_reference(
    dtype, partition, num_q_heads, num_kv_heads, head_size
) -> None:
    """Decode packs G = 1..8 query heads into one tile; long and ragged
    contexts cover partial tiles, partial pages and multiple partitions."""
    kv_lens = [1, 37, 700, 1500]
    out, ref = _run(
        [1] * len(kv_lens),
        kv_lens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        partition=partition,
    )
    _assert_matches(out, ref, dtype)


INSTANTIATED_CONFIGS = [
    (64, ("direct", 1, 16)),
    (64, ("direct", 2, 16)),
    (128, ("direct", 1, 8)),
    (128, ("direct", 2, 8)),
    (256, ("direct", 1, 8)),
    (256, ("direct", 2, 8)),
    (64, ("staged", 4, 32)),
    (64, ("staged", 4, 16)),
    (128, ("staged", 4, 16)),
    (128, ("staged", 4, 32)),
    (256, ("staged", 4, 16)),
    (256, ("staged", 2, 16)),
    (256, ("staged", 2, 8)),
]
# Staged FP32 grids whose tiles exceed 32 KB of threadgroup memory.
FP32_OVERSIZED_CONFIGS = [
    (128, ("staged", 4, 32)),
    (256, ("staged", 2, 16)),
    (256, ("staged", 4, 16)),
]


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32, mx.bfloat16])
@pytest.mark.parametrize("partition", [0, 1])
@pytest.mark.parametrize("head_size,config", INSTANTIATED_CONFIGS)
def test_every_simdgroup_grid_matches_reference(
    dtype, head_size, config, partition
) -> None:
    """Every instantiated grid computes the same attention, for both a
    decode batch and a prefill chunk with context."""
    if dtype == mx.float32 and (head_size, config) in FP32_OVERSIZED_CONFIGS:
        pytest.skip("covered by test_rejects_grid_beyond_threadgroup_memory")
    for query_lens, kv_lens in (([1, 1, 1], [5, 600, 1300]), ([3, 40], [900, 40])):
        out, ref = _run(
            query_lens,
            kv_lens,
            head_size=head_size,
            partition=partition,
            config=config,
            dtype=dtype,
        )
        _assert_matches(out, ref, dtype)


@pytest.mark.parametrize("head_size,config", FP32_OVERSIZED_CONFIGS)
def test_rejects_grid_beyond_threadgroup_memory(head_size, config) -> None:
    with pytest.raises(ValueError, match="threadgroup memory"):
        _run([40], [100], head_size=head_size, config=config, dtype=mx.float32)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(16, 16), (32, 8), (16, 2)])
def test_prefill_with_context_matches_reference(
    dtype, head_size, num_q_heads, num_kv_heads
) -> None:
    """Prefill rows (with and without cached context) take the row-parallel
    grid; packed rows span several tokens, so causal masking is per row."""
    query_lens = [1, 45, 130, 7]
    kv_lens = [300, 45, 900, 20]
    out, ref = _run(
        query_lens,
        kv_lens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        partition=0,
    )
    _assert_matches(out, ref, dtype)


@pytest.mark.parametrize("block_size", [8, 32])
@pytest.mark.parametrize("partition", [0, 1])
def test_kernel_block_sizes_match_reference(block_size, partition) -> None:
    """Pages of 8 and 32 keys: decode and a prefill chunk with context."""
    for query_lens, kv_lens in (([1, 1, 1], [7, 650, 1201]), ([5, 33], [500, 33])):
        out, ref = _run(query_lens, kv_lens, partition=partition, block_size=block_size)
        _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("partition", [0, 1, 3])
@pytest.mark.parametrize("window_q", [2, 4])
def test_verification_window_matches_reference(dtype, partition, window_q) -> None:
    """Declared K+1 verification windows run on the decode grid."""
    query_lens = [window_q, window_q, window_q]
    kv_lens = [64, 1100, 523]
    out, ref = _run(
        query_lens,
        kv_lens,
        num_q_heads=16,
        num_kv_heads=8,
        partition=partition,
        dtype=dtype,
        num_decode_tokens=sum(query_lens),
    )
    _assert_matches(out, ref, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("partition", [-1, 0, 1, 2])
@pytest.mark.parametrize("window", [100, 128, 700])
@pytest.mark.parametrize(
    "query_lens,kv_lens",
    [([1, 1], [2000, 90]), ([4, 4], [2048, 333]), ([64, 1], [1200, 1500])],
)
def test_sliding_window_matches_reference(
    dtype, partition, window, query_lens, kv_lens
) -> None:
    """Partitions count from each sequence's earliest window start, so window
    and partition boundaries rarely align; windowed keys of a partition may
    all be masked for some rows and none for others."""
    out, ref = _run(
        query_lens,
        kv_lens,
        sliding_window=window,
        partition=partition,
        dtype=dtype,
    )
    _assert_matches(out, ref, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("query_lens,kv_lens", [([1, 1], [900, 33]), ([40], [300])])
def test_softcap_matches_reference(dtype, query_lens, kv_lens) -> None:
    out, ref = _run(query_lens, kv_lens, softcap=30.0, partition=0, dtype=dtype)
    _assert_matches(out, ref, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("partition", [0, 1, 3])
def test_sinks_join_the_denominator(dtype, partition) -> None:
    """Sinks fold in-kernel on the single pass and in the reduce on split-KV."""
    kv_len, heads, kv_heads, head_size = 1300, 64, 8, 64
    query, key_cache, value_cache, tables_np, tables, lengths, cu = _inputs(
        [1], [kv_len], heads, kv_heads, head_size, dtype, seed=3
    )
    sinks = mx.random.normal((heads,)).astype(mx.float32) + 2.0
    scale = head_size**-0.5
    out = get_ops().unified_paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        kv_heads,
        scale,
        0.0,
        tables,
        lengths,
        cu,
        BLOCK_SIZE,
        kv_len,
        -1,
        1,
        sinks=sinks,
        partition=partition,
    )
    ref = _sink_reference(
        query, key_cache, value_cache, kv_len, tables_np[0], scale, sinks
    )
    mx.eval(out)
    mx.eval(ref)
    _assert_matches(out, ref, dtype)


@pytest.mark.parametrize("partition", [0, 1])
def test_zero_width_window_writes_zeros(partition) -> None:
    """A row that attends no key stores zeros, on the single pass and
    through the split-KV reduce."""
    query, key_cache, value_cache, tables, *_ = _inputs(
        [1, 3], [700, 300], 16, 8, 128, mx.float16, seed=2
    )
    out = _attend(
        query,
        key_cache,
        value_cache,
        tables,
        [1, 3],
        [700, 300],
        num_kv_heads=8,
        sliding_window=0,
        partition=partition,
    )
    assert mx.array_equal(out, mx.zeros_like(out)).item()


def test_rejects_unsupported_geometry() -> None:
    query, key_cache, value_cache, _, tables, lengths, cu = _inputs(
        [1], [16], 8, 8, 96, mx.float16, seed=0
    )
    with pytest.raises(ValueError, match="head sizes"):
        get_ops().unified_paged_attention_primitive(
            query,
            key_cache,
            value_cache,
            8,
            0.1,
            0.0,
            tables,
            lengths,
            cu,
            BLOCK_SIZE,
            16,
            -1,
            1,
        )


def _attend(query, key_cache, value_cache, tables, query_lens, kv_lens, **kwargs):
    """Auto-planned unified attention over `tables` rows (numpy int32)."""
    cu = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    out = get_ops().unified_paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        kwargs.pop("num_kv_heads"),
        query.shape[2] ** -0.5,
        0.0,
        mx.array(tables),
        mx.array(kv_lens, dtype=mx.int32),
        mx.array(cu),
        BLOCK_SIZE,
        max(kv_lens),
        kwargs.pop("sliding_window", -1),
        max(query_lens),
        **kwargs,
    )
    mx.eval(out)
    return out


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize(
    "query_lens,kv_lens,options",
    [
        ([1, 1, 1], [5, 700, 1500], {}),
        ([4, 4], [1100, 64], {"num_q_heads": 16, "num_decode_tokens": 8}),
        ([1, 45, 130, 7], [300, 45, 900, 20], {}),
        ([1, 64], [2000, 1200], {"sliding_window": 300}),
        ([1, 3], [1300, 600], {"head_size": 256}),
        ([1, 50], [1500, 400], {"head_size": 64, "sinks": True}),
    ],
)
def test_work_split_does_not_change_bits(dtype, query_lens, kv_lens, options) -> None:
    """Only the canonical partition tree fixes a row's bits: a single pass,
    split-KV groups of any size and the auto plan agree exactly, for direct,
    staged and mixed launches."""
    options = dict(options)
    num_q_heads = options.pop("num_q_heads", 32)
    head_size = options.pop("head_size", 128)
    query, key_cache, value_cache, tables, *_ = _inputs(
        query_lens, kv_lens, num_q_heads, 8, head_size, dtype, seed=9
    )
    if options.pop("sinks", False):
        options["sinks"] = mx.random.normal((num_q_heads,)).astype(mx.float32)

    def attend(partition):
        return _attend(
            query,
            key_cache,
            value_cache,
            tables,
            query_lens,
            kv_lens,
            num_kv_heads=8,
            partition=partition,
            **options,
        )

    single = attend(0)
    for partition in (1, 2, 3, -1):
        assert mx.array_equal(attend(partition), single).item(), partition


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_row_simdgroups_do_not_change_bits(dtype) -> None:
    """Stacking direct row simdgroups, so a verification window reads its KV
    once, leaves every row's arithmetic unchanged."""
    query_lens, kv_lens = [8, 3, 16], [2000, 300, 900]
    query, key_cache, value_cache, tables, *_ = _inputs(
        query_lens, kv_lens, 32, 8, 128, dtype, seed=4
    )
    outs = [
        _attend(
            query,
            key_cache,
            value_cache,
            tables,
            query_lens,
            kv_lens,
            num_kv_heads=8,
            config=("direct", sg_m, 8),
        )
        for sg_m in (1, 2, 4)
    ]
    for out in outs[1:]:
        assert mx.array_equal(out, outs[0]).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("target_len", [300, 1300])
@pytest.mark.parametrize("head_size,num_q_heads", [(64, 32), (128, 16), (256, 16)])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_decode_is_batch_invariant(
    dtype, target_len, head_size, num_q_heads, use_sinks
) -> None:
    """A decode row's output is bit-identical alone and inside batches that
    change the grid size, force split-KV (a 3000-key neighbour) or launch the
    prefill grid alongside (a 40-token neighbour)."""
    num_kv_heads = 8
    query_lens = [1, 1, 1, 40, 1]
    kv_lens = [target_len, 3000, 5, 700, 64]
    query, key_cache, value_cache, tables, *_ = _inputs(
        query_lens, kv_lens, num_q_heads, num_kv_heads, head_size, dtype, seed=7
    )
    kwargs = {"num_kv_heads": num_kv_heads}
    if use_sinks:
        kwargs["sinks"] = mx.random.normal((num_q_heads,)).astype(mx.float32)
    starts = np.concatenate([[0], np.cumsum(query_lens)])

    def target_rows(seqs):
        rows = np.concatenate(
            [np.arange(starts[i], starts[i] + query_lens[i]) for i in seqs]
        )
        out = _attend(
            query[mx.array(rows)],
            key_cache,
            value_cache,
            tables[seqs],
            [query_lens[i] for i in seqs],
            [kv_lens[i] for i in seqs],
            **dict(kwargs),
        )
        return out[sum(query_lens[i] for i in seqs[: seqs.index(0)])]

    alone = target_rows([0])
    for seqs in ([0, 2], [2, 0, 4], [0, 1], [1, 2, 0, 4], [0, 3], [3, 1, 2, 4, 0]):
        assert mx.array_equal(target_rows(seqs), alone).item(), seqs


def _window_and_expanded(
    windows, contexts, dtype, turboquant=None, prefill=None, **kwargs
):
    """The same verification windows as one declared segment each and as
    one-token decode segments (one per window row), optionally followed by a
    prefill segment (query_len, kv_len); returns the window rows of both."""
    num_q_heads, num_kv_heads, head_size = 32, 8, 128
    window_lens = [c + w for c, w in zip(contexts, windows, strict=True)]
    extra_q, extra_kv = ([prefill[0]], [prefill[1]]) if prefill else ([], [])
    query, key_cache, value_cache, tables, *_ = _inputs(
        windows + extra_q,
        window_lens + extra_kv,
        num_q_heads,
        num_kv_heads,
        head_size,
        dtype,
        seed=11,
    )
    if turboquant is not None:
        from vllm_metal.attention.caches.turboquant import (
            get_v_centroids,
            turbo_quant_encode,
        )

        quant, v_bits = turboquant
        (key_cache, k_scale, k_zero), (value_cache, v_scale) = turbo_quant_encode(
            key_cache, value_cache, quant, v_bits
        )
        kwargs.update(
            key_scale_cache=k_scale,
            value_scale_cache=v_scale,
            key_zero_cache=k_zero,
            v_centroids=get_v_centroids(v_bits),
            quant_type=quant,
            v_bits=v_bits,
        )
    rows = sum(windows)
    windowed = _attend(
        query,
        key_cache,
        value_cache,
        tables,
        windows + extra_q,
        window_lens + extra_kv,
        num_kv_heads=num_kv_heads,
        num_decode_tokens=rows,
        **dict(kwargs),
    )
    seq_of_row = np.repeat(np.arange(len(windows)), windows)
    if prefill:
        seq_of_row = np.append(seq_of_row, len(windows))
    lens = [c + j + 1 for c, w in zip(contexts, windows, strict=True) for j in range(w)]
    expanded = _attend(
        query,
        key_cache,
        value_cache,
        tables[seq_of_row],
        [1] * rows + extra_q,
        lens + extra_kv,
        num_kv_heads=num_kv_heads,
        num_decode_tokens=rows,
        **dict(kwargs),
    )
    return windowed[:rows], expanded[:rows]


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("windows", [[2, 4], [5, 8], [16, 3]])
@pytest.mark.parametrize("sliding_window", [-1, 300])
def test_verification_window_matches_decode_steps(
    dtype, windows, sliding_window
) -> None:
    """Speculative decoding stays lossless: each row of a K+1 verification
    window is bit-identical to decoding that token on its own, for contexts
    inside one split-KV partition and across several."""
    windowed, expanded = _window_and_expanded(
        windows, [200, 1900], dtype, sliding_window=sliding_window
    )
    assert mx.array_equal(windowed, expanded).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("windows", [[17, 24], [33]])
@pytest.mark.parametrize("prefill", [None, (40, 600)])
def test_long_verification_window_matches_decode_steps(dtype, windows, prefill) -> None:
    """A window takes the decode grid because it is a declared verification
    window, not because it is short: windows past 16 tokens, alone or next to
    a prefill segment, stay bit-identical to one-token decode."""
    windowed, expanded = _window_and_expanded(
        windows, [300, 1900][: len(windows)], dtype, prefill=prefill
    )
    assert mx.array_equal(windowed, expanded).item()


def _production(ops, query, key_cache, value_cache, tables, query_lens, kv_lens, **kw):
    cu = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    out = ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        key_cache.shape[2],
        query.shape[2] ** -0.5,
        0.0,
        mx.array(tables),
        mx.array(kv_lens, dtype=mx.int32),
        mx.array(cu),
        BLOCK_SIZE,
        max(kv_lens),
        -1,
        **kw,
    )
    mx.eval(out)
    return out


def test_production_route_keeps_long_windows_on_decode_grid() -> None:
    """Through paged_attention_primitive, the scheduler's decode metadata keeps
    a 20-token window next to a prefill segment bit-identical to its expanded
    decode rows; without that metadata a mixed batch declines the unified
    route instead of guessing which rows are decode rows."""
    ops = get_ops()
    window, ctx_len, prefill_q, prefill_kv = 20, 700, 48, 500
    query, key_cache, value_cache, tables, *_ = _inputs(
        [window, prefill_q],
        [ctx_len + window, prefill_kv],
        32,
        8,
        128,
        mx.float16,
        seed=13,
    )
    expanded_tables = tables[np.append(np.zeros(window, np.int64), 1)]
    expanded_lens = [ctx_len + j + 1 for j in range(window)] + [prefill_kv]
    previous = ops._set_paged_dispatch_diagnostics(True)
    ops.set_unified_attention_enabled(True)
    try:
        windowed = _production(
            ops,
            query,
            key_cache,
            value_cache,
            tables,
            [window, prefill_q],
            [ctx_len + window, prefill_kv],
            decode_query_tokens=window,
            max_decode_query_len=window,
        )
        assert ops.last_paged_dispatch() == "unified"
        expanded = _production(
            ops,
            query,
            key_cache,
            value_cache,
            expanded_tables,
            [1] * window + [prefill_q],
            expanded_lens,
            decode_query_tokens=window,
            max_decode_query_len=1,
        )
        assert mx.array_equal(windowed[:window], expanded[:window]).item()
        _production(
            ops,
            query,
            key_cache,
            value_cache,
            tables,
            [window, prefill_q],
            [ctx_len + window, prefill_kv],
        )
        assert ops.last_paged_dispatch() != "unified"
    finally:
        ops.set_unified_attention_enabled(False)
        ops._set_paged_dispatch_diagnostics(previous)


def test_one_token_prefill_takes_one_grid_in_every_batch() -> None:
    """A one-token prefill segment (a prefix-cache hit recomputing its last
    prompt token) is bit-identical alone, beside a decode row, and beside a
    decode row and a longer prefill: the declared decode count, not segment
    lengths, picks its grid."""
    ops = get_ops()
    query_lens, kv_lens = [1, 1, 40], [300, 700, 500]
    query, key_cache, value_cache, tables, *_ = _inputs(
        query_lens, kv_lens, 32, 8, 128, mx.float16, seed=1
    )
    ops.set_unified_attention_enabled(True)
    try:
        alone = _production(
            ops,
            query[1:2],
            key_cache,
            value_cache,
            tables[1:2],
            [1],
            [700],
            decode_query_tokens=0,
            max_decode_query_len=0,
        )[0]
        for seqs in ([0, 1], [0, 1, 2]):
            q_rows = int(np.sum([query_lens[i] for i in seqs]))
            out = _production(
                ops,
                query[:q_rows],
                key_cache,
                value_cache,
                tables[seqs],
                [query_lens[i] for i in seqs],
                [kv_lens[i] for i in seqs],
                decode_query_tokens=1,
                max_decode_query_len=1,
            )
            assert mx.array_equal(out[1], alone).item(), seqs
    finally:
        ops.set_unified_attention_enabled(False)


def test_verification_window_with_sinks_matches_decode_steps() -> None:
    sinks = mx.random.normal((32,)).astype(mx.float32)
    windowed, expanded = _window_and_expanded(
        [4, 6], [300, 1500], mx.bfloat16, sinks=sinks
    )
    assert mx.array_equal(windowed, expanded).item()


def _run_turboquant(
    query_lens,
    kv_lens,
    *,
    quant,
    v_bits,
    head_size,
    partition,
    sliding_window=-1,
    num_q_heads=16,
    num_kv_heads=4,
    seed=5,
    dtype=mx.float16,
):
    from vllm_metal.attention.caches.turboquant import (
        get_v_centroids,
        turbo_quant_decode,
        turbo_quant_encode,
    )

    query, key_cache, value_cache, tables_np, tables, lengths, cu = _inputs(
        query_lens, kv_lens, num_q_heads, num_kv_heads, head_size, dtype, seed
    )
    (k_q, k_scale, k_zero), (v_q, v_scale) = turbo_quant_encode(
        key_cache, value_cache, quant, v_bits
    )
    key_ref, value_ref = turbo_quant_decode(
        (k_q, k_scale, k_zero),
        (v_q, v_scale),
        output_dtype=dtype,
        key_quant_type=quant,
        value_bits=v_bits,
    )
    scale = head_size**-0.5
    out = get_ops().unified_paged_attention_primitive(
        query,
        k_q,
        v_q,
        num_kv_heads,
        scale,
        0.0,
        tables,
        lengths,
        cu,
        BLOCK_SIZE,
        max(kv_lens),
        sliding_window,
        max(query_lens),
        partition=partition,
        key_scale_cache=k_scale,
        value_scale_cache=v_scale,
        key_zero_cache=k_zero,
        v_centroids=get_v_centroids(v_bits),
        quant_type=quant,
        v_bits=v_bits,
    )
    ref = _grouped_paged_reference(
        query=query,
        key_cache=key_ref,
        value_cache=value_ref,
        query_lens=list(query_lens),
        kv_lens=list(kv_lens),
        block_tables=tables_np,
        scale=scale,
        sliding_window=None if sliding_window < 0 else sliding_window,
        compute_dtype=_oracle_dtype(dtype),
    )
    mx.eval(out)
    mx.eval(ref)
    return out, ref


@pytest.mark.parametrize("quant", ["q8_0", "uint8", "q5_0", "q4_0", "int2"])
@pytest.mark.parametrize("v_bits", [2, 3, 4, 8])
def test_turboquant_decode_matches_dequantized_reference(quant, v_bits) -> None:
    """Every key format and V width decodes to the same attention as the
    Python dequantization, including the deferred inverse FWHT."""
    for partition in (0, 1):
        out, ref = _run_turboquant(
            [1, 1, 1],
            [9, 700, 1300],
            quant=quant,
            v_bits=v_bits,
            head_size=128,
            partition=partition,
        )
        _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32, mx.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize(
    "query_lens,kv_lens,partition",
    [
        ([1, 1], [1100, 30], 0),
        ([1, 1], [1100, 30], 2),
        ([3, 3], [700, 64], 1),
        ([1, 70], [600, 400], 0),
        ([1, 70], [600, 400], 1),
    ],
)
def test_turboquant_paths_match_dequantized_reference(
    dtype, head_size, query_lens, kv_lens, partition
) -> None:
    """Direct (decode, verification) and staged (prefill) dequantization, on
    the single pass and through the split-KV reduce."""
    out, ref = _run_turboquant(
        query_lens,
        kv_lens,
        quant="q8_0",
        v_bits=3,
        head_size=head_size,
        partition=partition,
        dtype=dtype,
    )
    _assert_matches(out, ref, dtype)


def test_turboquant_sliding_window_matches_dequantized_reference() -> None:
    for partition in (0, 1):
        out, ref = _run_turboquant(
            [1, 4],
            [2000, 500],
            quant="q4_0",
            v_bits=3,
            head_size=128,
            partition=partition,
            sliding_window=300,
        )
        _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("quant,v_bits", [("q8_0", 3), ("q4_0", 4)])
def test_turboquant_verification_window_matches_decode_steps(quant, v_bits) -> None:
    windowed, expanded = _window_and_expanded(
        [4, 6], [300, 1500], mx.float16, turboquant=(quant, v_bits)
    )
    assert mx.array_equal(windowed, expanded).item()


@pytest.mark.parametrize("turboquant", [None, ("q8_0", 3)])
def test_repeated_launches_are_bit_identical(turboquant) -> None:
    """Cross-lane data movement inside a simdgroup must be ordered: a P^T
    transpose through threadgroup memory once changed the output between
    identical launches (two-token windows over 8-key tiles)."""
    first, _ = _window_and_expanded(
        [2, 2, 2, 2], [16, 40, 9, 100], mx.float16, turboquant=turboquant
    )
    for _ in range(8):
        again, _ = _window_and_expanded(
            [2, 2, 2, 2], [16, 40, 9, 100], mx.float16, turboquant=turboquant
        )
        assert mx.array_equal(again, first).item()
