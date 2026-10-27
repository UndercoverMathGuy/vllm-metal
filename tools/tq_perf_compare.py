"""TurboQuant perf A/B: current optimised kernels vs upstream/main baseline.

Single script with two roles:

  * Default (orchestrator): runs the worker against the *current* kernel
    sources, then swaps in upstream/main's ``turboquant.metal`` and
    ``paged_ops.cpp``, clears the build cache, runs the worker again in a
    fresh Python interpreter, restores the working tree, and prints a
    side-by-side comparison.

  * ``--mode {new,old}`` (worker): times encode + decode + paged-attention
    on the Qwen3-0.6B shape, writes results to ``--output``.

Usage::

    uv run python tools/tq_perf_compare.py                  # full A/B
    uv run python tools/tq_perf_compare.py --mode new --output new.json
    uv run python tools/tq_perf_compare.py --mode old --output old.json

Why a fresh interpreter per side: MLX caches the compiled metallib in the
loaded extension's library map for the process lifetime. Reloading
``_paged_ops.so`` does not invalidate it. The orchestrator therefore spawns
each worker as a subprocess so the swapped ``.metal`` source actually takes
effect.

Why we swap files instead of switching branches: the bench tool (this file)
only exists on the perf branch. Working-tree swap of just the two kernel
files keeps the bench code, Python helpers, and tests intact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL_FILES = [
    REPO / "vllm_metal" / "metal" / "kernels_v2" / "turboquant.metal",
    REPO / "vllm_metal" / "metal" / "kernels_v2" / "pagedattention.metal",
    REPO / "vllm_metal" / "metal" / "paged_ops.cpp",
]
BUILD_CACHE_GLOB = Path.home() / ".cache" / "vllm-metal"
BACKUP_DIR = Path("/tmp/tq_perf_compare_backup")
NEW_JSON = Path("/tmp/tq_perf_new.json")
OLD_JSON = Path("/tmp/tq_perf_old.json")

# Qwen3-0.6B shape — same as tools/tq_bench.py.
NUM_LAYERS = 28
NUM_KV_HEADS = 4
NUM_Q_HEADS = 16
HEAD_DIM = 128
BLOCK_SIZE = 16
K_QUANT = "q8_0"
K_BITS = 8
V_BITS = 3

ENCODE_N = [1, 16, 256, 1024, 4096]
ATTN_CTX = [128, 512, 2048, 8192]


# ---------------------------------------------------------------------------
# Worker — runs under each kernel state in its own Python interpreter.
# ---------------------------------------------------------------------------


def _sync():
    import mlx.core as mx

    mx.synchronize()


def _time_gpu(fn, warmup=3, iters=40):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e6  # µs


def _make_tq_cache(num_blocks):
    import mlx.core as mx

    from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache

    return MetalPagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=mx.float16,
        turboquant=True,
        k_quant=K_QUANT,
    )


def _bench_encode_new(n, cache, k, v, slot, v_centroids):
    """Fused Metal encode (`ops.tq_encode`)."""
    from vllm_metal.metal import get_ops

    ops = get_ops()

    def run():
        nk, nv, nks, nvs, nkz = ops.tq_encode(
            k,
            v,
            cache.key_caches[0],
            cache.value_caches[0],
            cache.key_scale_caches[0],
            cache.value_scale_caches[0],
            cache.key_zero_caches[0],
            slot,
            v_centroids,
            V_BITS,
            K_BITS,
            True,
        )
        cache.key_caches[0] = nk
        cache.value_caches[0] = nv
        cache.key_scale_caches[0] = nks
        cache.value_scale_caches[0] = nvs
        cache.key_zero_caches[0] = nkz

    def run_eval():
        run()
        import mlx.core as mx

        mx.eval(
            cache.key_caches[0],
            cache.value_caches[0],
            cache.key_scale_caches[0],
            cache.value_scale_caches[0],
            cache.key_zero_caches[0],
        )

    return _time_gpu(run_eval)


def _bench_encode_old(n, cache, k, v, slot, v_centroids):
    """Python `turbo_quant_encode` + 5 MLX scatters (the pre-fused-kernel path)."""
    import mlx.core as mx

    from vllm_metal.metal_kernel_backend.turboquant import turbo_quant_encode

    sg = HEAD_DIM // 32  # scale group count

    def run():
        (pk, ks, kz), (pv, vs) = turbo_quant_encode(
            k,
            v,
            K_QUANT,
            value_bits=V_BITS,
        )

        def sc(arr, data, d):
            flat = arr.reshape(-1, NUM_KV_HEADS, d)
            flat[slot] = data
            return flat.reshape(arr.shape)

        cache.key_caches[0] = sc(cache.key_caches[0], pk, cache.k_packed_dim)
        cache.value_caches[0] = sc(cache.value_caches[0], pv, cache.v_packed_dim)
        cache.key_scale_caches[0] = sc(cache.key_scale_caches[0], ks, sg)
        cache.value_scale_caches[0] = sc(cache.value_scale_caches[0], vs, sg)
        cache.key_zero_caches[0] = sc(cache.key_zero_caches[0], kz, sg)

    def run_eval():
        run()
        mx.eval(
            cache.key_caches[0],
            cache.value_caches[0],
            cache.key_scale_caches[0],
            cache.value_scale_caches[0],
            cache.key_zero_caches[0],
        )

    return _time_gpu(run_eval)


def _bench_decode_step(mode, k, v, slot, v_centroids):
    """28 × encode (one token per layer) — per-decode-step encode cost."""
    import mlx.core as mx

    from vllm_metal.metal import get_ops

    ops = get_ops()
    caches = [_make_tq_cache(num_blocks=8) for _ in range(NUM_LAYERS)]

    if mode == "new":

        def run():
            outs = []
            for c in caches:
                nk, nv, nks, nvs, nkz = ops.tq_encode(
                    k,
                    v,
                    c.key_caches[0],
                    c.value_caches[0],
                    c.key_scale_caches[0],
                    c.value_scale_caches[0],
                    c.key_zero_caches[0],
                    slot,
                    v_centroids,
                    V_BITS,
                    K_BITS,
                    True,
                )
                c.key_caches[0] = nk
                c.value_caches[0] = nv
                c.key_scale_caches[0] = nks
                c.value_scale_caches[0] = nvs
                c.key_zero_caches[0] = nkz
                outs.extend([nk, nv, nks, nvs, nkz])
            mx.eval(*outs)
    else:
        from vllm_metal.metal_kernel_backend.turboquant import turbo_quant_encode

        sg = HEAD_DIM // 32

        def run():
            outs = []
            for c in caches:
                (pk, ks, kz), (pv, vs) = turbo_quant_encode(
                    k,
                    v,
                    K_QUANT,
                    value_bits=V_BITS,
                )

                def sc(arr, data, d):
                    flat = arr.reshape(-1, NUM_KV_HEADS, d)
                    flat[slot] = data
                    return flat.reshape(arr.shape)

                c.key_caches[0] = sc(c.key_caches[0], pk, c.k_packed_dim)
                c.value_caches[0] = sc(c.value_caches[0], pv, c.v_packed_dim)
                c.key_scale_caches[0] = sc(c.key_scale_caches[0], ks, sg)
                c.value_scale_caches[0] = sc(c.value_scale_caches[0], vs, sg)
                c.key_zero_caches[0] = sc(c.key_zero_caches[0], kz, sg)
                outs.extend(
                    [
                        c.key_caches[0],
                        c.value_caches[0],
                        c.key_scale_caches[0],
                        c.value_scale_caches[0],
                        c.key_zero_caches[0],
                    ]
                )
            mx.eval(*outs)

    return _time_gpu(run, warmup=5, iters=20)


def _bench_paged_attention(mode, ctx, v_centroids):
    """Time `paged_attention_primitive(use_turboquant=True)` at the given ctx.

    On the *new* binding: register-resident inverse FWHT.
    On the *old* binding: threadgroup-memory inverse FWHT.

    The Python call is identical — the difference is which `.metal` source
    was JIT-compiled into the loaded `_paged_ops.so`.
    """
    import mlx.core as mx

    from vllm_metal.metal import get_ops

    ops = get_ops()
    num_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    cache = _make_tq_cache(num_blocks=num_blocks)

    k_fill = mx.random.normal(shape=(ctx, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
    v_fill = mx.random.normal(shape=(ctx, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
    slot_fill = mx.arange(ctx, dtype=mx.int64)
    mx.eval(k_fill, v_fill, slot_fill)

    # Populate cache. Mode-agnostic path via Python encode + scatters
    # (correctness check only — orthogonal to what we time).
    from vllm_metal.metal_kernel_backend.turboquant import turbo_quant_encode

    (pk, ks, kz), (pv, vs) = turbo_quant_encode(
        k_fill, v_fill, K_QUANT, value_bits=V_BITS
    )
    sg = HEAD_DIM // 32

    def sc(arr, data, d):
        flat = arr.reshape(-1, NUM_KV_HEADS, d)
        flat[slot_fill] = data
        return flat.reshape(arr.shape)

    cache.key_caches[0] = sc(cache.key_caches[0], pk, cache.k_packed_dim)
    cache.value_caches[0] = sc(cache.value_caches[0], pv, cache.v_packed_dim)
    cache.key_scale_caches[0] = sc(cache.key_scale_caches[0], ks, sg)
    cache.value_scale_caches[0] = sc(cache.value_scale_caches[0], vs, sg)
    cache.key_zero_caches[0] = sc(cache.key_zero_caches[0], kz, sg)
    mx.eval(
        cache.key_caches[0],
        cache.value_caches[0],
        cache.key_scale_caches[0],
        cache.value_scale_caches[0],
        cache.key_zero_caches[0],
    )

    q = mx.random.normal(shape=(1, NUM_Q_HEADS, HEAD_DIM)).astype(mx.float16)
    block_table = mx.arange(num_blocks, dtype=mx.int32).reshape(1, -1)
    seq_lens = mx.array([ctx], dtype=mx.int32)
    cu_seqlens = mx.array([0, 1], dtype=mx.int32)
    mx.eval(q, block_table, seq_lens, cu_seqlens)

    scale = 1.0 / (HEAD_DIM**0.5)

    def run():
        out = mx.array(0)
        ops.paged_attention_primitive(
            q,
            cache.key_caches[0],
            cache.value_caches[0],
            NUM_KV_HEADS,
            scale,
            0.0,
            block_table,
            seq_lens,
            cu_seqlens,
            BLOCK_SIZE,
            ctx,
            -1,
            out,
            key_scale_cache=cache.key_scale_caches[0],
            value_scale_cache=cache.value_scale_caches[0],
            key_zero_cache=cache.key_zero_caches[0],
            v_centroids=v_centroids,
            use_turboquant=True,
            quant_type=K_QUANT,
            v_bits=V_BITS,
        )
        mx.eval(out)

    return _time_gpu(run, warmup=3, iters=30)


def run_worker(mode, output_path):
    """Measurement entry point — invoked as a subprocess per side."""
    import mlx.core as mx

    from vllm_metal.metal_kernel_backend.turboquant import get_v_centroids

    print(f"[worker {mode}] starting (pid={os.getpid()})", flush=True)
    v_centroids = get_v_centroids(V_BITS)
    mx.eval(v_centroids)

    # Warm everything up so we don't bake JIT cost into the first measurement.
    if mode == "new":
        from vllm_metal.metal import get_ops

        ops = get_ops()  # triggers C++ build + Metal JIT
        warm_cache = _make_tq_cache(num_blocks=4)
        k = mx.random.normal(shape=(1, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
        v = mx.random.normal(shape=(1, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
        slot = mx.array([0], dtype=mx.int64)
        mx.eval(k, v, slot)
        for _ in range(3):
            nk, nv, nks, nvs, nkz = ops.tq_encode(
                k,
                v,
                warm_cache.key_caches[0],
                warm_cache.value_caches[0],
                warm_cache.key_scale_caches[0],
                warm_cache.value_scale_caches[0],
                warm_cache.key_zero_caches[0],
                slot,
                v_centroids,
                V_BITS,
                K_BITS,
                True,
            )
            mx.eval(nk, nv, nks, nvs, nkz)
        _sync()
        del warm_cache, k, v, slot

    results = {"mode": mode, "encode": [], "decode_step_us": None, "attn": []}

    # ─── Section 1: encode latency vs N ─────────────────────────────────────
    print(f"[worker {mode}] encode sweep", flush=True)
    for n in ENCODE_N:
        num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE + 2
        cache = _make_tq_cache(num_blocks=num_blocks)
        k = mx.random.normal(shape=(n, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
        v = mx.random.normal(shape=(n, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
        slot = mx.arange(n, dtype=mx.int64)
        mx.eval(k, v, slot)

        if mode == "new":
            us = _bench_encode_new(n, cache, k, v, slot, v_centroids)
        else:
            us = _bench_encode_old(n, cache, k, v, slot, v_centroids)

        results["encode"].append({"n": n, "us": us})
        print(f"[worker {mode}]   N={n:>5}  encode={us:>9.2f} µs", flush=True)
        del cache, k, v, slot

    # ─── Section 2: full decode-step encode cost (28 layers × 1 tok) ───────
    print(f"[worker {mode}] decode-step", flush=True)
    k = mx.random.normal(shape=(1, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
    v = mx.random.normal(shape=(1, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
    slot = mx.array([0], dtype=mx.int64)
    mx.eval(k, v, slot)
    results["decode_step_us"] = _bench_decode_step(mode, k, v, slot, v_centroids)
    print(
        f"[worker {mode}]   28 × encode = {results['decode_step_us']:>9.2f} µs",
        flush=True,
    )
    del k, v, slot

    # ─── Section 3: paged_attention read side ──────────────────────────────
    print(f"[worker {mode}] paged_attention sweep", flush=True)
    for ctx in ATTN_CTX:
        us = _bench_paged_attention(mode, ctx, v_centroids)
        results["attn"].append({"ctx": ctx, "us": us})
        print(f"[worker {mode}]   ctx={ctx:>5}  attn={us:>9.2f} µs", flush=True)
        try:
            mx.metal.clear_cache()
        except (AttributeError, RuntimeError):
            pass

    output_path.write_text(json.dumps(results, indent=2))
    print(f"[worker {mode}] wrote {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _sh(*cmd, **kw):
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, cwd=str(REPO), **kw)


def _backup_kernel_files():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in KERNEL_FILES:
        shutil.copy2(src, BACKUP_DIR / src.name)
    print(f"backed up kernel files → {BACKUP_DIR}", flush=True)


def _restore_kernel_files():
    if not BACKUP_DIR.exists():
        print("WARN: no backup dir to restore from", flush=True)
        return
    for src in KERNEL_FILES:
        backup = BACKUP_DIR / src.name
        if backup.exists():
            shutil.copy2(backup, src)
    print("restored kernel files from backup", flush=True)


def _checkout_upstream_kernels():
    for src in KERNEL_FILES:
        rel = src.relative_to(REPO)
        # `git show upstream/main:<path> > file` — replace contents in place.
        result = subprocess.run(
            ["git", "show", f"upstream/main:{rel}"],
            cwd=str(REPO),
            check=True,
            capture_output=True,
            text=True,
        )
        src.write_text(result.stdout)
    print("swapped in upstream/main kernel files", flush=True)


def _clear_build_cache():
    if not BUILD_CACHE_GLOB.exists():
        return
    for f in BUILD_CACHE_GLOB.glob("_paged_ops*"):
        f.unlink()
        print(f"removed {f}", flush=True)


def _spawn_worker(mode, output):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    _sh(*cmd)


def _print_compare(new, old):
    bar = "─" * 78
    print()
    print("=" * 78)
    print("  TurboQuant kernel A/B  —  current branch vs upstream/main")
    print("=" * 78)

    print()
    print("  Encode latency (µs per call)")
    print("  " + bar)
    print(
        f"  {'N':>6}  {'old (Py + 5 scatters)':>22}  {'new (fused kernel)':>20}  {'speedup':>10}"
    )
    print("  " + bar)
    for o, n in zip(old["encode"], new["encode"], strict=True):
        speedup = o["us"] / n["us"] if n["us"] > 0 else float("nan")
        print(
            f"  {o['n']:>6}  {o['us']:>20.2f}    {n['us']:>18.2f}    {speedup:>8.2f}x"
        )

    print()
    print("  Full decode-step encode cost (28 layers × 1 token)")
    print("  " + bar)
    o = old["decode_step_us"]
    n = new["decode_step_us"]
    print(f"  old: {o:>9.2f} µs    new: {n:>9.2f} µs    speedup: {o / n:>5.2f}x")

    print()
    print("  Paged attention read side (µs per dispatch)")
    print("  " + bar)
    print(
        f"  {'ctx':>6}  {'old (TG-mem FWHT)':>20}  {'new (register FWHT)':>20}  {'speedup':>10}"
    )
    print("  " + bar)
    for o, n in zip(old["attn"], new["attn"], strict=True):
        speedup = o["us"] / n["us"] if n["us"] > 0 else float("nan")
        print(
            f"  {o['ctx']:>6}  {o['us']:>18.2f}    {n['us']:>18.2f}    {speedup:>8.2f}x"
        )

    print()
    print("=" * 78)


def orchestrate():
    """Run new mode, swap kernels, run old mode, restore, compare."""
    print("=" * 78)
    print("  TurboQuant perf A/B  —  orchestrator")
    print("=" * 78)
    print(f"  repo:        {REPO}")
    print(f"  new output:  {NEW_JSON}")
    print(f"  old output:  {OLD_JSON}")
    print(f"  backup dir:  {BACKUP_DIR}")
    print()

    _backup_kernel_files()
    try:
        # ── New side ───────────────────────────────────────────────────────
        _clear_build_cache()  # force rebuild against current sources
        _spawn_worker("new", NEW_JSON)

        # ── Old side ───────────────────────────────────────────────────────
        _checkout_upstream_kernels()
        _clear_build_cache()  # force rebuild against upstream sources
        _spawn_worker("old", OLD_JSON)
    finally:
        # Always restore — even on Ctrl-C / worker crash.
        _restore_kernel_files()
        _clear_build_cache()  # next get_ops() call rebuilds against restored files

    new = json.loads(NEW_JSON.read_text())
    old = json.loads(OLD_JSON.read_text())
    _print_compare(new, old)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["new", "old"],
        default=None,
        help="worker mode (default: orchestrate)",
    )
    p.add_argument(
        "--output", type=Path, default=None, help="JSON output path for worker mode"
    )
    args = p.parse_args()

    if args.mode is None:
        orchestrate()
    else:
        if args.output is None:
            sys.exit("--output required in worker mode")
        run_worker(args.mode, args.output)


if __name__ == "__main__":
    main()
