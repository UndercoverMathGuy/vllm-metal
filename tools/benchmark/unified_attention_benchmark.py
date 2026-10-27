# SPDX-License-Identifier: Apache-2.0
"""A/B the unified packed-GQA paged attention kernel against the production
routes (per-token / split-KV / GQA decode / window / tiled prefill).

A timed step is a chain of CHAIN dependent calls (each query is the previous
output, like consecutive layers), so a kernel cannot hide latency behind
independent work it would not have in a forward pass.  The chain cycles
through up to eight KV caches sized to overflow the GPU's system-level cache,
so every call reads cold KV.  Long chains keep the GPU busy: short bursts
separated by host synchronization let Apple GPUs drop their clocks, which
inflates and destabilizes latency-bound timings.  The two kernels are timed in
alternating rounds (best median wins).

--prefill-vs-mlx instead times causal prefill through the unified staged grid
against MLX's fused SDPA (the kernel it is shaped after).  Run it after any
edit to the staged kernels, on a cool GPU, in every --dtype: their speed is
sensitive to codegen.

    VLLM_METAL_BUILD_FROM_SOURCE=1 python -m tools.benchmark.unified_attention_benchmark
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from vllm_metal.metal import get_ops

BLOCK_SIZE = 16
TARGET_BYTES = 96 << 20  # KV bytes across the distinct caches
CHAIN = 64
MAX_BYTES = 1 << 30


@dataclass(frozen=True)
class Model:
    label: str
    heads: int
    kv_heads: int
    head_size: int
    sinks: bool = False
    sliding_window: int = -1
    # TurboQuant (key quant type, V bits); None for a dense fp16 cache.
    turboquant: tuple[str, int] | None = None


MODELS = [
    Model("qwen3-0.6b", 16, 8, 128),
    Model("llama3-8b", 32, 8, 128),
    Model("gpt-oss sinks", 64, 8, 64, sinks=True),
    Model("gpt-oss swa128", 64, 8, 64, sinks=True, sliding_window=128),
    Model("gemma3-4b", 8, 4, 256),
    Model("gemma3 swa1024", 8, 4, 256, sliding_window=1024),
    Model("mha-hd64", 16, 16, 64),
    Model("llama3-8b tq8/3", 32, 8, 128, turboquant=("q8_0", 3)),
    Model("qwen3-0.6b tq4/3", 16, 8, 128, turboquant=("q4_0", 3)),
]


def _time(fn, iters, warmup=2):
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_case(model: Model, query_lens, kv_lens, *, iters, rounds, chain=CHAIN):
    ops = get_ops()
    dtype = mx.float16
    rng = np.random.default_rng(0)
    pages = [(n + BLOCK_SIZE - 1) // BLOCK_SIZE for n in kv_lens]
    n_blocks = sum(pages)
    kv_bytes = n_blocks * BLOCK_SIZE * model.kv_heads * model.head_size * 2 * 2
    if kv_bytes > MAX_BYTES:
        return None
    reps = int(min(8, max(1, -(-TARGET_BYTES // kv_bytes))))
    shape = (n_blocks, BLOCK_SIZE, model.kv_heads, model.head_size)
    caches = [
        (mx.random.normal(shape).astype(dtype), mx.random.normal(shape).astype(dtype))
        for _ in range(reps)
    ]
    perm = rng.permutation(n_blocks)
    tables_np = np.zeros((len(kv_lens), max(pages)), dtype=np.int32)
    taken = 0
    for i, n in enumerate(pages):
        tables_np[i, :n] = perm[taken : taken + n]
        taken += n
    tables = mx.array(tables_np)
    lengths = mx.array(kv_lens, dtype=mx.int32)
    cu = mx.array(np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32))
    query = mx.random.normal((sum(query_lens), model.heads, model.head_size))
    query = query.astype(dtype)
    sinks = mx.full((model.heads,), 1.0, dtype=mx.float32) if model.sinks else None
    tq_args = [{} for _ in caches]
    if model.turboquant is not None:
        from vllm_metal.attention.caches.turboquant import (
            get_v_centroids,
            turbo_quant_encode,
        )

        quant, v_bits = model.turboquant
        encoded = []
        for i, (k, v) in enumerate(caches):
            (kq, ks, kz), (vq, vs) = turbo_quant_encode(k, v, quant, v_bits)
            encoded.append((kq, vq))
            tq_args[i] = {
                "key_scale_cache": ks,
                "value_scale_cache": vs,
                "key_zero_cache": kz,
                "v_centroids": get_v_centroids(v_bits),
                "quant_type": quant,
                "v_bits": v_bits,
            }
        caches = encoded
        mx.eval(caches, tq_args)
        kq, vq = caches[0]
        kv_bytes = kq.nbytes + vq.nbytes + 3 * tq_args[0]["key_scale_cache"].nbytes
    mx.eval(caches, query, tables, lengths, cu)
    scale = model.head_size**-0.5
    max_q = max(query_lens)
    routing = {}
    if max_q == 1:
        routing = {
            "num_decode_requests": len(kv_lens),
            "num_decode_tokens": len(kv_lens),
            "max_decode_context_len": max(kv_lens),
            "gqa_length_plan": ops.gqa_decode_length_plan(kv_lens),
        }
    elif len(set(query_lens)) == 1 and max_q <= 8:
        routing = {"window_seqlen_q": max_q}

    def production():
        x = query
        for i in range(chain):
            k, v = caches[i % reps]
            tq = dict(tq_args[i % reps])
            if tq:
                tq["use_turboquant"] = True
            x = ops.paged_attention_primitive(
                x,
                k,
                v,
                model.kv_heads,
                scale,
                0.0,
                tables,
                lengths,
                cu,
                BLOCK_SIZE,
                max(kv_lens),
                model.sliding_window,
                sinks=sinks,
                **routing,
                **tq,
            )
        return x

    def unified():
        x = query
        for i in range(chain):
            k, v = caches[i % reps]
            x = ops.unified_paged_attention_primitive(
                x,
                k,
                v,
                model.kv_heads,
                scale,
                0.0,
                tables,
                lengths,
                cu,
                BLOCK_SIZE,
                max(kv_lens),
                model.sliding_window,
                max_q,
                sinks=sinks,
                **tq_args[i % reps],
            )
        return x

    ops._set_paged_dispatch_diagnostics(True)
    mx.eval(production())
    route = ops.last_paged_dispatch()
    t_prod = t_uni = float("inf")
    for _ in range(rounds):
        t_prod = min(t_prod, _time(production, iters))
        t_uni = min(t_uni, _time(unified, iters))
    return route, t_prod / chain, t_uni / chain, kv_bytes


def bench_prefill_vs_mlx(
    model: Model, length: int, *, iters: int, dtype=mx.float16, chain: int = 4
):
    """Causal prefill of one sequence: unified staged grid vs MLX SDPA.

    Returns (unified s, mlx s, flops) per call; samples alternate.
    """
    ops = get_ops()
    pages = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    shape = (pages, BLOCK_SIZE, model.kv_heads, model.head_size)
    k = mx.random.normal(shape).astype(dtype)
    v = mx.random.normal(shape).astype(dtype)
    q = mx.random.normal((length, model.heads, model.head_size)).astype(dtype)
    tables = mx.arange(pages, dtype=mx.int32)[None]
    lengths = mx.array([length], dtype=mx.int32)
    cu = mx.array([0, length], dtype=mx.int32)

    def dense(cache):
        flat = cache.reshape(1, pages * BLOCK_SIZE, model.kv_heads, model.head_size)
        return flat[:, :length].transpose(0, 2, 1, 3)

    kd, vd = dense(k), dense(v)
    mx.eval(k, v, q, kd, vd)
    scale = model.head_size**-0.5

    def unified():
        x = q
        for _ in range(chain):
            x = ops.unified_paged_attention_primitive(
                x,
                k,
                v,
                model.kv_heads,
                scale,
                0.0,
                tables,
                lengths,
                cu,
                BLOCK_SIZE,
                length,
                -1,
                length,
            )
        return x

    def sdpa():
        x = q[None].transpose(0, 2, 1, 3)
        for _ in range(chain):
            x = mx.fast.scaled_dot_product_attention(
                x, kd, vd, scale=scale, mask="causal"
            )
        return x

    t_uni, t_mlx = [], []
    for _ in range(iters):
        t_uni.append(_time(unified, 1, warmup=0 if t_uni else 2))
        t_mlx.append(_time(sdpa, 1, warmup=0 if t_mlx else 2))
    flops = 4 * model.heads * model.head_size * length * length / 2
    return statistics.median(t_uni) / chain, statistics.median(t_mlx) / chain, flops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--only", default="", help="substring filter on model label")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="four cases, short chains, one round (smoke-level timings)",
    )
    parser.add_argument(
        "--prefill-vs-mlx",
        action="store_true",
        help="causal prefill TFLOPS of the unified staged grid vs MLX SDPA",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="--prefill-vs-mlx dtype",
    )
    args = parser.parse_args()
    if args.prefill_vs_mlx:
        print(
            f"{'model':16s} {'length':>6s} {'unified TF':>10s} {'mlx TF':>7s} {'ratio':>6s}"
        )
        for model in MODELS:
            dense_causal = (
                model.turboquant is None
                and not model.sinks
                and model.sliding_window < 0
            )
            if args.only not in model.label or not dense_causal:
                continue
            for length in (1024, 2048):
                t_uni, t_mlx, flops = bench_prefill_vs_mlx(
                    model, length, iters=args.iters, dtype=getattr(mx, args.dtype)
                )
                print(
                    f"{model.label:16s} {length:6d} {flops / t_uni / 1e12:10.2f} "
                    f"{flops / t_mlx / 1e12:7.2f} {t_mlx / t_uni:5.2f}x",
                    flush=True,
                )
        return
    chain = CHAIN
    if args.quick:
        args.iters, args.rounds, chain = 3, 1, 16

    cases = [
        (f"decode b{b} ctx{c}", [1] * b, [c] * b)
        for b in (1, 4, 16)
        for c in (512, 2048, 8192)
    ] + [
        ("verify k4 b4 ctx2048", [4] * 4, [2048] * 4),
        ("prefill 512", [512], [512]),
        ("prefill 2048", [2048], [2048]),
        ("chunk 256 @ 4096", [256], [4096]),
    ]
    if args.quick:
        cases = [cases[1], cases[4], cases[9], cases[10]]
    print(
        f"{'model':16s} {'case':22s} {'production route':18s} "
        f"{'prod us':>9s} {'unified us':>11s} {'speedup':>8s} {'GB/s uni':>9s}"
    )
    for model in MODELS:
        if args.only not in model.label:
            continue
        for name, query_lens, kv_lens in cases:
            res = bench_case(
                model,
                query_lens,
                kv_lens,
                iters=args.iters,
                rounds=args.rounds,
                chain=chain,
            )
            if res is None:
                continue
            route, t_prod, t_uni, kv_bytes = res
            print(
                f"{model.label:16s} {name:22s} {route:18s} {t_prod * 1e6:9.1f} "
                f"{t_uni * 1e6:11.1f} {t_prod / t_uni:7.2f}x "
                f"{kv_bytes / t_uni / 1e9:9.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
