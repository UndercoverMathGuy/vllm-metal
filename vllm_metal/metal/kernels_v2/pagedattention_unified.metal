// SPDX-License-Identifier: Apache-2.0
// Unified paged attention: Flash-Attention-2 kernels for decode, spec-decode
// verification windows and prefill, plus their split-KV reduce.
//
// Packed GQA rows.  A threadgroup owns one KV head and a block of "packed"
// query rows of one sequence: packed row p maps to query token p / G and query
// head kv_head * G + p % G (G = query heads per KV head).  Every query head
// that shares the KV head is therefore a row of the same MMA tile, so each
// K/V tile is read once per KV head instead of once per query head, and a
// decode step (one token, G heads) is an ordinary G-row tile.
//
// Canonical reduction.  On the direct grid a row's keys fall into absolute
// partitions of partition_size keys.  Each partition is reduced by one
// simdgroup walking its TILE_KV-key tiles in key order with an online softmax,
// normalised, and the partitions are blended online in partition order
// (ua_blend_weights); the first live partition is copied, so a row inside one
// partition is exactly its partial.  Only that tree is fixed: which simdgroup
// or threadgroup reduces a partition, whether partials are blended in
// registers or written for paged_attention_unified_reduce, and how many rows
// share a launch are free, and none of them changes a bit.  The staged grid
// never splits; its rows run one online softmax over all their tiles.  A
// token's output therefore does not depend on batch size or composition, or
// on decoding alone versus inside a verification window, while the host still
// sizes the grid to the batch.
//
//   Direct (paged_attention_unified_direct; decode and verification): each
//   simdgroup owns 8 rows and a group of consecutive partitions, and reads
//   K/V fragments straight from device memory, without barriers.  An 8-token
//   fragment never crosses a page, so its rows sit kv_token_stride apart.
//   Scores are computed transposed, S^T = K x Q^T, so K fragments are read
//   row-major (contiguous) instead of as a strided device transpose; the
//   online-softmax state is per fragment column, and P^T and the per-row
//   factors move to the O layout by shuffle for O += P x V.  Row simdgroups
//   of a threadgroup (sg_m) share each K/V fragment through the cache, so a
//   verification window reads its KV once.
//
//   Staged (paged_attention_unified_staged; prefill): row parallelism,
//   structured like MLX steel attention (steel/attn/kernels/
//   steel_attention.h): each thread transposes one run of a key's head slice
//   into the tile's K^T.  Up to head 128 Q stays in registers and K^T and V
//   have their own buffers; head 256 stages Q too and alternates K^T and V
//   in one buffer.
//
// TurboQuant KV (ua_tq; formats in turboquant.metal).  The caches are bound as
// bytes.  K elements dequantize as (idx + zero) * scale per 32-element group;
// V is stored FWHT-rotated, so its Lloyd-Max dequant (centroid * rms scale)
// feeds the PV product in the rotated domain and, by linearity, the inverse
// FWHT runs once per output row: in the epilogue, or after the split-KV merge.
//
// Split-KV (direct grid).  With ua_partitioned, each simdgroup writes its
// partitions' log2-space (max, exp-sum) stats and normalised fp32 partials at tmp_out[token, head, partition, :], numbered
// from the partition holding the sequence's earliest attended key (its first
// query row's window start, or 0), so a short sliding window over a long
// context needs only a couple of partitions.

#include "utils.metal"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// Full unrolling, as MLX's STEEL_PRAGMA_UNROLL; the staged grid relies on it.
#define UA_UNROLL _Pragma("clang loop unroll(full)")

constant bool ua_partitioned [[function_constant(10)]];
constant bool ua_single_pass = !ua_partitioned;
constant bool ua_use_sinks [[function_constant(40)]];
// Softcap code, even untaken, costs the unrolled staged loop ~15% at head 256.
constant bool ua_softcap [[function_constant(41)]];
constant bool ua_tq [[function_constant(50)]];
constant int ua_k_bits [[function_constant(60)]];
constant int ua_v_bits [[function_constant(70)]];
constant bool ua_k_signed [[function_constant(71)]];

// Launch parameters; mirrors UnifiedKernelParams in unified_attention.cpp.
struct UaParams {
  int num_kv_heads;
  float scale;
  float softcap;
  int max_num_blocks_per_seq;
  // > 0: every sequence owns seq_blocks consecutive row blocks.
  int seq_blocks;
  int q_stride;
  int kv_block_stride;
  int kv_head_stride;
  int num_seqs;
  int sliding_window;
  int partition_size;
  int seq_filter;
  int decode_tokens;
  // Direct: row simdgroups per threadgroup.
  int sg_m;
  // Partition groups per row block, and partitions per group.
  int groups;
  int group_partitions;
  // Partition slots per (token, head) in tmp_out.
  int max_partitions;
  // TurboQuant strides.
  int v_block_stride;
  int v_head_stride;
  int scale_block_stride;
  int scale_head_stride;
};

struct UaReduceParams {
  int max_partitions;
  int num_seqs;
  int partition_size;
  int sliding_window;
  int seq_filter;
  int decode_tokens;
};

// Thread -> (col, row) of its two elements inside an 8x8 simdgroup_matrix
// fragment; see pagedattention_tiled.metal for the derivation.  The lane owns
// (fm, fn) and (fm, fn + 1).  fm takes lane bits 1, 2 and 4; fn takes lane
// bits 0 and 3.
inline short2 ua_frag_coord(ushort lane_id) {
  const short qid = short(lane_id) / 4;
  const short fm = (qid & 4) + (short(lane_id) / 2) % 4;
  const short fn = (qid & 2) * 2 + (short(lane_id) % 2) * 2;
  return short2{fn, fm};
}

// Reductions along a fragment row: its four lanes differ in lane bits 0, 3.
inline float ua_row_max(float v) {
  v = max(v, simd_shuffle_xor(v, ushort(1)));
  return max(v, simd_shuffle_xor(v, ushort(8)));
}

inline float ua_row_sum(float v) {
  v += simd_shuffle_xor(v, ushort(1));
  return v + simd_shuffle_xor(v, ushort(8));
}

// Reductions down a fragment column: its eight lanes differ in lane bits 1,
// 2 and 4.  The result is replicated to all eight.
inline float ua_col_max(float v) {
  v = max(v, simd_shuffle_xor(v, ushort(2)));
  v = max(v, simd_shuffle_xor(v, ushort(4)));
  return max(v, simd_shuffle_xor(v, ushort(16)));
}

inline float ua_col_sum(float v) {
  v += simd_shuffle_xor(v, ushort(2));
  v += simd_shuffle_xor(v, ushort(4));
  return v + simd_shuffle_xor(v, ushort(16));
}

// Lane holding fragment element (fm, fn) (fn even); inverse of ua_frag_coord.
inline ushort ua_frag_lane(short fm, short fn) {
  return ushort(((fn >> 1) & 1) | ((fm & 3) << 1) | (((fn >> 2) & 1) << 3) |
                (((fm >> 2) & 1) << 4));
}

// This lane's elements (fm, fn) and (fm, fn + 1) of P, given its elements
// t0 = P^T(fm, fn) and t1 = P^T(fm, fn + 1) of the transpose.  Shuffles,
// not a threadgroup round trip: a transposed simdgroup_store read back after
// simdgroup_barrier raced on M1.
inline float2 ua_transpose_pair(float t0, float t1, short fm, short fn) {
  const ushort src0 = ua_frag_lane(fn, fm & ~1);
  const ushort src1 = ua_frag_lane(fn + 1, fm & ~1);
  const float a0 = simd_shuffle(t0, src0);
  const float a1 = simd_shuffle(t1, src0);
  const float b0 = simd_shuffle(t0, src1);
  const float b1 = simd_shuffle(t1, src1);
  return (fm & 1) ? float2(a1, b1) : float2(a0, b0);
}

// Per-column values (col0 for column fn, col1 for fn + 1) re-read as the
// value of column `row`, i.e. this lane's O row.  The owner is any lane whose
// fn is row & 6; lane bit 0 is (row >> 1) & 1 and lane bit 3 is (row >> 2) & 1.
inline float ua_col_to_row(float col0, float col1, short row) {
  const ushort owner = ushort((((row >> 2) & 1) << 3) | ((row >> 1) & 1));
  const float v0 = simd_shuffle(col0, owner);
  const float v1 = simd_shuffle(col1, owner);
  return (row & 1) ? v1 : v0;
}

// Online-softmax step for one score row or column: returns the rescale factor
// of the previous accumulation and leaves exp2(s - max) in place.  A fully
// masked tile keeps the max at -INFINITY (see #876).
template <int N, typename Reduce>
inline float ua_softmax_step(thread float *scores, thread float &running_max,
                             thread float &running_sum, Reduce reduce) {
  float local_max = -INFINITY;
#pragma unroll
  for (int i = 0; i < N; i++) local_max = max(local_max, scores[i]);
  const float new_max = max(running_max, reduce.max(local_max));
  const float factor = new_max > running_max
      ? (running_max == -INFINITY ? 0.0f : fast::exp2(running_max - new_max))
      : 1.0f;
  running_max = new_max;
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < N; i++) {
    const float p =
        scores[i] == -INFINITY ? 0.0f : fast::exp2(scores[i] - new_max);
    scores[i] = p;
    local_sum += p;
  }
  running_sum = running_sum * factor + reduce.sum(local_sum);
  return factor;
}

struct UaRowReduce {
  float max(float v) const { return ua_row_max(v); }
  float sum(float v) const { return ua_row_sum(v); }
};

struct UaColReduce {
  float max(float v) const { return ua_col_max(v); }
  float sum(float v) const { return ua_col_sum(v); }
};

// First key of the sequence's partition grid: its first query row's window
// start, rounded down to a partition boundary.
inline int ua_partition_base(int context_len, int sliding_window,
                             int partition_size) {
  if (sliding_window < 0) return 0;
  return max(0, context_len + 1 - sliding_window) / partition_size *
         partition_size;
}

// Factor that takes a row normalised by its own exp-sum l (max m, log2
// space) to one whose denominator also holds the sink.
inline float ua_sink_scale(float m, float l, float sink) {
  const float m_new = max(m, sink);
  const float w = m == -INFINITY ? 0.0f : l * fast::exp2(m - m_new);
  return w / (w + fast::exp2(sink - m_new));
}

// Decode and verification segments are the batch's first decode_tokens
// query tokens, or, when the caller did not say (-1), its one-token
// segments.  They take the direct grid at any window length, so a token's
// output is the same decoded alone or inside a verification window.
inline bool ua_decode_segment(int seq_start, int q_len, int decode_tokens) {
  return decode_tokens < 0 ? q_len == 1 : seq_start < decode_tokens;
}

// Segments a launch serves: all (0), decode segments (> 0) or the rest
// (< 0).  Mixed batches launch each grid on its own share.
inline bool ua_serves(int filter, int seq_start, int q_len,
                      int decode_tokens) {
  return filter == 0 ||
      (filter > 0) == ua_decode_segment(seq_start, q_len, decode_tokens);
}

// Packed elements d and d + 1 (d even) of a sub-8-bit little-endian bit
// stream (pack_bits in turboquant.py).  `bits` is a function constant, so each
// width compiles to branch-free loads that are uniform across lanes: one byte
// for 2- and 4-bit, the pair's 3-byte group for 3-bit, two bytes for 5-bit
// (ten bits from an even bit offset).
inline uint2 ua_unpack_pair(const device uchar *row, int d, int bits) {
  if (bits == 4) {
    const uint b = row[d >> 1];
    return uint2(b & 15u, b >> 4);
  }
  if (bits == 2) {
    const uint b = row[d >> 2];
    const uint offset = uint(d & 3) * 2;
    return uint2((b >> offset) & 3u, (b >> (offset + 2)) & 3u);
  }
  if (bits == 3) {
    const device uchar *g = row + (d >> 3) * 3;
    const uint word = uint(g[0]) | (uint(g[1]) << 8) | (uint(g[2]) << 16);
    const uint offset = uint(d & 7) * 3;
    return uint2((word >> offset) & 7u, (word >> (offset + 3)) & 7u);
  }
  const int bit0 = d * bits;
  const device uchar *p = row + (bit0 >> 3);
  const uint word = uint(p[0]) | (uint(p[1]) << 8);
  const uint offset = uint(bit0 & 7);
  const uint mask = (1u << bits) - 1u;
  return uint2((word >> offset) & mask, (word >> (offset + bits)) & mask);
}

// TurboQuant K elements d and d + 1 (d even: one 32-element scale group).
inline float2 ua_tq_k_pair(const device uchar *row, int d, float scale,
                           float zero) {
  float2 idx;
  if (ua_k_bits == 8) {
    idx = ua_k_signed ? float2(*reinterpret_cast<const device char2 *>(row + d))
                      : float2(*reinterpret_cast<const device uchar2 *>(row + d));
  } else {
    idx = float2(ua_unpack_pair(row, d, ua_k_bits));
  }
  return (idx + zero) * scale;
}

// TurboQuant V elements d and d + 1 in the rotated domain.
inline float2 ua_tq_v_pair(const device uchar *row, int d, float scale,
                           const threadgroup float *centroids) {
  const uint2 idx = ua_v_bits == 8
      ? uint2(*reinterpret_cast<const device uchar2 *>(row + d))
      : ua_unpack_pair(row, d, ua_v_bits);
  return float2(centroids[idx.x], centroids[idx.y]) * scale;
}

// Inverse FWHT of each lane's O row in fragment layout: element 8d + fn + j
// is o[d][j].  Stage masks 1, 2, 4 map to j, lane bit 0 and lane bit 3, the
// rest to the fragment index.  The stages and butterflies are those of
// inverse_fwht_in_place, in its order, so the result is bitwise the one the
// split-KV reduce computes on a staged row.
template <int HEAD_SIZE, int TD>
inline void ua_inverse_fwht(thread float2 *o, short fn) {
#pragma unroll
  for (int d = 0; d < TD; d++) o[d] = float2(o[d].x + o[d].y, o[d].x - o[d].y);
#pragma unroll
  for (int stage = 0; stage < 2; stage++) {
    const ushort lane_mask = stage == 0 ? 1 : 8;
    const bool upper = fn & (2 << stage);
#pragma unroll
    for (int d = 0; d < TD; d++) {
      const float2 partner = simd_shuffle_xor(o[d], lane_mask);
      o[d] = upper ? partner - o[d] : o[d] + partner;
    }
  }
#pragma unroll
  for (int s = 1; s < TD; s <<= 1) {
    float2 r[TD];
#pragma unroll
    for (int d = 0; d < TD; d++)
      r[d] = (d & s) ? o[d ^ s] - o[d] : o[d] + o[d ^ s];
#pragma unroll
    for (int d = 0; d < TD; d++) o[d] = r[d];
  }
  constexpr float INV_SQRT_N = fwht_inv_sqrt_n<HEAD_SIZE>();
#pragma unroll
  for (int d = 0; d < TD; d++) {
#pragma unroll
    for (int j = 0; j < 2; j++)
      o[d][j] *= INV_SQRT_N * get_fwht_sign<HEAD_SIZE>(uint(d * 8 + fn + j));
  }
}

// Canonical cross-partition merge term of a partition (max m_p, exp-sum
// l_p, log2 space) under the row max m: its share of the row's exp-sum.
// Precise, so a lone live partition's term is l_p exactly.
inline float ua_merge_term(float m_p, float l_p, float m) {
  return m_p == -INFINITY ? 0.0f : l_p * precise::exp2(m_p - m);
}

// Merge weights per reduce pass, staged in threadgroup memory.
constant constexpr int UA_REDUCE_W = 256;

// Query rows of one threadgroup: its sequence, packed-row block, and the key
// range every row of the block can attend.
struct UaBlock {
  bool active;
  int seq_idx;
  int q_seq_start;
  int row0;
  int valid_rows;
  int context_len;
  int min_q_abs;
  int max_q_abs;
  int kv_begin;  // earliest row's window start
  int kv_end;    // latest row's causal frontier + 1
};

// With seq_blocks > 0 every sequence owns seq_blocks consecutive blocks
// (decode and verification windows).  Otherwise sequence s owns at most
// ceil(q_len * G / BM) blocks, starting at (cu_seqlens_q[s] * G) / BM + s
// (the tiled kernel's bound, scaled by G).
inline UaBlock ua_locate(constant UaParams &p,
                         device const int32_t *cu_seqlens_q,
                         device const uint32_t *context_lens, int block_global,
                         int group, int bm) {
  UaBlock b;
  if (p.seq_blocks > 0) {
    b.seq_idx = block_global / p.seq_blocks;
    b.row0 = (block_global % p.seq_blocks) * bm;
  } else {
    int lo = 0, hi = p.num_seqs;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if ((cu_seqlens_q[mid] * group) / bm + mid <= block_global) lo = mid;
      else hi = mid - 1;
    }
    b.seq_idx = lo;
    b.row0 = (block_global - ((cu_seqlens_q[lo] * group) / bm + lo)) * bm;
  }
  b.q_seq_start = cu_seqlens_q[b.seq_idx];
  const int q_len = cu_seqlens_q[b.seq_idx + 1] - b.q_seq_start;
  const int seq_rows = q_len * group;
  b.active = b.row0 < seq_rows &&
      ua_serves(p.seq_filter, b.q_seq_start, q_len, p.decode_tokens);
  b.valid_rows = min(bm, seq_rows - b.row0);
  b.context_len = int(context_lens[b.seq_idx]) - q_len;
  b.min_q_abs = b.context_len + b.row0 / group;
  b.max_q_abs = b.context_len + (b.row0 + b.valid_rows - 1) / group;
  b.kv_begin = p.sliding_window >= 0
      ? max(0, b.min_q_abs + 1 - p.sliding_window) : 0;
  b.kv_end = b.max_q_abs + 1;
  return b;
}

// Kernel arguments; both grids bind the same slots, and only the direct grid
// takes the split-KV outputs (0, 1, 6).
#define UA_KERNEL_ARGS(T)                                                      \
  device T *out [[buffer(2), function_constant(ua_single_pass)]],              \
      device const T *q [[buffer(3)]],                                         \
      device const uchar *k_cache [[buffer(4)]],                               \
      device const uchar *v_cache [[buffer(5)]],                               \
      constant UaParams &p [[buffer(7)]],                                      \
      device const uint32_t *block_tables [[buffer(8)]],                       \
      device const uint32_t *context_lens [[buffer(9)]],                       \
      device const float *sinks [[buffer(10), function_constant(ua_use_sinks)]], \
      device const int32_t *cu_seqlens_q [[buffer(11)]],                       \
      device const half *key_scale_cache                                       \
      [[buffer(12), function_constant(ua_tq)]],                                \
      device const half *value_scale_cache                                     \
      [[buffer(13), function_constant(ua_tq)]],                                \
      device const half *key_zero_cache                                        \
      [[buffer(14), function_constant(ua_tq)]],                                \
      device const float *v_centroids [[buffer(15), function_constant(ua_tq)]], \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 tgp [[threadgroup_position_in_grid]],                              \
      uint3 tgpg [[threadgroups_per_grid]],                                    \
      uint thread_idx [[thread_index_in_threadgroup]],                         \
      uint sg_idx [[simdgroup_index_in_threadgroup]],                          \
      uint lane [[thread_index_in_simdgroup]]

#define UA_DIRECT_ARGS(T)                                                      \
  device float *exp_sums [[buffer(0), function_constant(ua_partitioned)]],     \
      device float *max_logits                                                 \
      [[buffer(1), function_constant(ua_partitioned)]],                        \
      device float *tmp_out [[buffer(6), function_constant(ua_partitioned)]],  \
      UA_KERNEL_ARGS(T), uint3 tptg [[threads_per_threadgroup]]

// Where a lane's O elements live: element j of pair d sits at dim
// dim(d) + j.  Fragment order is {fn, 8, TD, 0}; chunked runs (ua_run_dim)
// lay pairs out in 16-byte chunks.
struct UaRowLayout {
  short base;
  short step;
  short pairs;   // pairs per chunk
  short stride;  // dims between chunks
  int dim(int d) const {
    return base + (d / pairs) * stride + (d % pairs) * step;
  }
};

// Finish a row: sink share, inverse FWHT (TurboQuant rows are still rotated,
// and in fragment order), and store this lane's two elements of every O
// fragment.
template <typename T, int HEAD_SIZE, int TD, bool WHOLE_ROW = true>
inline void ua_store_row(thread float2 *o, device T *out,
                         device const float *sinks, float m, float l,
                         bool row_valid, int token, int head, int q_stride,
                         short fn, UaRowLayout lay) {
  if (ua_use_sinks && row_valid) {
    // The sink joins the denominator once and contributes no value row.
    const float f = ua_sink_scale(m, l, sinks[head] * M_LOG2E_F);
#pragma unroll
    for (int d = 0; d < TD; d++) o[d] *= f;
  }
  if (ua_tq && WHOLE_ROW) ua_inverse_fwht<HEAD_SIZE, TD>(o, fn);
  if (!row_valid) return;
  device T *dst = out + int64_t(token) * q_stride + head * HEAD_SIZE;
#pragma unroll
  for (int d = 0; d < TD; d++)
    *reinterpret_cast<device vec<T, 2> *>(dst + lay.dim(d)) = vec<T, 2>(o[d]);
}

// Store one partition's stats and normalised partial for the reduce.
template <int TD>
inline void ua_store_partial(thread const float2 *o, device float *tmp_out,
                             device float *exp_sums, device float *max_logits,
                             int64_t stat, float m, float l, int head_size,
                             bool stats, UaRowLayout lay) {
  if (stats) {
    max_logits[stat] = m;
    exp_sums[stat] = l;
  }
  device float *dst = tmp_out + stat * head_size;
#pragma unroll
  for (int d = 0; d < TD; d++)
    *reinterpret_cast<device float2 *>(dst + lay.dim(d)) = o[d];
}

// Chunked runs.  Permuting the head dimension the same way in Q, K, V and O
// changes no dot product, so the direct grid gives fragment d column c the
// dim ua_run_dim(d, c).  The four lanes of a fragment row (c / 2 = 0..3)
// then hold adjacent 16-byte chunks of its K, V and O rows, chunk after
// chunk: one 16-byte load per lane reads 64 contiguous bytes of each of the
// tile's 8 rows, where fragment order (or one contiguous run per lane)
// scatters it over 32 cache lines, and the tile's loads issue together, not
// as one dependent simdgroup_load per fragment.
template <typename T>
constexpr int ua_chunk() {
  return 16 / sizeof(T);  // elements per 16-byte chunk
}

template <typename T>
inline int ua_run_dim(int d, int c) {
  constexpr int V = ua_chunk<T>();
  return (d / (V / 2)) * 4 * V + (c / 2) * V + (d % (V / 2)) * 2 + c % 2;
}

// Layout of chunked-run pairs [d0, d0 + n) of lane column fn.
template <typename T>
inline UaRowLayout ua_run_layout(short fn, int d0) {
  constexpr int V = ua_chunk<T>();
  return {short((fn / 2) * V + d0 / (V / 2) * 4 * V), 2, V / 2, 4 * V};
}

// This lane's pairs [d0, d0 + N) of a chunked-run row at src (layout from
// ua_run_layout, base included).
template <typename T, int N>
inline void ua_load_run(thread vec<T, 2> *dst, const device T *src) {
  constexpr int V = ua_chunk<T>();
  static_assert(N % (V / 2) == 0, "a run slice is whole 16-byte chunks");
  UA_UNROLL
  for (int c = 0; c < N / (V / 2); c++) {
    const vec<T, V> x =
        *reinterpret_cast<const device vec<T, V> *>(src + c * 4 * V);
    UA_UNROLL
    for (int j = 0; j < V / 2; j++)
      dst[c * V / 2 + j] = vec<T, 2>(x[2 * j], x[2 * j + 1]);
  }
}

// This lane's dense K and V runs (ua_load_run, N pairs each) of the TK
// fragments of the tile at tile_start: key row tile_start + 8k + fm.
template <typename T, int BLOCK_SIZE, int TK, int N>
inline void ua_load_tile(thread vec<T, 2> *k_run, thread vec<T, 2> *v_run,
                         const device T *k_dense, const device T *v_dense,
                         const device uint32_t *block_table, int tile_start,
                         int64_t row_off, int64_t block_stride,
                         int token_stride, short fm) {
  UA_UNROLL
  for (int k = 0; k < TK; k++) {
    const int kv0 = tile_start + k * 8;
    const int64_t off = int64_t(block_table[kv0 / BLOCK_SIZE]) * block_stride +
                        (kv0 % BLOCK_SIZE + fm) * token_stride + row_off;
    ua_load_run<T, N>(k_run + k * N, k_dense + off);
    ua_load_run<T, N>(v_run + k * N, v_dense + off);
  }
}

// ─── Direct (decode and verification windows) ────────────────────────────
// Grid (row blocks, KV heads, ceil(groups / SG_N)); a threadgroup is SG_M row
// simdgroups (p.sg_m, 8 rows each) x DS head slices x SG_N partition groups
// (SG_N > 1 only when DS is 1).  Simdgroup (m, h, n) reduces group
// z * SG_N + n: p.group_partitions consecutive partitions from the sequence's
// partition base, accumulating O for slice h, TD / DS of the head's
// fragments.  The DS simdgroups of a row block add their partial scores in
// slice order through threadgroup memory, so all of them hold the same S and
// run the same softmax; the slices cut each lane's O, K and V registers by
// DS and give a row block DS times the simdgroups.  With one group a
// simdgroup covers every partition and, with ua_blend, blends them in
// registers; otherwise it writes each partition for the reduce.
template <typename T, int HEAD_SIZE, int BLOCK_SIZE, int TILE_KV, int DS>
[[kernel]] void paged_attention_unified_direct(UA_DIRECT_ARGS(T)) {
  constexpr int TD = HEAD_SIZE / 8;
  constexpr int TDS = TD / DS;
  constexpr int TK = TILE_KV / 8;
  // Unsliced wide heads keep Q^T in threadgroup memory instead of registers:
  // the O accumulator alone is HEAD_SIZE / 4 floats per lane.
  constexpr bool Q_IN_SMEM = TDS >= 32;
  // Independent partial score accumulators per slice: chains of TDS / NA
  // dependent MMAs instead of TDS.
  constexpr int NA = TDS < 4 ? TDS : 4;
  static_assert(HEAD_SIZE % 8 == 0, "HEAD_SIZE must be a multiple of 8");
  static_assert(TD % DS == 0, "slices must split the fragments evenly");
  static_assert(TILE_KV % 8 == 0, "TILE_KV must be a multiple of 8");
  static_assert(BLOCK_SIZE % 8 == 0, "fragments must not cross a page");

  const int SG_M = p.sg_m;
  const int BM = 8 * SG_M;
  const int SG_N = int(tptg.x) / (32 * SG_M * DS);
  const int sg_m = int(sg_idx) % SG_M;
  const int sg_h = (int(sg_idx) / SG_M) % DS;
  const int sg_n = int(sg_idx) / (SG_M * DS);
  const int kv_head = tgp.y;
  // q_stride is one query token row: num_heads * HEAD_SIZE.
  const int group = p.q_stride / (p.num_kv_heads * HEAD_SIZE);
  const int heads = p.num_kv_heads * group;
  const UaBlock blk =
      ua_locate(p, cu_seqlens_q, context_lens, int(tgp.x), group, BM);
  if (!blk.active) return;

  // Strides are in elements of each cache: T for dense K/V, bytes for
  // TurboQuant K/V, halfs for its scale and zero-point caches.
  const int kv_token_stride = p.num_kv_heads * p.kv_head_stride;
  const device T *k_dense = reinterpret_cast<const device T *>(k_cache);
  const device T *v_dense = reinterpret_cast<const device T *>(v_cache);

  // ─ Threadgroup memory: score exchange (DS > 1), Q_smem BM x LDQ (T, wide
  // unsliced heads), then TurboQuant centroids ─────────────────────────────
  constexpr int LDQ = HEAD_SIZE + 16 / int(sizeof(T));
  const int x_bytes = DS > 1 ? 2 * BM * DS * TK * 32 * 8 : 0;
  threadgroup float2 *S_x = reinterpret_cast<threadgroup float2 *>(shared_mem);
  threadgroup T *Q_smem = reinterpret_cast<threadgroup T *>(shared_mem + x_bytes);
  threadgroup float *C_smem = reinterpret_cast<threadgroup float *>(
      shared_mem + x_bytes +
      (Q_IN_SMEM ? (BM * LDQ * int(sizeof(T)) + 15) & ~15 : 0));
  if (ua_tq) {
    for (int i = thread_idx; i < (1 << ua_v_bits); i += int(tptg.x))
      C_smem[i] = v_centroids[i];
  }
  using vec2T = vec<T, 2>;
  using vec2F = vec<float, 2>;
  const short2 fc = ua_frag_coord(ushort(lane));
  const short fm = fc.y;
  const short fn = fc.x;
  // This slice's first fragment.  Dense caches are read in chunked runs
  // (ua_run_dim); TurboQuant in fragment order.
  const int d0 = sg_h * TDS;
  const bool runs = !ua_tq;
  const UaRowLayout lay = runs ? ua_run_layout<T>(fn, d0)
                               : UaRowLayout{short(fn + 8 * d0), 8, TD, 0};

  // Q^T fragments (dims x rows) of this slice: this lane holds rows fn and
  // fn + 1 at dim (d, fm) of every fragment d.  Each lane loads its own, all
  // at once; wide heads park them in Q_smem, which the first partition group
  // of each row simdgroup fills.
  const device T *q_row[2];
  bool q_valid[2];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int r = sg_m * 8 + fn + j;
    const int pr = blk.row0 + r;
    q_valid[j] = r < blk.valid_rows;
    q_row[j] = q + int64_t(blk.q_seq_start + pr / group) * p.q_stride +
               (kv_head * group + pr % group) * HEAD_SIZE;
  }
  threadgroup T *Q_rows = Q_smem + sg_m * 8 * LDQ;
  vec2T Qreg[Q_IN_SMEM ? 1 : TDS];
  if (!Q_IN_SMEM || sg_n == 0) {
    UA_UNROLL
    for (int d = 0; d < TDS; d++) {
      const int dim = runs ? ua_run_dim<T>(d0 + d, fm) : (d0 + d) * 8 + fm;
      const vec2T qt = vec2T(q_valid[0] ? q_row[0][dim] : T(0),
                             q_valid[1] ? q_row[1][dim] : T(0));
      if (Q_IN_SMEM) {
        Q_rows[fn * LDQ + d * 8 + fm] = qt.x;
        Q_rows[(fn + 1) * LDQ + d * 8 + fm] = qt.y;
      } else {
        Qreg[d] = qt;
      }
    }
  }
  if (Q_IN_SMEM || ua_tq) threadgroup_barrier(mem_flags::mem_threadgroup);
  // No barrier follows outside a slice group, whose simdgroups share a
  // partition group and so every tile: simdgroups past the last group leave
  // now.
  const int g = int(tgp.z) * SG_N + sg_n;
  if (g >= p.groups) return;

  const int my_row = sg_m * 8 + fm;
  const bool row_valid = my_row < blk.valid_rows;
  // Score columns fn and fn + 1 are query rows of S^T.
  bool col_valid[2];
  int col_q_abs[2];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int r = sg_m * 8 + fn + j;
    col_valid[j] = r < blk.valid_rows;
    col_q_abs[j] = blk.context_len + (blk.row0 + r) / group;
  }

  const float scale_log2 = p.scale * M_LOG2E_F;
  const float softcapping = p.softcap;
  const int sliding_window = p.sliding_window;
  const device uint32_t *block_table =
      block_tables + blk.seq_idx * p.max_num_blocks_per_seq;
  const int P = p.partition_size;
  const int base_part =
      ua_partition_base(blk.context_len, sliding_window, P) / P;
  const int last_part = (blk.kv_end - 1) / P;
  const int my_head = kv_head * group + (blk.row0 + my_row) % group;
  const int token = blk.q_seq_start + (blk.row0 + my_row) / group;
  // This slice group's exchange slots, double-buffered by tile parity.
  threadgroup float2 *x_slots = S_x + (sg_m * DS) * TK * 32 + lane;
  int tile_parity = 0;

  for (int pi = 0; pi < p.group_partitions; pi++) {
    const int part = base_part + g * p.group_partitions + pi;
    if (part > last_part) break;
    // Keys outside [lo, hi) are masked for every row of the block.
    const int lo = max(blk.kv_begin, part * P);
    const int hi = min(blk.kv_end, part * P + P);
    float m_col[2] = {-INFINITY, -INFINITY};
    float l_col[2] = {0.0f, 0.0f};
    vec2F Oreg[TDS];
#pragma unroll
    for (int d = 0; d < TDS; d++) Oreg[d] = vec2F(0.0f);

    // Dense runs of the next tile load while this one computes.
    const int64_t row_off = kv_head * p.kv_head_stride + lay.base;
    vec2T k_next[TK][TDS];
    vec2T v_next[TK][TDS];
    if (!ua_tq) {
      ua_load_tile<T, BLOCK_SIZE, TK, TDS>(
          &k_next[0][0], &v_next[0][0], k_dense, v_dense, block_table,
          lo / TILE_KV * TILE_KV, row_off, p.kv_block_stride, kv_token_stride,
          fm);
    }
    for (int tile_start = lo / TILE_KV * TILE_KV; tile_start < hi;
         tile_start += TILE_KV) {
      // A tile inside every row's causal frontier and window needs no
      // per-element mask.
      const bool tile_no_mask =
          (tile_start + TILE_KV - 1) <= blk.min_q_abs && tile_start >= lo &&
          !ua_softcap &&
          (sliding_window < 0 ||
           tile_start >= blk.max_q_abs + 1 - sliding_window);

      // Lanes read their own row, key kv0 + fm: this slice of a dense run,
      // or a TurboQuant row to dequantize.
      vec2T k_run[TK][TDS];
      vec2T v_run[TK][TDS];
      if (!ua_tq) {
#pragma unroll
        for (int k = 0; k < TK; k++) {
#pragma unroll
          for (int d = 0; d < TDS; d++) {
            k_run[k][d] = k_next[k][d];
            v_run[k][d] = v_next[k][d];
          }
        }
        if (tile_start + TILE_KV < hi) {
          ua_load_tile<T, BLOCK_SIZE, TK, TDS>(
              &k_next[0][0], &v_next[0][0], k_dense, v_dense, block_table,
              tile_start + TILE_KV, row_off, p.kv_block_stride,
              kv_token_stride, fm);
        }
      }
      const device uchar *tq_k_row[TK];
      const device uchar *tq_v_row[TK];
      int64_t tq_scale[TK];
#pragma unroll
      for (int k = 0; k < (ua_tq ? TK : 0); k++) {
        const int kv0 = tile_start + k * 8;
        const int64_t pb = int64_t(block_table[kv0 / BLOCK_SIZE]);
        {
          const int64_t tok = kv0 % BLOCK_SIZE + fm;
          tq_k_row[k] = k_cache + pb * p.kv_block_stride +
                        tok * kv_token_stride + kv_head * p.kv_head_stride;
          tq_v_row[k] = v_cache + pb * p.v_block_stride +
                        tok * (p.num_kv_heads * p.v_head_stride) +
                        kv_head * p.v_head_stride;
          tq_scale[k] = pb * p.scale_block_stride +
                        tok * (p.num_kv_heads * p.scale_head_stride) +
                        kv_head * p.scale_head_stride;
        }
      }

      // S^T[TILE_KV, 8] = K x Q^T over this slice, as NA interleaved
      // partial sums added in order.
      vec2F STacc[NA][TK];
#pragma unroll
      for (int a = 0; a < NA; a++) {
#pragma unroll
        for (int k = 0; k < TK; k++) STacc[a][k] = vec2F(0.0f);
      }
#pragma unroll
      for (int d = 0; d < TDS; d++) {
        simdgroup_matrix<T, 8, 8> qt_frag;
        if (Q_IN_SMEM) {
          simdgroup_load(qt_frag, Q_rows + d * 8, LDQ, ulong2(0),
                         /*transpose=*/true);
        } else {
          reinterpret_cast<thread vec2T &>(qt_frag.thread_elements()) =
              Qreg[d];
        }
#pragma unroll
        for (int k = 0; k < TK; k++) {
          simdgroup_matrix<T, 8, 8> k_frag;
          if (ua_tq) {
            const int64_t gi = tq_scale[k] + (d0 + d) / 4;
            reinterpret_cast<thread vec2T &>(k_frag.thread_elements()) =
                vec2T(ua_tq_k_pair(tq_k_row[k], (d0 + d) * 8 + fn,
                                   float(key_scale_cache[gi]),
                                   float(key_zero_cache[gi])));
          } else {
            reinterpret_cast<thread vec2T &>(k_frag.thread_elements()) =
                k_run[k][d];
          }
          simdgroup_matrix<float, 8, 8> st_frag;
          reinterpret_cast<thread vec2F &>(st_frag.thread_elements()) =
              STacc[d % NA][k];
          simdgroup_multiply_accumulate(st_frag, k_frag, qt_frag, st_frag);
          STacc[d % NA][k] =
              reinterpret_cast<thread vec2F &>(st_frag.thread_elements());
        }
      }
      vec2F STreg[TK];
#pragma unroll
      for (int k = 0; k < TK; k++) {
        STreg[k] = STacc[0][k];
#pragma unroll
        for (int a = 1; a < NA; a++) STreg[k] += STacc[a][k];
      }
      if (DS > 1) {
        // Whole-head scores: the slices' partial sums, added in slice order
        // by every simdgroup of the slice group alike.  The slot buffer
        // alternates by tile, so a write never races the previous tile's
        // reads (each tile's barrier orders them).
        threadgroup float2 *slots = x_slots + tile_parity * BM * DS * TK * 32;
#pragma unroll
        for (int k = 0; k < TK; k++) slots[(sg_h * TK + k) * 32] = STreg[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (int k = 0; k < TK; k++) {
          STreg[k] = slots[k * 32];
#pragma unroll
          for (int h = 1; h < DS; h++) STreg[k] += slots[(h * TK + k) * 32];
        }
        tile_parity ^= 1;
      }

      // Element (k, j): key tile_start + 8k + fm against query row fn + j.
      float cols[2][TK];
#pragma unroll
      for (int k = 0; k < TK; k++) {
        const int kv_pos = tile_start + k * 8 + fm;
#pragma unroll
        for (int j = 0; j < 2; j++) {
          float s = STreg[k][j] * scale_log2;
          if (!(tile_no_mask && col_valid[0] && col_valid[1])) {
            if (ua_softcap) {
              s = softcapping * precise::tanh(s / (M_LOG2E_F * softcapping)) *
                  M_LOG2E_F;
            }
            bool masked = !col_valid[j] || kv_pos > col_q_abs[j] ||
                          kv_pos < lo || kv_pos >= hi;
            if (sliding_window >= 0)
              masked = masked || kv_pos < col_q_abs[j] + 1 - sliding_window;
            s = masked ? -INFINITY : s;
          }
          cols[j][k] = s;
        }
      }
      float factor[2];
#pragma unroll
      for (int j = 0; j < 2; j++) {
        factor[j] = ua_softmax_step<TK>(cols[j], m_col[j], l_col[j],
                                        UaColReduce());
      }
      const float row_factor = ua_col_to_row(factor[0], factor[1], fm);
#pragma unroll
      for (int d = 0; d < TDS; d++) Oreg[d] *= row_factor;

      // O += P x V over this slice, with P transposed back from P^T by
      // shuffle.
#pragma unroll
      for (int k = 0; k < TK; k++) {
        simdgroup_matrix<float, 8, 8> p_frag;
        reinterpret_cast<thread vec2F &>(p_frag.thread_elements()) =
            ua_transpose_pair(cols[0][k], cols[1][k], fm, fn);
        // V rows outside [lo, hi) are zeroed: their P is 0, and a stale
        // non-finite cache value must not turn 0 * V into NaN.  This lane
        // holds V row (key) tile_start + 8k + fm.
        const int v_tok = tile_start + k * 8 + fm;
        const bool v_in_range = v_tok >= lo && v_tok < hi;
#pragma unroll
        for (int d = 0; d < TDS; d++) {
          simdgroup_matrix<T, 8, 8> v_frag;
          if (ua_tq) {
            float2 vv = float2(0.0f);
            if (v_in_range) {
              vv = ua_tq_v_pair(
                  tq_v_row[k], (d0 + d) * 8 + fn,
                  float(value_scale_cache[tq_scale[k] + (d0 + d) / 4]),
                  C_smem);
            }
            reinterpret_cast<thread vec2T &>(v_frag.thread_elements()) =
                vec2T(vv);
          } else {
            reinterpret_cast<thread vec2T &>(v_frag.thread_elements()) =
                v_in_range ? v_run[k][d] : vec2T(T(0));
          }
          simdgroup_matrix<float, 8, 8> o_frag;
          reinterpret_cast<thread vec2F &>(o_frag.thread_elements()) = Oreg[d];
          simdgroup_multiply_accumulate(o_frag, p_frag, v_frag, o_frag);
          Oreg[d] = reinterpret_cast<thread vec2F &>(o_frag.thread_elements());
        }
      }
    }

    // Partition done: column state -> this lane's O row (fm), normalised.
    // A row with no attended key here has sum 0 and stays zero.
    const float m_row = ua_col_to_row(m_col[0], m_col[1], fm);
    const float l_row = ua_col_to_row(l_col[0], l_col[1], fm);
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
#pragma unroll
    for (int d = 0; d < TDS; d++) Oreg[d] *= inv;
    if (ua_partitioned) {
      if (row_valid) {
        const int64_t stat =
            (int64_t(token) * heads + my_head) * p.max_partitions +
            (part - base_part);
        ua_store_partial<TDS>(Oreg, tmp_out, exp_sums, max_logits, stat,
                              m_row, l_row, HEAD_SIZE, fn == 0 && sg_h == 0,
                              lay);
      }
    } else {
      // A single pass covers one partition per row: it is the row.
      ua_store_row<T, HEAD_SIZE, TDS, DS == 1>(Oreg, out, sinks, m_row, l_row,
                                               row_valid, token, my_head,
                                               p.q_stride, fn, lay);
    }
  }
}

// ─── Staged (prefill) ────────────────────────────────────────────────────
// Read one 8x8 fragment of a staged tile at src (leading dimension ld): with
// simdgroup_load in the tile's type, or with LANE each lane reads its pair at
// src + lane_off into fp32.
template <bool LANE, typename F, typename T>
inline void ua_read_frag(thread simdgroup_matrix<F, 8, 8> &frag,
                         const threadgroup T *src, short ld, short lane_off) {
  if constexpr (LANE) {
    reinterpret_cast<thread vec<F, 2> &>(frag.thread_elements()) =
        vec<F, 2>(*reinterpret_cast<const threadgroup vec<T, 2> *>(src + lane_off));
  } else {
    simdgroup_load(frag, src, ld);
  }
}

// Stage this thread's run [st_d0, st_d0 + STAGE_ELEMS) of the V row of
// (page, tok) into tile row st_key; TurboQuant V dequantizes on the way.
// kv_off is the run's dense element offset, computed once with K's (head
// 256 lost half its speed recomputing it here).
template <typename T, int LDV, int STAGE_ELEMS>
inline void ua_stage_v(threadgroup T *V_smem, const device uchar *v_cache,
                       const device half *value_scale_cache,
                       const threadgroup float *C_smem, constant UaParams &p,
                       int64_t page, int64_t tok, int64_t kv_off, int kv_head,
                       short st_key, short st_d0) {
  threadgroup T *dst = V_smem + st_key * LDV + st_d0;
  if (ua_tq) {
    const device uchar *v_row = v_cache + page * p.v_block_stride +
                                tok * (p.num_kv_heads * p.v_head_stride) +
                                kv_head * p.v_head_stride;
    const int64_t s_row = page * p.scale_block_stride +
                          tok * (p.num_kv_heads * p.scale_head_stride) +
                          kv_head * p.scale_head_stride;
    UA_UNROLL
    for (short e = 0; e < STAGE_ELEMS; e += 2) {
      const float2 vv = ua_tq_v_pair(
          v_row, st_d0 + e,
          float(value_scale_cache[s_row + (st_d0 + e) / 32]), C_smem);
      *reinterpret_cast<threadgroup vec<T, 2> *>(dst + e) = vec<T, 2>(vv);
    }
  } else {
    const device T *src = reinterpret_cast<const device T *>(v_cache) + kv_off;
    UA_UNROLL
    for (short e = 0; e < STAGE_ELEMS; e += 4) {
      *reinterpret_cast<threadgroup vec<T, 4> *>(dst + e) =
          *reinterpret_cast<const device vec<T, 4> *>(src + e);
    }
  }
}

// Grid (row blocks, KV heads); a threadgroup is SG_M simdgroups of 8 rows
// sharing each staged tile, structured like MLX steel attention
// (steel/attn/kernels/steel_attention.h).  Its rows' tree is one online
// softmax over the block's tiles in key order: a staged launch never splits,
// so no choice of the host reaches its bits.  Masked scores are -FLT_MAX as
// in MLX; a row whose window has not started yet (all of its scores so far
// masked) keeps p = 0, so a fully masked tile leaves a row's state
// bit-for-bit unchanged.
//
// Below head 256 K^T and V are staged together into their own buffers: two
// barriers per tile.  Q is staged once from head 128 (16-bit; fp32 does not
// fit 32 KB): in registers next to the O accumulator it cost head-128 fp16
// prefill ~8% on M1.  Head 256 also cannot fit K^T and V side by side, so
// they take turns in one buffer as in MLX: V is staged once the scores are
// in, its loads in flight during the softmax (four barriers).  Fragments keep the
// cache's type so a GPU with bf16 matrix hardware can use it; M1 has none,
// and its head-128 bf16 prefill runs 14% slower than with fp32 fragments.
// fp16 and fp32 at head 256 read fp32 fragments lane-wise as MLX does:
// simdgroup_load measured 15-47% slower there on M1.  Head-256 speed hangs on
// register allocation (edits that looked neutral have moved it by 40%):
// re-run tools/benchmark/unified_attention_benchmark.py --prefill-vs-mlx
// (each --dtype) after any edit, on a cool GPU.
template <typename T, int HEAD_SIZE, int BLOCK_SIZE, int SG_M, int TILE_KV>
[[kernel, max_total_threads_per_threadgroup(SG_M * 32)]] void
paged_attention_unified_staged(UA_KERNEL_ARGS(T)) {
  constexpr int BQ = 8 * SG_M;
  constexpr int BK = TILE_KV;
  constexpr int NUM_THREADS = SG_M * 32;
  constexpr int TD = HEAD_SIZE / 8;
  constexpr int TK = BK / 8;
  constexpr bool KV_SHARED = HEAD_SIZE >= 256;
  constexpr bool Q_IN_SMEM = KV_SHARED || (HEAD_SIZE >= 128 && sizeof(T) == 2);
  constexpr bool LANE_READS = KV_SHARED && !is_same_v<T, bfloat16_t>;
  using F = conditional_t<LANE_READS, float, T>;
  using vec2T = vec<T, 2>;
  using vec4T = vec<T, 4>;
  using vec2F = vec<float, 2>;
  const int group = p.q_stride / (p.num_kv_heads * HEAD_SIZE);
  const int kv_head = tgp.y;
  // Row blocks launch last first: causal rows attend more keys the later they
  // sit, so the longest blocks start first and the grid ends on short ones.
  const UaBlock blk = ua_locate(p, cu_seqlens_q, context_lens,
                                int(tgpg.x) - 1 - int(tgp.x), group, BQ);
  if (!blk.active) return;

  // ─ Threadgroup memory: staged Q, K^T, and V (its own buffer or K^T's) ──
  // Static arrays, as in MLX: tile addresses are immediates, and the
  // unrolled loop's register allocation (head 256) depends on it.
  constexpr int PAD = 16 / sizeof(T);
  constexpr int LDQ = HEAD_SIZE + PAD;
  constexpr int LDK = BK + PAD;
  constexpr int LDV = HEAD_SIZE + PAD;
  constexpr int K_ELEMS = HEAD_SIZE * LDK;
  constexpr int V_ELEMS = BK * LDV;
  threadgroup T Q_smem[Q_IN_SMEM ? BQ * LDQ : 1];
  threadgroup T K_smem[KV_SHARED && V_ELEMS > K_ELEMS ? V_ELEMS : K_ELEMS];
  threadgroup T V_own[KV_SHARED ? 1 : V_ELEMS];
  threadgroup T *V_smem = KV_SHARED ? K_smem : V_own;
  // TurboQuant V centroids: dynamic, so dense grids do not reserve them.
  threadgroup float *C_smem = reinterpret_cast<threadgroup float *>(shared_mem);
  if (ua_tq) {
    for (int i = thread_idx; i < (1 << ua_v_bits); i += NUM_THREADS)
      C_smem[i] = v_centroids[i];
  }
  // This lane's fragment element (row fm, columns fn and fn + 1).
  const short2 fc = ua_frag_coord(ushort(lane));
  const short fm = fc.y;
  const short fn = fc.x;
  const int my_row = 8 * int(sg_idx) + fm;
  const bool row_valid = my_row < blk.valid_rows;
  const int token = blk.q_seq_start + (blk.row0 + my_row) / group;
  const int my_head = kv_head * group + (blk.row0 + my_row) % group;
  // Q: packed rows; idle rows are zero.
  vec2T q_reg[TD];
  if (Q_IN_SMEM) {
    for (int i = thread_idx * 4; i < BQ * HEAD_SIZE; i += NUM_THREADS * 4) {
      const int r = i / HEAD_SIZE;
      const int d = i % HEAD_SIZE;
      const int pr = blk.row0 + r;
      const int tk = blk.q_seq_start + pr / group;
      const int head = kv_head * group + pr % group;
      *reinterpret_cast<threadgroup vec4T *>(Q_smem + r * LDQ + d) =
          r < blk.valid_rows
              ? *reinterpret_cast<const device vec4T *>(
                    q + int64_t(tk) * p.q_stride + head * HEAD_SIZE + d)
              : vec4T(T(0));
    }
  } else {
    const device T *q_row =
        q + int64_t(token) * p.q_stride + my_head * HEAD_SIZE + fn;
    UA_UNROLL
    for (int d = 0; d < TD; d++) {
      q_reg[d] = row_valid
          ? *reinterpret_cast<const device vec2T *>(q_row + d * 8)
          : vec2T(T(0));
    }
  }
  // Lane reads fold this simdgroup's rows into q_lane (a per-simdgroup base
  // pointer cost head 256 40%).
  const int q_frag0 = LANE_READS ? 0 : 8 * int(sg_idx) * LDQ;
  const short q_lane = my_row * LDQ + fn;
  const short k_lane = fm * LDK + fn;
  const short v_lane = fm * LDV + fn;

  // Staged loads (MLX BlockLoaderT layout): key st_key of the tile,
  // elements [st_d0, st_d0 + STAGE_ELEMS) of its head slice.
  constexpr int STAGE_ELEMS = BK * HEAD_SIZE / NUM_THREADS;
  constexpr int STAGE_COLS = HEAD_SIZE / STAGE_ELEMS;
  static_assert(STAGE_ELEMS % 4 == 0 && STAGE_ELEMS * NUM_THREADS == BK * HEAD_SIZE,
                "staged loaders must tile the K/V block exactly");
  const short st_key = thread_idx / STAGE_COLS;
  const short st_d0 = STAGE_ELEMS * (thread_idx % STAGE_COLS);
  vec2F Oreg[TD];
  UA_UNROLL
  for (int d = 0; d < TD; d++) Oreg[d] = vec2F(0.0f);

  float max_score = -FLT_MAX;
  float sum_score = 0.0f;
  // Keys outside [lo, hi) are masked for every row of the block; this row
  // attends [row_lo, row_hi) (empty for an idle row), and a tile inside
  // [all_lo, all_hi) is attended by every row.
  const int q_abs = blk.context_len + (blk.row0 + my_row) / group;
  const int sliding_window = p.sliding_window;
  const int lo = blk.kv_begin;
  const int hi = blk.kv_end;
  const int row_lo = !row_valid ? 0
      : sliding_window >= 0 ? max(lo, q_abs + 1 - sliding_window) : lo;
  const int row_hi = row_valid ? min(hi, q_abs + 1) : 0;
  const int all_lo =
      sliding_window >= 0 ? max(lo, blk.max_q_abs + 1 - sliding_window) : lo;
  const int all_hi = blk.min_q_abs + 1;
  const bool all_rows = blk.valid_rows == BQ;
  const float scale_log2 = p.scale * M_LOG2E_F;
  const device uint32_t *block_table =
      block_tables + blk.seq_idx * p.max_num_blocks_per_seq;
  const int kv_token_stride = p.num_kv_heads * p.kv_head_stride;
  const device T *k_dense = reinterpret_cast<const device T *>(k_cache);

  const int kb_end = (hi + BK - 1) / BK;
  for (int kb = lo / BK; kb < kb_end; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);  // previous tile consumed
    // Keys outside [lo, hi) read a clamped in-range row instead: their
    // scores are masked and their p is 0, and no value select enters the
    // staging loop (selects there cost head 256 a fifth of its throughput).
    const int key = clamp(kb * BK + st_key, lo, hi - 1);
    const int64_t page = int64_t(block_table[key / BLOCK_SIZE]);
    const int64_t tok = key % BLOCK_SIZE;
    const int64_t kv_off = page * p.kv_block_stride + tok * kv_token_stride +
                           kv_head * p.kv_head_stride + st_d0;
    if (ua_tq) {
      const device uchar *k_row = k_cache + page * p.kv_block_stride +
                                  tok * kv_token_stride +
                                  kv_head * p.kv_head_stride;
      const int64_t s_row = page * p.scale_block_stride +
                            tok * (p.num_kv_heads * p.scale_head_stride) +
                            kv_head * p.scale_head_stride;
      UA_UNROLL
      for (short e = 0; e < STAGE_ELEMS; e += 2) {
        const int64_t gi = s_row + (st_d0 + e) / 32;
        const float2 kv = ua_tq_k_pair(k_row, st_d0 + e,
                                       float(key_scale_cache[gi]),
                                       float(key_zero_cache[gi]));
        K_smem[(st_d0 + e) * LDK + st_key] = T(kv.x);
        K_smem[(st_d0 + e + 1) * LDK + st_key] = T(kv.y);
      }
    } else {
      const device T *src = k_dense + kv_off;
      UA_UNROLL
      for (short e = 0; e < STAGE_ELEMS; e += 4) {
        const vec4T k4 = *reinterpret_cast<const device vec4T *>(src + e);
        K_smem[(st_d0 + e) * LDK + st_key] = k4.x;
        K_smem[(st_d0 + e + 1) * LDK + st_key] = k4.y;
        K_smem[(st_d0 + e + 2) * LDK + st_key] = k4.z;
        K_smem[(st_d0 + e + 3) * LDK + st_key] = k4.w;
      }
    }
    if (!KV_SHARED) {
      ua_stage_v<T, LDV, STAGE_ELEMS>(V_smem, v_cache, value_scale_cache,
                                      C_smem, p, page, tok, kv_off, kv_head, st_key,
                                      st_d0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // staged tiles visible

    // S[8, BK] = Q x K^T.
    simdgroup_matrix<float, 8, 8> s_frag[TK];
    UA_UNROLL
    for (int k = 0; k < TK; k++) s_frag[k] = simdgroup_matrix<float, 8, 8>(0);
    UA_UNROLL
    for (short d = 0; d < TD; d++) {
      simdgroup_barrier(mem_flags::mem_none);
      simdgroup_matrix<F, 8, 8> q_frag;
      if (Q_IN_SMEM) {
        ua_read_frag<LANE_READS>(q_frag, Q_smem + q_frag0 + d * 8, LDQ, q_lane);
      } else {
        reinterpret_cast<thread vec<F, 2> &>(q_frag.thread_elements()) =
            vec<F, 2>(q_reg[d]);
      }
      simdgroup_matrix<F, 8, 8> k_frag[TK];
      UA_UNROLL
      for (int k = 0; k < TK; k++) {
        ua_read_frag<LANE_READS>(k_frag[k], K_smem + d * 8 * LDK + k * 8, LDK,
                                 k_lane);
      }
      simdgroup_barrier(mem_flags::mem_none);
      UA_UNROLL
      for (int k = 0; k < TK; k++)
        simdgroup_multiply_accumulate(s_frag[k], q_frag, k_frag[k], s_frag[k]);
    }
    // Element (k, jj): key kb * BK + 8k + fn + jj against row fm.
    float row[2 * TK];
    UA_UNROLL
    for (int k = 0; k < TK; k++) {
      UA_UNROLL
      for (int jj = 0; jj < 2; jj++)
        row[2 * k + jj] = s_frag[k].thread_elements()[jj] * scale_log2;
    }
    if (ua_softcap) {
      const float softcapping = p.softcap;
      UA_UNROLL
      for (int i = 0; i < 2 * TK; i++) {
        row[i] = softcapping *
            precise::tanh(row[i] / (M_LOG2E_F * softcapping)) * M_LOG2E_F;
      }
    }
    if (!(kb * BK >= all_lo && kb * BK + BK <= all_hi && all_rows)) {
      UA_UNROLL
      for (int k = 0; k < TK; k++) {
        UA_UNROLL
        for (int jj = 0; jj < 2; jj++) {
          const int key_jj = kb * BK + fn + k * 8 + jj;
          if (key_jj < row_lo || key_jj >= row_hi) row[2 * k + jj] = -FLT_MAX;
        }
      }
    }
    if (KV_SHARED) {
      threadgroup_barrier(mem_flags::mem_threadgroup);  // K^T consumed
      // Stage V; its loads are in flight while the softmax below runs.
      ua_stage_v<T, LDV, STAGE_ELEMS>(V_smem, v_cache, value_scale_cache,
                                      C_smem, p, page, tok, kv_off, kv_head, st_key,
                                      st_d0);
    }
    // Online softmax, log2 domain.  A row whose window has not started yet
    // (every score so far masked) keeps p = 0 rather than exp2(0).
    float local_max = -FLT_MAX;
    UA_UNROLL
    for (int i = 0; i < 2 * TK; i++) local_max = max(local_max, row[i]);
    local_max = ua_row_max(local_max);
    const float new_max = max(max_score, local_max);
    const bool dead = new_max == -FLT_MAX;
    float local_sum = 0.0f;
    UA_UNROLL
    for (int i = 0; i < 2 * TK; i++) {
      row[i] = dead ? 0.0f : fast::exp2(row[i] - new_max);
      local_sum += row[i];
    }
    local_sum = ua_row_sum(local_sum);
    const float factor = fast::exp2(max_score - new_max);
    max_score = new_max;
    sum_score = sum_score * factor + local_sum;
    UA_UNROLL
    for (int d = 0; d < TD; d++) Oreg[d] *= factor;
    if (KV_SHARED) threadgroup_barrier(mem_flags::mem_threadgroup);  // V visible

    // O += P x V, one O fragment at a time over the tile's keys.
    simdgroup_matrix<float, 8, 8> p_frag[TK];
    UA_UNROLL
    for (int k = 0; k < TK; k++) {
      reinterpret_cast<thread vec2F &>(p_frag[k].thread_elements()) =
          vec2F(row[2 * k], row[2 * k + 1]);
    }
    UA_UNROLL
    for (short d = 0; d < TD; d++) {
      simdgroup_matrix<float, 8, 8> o_frag;
      reinterpret_cast<thread vec2F &>(o_frag.thread_elements()) = Oreg[d];
      UA_UNROLL
      for (short k = 0; k < TK; k++) {
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_matrix<F, 8, 8> v_frag;
        ua_read_frag<LANE_READS>(v_frag, V_smem + k * 8 * LDV + d * 8, LDV,
                                 v_lane);
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_multiply_accumulate(o_frag, p_frag[k], v_frag, o_frag);
      }
      Oreg[d] = reinterpret_cast<thread vec2F &>(o_frag.thread_elements());
    }
  }

  // A row with no attended key (an idle row, a zero-width window) stays zero.
  const float inv = sum_score > 0.0f ? 1.0f / sum_score : 0.0f;
  UA_UNROLL
  for (int d = 0; d < TD; d++) Oreg[d] *= inv;
  ua_store_row<T, HEAD_SIZE, TD>(Oreg, out, sinks, max_score, sum_score,
                                 row_valid, token, my_head, p.q_stride, fn,
                                 UaRowLayout{fn, 8, TD, 0});
}

// ─── Split-KV reduce ─────────────────────────────────────────────────────
// One threadgroup of 128 threads per (query head, query token) merges the
// token's partitions max-first (ua_merge_weight): the exact max M, then
// L = sum of l_p * 2^(m_p - M) over a fixed tree, then O = sum of
// w_p * partial_p in partition order, w_p = l_p * 2^(m_p - M) / L.  The
// weights are independent, so the only chain is the FMA over partitions
// (a sequential online blend chained an exp2 and a divide per partition).
// Precise exp2 and divide give a lone live partition w = 1 exactly: a row
// inside one partition is bitwise its partial, as the single pass writes it.
// Each thread keeps dims thread_idx + 128 * e.  shared_mem holds
// max(HEAD_SIZE, UA_REDUCE_W + 4) floats.
template <typename T, int HEAD_SIZE>
[[kernel, max_total_threads_per_threadgroup(128)]] void
paged_attention_unified_reduce(
    device T *out [[buffer(0)]],
    const device float *exp_sums [[buffer(1)]],
    const device float *max_logits [[buffer(2)]],
    const device float *tmp_out [[buffer(3)]],
    device const uint32_t *context_lens [[buffer(4)]],
    constant UaReduceParams &rp [[buffer(5)]],
    device const float *sinks [[buffer(6), function_constant(ua_use_sinks)]],
    device const int32_t *cu_seqlens_q [[buffer(7)]],
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NUM_THREADS = 128;
  constexpr int PER_THREAD = (HEAD_SIZE + NUM_THREADS - 1) / NUM_THREADS;
  const int head = tgp.x;
  const int heads = tgpg.x;
  const int token = tgp.y;

  int lo = 0, hi = rp.num_seqs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (cu_seqlens_q[mid] <= token) lo = mid;
    else hi = mid - 1;
  }
  const int q_len = cu_seqlens_q[lo + 1] - cu_seqlens_q[lo];
  if (!ua_serves(rp.seq_filter, cu_seqlens_q[lo], q_len, rp.decode_tokens))
    return;
  const int context_len = int(context_lens[lo]) - q_len;
  const int q_abs = context_len + token - cu_seqlens_q[lo];
  const int P = rp.partition_size;
  const int base = ua_partition_base(context_len, rp.sliding_window, P);
  const int parts = (q_abs + 1 - base + P - 1) / P;

  const int64_t stat0 = (int64_t(token) * heads + head) * rp.max_partitions;
  const device float *partials = tmp_out + stat0 * HEAD_SIZE;
  threadgroup float *sg_part = shared_mem;     // per-simdgroup max, sum
  threadgroup float *w = shared_mem + 4;       // a chunk of merge weights

  float m = -INFINITY;
  for (int pp = thread_idx; pp < parts; pp += NUM_THREADS)
    m = max(m, max_logits[stat0 + pp]);
  m = simd_max(m);
  if (lane == 0) sg_part[sg_idx] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(sg_part[0], sg_part[1]), max(sg_part[2], sg_part[3]));
  float l = 0.0f;
  if (m != -INFINITY) {
    for (int pp = thread_idx; pp < parts; pp += NUM_THREADS)
      l += ua_merge_term(max_logits[stat0 + pp], exp_sums[stat0 + pp], m);
  }
  l = simd_sum(l);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sg_part[sg_idx] = l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  l = ((sg_part[0] + sg_part[1]) + sg_part[2]) + sg_part[3];

  float a[PER_THREAD];
#pragma unroll
  for (int e = 0; e < PER_THREAD; e++) a[e] = 0.0f;
  for (int c0 = 0; l > 0.0f && c0 < parts; c0 += UA_REDUCE_W) {
    const int n = min(parts - c0, UA_REDUCE_W);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = thread_idx; c < n; c += NUM_THREADS) {
      w[c] = precise::divide(
          ua_merge_term(max_logits[stat0 + c0 + c], exp_sums[stat0 + c0 + c], m),
          l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Loads go out CHUNK partitions at a time; the FMAs run in order.
    constexpr int CHUNK = 16;
    for (int c1 = 0; c1 < n; c1 += CHUNK) {
      float x[CHUNK][PER_THREAD];
#pragma unroll
      for (int c = 0; c < CHUNK; c++) {
#pragma unroll
        for (int e = 0; e < PER_THREAD; e++) {
          const int d = int(thread_idx) + e * NUM_THREADS;
          x[c][e] = c1 + c < n && d < HEAD_SIZE
              ? partials[int64_t(c0 + c1 + c) * HEAD_SIZE + d] : 0.0f;
        }
      }
#pragma unroll
      for (int c = 0; c < CHUNK; c++) {
        if (c1 + c >= n) break;
#pragma unroll
        for (int e = 0; e < PER_THREAD; e++)
          a[e] = fma(w[c1 + c], x[c][e], a[e]);
      }
    }
  }
  if (ua_use_sinks) {
    const float f = ua_sink_scale(m, l, sinks[head] * M_LOG2E_F);
#pragma unroll
    for (int e = 0; e < PER_THREAD; e++) a[e] *= f;
  }

  device T *dst = out + (int64_t(token) * heads + head) * HEAD_SIZE;
  if (!ua_tq) {
#pragma unroll
    for (int e = 0; e < PER_THREAD; e++) {
      const int d = int(thread_idx) + e * NUM_THREADS;
      if (d < HEAD_SIZE) dst[d] = T(a[e]);
    }
    return;
  }
  // TurboQuant rows are rotated: simdgroup 0 applies the row's inverse FWHT.
  // The row reuses the merge weights' memory.
  threadgroup float *merged = shared_mem;
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (int e = 0; e < PER_THREAD; e++) {
    const int d = int(thread_idx) + e * NUM_THREADS;
    if (d < HEAD_SIZE) merged[d] = a[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_idx == 0) {
    constexpr int ELEMS = HEAD_SIZE / 32;
    float vals[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; e++) vals[e] = merged[int(lane) + e * 32];
    inverse_fwht_in_place<HEAD_SIZE, ELEMS>(vals, lane);
#pragma unroll
    for (int e = 0; e < ELEMS; e++) dst[int(lane) + e * 32] = T(vals[e]);
  }
}

// ─── Template instantiation ──────────────────────────────────────────────

#define instantiate_unified_direct(type, head_size, block_size, tile_kv, ds)   \
  template [[host_name("paged_attention_unified_direct_" #type "_hs"          \
                       #head_size "_bs" #block_size "_tk" #tile_kv            \
                       "_ds" #ds)]] [[kernel]] void                            \
  paged_attention_unified_direct<type, head_size, block_size, tile_kv, ds>(    \
      UA_DIRECT_ARGS(type));

#define instantiate_unified_staged(type, head_size, block_size, sg_m, tile_kv) \
  template [[host_name("paged_attention_unified_staged_" #type "_hs"          \
                       #head_size "_bs" #block_size "_m" #sg_m                 \
                       "_tk" #tile_kv)]] [[kernel]] void                       \
  paged_attention_unified_staged<type, head_size, block_size, sg_m, tile_kv>(  \
      UA_KERNEL_ARGS(type));

#define instantiate_unified_reduce(type, head_size)                            \
  template [[host_name("paged_attention_unified_reduce_" #type                 \
                       "_hs" #head_size)]] [[kernel]] void                     \
  paged_attention_unified_reduce<type, head_size>(                             \
      device type * out [[buffer(0)]],                                         \
      const device float *exp_sums [[buffer(1)]],                              \
      const device float *max_logits [[buffer(2)]],                            \
      const device float *tmp_out [[buffer(3)]],                               \
      device const uint32_t *context_lens [[buffer(4)]],                       \
      constant UaReduceParams &rp [[buffer(5)]],                               \
      device const float *sinks [[buffer(6), function_constant(ua_use_sinks)]],  \
      device const int32_t *cu_seqlens_q [[buffer(7)]],                        \
      threadgroup float *shared_mem [[threadgroup(0)]],                        \
      uint3 tgp [[threadgroup_position_in_grid]],                              \
      uint3 tgpg [[threadgroups_per_grid]],                                    \
      uint thread_idx [[thread_index_in_threadgroup]],                         \
      uint sg_idx [[simdgroup_index_in_threadgroup]],                          \
      uint lane [[thread_index_in_simdgroup]]);

// Each grid must match kUnifiedInstances in unified_attention.cpp.
#define instantiate_unified_grids(type, block_size)                            \
  instantiate_unified_direct(type, 64, block_size, 16, 1);                     \
  instantiate_unified_direct(type, 128, block_size, 8, 1);                     \
  instantiate_unified_direct(type, 128, block_size, 8, 2);                     \
  instantiate_unified_direct(type, 256, block_size, 8, 1);                     \
  instantiate_unified_direct(type, 256, block_size, 8, 4);                     \
  instantiate_unified_staged(type, 64, block_size, 4, 32);                     \
  instantiate_unified_staged(type, 64, block_size, 4, 16);                     \
  instantiate_unified_staged(type, 128, block_size, 4, 16);                    \
  instantiate_unified_staged(type, 128, block_size, 4, 32);                    \
  instantiate_unified_staged(type, 256, block_size, 4, 16);                    \
  instantiate_unified_staged(type, 256, block_size, 2, 16);                    \
  instantiate_unified_staged(type, 256, block_size, 2, 8);

#define instantiate_unified_all(type)                                          \
  instantiate_unified_grids(type, 8);                                          \
  instantiate_unified_grids(type, 16);                                         \
  instantiate_unified_grids(type, 32);                                         \
  instantiate_unified_reduce(type, 64);                                        \
  instantiate_unified_reduce(type, 128);                                       \
  instantiate_unified_reduce(type, 256);

instantiate_unified_all(float);
instantiate_unified_all(half);
instantiate_unified_all(bfloat16_t);
