// SPDX-License-Identifier: Apache-2.0
// Unified paged attention: the packed-GQA Flash-Attention-2 kernel in
// kernels_v2/pagedattention_unified.metal, plus its split-KV reduce.
#include "ops_internal.h"

static std::string unified_source_;

void init_unified_library(const std::string& src) {
  unified_source_ = src;
  auto& d = metal::device(Device::gpu);
  d.get_library("paged_attention_unified_kern",
                [&]() { return unified_source_; });
}

namespace {

// A simdgroup grid of pagedattention_unified.metal.  Staged grids are
// instantiated per sg_m; direct grids take their row simdgroups at launch
// (sg_m 0: pick from the rows) and are instantiated per head slice count ds.
struct UnifiedConfig {
  bool staged;
  int sg_m;
  int tile_kv;
  int ds = 1;
  bool operator==(const UnifiedConfig& o) const {
    return staged == o.staged && sg_m == o.sg_m && tile_kv == o.tile_kv &&
        ds == o.ds;
  }
};

// Mirrors instantiate_unified_grids (direct entries carry sg_m 0).
struct UnifiedInstance {
  int head_size;
  UnifiedConfig config;
};
constexpr std::array<UnifiedInstance, 12> kUnifiedInstances = {{
    {64, {false, 0, 16, 1}}, {128, {false, 0, 8, 1}},
    {128, {false, 0, 8, 2}}, {256, {false, 0, 8, 1}},
    {256, {false, 0, 8, 4}}, {64, {true, 4, 32}},
    {64, {true, 4, 16}},     {128, {true, 4, 16}},
    {128, {true, 4, 32}},    {256, {true, 4, 16}},
    {256, {true, 2, 16}},    {256, {true, 2, 8}},
}};
constexpr int kMaxDirectRowSimdgroups = 4;

// Kernel page sizes instantiated by instantiate_unified_all.
bool unified_block_size(int block_size) {
  return block_size == 8 || block_size == 16 || block_size == 32;
}

bool unified_head_size(int head_size) {
  return head_size == 64 || head_size == 128 || head_size == 256;
}

bool unified_instantiated(int head_size, UnifiedConfig c) {
  if (!c.staged && (c.sg_m < 0 || c.sg_m > kMaxDirectRowSimdgroups)) return false;
  return std::any_of(kUnifiedInstances.begin(), kUnifiedInstances.end(),
                     [&](const UnifiedInstance& i) {
                       return i.head_size == head_size &&
                           i.config.staged == c.staged &&
                           i.config.tile_kv == c.tile_kv &&
                           (c.staged ? i.config.sg_m == c.sg_m
                                     : i.config.ds == c.ds);
                     });
}

// The canonical reduction tree: absolute partitions of partition_keys keys,
// each walked in tiles of the grid's tile_kv, blended in partition order.
// These fix a row's bits on a given GPU; everything else about a launch is
// free.  A row's keys split no finer than its partitions, and each one costs
// a split-KV partial.  128 keys let base and Pro GPUs (device class as in
// mlx_decode_splits) reach MLX's 32-64 splits from 4K-8K keys; Max and Ultra
// GPUs, which MLX splits 64-128 ways from 1K keys, take 64 (on M1, which
// those splits cannot speed up, 64-key partials cost 4-11%).
int partition_keys(char device_class) {
  return device_class == 's' || device_class == 'd' ? 64 : 128;
}

// Head slices per direct row block (part of the canonical tree: the slices'
// partial scores add in slice order).  Slicing cuts each lane's O, K and V
// registers and multiplies the simdgroups reading a KV head; on M1 decode it
// lifted few-KV-head models most (Gemma 3 1B, head 256: 0.54x -> 1.05x MLX at
// 2K keys) and held head 64, whose fragments are already small, back.
// TurboQuant rows stay whole for the epilogue's in-register inverse FWHT.
UnifiedConfig direct_config(int head_size, bool tq) {
  const int ds = tq ? 1 : head_size == 256 ? 4 : head_size == 128 ? 2 : 1;
  return {false, 0, head_size == 64 ? 16 : 8, ds};
}

// FP32 head-256 prefill stages half the rows and keys to fit 32 KB.
UnifiedConfig staged_config(int head_size, Dtype dtype) {
  switch (head_size) {
    case 64:
      return {true, 4, 32};
    case 128:
      return {true, 4, 16};
    default:
      return dtype == float32 ? UnifiedConfig{true, 2, 8}
                              : UnifiedConfig{true, 4, 16};
  }
}

// Must mirror the shared_mem carves in pagedattention_unified.metal: the
// direct grid's score exchange and Q (unsliced head 256), or the staged
// Q, K^T and V tiles; then TurboQuant centroids.
size_t unified_shared_bytes(UnifiedConfig c, int sg_m, int head_size,
                            int t_size, size_t centroid_bytes) {
  const size_t pad = 16 / t_size;
  const size_t q = 8 * sg_m * (head_size + pad);
  // Staged grids allocate their tiles statically: at head 256 Q and one
  // buffer K^T and V take turns in, below it K^T and V each, plus Q from
  // 16-bit head 128.  TurboQuant centroids are dynamic.
  if (c.staged) {
    const size_t k = head_size * (c.tile_kv + pad);
    const size_t v = c.tile_kv * (head_size + pad);
    const bool q_staged = head_size >= 256 || (head_size >= 128 && t_size == 2);
    const size_t tiles = (q_staged ? q : 0) +
        (head_size >= 256 ? std::max(k, v) : k + v);
    return tiles * t_size + centroid_bytes;
  }
  const size_t exchange =
      c.ds > 1 ? size_t(2) * 8 * sg_m * c.ds * (c.tile_kv / 8) * 32 * 8 : 0;
  const size_t q_bytes =
      head_size / c.ds >= 256 ? (q * t_size + 15) & ~size_t(15) : 0;
  return (exchange + q_bytes + centroid_bytes + 15) & ~size_t(15);
}

std::string unified_name(const std::string& dt, int head_size, int block_size,
                         UnifiedConfig c) {
  return std::string("paged_attention_unified_") +
      (c.staged ? "staged_" : "direct_") + dt + "_hs" +
      std::to_string(head_size) + "_bs" + std::to_string(block_size) +
      (c.staged ? "_m" + std::to_string(c.sg_m) : std::string()) + "_tk" +
      std::to_string(c.tile_kv) +
      (c.staged ? std::string() : "_ds" + std::to_string(c.ds));
}

// How a launch spreads the canonical partitions; none of it changes a bit.
struct WorkPlan {
  int groups;            // partition groups per row block
  int group_partitions;  // consecutive partitions per group
  int groups_per_tg;     // direct: groups sharing a threadgroup
  bool split;            // write partials for the reduce
};

// Key splits per (row block, KV head) of a decode launch: MLX's decode policy
// (v0.32.2, mlx/backend/metal/scaled_dot_product_attention.cpp).  Its single
// pass splits a head's keys over the 32 simdgroups of a threadgroup; its two
// passes over `blocks` threadgroups, picked from the device class (the GPU
// architecture's last letter: s Max, d Ultra, otherwise base or Pro), the
// keys and the query rows sharing a KV head.  Like MLX it ignores
// the batch and needs no per-chip calibration.  On M1, against the occupancy
// cap it replaced, it about doubled head-256 decode at batch 16-64 and cost
// head-128 decode 1-9% at batch 4-64.
int mlx_decode_splits(char device_class, int64_t keys, int rows_per_kv_head,
                      bool gqa) {
  const bool large = device_class == 's' || device_class == 'd';
  if (!((large && keys >= 1024) || (gqa && keys >= 4096))) return 32;
  if (device_class == 's') {
    if (keys <= 1024 || rows_per_kv_head <= 4) return 64;
    return keys <= 8192 ? 128 : keys <= 32768 ? 256 : keys <= 65536 ? 512 : 1024;
  }
  if (device_class == 'd') {
    if (rows_per_kv_head <= 2 && keys > 8192) return 256;
    if (rows_per_kv_head >= 6 && keys >= 65536) return 1024;
    if (rows_per_kv_head >= 6 && keys >= 16384) return 512;
    return 128;
  }
  return rows_per_kv_head >= 4 ? 64 : 32;
}

constexpr int kMaxGroupsPerThreadgroup = 4;

// partition > 0 forces direct groups of that many partitions (and the
// reduce); 0 forces one group; -1 spreads the partitions over `splits`
// groups.  Rows spanning several partitions always merge in the reduce.
WorkPlan plan_work(int partition, bool staged, int splits, int partitions) {
  if (staged) return {1, partitions, 1, false};
  const int target = std::max(1, std::min(splits, partitions));
  const int group_partitions = partition > 0 ? partition
      : partition == 0 ? partitions
                       : (partitions + target - 1) / target;
  const int groups = (partitions + group_partitions - 1) / group_partitions;
  const bool split = partition > 0 || partitions > 1;
  return {groups, group_partitions,
          split ? std::min(groups, kMaxGroupsPerThreadgroup) : 1, split};
}

// Partitions of P keys any row of a launch can span, counted from its
// sequence's base.
int partition_span(int P, int max_seq_len, int max_query_len,
                   int sliding_window) {
  // A window's base rounds down by up to P - 1 keys.
  const int64_t span = sliding_window < 0
      ? max_seq_len
      : std::min<int64_t>(max_seq_len,
                          int64_t(sliding_window) + max_query_len + P - 2);
  return static_cast<int>(std::max<int64_t>(1, (span + P - 1) / P));
}


// Mirrors UaParams and UaReduceParams in pagedattention_unified.metal.
struct UnifiedKernelParams {
  int32_t num_kv_heads;
  float scale;
  float softcap;
  int32_t max_num_blocks_per_seq;
  int32_t seq_blocks;
  int32_t q_stride;
  int32_t kv_block_stride;
  int32_t kv_head_stride;
  int32_t num_seqs;
  int32_t sliding_window;
  int32_t partition_size;
  int32_t seq_filter;
  int32_t decode_tokens;
  int32_t sg_m;
  int32_t groups;
  int32_t group_partitions;
  int32_t max_partitions;
  int32_t v_block_stride;
  int32_t v_head_stride;
  int32_t scale_block_stride;
  int32_t scale_head_stride;
};

struct UnifiedReduceParams {
  int32_t max_partitions;
  int32_t num_seqs;
  int32_t partition_size;
  int32_t sliding_window;
  int32_t seq_filter;
  int32_t decode_tokens;
};

// TurboQuant cache format (vllm_metal/attention/caches/turboquant.py).
struct TurboQuantSpec {
  bool enabled = false;
  int k_bits = 8;
  bool k_signed = false;
  int v_bits = 8;
  bool operator==(const TurboQuantSpec& o) const {
    return enabled == o.enabled && k_bits == o.k_bits &&
        k_signed == o.k_signed && v_bits == o.v_bits;
  }
};

// Element stride between pages; a cache viewed with larger upstream pages
// (shape(1) > block_size) is addressed as consecutive kernel-sized subpages.
int32_t page_stride(const array& cache, int block_size) {
  return static_cast<int32_t>(static_cast<int>(cache.shape(1)) > block_size
                                  ? block_size * cache.strides()[1]
                                  : cache.strides()[0]);
}

array scratch(metal::CommandEncoder& enc, Shape shape, Dtype dtype) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  enc.add_temporary(a);
  return a;
}

struct UnifiedArrays {
  const array& query;
  const array& key_cache;
  const array& value_cache;
  const array& block_tables;
  const array& seq_lens;
  const array& cu_seqlens_q;
  const array* sinks;
  // TurboQuant only.
  const array* key_scale;
  const array* value_scale;
  const array* key_zero;
  const array* centroids;
};

struct UnifiedParams {
  int num_kv_heads;
  float scale;
  float softcap;
  int block_size;
  int max_seq_len;
  int sliding_window;
  // Upper bound on every cu_seqlens_q segment; 1 for decode.
  int max_query_len;
  // -1 auto, 0 single pass, > 0 split-KV groups of that many partitions.
  int partition;
  std::optional<UnifiedConfig> config;
  TurboQuantSpec tq;
  // Decode and verification rows: the first decode_tokens query tokens, or
  // (-1) the one-token segments; ua_decode_segment in the kernel.  They take
  // the direct grid at any length, so verification windows of any size stay
  // bit-identical to one-token decode.  max_decode_query_len bounds those
  // segments (0: unknown).
  int decode_tokens = -1;
  int max_decode_query_len = 0;
  bool operator==(const UnifiedParams& o) const {
    return num_kv_heads == o.num_kv_heads && scale == o.scale &&
        softcap == o.softcap && block_size == o.block_size &&
        max_seq_len == o.max_seq_len && sliding_window == o.sliding_window &&
        max_query_len == o.max_query_len && partition == o.partition &&
        config == o.config && tq == o.tq && decode_tokens == o.decode_tokens &&
        max_decode_query_len == o.max_decode_query_len;
  }
};

// Encode one grid (and its reduce when split) for the sequences `seq_filter`
// selects (ua_serves; each at most max_query_len tokens) into `out`, whose
// buffer the caller has allocated.
void dispatch_grid(array& out, const UnifiedArrays& a, const UnifiedParams& p,
                   UnifiedConfig cfg, int seq_filter, int max_query_len,
                   Stream s) {
  auto& d = metal::device(s.device);
  auto& enc = metal::get_command_encoder(s);

  const int total_q = static_cast<int>(a.query.shape(0));
  const int num_heads = static_cast<int>(a.query.shape(1));
  const int head_size = static_cast<int>(a.query.shape(2));
  const int num_seqs = static_cast<int>(a.cu_seqlens_q.shape(0)) - 1;
  const int group = num_heads / p.num_kv_heads;
  const int t_size = static_cast<int>(a.query.itemsize());
  const int max_rows = max_query_len * group;
  const bool use_tq = p.tq.enabled;
  const size_t centroid_bytes = use_tq ? (size_t(1) << p.tq.v_bits) * 4 : 0;
  const size_t max_shared = d.mtl_device()->maxThreadgroupMemoryLength();
  const char device_class = d.get_architecture().back();
  const int part_keys = partition_keys(device_class);
  const int partitions =
      partition_span(part_keys, p.max_seq_len, max_query_len, p.sliding_window);
  const int64_t keys = p.sliding_window < 0
      ? p.max_seq_len
      : std::min<int64_t>(p.max_seq_len,
                          int64_t(p.sliding_window) + max_query_len - 1);
  const int splits = mlx_decode_splits(device_class, keys, max_rows, group > 1);
  WorkPlan plan = plan_work(p.partition, cfg.staged, splits, partitions);
  bool partitioned = plan.split;
  const int max_partitions = plan.groups * plan.group_partitions;

  // Split-KV folds the sink once, in the reduce.
  bool use_sinks = a.sinks != nullptr && !partitioned;
  bool use_softcap = p.softcap > 0.0f;
  bool tq_on = use_tq;
  int k_bits = p.tq.k_bits;
  int v_bits = p.tq.v_bits;
  bool k_signed = p.tq.k_signed;
  const std::string dt = dtype_to_metal(a.query.dtype());
  const std::string name = unified_name(dt, head_size, p.block_size, cfg);
  const std::string hash = name + "_p" + (partitioned ? "1" : "0") + "_sk" +
      (use_sinks ? "1" : "0") + "_sc" + (use_softcap ? "1" : "0") + "_tq" +
      (use_tq ? "1" : "0") + "_kb" +
      std::to_string(k_bits) + (k_signed ? "s" : "u") + "_vb" +
      std::to_string(v_bits);
  // Staged grids allocate statically: reject one beyond the device before
  // its pipeline fails to build.
  if (cfg.staged) {
    const size_t staged_bytes = unified_shared_bytes(cfg, cfg.sg_m, head_size,
                                                     t_size, centroid_bytes);
    if (staged_bytes > max_shared) {
      throw std::invalid_argument(
          name + " needs " + std::to_string(staged_bytes) +
          " bytes of threadgroup memory, more than the device allows");
    }
  }
  auto* lib = d.get_library("paged_attention_unified_kern");
  auto* kernel = d.get_kernel(
      name, lib, hash,
      {{&partitioned, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&use_softcap, MTL::DataType::DataTypeBool, NS::UInteger(41)},
       {&tq_on, MTL::DataType::DataTypeBool, NS::UInteger(50)},
       {&k_bits, MTL::DataType::DataTypeInt, NS::UInteger(60)},
       {&v_bits, MTL::DataType::DataTypeInt, NS::UInteger(70)},
       {&k_signed, MTL::DataType::DataTypeBool, NS::UInteger(71)}});
  const int max_threads =
      static_cast<int>(kernel->maxTotalThreadsPerThreadgroup());
  // A direct threadgroup stacks enough row simdgroups (each with its ds head
  // slices) to read a window's KV once, within threadgroup memory and the
  // pipeline's register-limited threadgroup size.
  // A requested sg_m is an upper bound; none of this changes a bit.
  int sg_m = cfg.sg_m;
  if (!cfg.staged) {
    sg_m = std::clamp((max_rows + 7) / 8, 1,
                      sg_m > 0 ? sg_m : kMaxDirectRowSimdgroups);
    while (sg_m > 1 &&
           (unified_shared_bytes(cfg, sg_m, head_size, t_size,
                                 centroid_bytes) > max_shared ||
            32 * sg_m * cfg.ds > max_threads)) {
      sg_m--;
    }
  }
  const size_t shared_bytes =
      unified_shared_bytes(cfg, sg_m, head_size, t_size, centroid_bytes);
  if (shared_bytes > max_shared) {
    throw std::invalid_argument(
        name + " needs " + std::to_string(shared_bytes) +
        " bytes of threadgroup memory, more than the device allows");
  }
  const int bm = 8 * sg_m;
  // Up to 8 blocks per sequence index directly; longer sequences use the
  // cu_seqlens-bounded block layout.
  const int seq_blocks = max_rows <= 8 * bm ? (max_rows + bm - 1) / bm : 0;
  const int row_blocks = seq_blocks > 0
      ? num_seqs * seq_blocks : (total_q * group) / bm + num_seqs;
  // Packed partition groups must fit the pipeline's register-limited
  // threadgroup size (384 threads for most direct grids on M1).  Head slices
  // synchronise per tile, so a sliced threadgroup holds one group.
  plan.groups_per_tg = cfg.ds > 1
      ? 1
      : std::max(1, std::min(plan.groups_per_tg, max_threads / (32 * sg_m)));

  // Declared decode segments lead the batch, so a decode launch's partials
  // (and its reduce) cover only those rows, not the prefill after them.
  const int partial_rows =
      seq_filter > 0 && p.decode_tokens >= 0 ? p.decode_tokens : total_q;
  std::optional<array> tmp_out, exp_sums, max_logits;
  if (partitioned) {
    Shape stats{partial_rows, num_heads, max_partitions};
    Shape partial = stats;
    partial.push_back(head_size);
    tmp_out.emplace(scratch(enc, partial, float32));
    exp_sums.emplace(scratch(enc, stats, float32));
    max_logits.emplace(scratch(enc, stats, float32));
  }

  UnifiedKernelParams kp{};
  kp.num_kv_heads = p.num_kv_heads;
  kp.scale = p.scale;
  kp.softcap = p.softcap;
  kp.max_num_blocks_per_seq = static_cast<int32_t>(a.block_tables.shape(1));
  kp.seq_blocks = seq_blocks;
  kp.q_stride = num_heads * head_size;
  kp.kv_block_stride = page_stride(a.key_cache, p.block_size);
  kp.kv_head_stride = static_cast<int32_t>(a.key_cache.strides()[2]);
  kp.num_seqs = num_seqs;
  kp.sliding_window = p.sliding_window;
  kp.partition_size = part_keys;
  kp.seq_filter = seq_filter;
  kp.decode_tokens = p.decode_tokens;
  kp.sg_m = sg_m;
  kp.groups = plan.groups;
  kp.group_partitions = plan.group_partitions;
  kp.max_partitions = max_partitions;
  if (use_tq) {
    kp.v_block_stride = page_stride(a.value_cache, p.block_size);
    kp.v_head_stride = static_cast<int32_t>(a.value_cache.strides()[2]);
    kp.scale_block_stride = page_stride(*a.key_scale, p.block_size);
    kp.scale_head_stride = static_cast<int32_t>(a.key_scale->strides()[2]);
  }

  enc.set_compute_pipeline_state(kernel);
  // Staged grids allocate their tiles statically and only the TurboQuant
  // centroids dynamically.
  if (!cfg.staged) {
    enc.set_threadgroup_memory_length(shared_bytes, 0);
  } else if (centroid_bytes > 0) {
    enc.set_threadgroup_memory_length(centroid_bytes, 0);
  }
  if (partitioned) {
    enc.set_output_array(*exp_sums, 0);
    enc.set_output_array(*max_logits, 1);
    enc.set_output_array(*tmp_out, 6);
  } else {
    enc.set_output_array(out, 2);
  }
  enc.set_input_array(a.query, 3);
  enc.set_input_array(a.key_cache, 4);
  enc.set_input_array(a.value_cache, 5);
  enc.set_bytes(kp, 7);
  enc.set_input_array(a.block_tables, 8);
  enc.set_input_array(a.seq_lens, 9);
  if (use_sinks) enc.set_input_array(*a.sinks, 10);
  enc.set_input_array(a.cu_seqlens_q, 11);
  if (use_tq) {
    enc.set_input_array(*a.key_scale, 12);
    enc.set_input_array(*a.value_scale, 13);
    enc.set_input_array(*a.key_zero, 14);
    enc.set_input_array(*a.centroids, 15);
  }
  const int tg_groups = cfg.staged
      ? plan.groups
      : (plan.groups + plan.groups_per_tg - 1) / plan.groups_per_tg;
  enc.dispatch_threadgroups(
      MTL::Size::Make(row_blocks, p.num_kv_heads, tg_groups),
      MTL::Size::Make(32 * sg_m * cfg.ds * plan.groups_per_tg, 1, 1));
  if (!partitioned) return;

  // Pass 2: blend each (token, head)'s partitions into `out`.
  bool reduce_sinks = a.sinks != nullptr;
  const std::string rname =
      "paged_attention_unified_reduce_" + dt + "_hs" + std::to_string(head_size);
  auto* rkernel = d.get_kernel(
      rname, lib,
      rname + "_sk" + (reduce_sinks ? "1" : "0") + "_tq" + (use_tq ? "1" : "0"),
      {{&reduce_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&tq_on, MTL::DataType::DataTypeBool, NS::UInteger(50)}});
  const UnifiedReduceParams rp{max_partitions, num_seqs,  part_keys,
                               p.sliding_window, seq_filter, p.decode_tokens};
  enc.set_compute_pipeline_state(rkernel);
  // Max-first merge weights (UA_REDUCE_W + 4 floats), or a TurboQuant row.
  enc.set_threadgroup_memory_length(
      (std::max<size_t>(head_size, 256 + 4) * sizeof(float) + 15) & ~size_t(15),
      0);
  enc.set_output_array(out, 0);
  enc.set_input_array(*exp_sums, 1);
  enc.set_input_array(*max_logits, 2);
  enc.set_input_array(*tmp_out, 3);
  enc.set_input_array(a.seq_lens, 4);
  enc.set_bytes(rp, 5);
  if (reduce_sinks) enc.set_input_array(*a.sinks, 6);
  enc.set_input_array(a.cu_seqlens_q, 7);
  enc.dispatch_threadgroups(MTL::Size::Make(num_heads, partial_rows, 1),
                            MTL::Size::Make(128, 1, 1));
}

// Encode the unified kernel into `out`: decode and verification segments on
// the direct grid, the rest (prefill) on the staged grid, launching only the
// grids the batch needs.
void dispatch_unified(array& out, const UnifiedArrays& a, const UnifiedParams& p,
                      Stream s) {
  if (p.config) {
    dispatch_grid(out, a, p, *p.config, 0, p.max_query_len, s);
    return;
  }
  const int total_q = static_cast<int>(a.query.shape(0));
  const int head_size = static_cast<int>(a.query.shape(2));
  const UnifiedConfig direct = direct_config(head_size, p.tq.enabled);
  const UnifiedConfig staged = staged_config(head_size, a.query.dtype());
  const bool known = p.decode_tokens >= 0;
  const int decode_max_q = !known ? 1
      : std::min(p.max_query_len, p.max_decode_query_len > 0
                                      ? p.max_decode_query_len
                                      : std::max(1, p.decode_tokens));
  if (known ? p.decode_tokens >= total_q : p.max_query_len == 1) {
    dispatch_grid(out, a, p, direct, 0, decode_max_q, s);
  } else if (known && p.decode_tokens == 0) {
    dispatch_grid(out, a, p, staged, 0, p.max_query_len, s);
  } else {
    dispatch_grid(out, a, p, staged, -1, p.max_query_len, s);
    dispatch_grid(out, a, p, direct, 1, decode_max_q, s);
  }
}

}  // namespace

bool unified_attention_enabled_ = false;

bool dispatch_unified_paged_attention_if_supported(
    array& out, const array& query, const array& key_cache,
    const array& value_cache, const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q, const array* sinks, const array* key_scale,
    const array* value_scale, const array* key_zero, const array* centroids,
    int num_kv_heads, float scale, float softcap, int block_size,
    int max_seq_len, int sliding_window, int max_query_len,
    int window_seqlen_q, int decode_query_tokens, int max_decode_query_len,
    bool use_turboquant, int k_bits, int v_bits, Stream s) {
  const int head_size = static_cast<int>(query.shape(2));
  const bool dtype_ok = query.dtype() == float32 || query.dtype() == float16 ||
      query.dtype() == bfloat16;
  const bool cache_ok = use_turboquant
      ? (key_cache.dtype() == int8 || key_cache.dtype() == uint8) &&
            value_cache.dtype() == uint8
      : key_cache.dtype() == query.dtype() && value_cache.dtype() == query.dtype();
  if (!dtype_ok || !cache_ok || !unified_block_size(block_size) ||
      !unified_head_size(head_size)) {
    return false;
  }
  // Which rows are decode or verification rows decides the grid, and with it
  // the bits: a window must take the grid its tokens take when decoded one by
  // one, and a one-token prefill segment the grid it takes in any batch.  A
  // verification batch (window hint) holds no prefill; otherwise the caller's
  // count decides, and without one only a batch of one-token segments can be
  // taken as all decode.  Failing that, the established kernels keep spec
  // decode exact.
  const int total_q = static_cast<int>(query.shape(0));
  const int num_seqs = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  int decode_tokens = decode_query_tokens;
  int decode_max_q = max_decode_query_len;
  if (window_seqlen_q > 1) {
    decode_tokens = total_q;
    decode_max_q = window_seqlen_q;
  } else if (decode_query_tokens < 0) {
    if (total_q != num_seqs) return false;
    decode_tokens = total_q;
    decode_max_q = 1;
  }
  UnifiedParams params{num_kv_heads, scale, softcap, block_size, max_seq_len,
                       sliding_window, max_query_len, -1, std::nullopt, {}};
  params.decode_tokens = decode_tokens;
  params.max_decode_query_len = decode_max_q;
  if (use_turboquant) {
    params.tq = {true, k_bits, key_cache.dtype() == int8, v_bits};
  }
  dispatch_unified(out,
                   {query, key_cache, value_cache, block_tables, seq_lens,
                    cu_seqlens_q, sinks, key_scale, value_scale, key_zero,
                    centroids},
                   params, s);
  return true;
}

class UnifiedPagedAttentionPrimitive : public UnaryPrimitive {
 public:
  UnifiedPagedAttentionPrimitive(Stream stream, UnifiedParams params,
                                 bool use_sinks)
      : UnaryPrimitive(stream), params_(std::move(params)),
        use_sinks_(use_sinks) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("UnifiedPagedAttentionPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    out.set_data(allocator::malloc(out.nbytes()));
    // Optional inputs: sinks at 6, or the four TurboQuant arrays at 6-9
    // (the two are mutually exclusive).
    const bool tq = params_.tq.enabled;
    dispatch_unified(out,
                     {inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                      inputs[5], use_sinks_ ? &inputs[6] : nullptr,
                      tq ? &inputs[6] : nullptr, tq ? &inputs[7] : nullptr,
                      tq ? &inputs[8] : nullptr, tq ? &inputs[9] : nullptr},
                     params_, stream());
  }

  const char* name() const override { return "UnifiedPagedAttention"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const UnifiedPagedAttentionPrimitive*>(&other);
    return rhs && rhs->params_ == params_ && rhs->use_sinks_ == use_sinks_;
  }

 private:
  UnifiedParams params_;
  bool use_sinks_;
};

namespace {

// k_bits and signedness of each TurboQuant key format (QUANT_PARAMS in
// vllm_metal/attention/caches/turboquant.py).
std::optional<std::pair<int, bool>> turboquant_key_format(const std::string& q) {
  if (q == "q8_0" || q == "int8") return std::make_pair(8, true);
  if (q == "uint8") return std::make_pair(8, false);
  if (q == "q5_0") return std::make_pair(5, false);
  if (q == "q4_0" || q == "int4" || q == "uint4") return std::make_pair(4, false);
  if (q == "int2" || q == "uint2") return std::make_pair(2, false);
  return std::nullopt;
}

TurboQuantSpec validate_turboquant(
    const array& key_cache, const array& value_cache, const array* key_scale,
    const array* value_scale, const array* key_zero, const array* centroids,
    const std::string& quant_type, int v_bits, int head_size, int num_kv_heads) {
  const auto format = turboquant_key_format(quant_type);
  if (!format) {
    throw std::invalid_argument("unknown TurboQuant key quant_type: " + quant_type);
  }
  if (key_scale == nullptr || value_scale == nullptr || key_zero == nullptr ||
      centroids == nullptr) {
    throw std::invalid_argument(
        "TurboQuant requires key_scale_cache, value_scale_cache, "
        "key_zero_cache and v_centroids");
  }
  const auto [k_bits, k_signed] = *format;
  if (!(v_bits == 2 || v_bits == 3 || v_bits == 4 || v_bits == 5 || v_bits == 8)) {
    throw std::invalid_argument("TurboQuant v_bits must be 2, 3, 4, 5 or 8");
  }
  const Shape& ks = key_cache.shape();
  const Shape scales{ks[0], ks[1], ks[2], head_size / 32};
  if (key_cache.dtype() != (k_signed ? int8 : uint8) ||
      value_cache.dtype() != uint8 || ks[2] != num_kv_heads ||
      ks[3] != head_size * k_bits / 8 || value_cache.ndim() != 4 ||
      value_cache.shape(0) != ks[0] || value_cache.shape(1) != ks[1] ||
      value_cache.shape(2) != ks[2] ||
      value_cache.shape(3) != head_size * v_bits / 8 ||
      key_scale->shape() != scales || value_scale->shape() != scales ||
      key_zero->shape() != scales || key_scale->dtype() != float16 ||
      value_scale->dtype() != float16 || key_zero->dtype() != float16 ||
      value_scale->strides() != key_scale->strides() ||
      key_zero->strides() != key_scale->strides() ||
      centroids->dtype() != float32 || centroids->ndim() != 1 ||
      centroids->shape(0) != (1 << v_bits)) {
    throw std::invalid_argument(
        "TurboQuant caches must match turboquant.py: K packed to head_size * "
        "k_bits / 8 bytes (int8 when signed), V packed to head_size * v_bits / 8 "
        "uint8, float16 scale/zero caches of head_size / 32 groups sharing one "
        "layout, and 2^v_bits float32 centroids");
  }
  return {true, k_bits, k_signed, v_bits};
}

}  // namespace

array unified_paged_attention_primitive_fn(
    const array& query, const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap, const array& block_tables,
    const array& seq_lens, const array& cu_seqlens_q, int block_size,
    int max_seq_len, int sliding_window, int max_query_len,
    const array* sinks, int partition,
    const std::optional<std::tuple<std::string, int, int>>& config,
    const array* key_scale_cache, const array* value_scale_cache,
    const array* key_zero_cache, const array* v_centroids,
    const std::string& quant_type, int v_bits, int num_decode_tokens) {
  if (query.ndim() != 3 || key_cache.ndim() != 4 || value_cache.ndim() != 4) {
    throw std::invalid_argument(
        "unified paged attention expects query [tokens, heads, head_size] and "
        "[blocks, block_size, kv_heads, dim] caches");
  }
  if (query.dtype() != float32 && query.dtype() != float16 &&
      query.dtype() != bfloat16) {
    throw std::invalid_argument(
        "unified paged attention supports float32/float16/bfloat16 queries");
  }
  const int num_heads = static_cast<int>(query.shape(1));
  const int head_size = static_cast<int>(query.shape(2));
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    throw std::invalid_argument(
        "unified paged attention needs query heads divisible by KV heads");
  }
  if (!unified_head_size(head_size) || !unified_block_size(block_size)) {
    throw std::invalid_argument(
        "unified paged attention is instantiated for head sizes {64, 128, 256} "
        "and kernel block sizes {8, 16, 32}");
  }
  TurboQuantSpec tq;
  if (quant_type.empty()) {
    if (key_cache.shape() != value_cache.shape() ||
        key_cache.strides() != value_cache.strides() ||
        key_cache.dtype() != query.dtype() || value_cache.dtype() != query.dtype() ||
        static_cast<int>(key_cache.shape(2)) != num_kv_heads ||
        static_cast<int>(key_cache.shape(3)) != head_size) {
      throw std::invalid_argument(
          "dense caches must share one layout, match the query dtype, and have "
          "num_kv_heads heads of head_size");
    }
  } else {
    tq = validate_turboquant(key_cache, value_cache, key_scale_cache,
                             value_scale_cache, key_zero_cache, v_centroids,
                             quant_type, v_bits, head_size, num_kv_heads);
    if (sinks != nullptr) {
      throw std::invalid_argument(
          "attention sinks are not supported with TurboQuant KV");
    }
  }
  if (max_query_len < 1 || max_seq_len < 1) {
    throw std::invalid_argument("max_query_len and max_seq_len must be >= 1");
  }
  if (partition < -1) {
    throw std::invalid_argument(
        "partition must be -1 (auto), 0 (single pass) or a positive number of "
        "partitions per split-KV group");
  }
  if (num_decode_tokens < -1 ||
      num_decode_tokens > static_cast<int>(query.shape(0))) {
    throw std::invalid_argument(
        "num_decode_tokens must be -1 (one-token segments are decode rows) or "
        "the number of leading decode/verification query tokens");
  }
  if (sinks != nullptr &&
      (sinks->ndim() != 1 || static_cast<int>(sinks->shape(0)) != num_heads ||
       sinks->dtype() != float32)) {
    throw std::invalid_argument("sinks must be float32 [num_heads]");
  }
  std::optional<UnifiedConfig> cfg;
  if (config) {
    const std::string& kind = std::get<0>(*config);
    cfg = UnifiedConfig{kind == "staged", std::get<1>(*config),
                        std::get<2>(*config),
                        kind == "staged"
                            ? 1
                            : direct_config(head_size, tq.enabled).ds};
    if ((kind != "direct" && kind != "staged") ||
        !unified_instantiated(head_size, *cfg)) {
      throw std::invalid_argument(
          "config (\"direct\" | \"staged\", sg_m, tile_kv) is not instantiated "
          "for this head size");
    }
  }
  UnifiedParams params{num_kv_heads, scale, softcap, block_size, max_seq_len,
                       sliding_window, max_query_len, partition, cfg, tq};
  params.decode_tokens = num_decode_tokens;
  auto prim = std::make_shared<UnifiedPagedAttentionPrimitive>(
      default_stream(Device::gpu), std::move(params), sinks != nullptr);
  std::vector<array> inputs = {query, key_cache, value_cache,
                               block_tables, seq_lens, cu_seqlens_q};
  if (sinks != nullptr) inputs.push_back(*sinks);
  if (tq.enabled) {
    inputs.insert(inputs.end(), {*key_scale_cache, *value_scale_cache,
                                 *key_zero_cache, *v_centroids});
  }
  return array(query.shape(), query.dtype(), std::move(prim), std::move(inputs));
}
