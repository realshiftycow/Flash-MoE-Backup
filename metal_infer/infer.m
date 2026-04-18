/*
 * infer.m — MiniMax M2.7 MoE inference engine using Metal
 *
 * Full forward pass: embedding -> 62 transformer layers -> norm -> lm_head -> sample
 * Non-expert weights loaded from flash_moe_dense_q4/model_weights.bin (mmap'd at startup)
 * Expert weights loaded from packed_experts_minimax_q4/ per layer per token (pread)
 *
 * Architecture: MiniMax-M2.7 (MoE)
 *   - 62 full-attention layers
 *   - hidden_size=3072, head_dim=128, num_attention_heads=48, num_kv_heads=8
 *   - 256 experts/layer, router top-k=8 (we default to K=4 for speed)
 *   - No shared expert path
 *   - Full attention: standard QKV + GQA + RoPE
 *
 * Command buffer optimization (fused_layer_forward):
 *   Per-layer Metal command buffer structure:
 *     CMD1: attention input projections (3-4 dispatches, 1 commit)
 *     CPU:  attention compute (RoPE/softmax/delta-net)
 *     CMD2: o_proj + residual_add + rms_norm + routing + shared gate/up (8 encoders, 1 commit)
 *           GPU handles residual connection and post-attn norm internally,
 *           eliminating the CPU round-trip that previously split this into 2 cmd buffers.
 *     CPU:  softmax + top-K + pread all K experts (4 pthreads parallel)
 *     CMD3: all K expert forwards + shared SwiGLU + shared down
 *           + GPU-side combine + residual_add + rms_norm -> buf_input (DEFERRED commit)
 *           Batched encoding: 4 encoders for K experts + 2 shared + 3 combine = 9 total
 *   Total: 3 cmd buffers per layer. CMD3 is submitted async (commit without wait).
 *   GPU-side combine in CMD3: for non-last layers, CMD3 also computes:
 *     moe_combine_residual (weighted sum + residual + shared gate -> hidden)
 *     rms_norm (hidden -> buf_input using NEXT layer's input_norm weights)
 *   This allows the next layer's CMD1 to submit immediately without waiting
 *   for CMD3 completion — the GPU queue serializes CMD3(N-1) then CMD1(N).
 *   Saves ~0.83ms/layer deferred_wait + CPU combine + input_norm overhead.
 *   Multi-expert buffers (MAX_K=8 independent slots) allow all K expert
 *   forwards to be encoded into a single command buffer.
 *   Batched encoding: 2 encoders per expert (gate+up fused, SwiGLU+down fused)
 *   + 2 for shared expert = K*2 + 2 total encoders in CMD3.
 *   Double-buffered expert data (buf_multi_expert_data / data_B) for future
 *   async pread overlap with GPU compute.
 *
 * Build:  clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation -lpthread infer.m -o infer
 * Run:    ./infer --prompt "Explain relativity" --tokens 50
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <compression.h>
// ============================================================================
// Model constants
// ============================================================================
#define HIDDEN_DIM          3072
#define NUM_LAYERS          62
#define NUM_ATTN_HEADS      48
#define NUM_KV_HEADS        8
#define HEAD_DIM            128
#define VOCAB_SIZE          200064
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         256
#define NUM_EXPERTS_PER_TOK 8
#define MOE_INTERMEDIATE    1536
#define SHARED_INTERMEDIATE 1536
#define FULL_ATTN_INTERVAL  1
#define GROUP_SIZE          64
#define BITS                4
// Linear attention (GatedDeltaNet) constants
#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128   // head_k_dim
#define LINEAR_VALUE_DIM    128   // head_v_dim
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)   // 2048
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) // 8192
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE) // 12288
#define CONV_KERNEL_SIZE    4
// Full attention constants
#define ROPE_THETA          5000000.0f
// MiniMax M2.7 uses rotary_dim=64 in the released config. Only the first 64
// dimensions of each 128-d head participate in RoPE.
#define ROTARY_DIM          64
// MiniMax q4 expert packed binary layout
#define EXPERT_SIZE         7962624
#define EXPERT_W1_W_OFF     0
#define EXPERT_W1_S_OFF     2359296
#define EXPERT_W1_B_OFF     2506752
#define EXPERT_W3_W_OFF     2654208
#define EXPERT_W3_S_OFF     5013504
#define EXPERT_W3_B_OFF     5160960
#define EXPERT_W2_W_OFF     5308416
#define EXPERT_W2_S_OFF     7667712
#define EXPERT_W2_B_OFF     7815168
// 2-bit expert layout is not wired up for MiniMax yet.
#define EXPERT_SIZE_2BIT    3932160
#define GATE_W_OFF_2  0
#define GATE_S_OFF_2  1048576
#define GATE_B_OFF_2  1179648
#define UP_W_OFF_2    1310720
#define UP_S_OFF_2    2359296
#define UP_B_OFF_2    2490368
#define DOWN_W_OFF_2  2621440
#define DOWN_S_OFF_2  3670016
#define DOWN_B_OFF_2  3801088
// KV cache maximum context length
// MiniMax currently keeps KV cache in float32, so keep the first runtime bring-up
// on a moderate context length until KV quantization is added.
#define MAX_SEQ_LEN 16384
#define GPU_KV_SEQ  2048
// Special tokens (MiniMax M2.7)
#define EOS_TOKEN_1         200020      // [e~[
#define EOS_TOKEN_2         200020
#define THINK_START_TOKEN   200050      // think start
#define THINK_END_TOKEN     200051      // think end
#define MODEL_PATH_DEFAULT "."
// ============================================================================
// Timing helper
// ============================================================================
static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
// ============================================================================
// Per-phase timing accumulators for fused_layer_forward
// Tracks time spent in each pipeline phase across all layers per token.
// Reset at token boundary, printed as summary.
// ============================================================================
typedef struct {
    double deferred_wait;    // waiting for previous CMD3 GPU
    double deferred_cpu;     // CPU readback + combine for deferred experts
    double input_norm;       // CPU RMS norm + CMD1 prep
    double cmd1_submit;      // CMD1 encode + commit
    double cmd1_wait;        // CMD1 waitUntilCompleted
    double cpu_attn;         // CPU attention compute (delta-net or full-attn)
    double cmd2_encode;      // CMD2 encode (o_proj + residual + norm + routing)
    double cmd2_wait;        // CMD2 commit + waitUntilCompleted
    double routing_cpu;      // CPU softmax + topK
    double spec_route;       // speculative early routing (gate matvec + topK)
    double expert_io;        // parallel pread + cache lookup
    double cmd3_encode;      // CMD3 encode experts + submit (deferred)
    double total;            // total per-layer time
    int count;               // number of layers timed
} LayerTimingAccum;
static LayerTimingAccum g_timing = {0};
static int g_timing_enabled = 0;
// Temporal prediction pipeline counters (declared early for timing_print access)
static int g_pred_enabled = 0;
static int g_pred_generating = 0;   // only set to 1 after prefill (predictions only help during generation)
static uint64_t g_pred_hits = 0;
static uint64_t g_pred_misses = 0;
static uint64_t g_pred_layers = 0;
// Routing data collection for training an expert predictor
// Binary format per sample: int32 layer_idx, int32 K, float32[4096] hidden, int32[K] expert_indices
static FILE *g_routing_log = NULL;
static int g_routing_log_samples = 0;
// LZ4 compressed expert support
// File format: [LZ4IndexEntry × 512] + [compressed blobs]
typedef struct {
    uint64_t offset;
    uint32_t comp_size;
    uint32_t raw_size;
} LZ4IndexEntry;
static LZ4IndexEntry *g_lz4_index[NUM_LAYERS];  // per-layer index (NULL if not using LZ4)
static void *g_lz4_comp_bufs[8];                 // pre-allocated compressed read buffers (MAX_K=8)
static int g_use_lz4 = 0;                        // auto-detected from packed_experts_lz4/
// ============================================================================
// Expert frequency tracking (diagnostic: --freq flag)
// ============================================================================
static int g_expert_freq[NUM_LAYERS][NUM_EXPERTS];  // activation count per (layer, expert)
static int g_freq_tracking = 0;  // enabled by --freq flag
static int g_use_2bit = 0;       // enabled by --2bit flag: use packed_experts_2bit/ + 2-bit kernel
static int g_cache_telemetry_enabled = 0;  // enabled by --cache-telemetry flag
static int g_think_budget = 2048; // max thinking tokens before force-emitting </think>
static int linear_attn_bypass = 1;      // MiniMax has no linear attention path
static int gpu_linear_attn_enabled = 0; // kept for legacy Qwen code paths only
static int gpu_full_attn_enabled = 1;   // default on for MiniMax; use --cpu-full-attn to force CPU fallback
static int g_force_gpu_prefill_attn = 0; // when set, allow GPU full attention during token-serial prefill from token 0
static int g_prefill_chunk_tokens = 0;  // experimental serve/prefill chunk-local expert cache (0 disables)
static int g_prefill_chunk_cache_entries = 0; // experimental small per-chunk expert cache (0 disables)
static int g_prefill_microbatch_tokens = 0; // scaffold only: collect per-chunk routing/dedup stats
static int g_prefill_microbatch_exec = 0; // experimental layer-major prefill executor
static int g_prefill_microbatch_cache_entries = 0; // 0 = auto (chunk_tokens * MAX_K)
typedef struct {
    int chunk_tokens_capacity;
    int routed_slots_capacity;
    id<MTLBuffer> inputs;
    id<MTLBuffer> gate;
    id<MTLBuffer> up;
    id<MTLBuffer> act;
    id<MTLBuffer> out;
} PrefillMicrobatchGpuScratch;
static PrefillMicrobatchGpuScratch g_prefill_microbatch_gpu = {0};
// Tiered I/O: cold fds (F_NOCACHE) for first reads, warm fds (page cached) for repeats
static int *g_layer_fds_cold = NULL;    // [NUM_LAYERS] cold fds (set in main)
static uint8_t g_expert_seen[NUM_LAYERS][NUM_EXPERTS / 8];  // bitset: seen before?
// Async pread state defined after InferPreadTask (see below)
static inline int expert_is_seen(int layer, int expert) {
    return (g_expert_seen[layer][expert >> 3] >> (expert & 7)) & 1;
}
static inline void expert_mark_seen(int layer, int expert) {
    g_expert_seen[layer][expert >> 3] |= (1 << (expert & 7));
}
// Active expert size based on quantization mode
static inline size_t active_expert_size(void) {
    return g_use_2bit ? EXPERT_SIZE_2BIT : EXPERT_SIZE;
}
typedef struct {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
} ExpertLayout;
static inline ExpertLayout active_expert_layout(void) {
    if (g_use_2bit) {
        return (ExpertLayout){
            .gate_w_off = GATE_W_OFF_2, .gate_s_off = GATE_S_OFF_2, .gate_b_off = GATE_B_OFF_2,
            .up_w_off   = UP_W_OFF_2,   .up_s_off   = UP_S_OFF_2,   .up_b_off   = UP_B_OFF_2,
            .down_w_off = DOWN_W_OFF_2, .down_s_off = DOWN_S_OFF_2, .down_b_off = DOWN_B_OFF_2,
        };
    }
    return (ExpertLayout){
        .gate_w_off = EXPERT_W1_W_OFF, .gate_s_off = EXPERT_W1_S_OFF, .gate_b_off = EXPERT_W1_B_OFF,
        .up_w_off   = EXPERT_W3_W_OFF, .up_s_off   = EXPERT_W3_S_OFF, .up_b_off   = EXPERT_W3_B_OFF,
        .down_w_off = EXPERT_W2_W_OFF, .down_s_off = EXPERT_W2_S_OFF, .down_b_off = EXPERT_W2_B_OFF,
    };
}
static int g_freq_total_tokens = 0;  // total tokens processed while tracking
typedef struct {
    uint64_t token_clock;
    uint64_t unique_experts_touched;
    uint64_t cold_misses;
    uint64_t eviction_misses;
    uint64_t evictions;
    uint64_t reuse_le_1;
    uint64_t reuse_le_4;
    uint64_t reuse_le_16;
    uint64_t reuse_le_64;
    uint64_t reuse_gt_64;
    uint64_t reuse_distance_sum;
    uint64_t reuse_distance_samples;
} CacheTelemetry;
static CacheTelemetry g_cache_telemetry = {0};
static uint8_t g_cache_seen[NUM_LAYERS][NUM_EXPERTS];
static uint64_t g_cache_last_touch_token[NUM_LAYERS][NUM_EXPERTS];
static uint64_t g_cache_last_evict_token[NUM_LAYERS][NUM_EXPERTS];
static void cache_telemetry_reset(void) {
    memset(&g_cache_telemetry, 0, sizeof(g_cache_telemetry));
    memset(g_cache_seen, 0, sizeof(g_cache_seen));
    memset(g_cache_last_touch_token, 0, sizeof(g_cache_last_touch_token));
    memset(g_cache_last_evict_token, 0, sizeof(g_cache_last_evict_token));
}
static void cache_telemetry_note_token(void) {
    if (!g_cache_telemetry_enabled) return;
    g_cache_telemetry.token_clock++;
}
static void cache_telemetry_touch(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS || expert_idx < 0 || expert_idx >= NUM_EXPERTS) return;
    if (!g_cache_seen[layer_idx][expert_idx]) {
        g_cache_seen[layer_idx][expert_idx] = 1;
        g_cache_telemetry.unique_experts_touched++;
    }
    g_cache_last_touch_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}
static void cache_telemetry_miss(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS || expert_idx < 0 || expert_idx >= NUM_EXPERTS) return;
    if (!g_cache_seen[layer_idx][expert_idx]) {
        g_cache_telemetry.cold_misses++;
        g_cache_seen[layer_idx][expert_idx] = 1;
        g_cache_telemetry.unique_experts_touched++;
    } else {
        g_cache_telemetry.eviction_misses++;
        uint64_t dist = 0;
        if (g_cache_last_evict_token[layer_idx][expert_idx] > 0 &&
            g_cache_telemetry.token_clock >= g_cache_last_evict_token[layer_idx][expert_idx]) {
            dist = g_cache_telemetry.token_clock - g_cache_last_evict_token[layer_idx][expert_idx];
        }
        if (dist <= 1) g_cache_telemetry.reuse_le_1++;
        else if (dist <= 4) g_cache_telemetry.reuse_le_4++;
        else if (dist <= 16) g_cache_telemetry.reuse_le_16++;
        else if (dist <= 64) g_cache_telemetry.reuse_le_64++;
        else g_cache_telemetry.reuse_gt_64++;
        g_cache_telemetry.reuse_distance_sum += dist;
        g_cache_telemetry.reuse_distance_samples++;
    }
    g_cache_last_touch_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}
static void cache_telemetry_evict(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS || expert_idx < 0 || expert_idx >= NUM_EXPERTS) return;
    g_cache_telemetry.evictions++;
    g_cache_last_evict_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}
static void cache_telemetry_print(uint64_t hits, uint64_t misses) {
    if (!g_cache_telemetry_enabled) return;
    uint64_t total = hits + misses;
    fprintf(stderr, "\n=== Cache Telemetry ===\n");
    fprintf(stderr, "Tokens tracked: %llu\n", g_cache_telemetry.token_clock);
    fprintf(stderr, "Unique experts touched: %llu / %d (%.1f%%)\n",
            g_cache_telemetry.unique_experts_touched,
            NUM_LAYERS * NUM_EXPERTS,
            100.0 * g_cache_telemetry.unique_experts_touched / (NUM_LAYERS * NUM_EXPERTS));
    fprintf(stderr, "Miss breakdown: cold %llu (%.1f%% of misses), eviction %llu (%.1f%% of misses)\n",
            g_cache_telemetry.cold_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.cold_misses / misses : 0.0,
            g_cache_telemetry.eviction_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.eviction_misses / misses : 0.0);
    fprintf(stderr, "Evictions: %llu\n", g_cache_telemetry.evictions);
    fprintf(stderr, "Eviction reuse distance: <=1 tok %llu, <=4 %llu, <=16 %llu, <=64 %llu, >64 %llu",
            g_cache_telemetry.reuse_le_1,
            g_cache_telemetry.reuse_le_4,
            g_cache_telemetry.reuse_le_16,
            g_cache_telemetry.reuse_le_64,
            g_cache_telemetry.reuse_gt_64);
    if (g_cache_telemetry.reuse_distance_samples > 0) {
        fprintf(stderr, " (avg %.1f tok)\n",
                (double)g_cache_telemetry.reuse_distance_sum / g_cache_telemetry.reuse_distance_samples);
    } else {
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "Effective hit rate: %.1f%%\n",
            total > 0 ? 100.0 * hits / total : 0.0);
}
static void timing_reset(void) {
    memset(&g_timing, 0, sizeof(g_timing));
}
static void timing_print(void) {
    if (g_timing.count == 0) return;
    int n = g_timing.count;
    fprintf(stderr, "\n[timing] Per-layer breakdown (avg of %d layers, ms):\n", n);
    fprintf(stderr, "  deferred_wait:  %6.3f\n", g_timing.deferred_wait / n);
    fprintf(stderr, "  deferred_cpu:   %6.3f\n", g_timing.deferred_cpu / n);
    fprintf(stderr, "  input_norm:     %6.3f\n", g_timing.input_norm / n);
    fprintf(stderr, "  cmd1_submit:    %6.3f\n", g_timing.cmd1_submit / n);
    fprintf(stderr, "  cmd1_wait:      %6.3f\n", g_timing.cmd1_wait / n);
    fprintf(stderr, "  spec_route:     %6.3f\n", g_timing.spec_route / n);
    fprintf(stderr, "  cpu_attn:       %6.3f\n", g_timing.cpu_attn / n);
    fprintf(stderr, "  cmd2_encode:    %6.3f\n", g_timing.cmd2_encode / n);
    fprintf(stderr, "  cmd2_wait:      %6.3f\n", g_timing.cmd2_wait / n);
    fprintf(stderr, "  routing_cpu:    %6.3f\n", g_timing.routing_cpu / n);
    fprintf(stderr, "  expert_io:      %6.3f\n", g_timing.expert_io / n);
    fprintf(stderr, "  cmd3_encode:    %6.3f\n", g_timing.cmd3_encode / n);
    fprintf(stderr, "  total_layer:    %6.3f\n", g_timing.total / n);
    fprintf(stderr, "  sum_phases:     %6.3f\n",
            (g_timing.deferred_wait + g_timing.deferred_cpu + g_timing.input_norm +
             g_timing.cmd1_submit + g_timing.cmd1_wait + g_timing.spec_route +
             g_timing.cpu_attn +
             g_timing.cmd2_encode + g_timing.cmd2_wait + g_timing.routing_cpu +
             g_timing.expert_io + g_timing.cmd3_encode) / n);
    fprintf(stderr, "  cmd_buffers:    %d (3 per layer: CMD1+CMD2+CMD3)\n", n * 3);
    fprintf(stderr, "  sync_waits:     %d (2 per layer: CMD1+CMD2, CMD3 deferred)\n", n * 2);
    fprintf(stderr, "  gpu_encoders:   ~%d per layer (CMD1:3-4, CMD2:8-12, CMD3:~10)\n",
            22);  // approximate
    if (g_pred_enabled && g_pred_layers > 0) {
        uint64_t total = g_pred_hits + g_pred_misses;
        double hit_rate = total > 0 ? (double)g_pred_hits / total * 100.0 : 0;
        fprintf(stderr, "  [predict] hits=%llu misses=%llu rate=%.1f%% layers=%llu\n",
                g_pred_hits, g_pred_misses, hit_rate, g_pred_layers);
    }
}

static uint64_t detect_physical_memory_bytes(void) {
    uint64_t mem = 0;
    size_t size = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &size, NULL, 0) == 0 && mem > 0) {
        return mem;
    }
    return 0;
}

static int auto_malloc_cache_entries_for_serve(void) {
    uint64_t mem = detect_physical_memory_bytes();
    size_t expert_size = active_expert_size();
    if (mem == 0 || expert_size == 0) return 0;
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    const uint64_t reserve_bytes = 24ull * gib;  // leave room for weights, KV, app, and OS
    const uint64_t cap_bytes = 20ull * gib;      // match earlier proven-good serve tuning on 48GB systems
    if (mem <= reserve_bytes + expert_size) return 0;
    uint64_t budget = mem - reserve_bytes;
    if (budget > cap_bytes) budget = cap_bytes;
    int entries = (int)(budget / expert_size);
    if (entries < 256) return 0;  // too small to justify the up-front allocation cost
    return entries;
}
// ============================================================================
// bf16 <-> f32 conversion (CPU side)
// ============================================================================
static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
__attribute__((unused))
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}
// ============================================================================
// JSON parser (minimal, for model_weights.json)
// ============================================================================
// We use NSJSONSerialization via ObjC since we already link Foundation
typedef struct {
    const char *name;
    size_t offset;
    size_t size;
    int ndim;
    int shape[4];
    char dtype[8];  // "U32", "BF16", "F32"
} TensorInfo;
typedef struct {
    TensorInfo *tensors;
    int num_tensors;
    int capacity;
} TensorManifest;
static TensorManifest *load_manifest(const char *json_path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:json_path]];
        if (!data) {
            fprintf(stderr, "ERROR: Cannot read %s\n", json_path);
            return NULL;
        }
        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                                                             options:0
                                                               error:&error];
        if (!root) {
            fprintf(stderr, "ERROR: JSON parse failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }
        NSDictionary *tensors = root[@"tensors"];
        if (!tensors) {
            fprintf(stderr, "ERROR: No 'tensors' key in manifest\n");
            return NULL;
        }
        TensorManifest *m = calloc(1, sizeof(TensorManifest));
        m->capacity = (int)[tensors count] + 16;
        m->tensors = calloc(m->capacity, sizeof(TensorInfo));
        m->num_tensors = 0;
        for (NSString *key in tensors) {
            NSDictionary *info = tensors[key];
            TensorInfo *t = &m->tensors[m->num_tensors];
            const char *name = [key UTF8String];
            t->name = strdup(name);
            t->offset = [info[@"offset"] unsignedLongLongValue];
            t->size = [info[@"size"] unsignedLongLongValue];
            NSArray *shape = info[@"shape"];
            t->ndim = (int)[shape count];
            for (int i = 0; i < t->ndim && i < 4; i++) {
                t->shape[i] = [shape[i] intValue];
            }
            const char *dtype = [info[@"dtype"] UTF8String];
            strncpy(t->dtype, dtype, 7);
            m->num_tensors++;
        }
        printf("[manifest] Loaded %d tensors from %s\n", m->num_tensors, json_path);
        return m;
    }
}
// Hash table for O(1) tensor lookup (replaces O(N) linear scan).
// FNV-1a hash, open addressing with linear probing.
#define TENSOR_HT_SIZE 8192  // power of 2, > 4x num_tensors (2092)
typedef struct {
    const char *key;     // tensor name (pointer into TensorInfo)
    TensorInfo *value;   // pointer to tensor info
} TensorHTEntry;
static TensorHTEntry tensor_ht[TENSOR_HT_SIZE];
static int tensor_ht_built = 0;
static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}
static void build_tensor_ht(TensorManifest *m) {
    if (tensor_ht_built) return;
    memset(tensor_ht, 0, sizeof(tensor_ht));
    for (int i = 0; i < m->num_tensors; i++) {
        uint32_t idx = fnv1a(m->tensors[i].name) & (TENSOR_HT_SIZE - 1);
        while (tensor_ht[idx].key) {
            idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
        }
        tensor_ht[idx].key = m->tensors[i].name;
        tensor_ht[idx].value = &m->tensors[i];
    }
    tensor_ht_built = 1;
}
static TensorInfo *find_tensor(TensorManifest *m, const char *name) {
    if (!tensor_ht_built) build_tensor_ht(m);
    uint32_t idx = fnv1a(name) & (TENSOR_HT_SIZE - 1);
    while (tensor_ht[idx].key) {
        if (strcmp(tensor_ht[idx].key, name) == 0) {
            return tensor_ht[idx].value;
        }
        idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
    }
    return NULL;
}
// ============================================================================
// Weight file: mmap'd binary blob
// ============================================================================
typedef struct {
    void *data;
    size_t size;
    TensorManifest *manifest;
} WeightFile;
static WeightFile *open_weights(const char *bin_path, const char *json_path) {
    // mmap the binary file
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n", bin_path, strerror(errno));
        return NULL;
    }
    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;
    void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
        return NULL;
    }
    // Advise sequential access
    madvise(data, size, MADV_SEQUENTIAL);
    TensorManifest *manifest = load_manifest(json_path);
    if (!manifest) {
        munmap(data, size);
        return NULL;
    }
    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = size;
    wf->manifest = manifest;
    printf("[weights] mmap'd %.2f GB from %s\n", size / 1e9, bin_path);
    return wf;
}
static void *get_tensor_ptr(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf->manifest, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return NULL;
    }
    return (char *)wf->data + t->offset;
}
static TensorInfo *get_tensor_info(WeightFile *wf, const char *name) {
    return find_tensor(wf->manifest, name);
}
// ============================================================================
// Vocabulary for token decoding
// ============================================================================
typedef struct {
    char **tokens;   // token_id -> UTF-8 string
    int *lengths;    // token_id -> byte length
    int num_tokens;
} Vocabulary;
static Vocabulary *load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open vocab %s\n", path);
        return NULL;
    }
    uint32_t num_entries, max_id;
    fread(&num_entries, 4, 1, f);
    fread(&max_id, 4, 1, f);
    Vocabulary *v = calloc(1, sizeof(Vocabulary));
    v->num_tokens = num_entries;
    v->tokens = calloc(num_entries, sizeof(char *));
    v->lengths = calloc(num_entries, sizeof(int));
    for (uint32_t i = 0; i < num_entries; i++) {
        uint16_t byte_len;
        fread(&byte_len, 2, 1, f);
        if (byte_len > 0) {
            v->tokens[i] = malloc(byte_len + 1);
            fread(v->tokens[i], 1, byte_len, f);
            v->tokens[i][byte_len] = '\0';
            v->lengths[i] = byte_len;
        }
    }
    fclose(f);
    printf("[vocab] Loaded %d tokens\n", num_entries);
    return v;
}
static const char *decode_token(Vocabulary *v, int token_id) {
    if (token_id < 0 || token_id >= v->num_tokens || !v->tokens[token_id]) {
        return "<unk>";
    }
    return v->tokens[token_id];
}
// ============================================================================
// Prompt tokens loader
// ============================================================================
typedef struct {
    uint32_t *ids;
    int count;
} PromptTokens;
typedef struct {
    int enabled;
    int capture_active;
    char dir[1024];
} TokenDumpState;
static TokenDumpState g_dump = {0};

typedef struct {
    int enabled;
    int capture_active;
    float *layer_hidden_out;  // NUM_LAYERS * HIDDEN_DIM
    float *layer_attn_out;    // NUM_LAYERS * HIDDEN_DIM
    float *layer_h_mid;       // NUM_LAYERS * HIDDEN_DIM
    float *layer_h_post;      // NUM_LAYERS * HIDDEN_DIM
} PrefillLayerCaptureState;
static PrefillLayerCaptureState g_prefill_capture = {0};

typedef struct {
    int enabled;
    float *h_mid;
    float *h_post;
    int *expert_indices;
    float *expert_weights;
    int actual_k;
} PrefillRouteCaptureState;
static PrefillRouteCaptureState g_prefill_route_capture = {0};

typedef struct {
    int enabled;
    int chunk_tokens;
    int chunk_open;
    uint8_t layer_seen[NUM_LAYERS][NUM_EXPERTS];
    uint16_t layer_unique[NUM_LAYERS];
    uint64_t chunk_count;
    uint64_t total_layer_chunks;
    uint64_t total_routes;
    uint64_t total_unique_routes;
    int max_unique_experts;
} PrefillMicrobatchScaffoldState;
static PrefillMicrobatchScaffoldState g_prefill_microbatch = {0};

static void prefill_capture_set_storage(float *hidden_storage,
                                        float *attn_storage,
                                        float *h_mid_storage,
                                        float *h_post_storage) {
    memset(&g_prefill_capture, 0, sizeof(g_prefill_capture));
    if (!hidden_storage) return;
    g_prefill_capture.enabled = 1;
    g_prefill_capture.layer_hidden_out = hidden_storage;
    g_prefill_capture.layer_attn_out = attn_storage;
    g_prefill_capture.layer_h_mid = h_mid_storage;
    g_prefill_capture.layer_h_post = h_post_storage;
}

static void prefill_capture_disable(void) {
    memset(&g_prefill_capture, 0, sizeof(g_prefill_capture));
}

static void prefill_capture_activate(void) {
    if (!g_prefill_capture.enabled) return;
    g_prefill_capture.capture_active = 1;
}

static void prefill_capture_finish(void) {
    g_prefill_capture.capture_active = 0;
}

static void prefill_capture_write_hidden_out(int layer_idx, const float *hidden_out) {
    if (!g_prefill_capture.enabled || !g_prefill_capture.capture_active ||
        !g_prefill_capture.layer_hidden_out || !hidden_out ||
        layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return;
    }
    memcpy(g_prefill_capture.layer_hidden_out + (size_t)layer_idx * HIDDEN_DIM,
           hidden_out, HIDDEN_DIM * sizeof(float));
}

static void prefill_capture_write_attn_out(int layer_idx, const float *attn_out, size_t count) {
    if (!g_prefill_capture.enabled || !g_prefill_capture.capture_active ||
        !g_prefill_capture.layer_attn_out || !attn_out ||
        layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return;
    }
    size_t n = count;
    if (n > HIDDEN_DIM) n = HIDDEN_DIM;
    float *dst = g_prefill_capture.layer_attn_out + (size_t)layer_idx * HIDDEN_DIM;
    memset(dst, 0, HIDDEN_DIM * sizeof(float));
    memcpy(dst, attn_out, n * sizeof(float));
}

static void prefill_capture_write_h_mid(int layer_idx, const float *h_mid) {
    if (!g_prefill_capture.enabled || !g_prefill_capture.capture_active ||
        !g_prefill_capture.layer_h_mid || !h_mid ||
        layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return;
    }
    memcpy(g_prefill_capture.layer_h_mid + (size_t)layer_idx * HIDDEN_DIM,
           h_mid, HIDDEN_DIM * sizeof(float));
}

static void prefill_capture_write_h_post(int layer_idx, const float *h_post) {
    if (!g_prefill_capture.enabled || !g_prefill_capture.capture_active ||
        !g_prefill_capture.layer_h_post || !h_post ||
        layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return;
    }
    memcpy(g_prefill_capture.layer_h_post + (size_t)layer_idx * HIDDEN_DIM,
           h_post, HIDDEN_DIM * sizeof(float));
}

static void prefill_route_capture_enable(float *h_mid, float *h_post,
                                         int *expert_indices, float *expert_weights) {
    memset(&g_prefill_route_capture, 0, sizeof(g_prefill_route_capture));
    if (!h_mid || !h_post || !expert_indices || !expert_weights) return;
    g_prefill_route_capture.enabled = 1;
    g_prefill_route_capture.h_mid = h_mid;
    g_prefill_route_capture.h_post = h_post;
    g_prefill_route_capture.expert_indices = expert_indices;
    g_prefill_route_capture.expert_weights = expert_weights;
}

static void prefill_route_capture_disable(void) {
    memset(&g_prefill_route_capture, 0, sizeof(g_prefill_route_capture));
}

static void prefill_microbatch_scaffold_disable(void) {
    memset(&g_prefill_microbatch, 0, sizeof(g_prefill_microbatch));
}

static void prefill_microbatch_scaffold_begin(int chunk_tokens) {
    memset(&g_prefill_microbatch, 0, sizeof(g_prefill_microbatch));
    if (chunk_tokens <= 0) return;
    g_prefill_microbatch.enabled = 1;
    g_prefill_microbatch.chunk_tokens = chunk_tokens;
}

static void prefill_microbatch_scaffold_begin_chunk(void) {
    if (!g_prefill_microbatch.enabled) return;
    memset(g_prefill_microbatch.layer_seen, 0, sizeof(g_prefill_microbatch.layer_seen));
    memset(g_prefill_microbatch.layer_unique, 0, sizeof(g_prefill_microbatch.layer_unique));
    g_prefill_microbatch.chunk_open = 1;
    g_prefill_microbatch.chunk_count++;
}

static void prefill_microbatch_scaffold_note_route(int layer_idx, const int *expert_indices, int K) {
    if (!g_prefill_microbatch.enabled || !g_prefill_microbatch.chunk_open ||
        !expert_indices || layer_idx < 0 || layer_idx >= NUM_LAYERS || K <= 0) {
        return;
    }
    g_prefill_microbatch.total_routes += (uint64_t)K;
    for (int k = 0; k < K; k++) {
        int expert_idx = expert_indices[k];
        if (expert_idx < 0 || expert_idx >= NUM_EXPERTS) continue;
        if (!g_prefill_microbatch.layer_seen[layer_idx][expert_idx]) {
            g_prefill_microbatch.layer_seen[layer_idx][expert_idx] = 1;
            g_prefill_microbatch.layer_unique[layer_idx]++;
        }
    }
}

static void prefill_microbatch_scaffold_end_chunk(void) {
    if (!g_prefill_microbatch.enabled || !g_prefill_microbatch.chunk_open) return;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        g_prefill_microbatch.total_unique_routes += g_prefill_microbatch.layer_unique[layer];
        g_prefill_microbatch.total_layer_chunks++;
        if (g_prefill_microbatch.layer_unique[layer] > g_prefill_microbatch.max_unique_experts) {
            g_prefill_microbatch.max_unique_experts = g_prefill_microbatch.layer_unique[layer];
        }
    }
    g_prefill_microbatch.chunk_open = 0;
}
static PromptTokens *load_prompt_tokens(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    fread(&pt->count, 4, 1, f);
    pt->ids = malloc(pt->count * sizeof(uint32_t));
    fread(pt->ids, 4, pt->count, f);
    fclose(f);
    return pt;
}
static int dump_is_active(void) {
    return g_dump.enabled && g_dump.capture_active;
}
static int dump_ensure_dir(const char *path) {
    if (!path || !path[0]) return -1;
    if (mkdir(path, 0755) == 0) return 0;
    if (errno == EEXIST) return 0;
    fprintf(stderr, "WARNING: failed to create dump dir %s: %s\n", path, strerror(errno));
    return -1;
}
static void dump_write_bytes(const char *filename, const void *data, size_t size) {
    if (!g_dump.enabled) return;
    char path[1400];
    snprintf(path, sizeof(path), "%s/%s", g_dump.dir, filename);
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "WARNING: failed to open dump file %s: %s\n", path, strerror(errno));
        return;
    }
    fwrite(data, 1, size, f);
    fclose(f);
}
static void dump_write_f32(const char *filename, const float *data, size_t count) {
    dump_write_bytes(filename, data, count * sizeof(float));
}
static void dump_write_i32(const char *filename, const int32_t *data, size_t count) {
    dump_write_bytes(filename, data, count * sizeof(int32_t));
}
static void dump_write_layer_f32(int layer_idx, const char *stage, const float *data, size_t count) {
    if (!dump_is_active()) return;
    char filename[256];
    snprintf(filename, sizeof(filename), "layer_%02d_%s.f32", layer_idx, stage);
    dump_write_f32(filename, data, count);
}
static void dump_write_layer_i32(int layer_idx, const char *stage, const int32_t *data, size_t count) {
    if (!dump_is_active()) return;
    char filename[256];
    snprintf(filename, sizeof(filename), "layer_%02d_%s.i32", layer_idx, stage);
    dump_write_i32(filename, data, count);
}
static void dump_begin(const char *dir, const PromptTokens *pt,
                       int target_prompt_index, int target_pos,
                       int K, const char *model_path) {
    memset(&g_dump, 0, sizeof(g_dump));
    if (!dir || !dir[0]) return;
    snprintf(g_dump.dir, sizeof(g_dump.dir), "%s", dir);
    if (dump_ensure_dir(g_dump.dir) != 0) return;
    g_dump.enabled = 1;
    char meta_path[1400];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.json", g_dump.dir);
    FILE *f = fopen(meta_path, "w");
    if (!f) {
        fprintf(stderr, "WARNING: failed to open %s: %s\n", meta_path, strerror(errno));
        g_dump.enabled = 0;
        return;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"capture\": \"prefill_last\",\n");
    fprintf(f, "  \"model_path\": \"%s\",\n", model_path ? model_path : "");
    fprintf(f, "  \"hidden_dim\": %d,\n", HIDDEN_DIM);
    fprintf(f, "  \"num_layers\": %d,\n", NUM_LAYERS);
    fprintf(f, "  \"num_attention_heads\": %d,\n", NUM_ATTN_HEADS);
    fprintf(f, "  \"num_kv_heads\": %d,\n", NUM_KV_HEADS);
    fprintf(f, "  \"head_dim\": %d,\n", HEAD_DIM);
    fprintf(f, "  \"vocab_size\": %d,\n", VOCAB_SIZE);
    fprintf(f, "  \"num_experts\": %d,\n", NUM_EXPERTS);
    fprintf(f, "  \"moe_intermediate\": %d,\n", MOE_INTERMEDIATE);
    fprintf(f, "  \"group_size\": %d,\n", GROUP_SIZE);
    fprintf(f, "  \"k\": %d,\n", K);
    fprintf(f, "  \"target_prompt_index\": %d,\n", target_prompt_index);
    fprintf(f, "  \"target_pos\": %d,\n", target_pos);
    fprintf(f, "  \"prompt_tokens\": [");
    for (int i = 0; i < pt->count; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%u", pt->ids[i]);
    }
    fprintf(f, "]\n");
    fprintf(f, "}\n");
    fclose(f);
}
static void dump_activate_capture(void) {
    if (!g_dump.enabled) return;
    g_dump.capture_active = 1;
}
static void dump_finish_capture(void) {
    g_dump.capture_active = 0;
}
static void dump_write_result(const float *final_hidden, const float *logits, int next_token) {
    if (!dump_is_active()) return;
    dump_write_f32("final_norm_hidden.f32", final_hidden, HIDDEN_DIM);
    dump_write_f32("logits.f32", logits, VOCAB_SIZE);
    int32_t tok = next_token;
    dump_write_i32("next_token.i32", &tok, 1);
}
// ============================================================================
// C BPE tokenizer (replaces Python encode_prompt.py)
// ============================================================================
#define TOKENIZER_IMPL
#include "tokenizer.h"
static bpe_tokenizer g_tokenizer;
static int g_tokenizer_loaded = 0;
static char g_tokenizer_path[1024] = {0};
static void set_tokenizer_path(const char *path) {
    if (!path || !path[0]) {
        g_tokenizer_path[0] = '\0';
        return;
    }
    snprintf(g_tokenizer_path, sizeof(g_tokenizer_path), "%s", path);
}
static void init_tokenizer(void) {
    if (g_tokenizer_loaded) return;
    const char *paths[4];
    int path_count = 0;
    if (g_tokenizer_path[0]) paths[path_count++] = g_tokenizer_path;
    paths[path_count++] = "tokenizer.bin";
    paths[path_count++] = "metal_infer/tokenizer.bin";
    paths[path_count] = NULL;
    for (int i = 0; i < path_count; i++) {
        if (access(paths[i], R_OK) == 0) {
            if (bpe_load(&g_tokenizer, paths[i]) == 0) {
                g_tokenizer_loaded = 1;
                return;
            }
        }
    }
    fprintf(stderr, "WARNING: tokenizer.bin not found, tokenization will fail\n");
}
static PromptTokens *encode_prompt_text_to_tokens(const char *text) {
    init_tokenizer();
    if (!g_tokenizer_loaded) return NULL;
    // Allocate output buffer (generous: 4 tokens per character worst case)
    int max_ids = (int)strlen(text) * 4 + 256;
    uint32_t *ids = malloc(max_ids * sizeof(uint32_t));
    if (!ids) return NULL;
    int n = bpe_encode(&g_tokenizer, text, ids, max_ids);
    if (n < 0) { free(ids); return NULL; }
    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = ids;
    pt->count = n;
    fprintf(stderr, "Tokens (%d): [", n);
    for (int i = 0; i < n && i < 20; i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%u", ids[i]);
    }
    if (n > 20) fprintf(stderr, ", ...");
    fprintf(stderr, "]\n");
    return pt;
}
// ============================================================================
// CPU computation kernels
// ============================================================================
// 4-bit dequant matvec: out[out_dim] = W * x[in_dim]
// W is stored as packed uint32 (8 x 4-bit values per uint32)
// scales/biases are bfloat16 per group
static void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;
    int packed_cols = in_dim / 8;
    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;
        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias = bf16_to_f32(b_row[g]);
            int base_packed = g * packed_per_group;
            int base_x = g * group_size;
            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                int x_base = base_x + p * 8;
                for (int n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    acc += ((float)nibble * scale + bias) * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}
// RMS normalization: out = x * w / rms(x)
static void cpu_rms_norm(const float *x, const uint16_t *w_bf16, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < dim; i++) {
        float weight = bf16_to_f32(w_bf16[i]);
        out[i] = x[i] * inv_rms * weight;
    }
}
static void cpu_rms_norm_inplace(const uint16_t *w_bf16, float *x, int dim, float eps) {
    if (!w_bf16) return;
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        x[i] = x[i] * inv_rms * bf16_to_f32(w_bf16[i]);
    }
}
// SwiGLU: out = silu(gate) * up
static void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}
// Sigmoid
static float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
// Softmax over a vector
static void cpu_softmax(float *x, int dim) {
    float max_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim; i++) {
        x[i] *= inv_sum;
    }
}
// Top-K: find K largest indices from scores[dim]
static void cpu_topk(const float *scores, int dim, int K, int *indices, float *values) {
    // Simple selection sort for small K
    // Initialize with -inf
    for (int k = 0; k < K; k++) {
        values[k] = -1e30f;
        indices[k] = 0;
    }
    for (int i = 0; i < dim; i++) {
        // Check if this score beats the smallest in our top-K
        int min_k = 0;
        for (int k = 1; k < K; k++) {
            if (values[k] < values[min_k]) min_k = k;
        }
        if (scores[i] > values[min_k]) {
            values[min_k] = scores[i];
            indices[min_k] = i;
        }
    }
}
// Normalize top-K weights to sum to 1
static void cpu_normalize_weights(float *weights, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += weights[k];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int k = 0; k < K; k++) weights[k] *= inv;
    }
}
static void cpu_route_minimax(
    float *router_logits,
    const float *e_score_correction_bias,
    int dim,
    int K,
    float *choice_scores_scratch,
    int *indices,
    float *weights
) {
    for (int i = 0; i < dim; i++) {
        float routing_weight = cpu_sigmoid(router_logits[i]);
        router_logits[i] = routing_weight;
        choice_scores_scratch[i] = routing_weight +
            (e_score_correction_bias ? e_score_correction_bias[i] : 0.0f);
    }
    cpu_topk(choice_scores_scratch, dim, K, indices, weights);
    for (int k = 0; k < K; k++) {
        weights[k] = router_logits[indices[k]];
    }
    cpu_normalize_weights(weights, K);
}
// Element-wise add: dst += src
__attribute__((unused))
static void cpu_vec_add(float *dst, const float *src, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += src[i];
}
// Element-wise multiply-add: dst += scale * src
static void cpu_vec_madd(float *dst, const float *src, float scale, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += scale * src[i];
}
// Element-wise multiply: dst = a * b
__attribute__((unused))
static void cpu_vec_mul(float *dst, const float *a, const float *b, int dim) {
    for (int i = 0; i < dim; i++) dst[i] = a[i] * b[i];
}
// Copy
static void cpu_vec_copy(float *dst, const float *src, int dim) {
    memcpy(dst, src, dim * sizeof(float));
}
// Zero
__attribute__((unused))
static void cpu_vec_zero(float *dst, int dim) {
    memset(dst, 0, dim * sizeof(float));
}
// Argmax
static int cpu_argmax(const float *x, int dim) {
    int best = 0;
    float best_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    return best;
}
// SiLU activation
static void cpu_silu(float *x, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}
// Conv1d depthwise: one step (for incremental inference)
// Input: conv_state[kernel_size-1][channels] + new_input[channels]
// Output: result[channels]
// Weight: [channels, kernel_size, 1] stored as bf16
// This is a depthwise conv1d: each channel is independent
static void cpu_conv1d_step(
    const float *conv_state,    // [(kernel_size-1) * channels] row-major
    const float *new_input,     // [channels]
    const uint16_t *weight_bf16, // [channels * kernel_size] flattened
    float *out,                 // [channels]
    int channels,
    int kernel_size
) {
    // For each channel, compute dot product of [conv_state..., new_input] with weight
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        // Process previous states from conv_state
        for (int k = 0; k < kernel_size - 1; k++) {
            float w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }
        // Process new input (last position in kernel)
        float w = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w;
        out[c] = acc;
    }
    // Apply SiLU
    cpu_silu(out, channels);
}
// ============================================================================
// Metal context for GPU-accelerated matmuls
// ============================================================================
// Maximum number of batched matmul output slots.
// Used for encoding multiple matmuls into one command buffer.
#define MAX_BATCH_SLOTS 8
typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> matvec_v3;
    id<MTLComputePipelineState> matvec_v5;  // LUT dequant variant
    id<MTLComputePipelineState> matvec_fast;  // for in_dim > 4096
    id<MTLComputePipelineState> matvec_2bit;  // 2-bit expert dequant kernel
    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
    id<MTLComputePipelineState> rms_norm_apply_bf16;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> swiglu;
    // GPU attention pipelines
    id<MTLComputePipelineState> attn_scores_pipe;
    id<MTLComputePipelineState> attn_softmax_pipe;
    id<MTLComputePipelineState> attn_values_pipe;
    id<MTLComputePipelineState> sigmoid_gate_pipe;
    // Reusable buffers for attention matmuls
    id<MTLBuffer> buf_input;     // input vector [HIDDEN_DIM or max projection input]
    id<MTLBuffer> buf_output;    // output vector [max projection output]
    id<MTLBuffer> wf_buf;        // the mmap'd weight file as a Metal buffer
    // Batched matmul output slots (preallocated, reused across dispatches)
    id<MTLBuffer> batch_out[MAX_BATCH_SLOTS];
    // Reusable buffers for expert computation (avoids per-expert alloc)
    // Legacy single-expert buffers (kept for gpu_expert_forward compat)
    id<MTLBuffer> buf_expert_data;   // holds one expert's packed weights (EXPERT_SIZE bytes)
    id<MTLBuffer> buf_expert_input;  // h_post input [HIDDEN_DIM floats]
    id<MTLBuffer> buf_expert_gate;   // gate_proj output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_up;     // up_proj output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_act;    // SwiGLU output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_out;    // down_proj output [HIDDEN_DIM floats]
    // Multi-expert buffers: K independent sets so all experts can be encoded
    // into a SINGLE command buffer (no per-expert commit+wait).
    // Each expert k uses slot [k].
    // Double-buffered: set A (data) for GPU compute, set B (data_B) for background pread.
    // Gate/up/act/out only need one set (GPU uses them after pread completes).
    #define MAX_K 8
    id<MTLBuffer> buf_multi_expert_data[MAX_K];   // [EXPERT_SIZE bytes] each — buffer set A
    id<MTLBuffer> buf_multi_expert_data_B[MAX_K]; // [EXPERT_SIZE bytes] each — buffer set B (prefetch)
    id<MTLBuffer> buf_multi_expert_gate[MAX_K];   // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_up[MAX_K];     // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_act[MAX_K];    // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_out[MAX_K];    // [HIDDEN_DIM floats]
    id<MTLBuffer> buf_multi_expert_input;         // [HIDDEN_DIM floats] (shared, read-only during dispatch)
    // Shared expert buffers for fused CMD2 (shared gate/up computed in CMD1,
    // SwiGLU + down_proj in CMD2 alongside routed experts)
    id<MTLBuffer> buf_shared_gate;   // [SHARED_INTERMEDIATE floats]
    id<MTLBuffer> buf_shared_up;     // [SHARED_INTERMEDIATE floats]
    id<MTLBuffer> buf_shared_act;    // [SHARED_INTERMEDIATE floats] (SwiGLU output)
    id<MTLBuffer> buf_shared_out;    // [HIDDEN_DIM floats] (down_proj output)
    // Fused o_proj+norm+routing buffers (eliminates 1 cmd buffer per layer)
    id<MTLBuffer> buf_residual;     // [HIDDEN_DIM floats] holds residual for GPU add
    id<MTLBuffer> buf_h_mid;        // [HIDDEN_DIM floats] residual+oproj result
    id<MTLBuffer> buf_sum_sq;       // [1 float] for RMS norm reduction
    // GPU attention buffers (for full attention layers)
    #define NUM_FULL_ATTN_LAYERS NUM_LAYERS
    id<MTLBuffer> buf_kv_k[NUM_FULL_ATTN_LAYERS];  // K cache per full-attn layer
    id<MTLBuffer> buf_kv_v[NUM_FULL_ATTN_LAYERS];  // V cache per full-attn layer
    id<MTLBuffer> buf_attn_q;       // [NUM_ATTN_HEADS * HEAD_DIM floats] all query heads
    id<MTLBuffer> buf_attn_scores;  // [NUM_ATTN_HEADS * MAX_SEQ_LEN floats] all heads' scores
    id<MTLBuffer> buf_attn_out;     // [NUM_ATTN_HEADS * HEAD_DIM floats] full attention output
    id<MTLBuffer> buf_attn_gate;    // [NUM_ATTN_HEADS * HEAD_DIM floats] sigmoid gate
    // CMD3 GPU-side combine buffers (weighted_sum + residual + norm on GPU)
    id<MTLComputePipelineState> moe_combine_residual;  // fused combine kernel
    id<MTLBuffer> buf_moe_hidden;     // [HIDDEN_DIM floats] GPU combine output (hidden state)
    id<MTLBuffer> buf_combine_params; // [10 floats] expert weights[8] + shared_gate_score + padding
    id<MTLBuffer> buf_cmd3_sum_sq;    // [1 float] for RMS norm reduction in CMD3
    // Shared event for CPU-GPU synchronization (async pipeline)
    id<MTLSharedEvent> pipeline_event;   // CPU signals when buf_input is ready
    uint64_t event_value;                // monotonically increasing event counter
    // GPU delta-net (gated_delta_net_step) and conv1d pipelines
    id<MTLComputePipelineState> delta_net_step;  // gated_delta_net_step kernel
    id<MTLComputePipelineState> conv1d_step;     // conv1d_step kernel
    id<MTLComputePipelineState> rms_norm_qk;     // per-head RMS normalize for q and k
    id<MTLComputePipelineState> compute_decay_beta; // g_decay and beta_gate for delta-net
    id<MTLComputePipelineState> gated_rms_norm;  // z-gated output normalization
    // Persistent GPU state buffers for linear attention layers
    #define NUM_LINEAR_LAYERS 1
    id<MTLBuffer> buf_delta_state[NUM_LINEAR_LAYERS];   // [64*128*128] float per layer
    id<MTLBuffer> buf_conv_state[NUM_LINEAR_LAYERS];     // [3*12288] float per layer
    // Scratch buffers for delta-net inputs/outputs
    id<MTLBuffer> buf_delta_q;        // [2048] float
    id<MTLBuffer> buf_delta_k;        // [2048] float
    id<MTLBuffer> buf_delta_v;        // [8192] float
    id<MTLBuffer> buf_delta_g_decay;  // [64] float
    id<MTLBuffer> buf_delta_beta;     // [64] float
    id<MTLBuffer> buf_delta_output;   // [8192] float
    id<MTLBuffer> buf_conv_input;     // [12288] float
    id<MTLBuffer> buf_conv_output;    // [12288] float
} MetalCtx;
static MetalCtx *g_metal = NULL;
static MetalCtx *metal_setup(void) {
    MetalCtx *ctx = calloc(1, sizeof(MetalCtx));
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device\n");
        free(ctx); return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);
    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: No command queue\n");
        free(ctx); return NULL;
    }
    // Compile shaders from source
    NSError *error = nil;
    NSArray *paths = @[@"shaders.metal", @"metal_infer/shaders.metal"];
    NSString *src = nil;
    for (NSString *p in paths) {
        src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
        if (src) break;
    }
    if (!src) {
        fprintf(stderr, "ERROR: Cannot find shaders.metal\n");
        free(ctx); return NULL;
    }
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_1;
    double t0 = now_ms();
    ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
    if (!ctx->library) {
        fprintf(stderr, "ERROR: Shader compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        free(ctx); return NULL;
    }
    printf("[metal] Shader compile: %.0f ms\n", now_ms() - t0);
    // Create pipelines
    id<MTLComputePipelineState> (^makePipe)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) { fprintf(stderr, "ERROR: shader '%s' not found\n", [name UTF8String]); return (id<MTLComputePipelineState>)nil; }
        NSError *e2 = nil;
        id<MTLComputePipelineState> ps = [ctx->device newComputePipelineStateWithFunction:fn error:&e2];
        if (!ps) { fprintf(stderr, "ERROR: pipeline '%s': %s\n", [name UTF8String], [[e2 localizedDescription] UTF8String]); }
        return ps;
    };
    ctx->matvec_v3     = makePipe(@"dequant_matvec_4bit_v3");
    ctx->matvec_v5     = makePipe(@"dequant_matvec_4bit_v5");  // LUT variant (no uint→float conversions)
    ctx->matvec_fast   = makePipe(@"dequant_matvec_4bit_fast");
    ctx->matvec_2bit   = makePipe(@"dequant_matvec_2bit");
    ctx->rms_norm_sum  = makePipe(@"rms_norm_sum_sq");
    ctx->rms_norm_apply = makePipe(@"rms_norm_apply");
    ctx->rms_norm_apply_bf16 = makePipe(@"rms_norm_apply_bf16");
    ctx->residual_add  = makePipe(@"residual_add");
    ctx->swiglu        = makePipe(@"swiglu_fused");
    ctx->attn_scores_pipe  = makePipe(@"attn_scores_batched");
    ctx->attn_softmax_pipe = makePipe(@"attn_softmax_batched");
    ctx->attn_values_pipe  = makePipe(@"attn_values_batched");
    ctx->sigmoid_gate_pipe = makePipe(@"sigmoid_gate");
    ctx->moe_combine_residual = makePipe(@"moe_combine_residual");
    ctx->delta_net_step    = makePipe(@"gated_delta_net_step");
    ctx->conv1d_step       = makePipe(@"conv1d_step");
    ctx->rms_norm_qk       = makePipe(@"rms_norm_qk");
    ctx->compute_decay_beta = makePipe(@"compute_decay_beta");
    ctx->gated_rms_norm    = makePipe(@"gated_rms_norm");
    if (!ctx->moe_combine_residual) fprintf(stderr, "[metal] WARNING: moe_combine_residual pipeline failed\n");
    if (!ctx->delta_net_step) fprintf(stderr, "[metal] WARNING: gated_delta_net_step pipeline failed (CPU fallback)\n");
    if (!ctx->conv1d_step)    fprintf(stderr, "[metal] WARNING: conv1d_step pipeline failed (CPU fallback)\n");
    if (!ctx->rms_norm_qk)       fprintf(stderr, "[metal] WARNING: rms_norm_qk pipeline failed (CPU fallback)\n");
    if (!ctx->compute_decay_beta) fprintf(stderr, "[metal] WARNING: compute_decay_beta pipeline failed (CPU fallback)\n");
    if (!ctx->gated_rms_norm)     fprintf(stderr, "[metal] WARNING: gated_rms_norm pipeline failed (CPU fallback)\n");
    if (!ctx->matvec_v3 || !ctx->matvec_fast) {
        fprintf(stderr, "ERROR: Required Metal pipeline missing\n");
        free(ctx); return NULL;
    }
    // Allocate reusable buffers (large enough for biggest projection)
    // Q proj output is 16384 floats, lm_head output is 248320 floats
    // o_proj input is 8192, linear attn out_proj input is 8192
    size_t max_out = VOCAB_SIZE * sizeof(float);  // lm_head is largest
    size_t max_in = LINEAR_TOTAL_VALUE * sizeof(float);  // 8192 floats (linear_attn out_proj)
    if (max_in < (size_t)(NUM_ATTN_HEADS * HEAD_DIM) * sizeof(float)) {
        max_in = (size_t)(NUM_ATTN_HEADS * HEAD_DIM) * sizeof(float);  // o_proj input = 8192
    }
    ctx->buf_input  = [ctx->device newBufferWithLength:max_in  options:MTLResourceStorageModeShared];
    ctx->buf_output = [ctx->device newBufferWithLength:max_out options:MTLResourceStorageModeShared];
    // Batched matmul output slots — each large enough for the biggest projection
    // q_proj = 16384 floats, qkv_proj = 12288, z_proj = 8192, o_proj = 4096
    // lm_head (248320) uses buf_output directly, not batched.
    {
        size_t slot_size = (size_t)(NUM_ATTN_HEADS * HEAD_DIM * 2) * sizeof(float);  // 16384 floats
        if (slot_size < (size_t)LINEAR_CONV_DIM * sizeof(float))
            slot_size = (size_t)LINEAR_CONV_DIM * sizeof(float);  // 12288 floats
        for (int i = 0; i < MAX_BATCH_SLOTS; i++) {
            ctx->batch_out[i] = [ctx->device newBufferWithLength:slot_size
                                                         options:MTLResourceStorageModeShared];
        }
    }
    // Expert computation buffers (reused across all experts and layers)
    ctx->buf_expert_data  = [ctx->device newBufferWithLength:EXPERT_SIZE
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_input = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_gate  = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_up    = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_act   = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_out   = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    // Multi-expert buffers: K independent slots (double-buffered data)
    // Expert data buffers use 2MB-aligned backing memory for DMA efficiency.
    // The pread DMA controller transfers 3.6x faster with 2MB alignment vs 16KB.
    ctx->buf_multi_expert_input = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    size_t expert_alloc_size = (EXPERT_SIZE + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);  // round up to 2MB
    for (int k = 0; k < MAX_K; k++) {
        // 2MB-aligned allocation for optimal DMA throughput
        void *aligned_data = NULL, *aligned_data_b = NULL;
        posix_memalign(&aligned_data,   2*1024*1024, expert_alloc_size);
        posix_memalign(&aligned_data_b, 2*1024*1024, expert_alloc_size);
        memset(aligned_data, 0, expert_alloc_size);
        memset(aligned_data_b, 0, expert_alloc_size);
        ctx->buf_multi_expert_data[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data
                                                                       length:expert_alloc_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
        ctx->buf_multi_expert_data_B[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data_b
                                                                         length:expert_alloc_size
                                                                        options:MTLResourceStorageModeShared
                                                                    deallocator:nil];
        ctx->buf_multi_expert_gate[k] = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_up[k]   = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_act[k]  = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_out[k]  = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
    }
    // Shared expert buffers (for fused CMD2)
    ctx->buf_shared_gate = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up   = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_act  = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_out  = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    // Fused o_proj+norm+routing buffers
    ctx->buf_residual = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_h_mid    = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_sum_sq   = [ctx->device newBufferWithLength:sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    // CMD3 GPU-side combine buffers
    ctx->buf_moe_hidden    = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_combine_params = [ctx->device newBufferWithLength:10 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    ctx->buf_cmd3_sum_sq    = [ctx->device newBufferWithLength:sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    // GPU attention buffers (disabled for the first MiniMax bring-up)
    if (gpu_full_attn_enabled) {
        size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;  // 512
        size_t kv_cache_size = GPU_KV_SEQ * kv_dim * sizeof(float);
        for (int i = 0; i < NUM_FULL_ATTN_LAYERS; i++) {
            ctx->buf_kv_k[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
            ctx->buf_kv_v[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
        }
        ctx->buf_attn_q      = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_scores = [ctx->device newBufferWithLength:(size_t)NUM_ATTN_HEADS * GPU_KV_SEQ * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_out    = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_gate   = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        printf("[metal] GPU attention buffers: %d KV caches (%.1f MB each), scores buf %.1f MB\n",
               NUM_FULL_ATTN_LAYERS, kv_cache_size / 1e6,
               (double)(NUM_ATTN_HEADS * MAX_SEQ_LEN * sizeof(float)) / 1e6);
    }
    // Persistent GPU state buffers for delta-net (linear attention layers)
    if (gpu_linear_attn_enabled && ctx->delta_net_step) {
        for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
            ctx->buf_delta_state[i] = [ctx->device newBufferWithLength:64*128*128*sizeof(float)
                                                               options:MTLResourceStorageModeShared];
            memset([ctx->buf_delta_state[i] contents], 0, 64*128*128*sizeof(float));
            ctx->buf_conv_state[i] = [ctx->device newBufferWithLength:3*12288*sizeof(float)
                                                              options:MTLResourceStorageModeShared];
            memset([ctx->buf_conv_state[i] contents], 0, 3*12288*sizeof(float));
        }
        // Scratch buffers for delta-net inputs/outputs (allocated once, reused)
        ctx->buf_delta_q       = [ctx->device newBufferWithLength:2048*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_k       = [ctx->device newBufferWithLength:2048*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_v       = [ctx->device newBufferWithLength:8192*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_g_decay = [ctx->device newBufferWithLength:64*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_beta    = [ctx->device newBufferWithLength:64*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_output  = [ctx->device newBufferWithLength:8192*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_conv_input    = [ctx->device newBufferWithLength:12288*sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_conv_output   = [ctx->device newBufferWithLength:12288*sizeof(float) options:MTLResourceStorageModeShared];
        printf("[metal] Delta-net GPU buffers: %d layers (%.1f MB state + %.1f MB scratch)\n",
               NUM_LINEAR_LAYERS,
               NUM_LINEAR_LAYERS * (64*128*128*4 + 3*12288*4) / 1e6,
               (2048+2048+8192+64+64+8192+12288+12288) * 4 / 1e6);
    }
    // Create shared event for CPU-GPU async pipeline
    ctx->pipeline_event = [ctx->device newSharedEvent];
    ctx->event_value = 0;
    printf("[metal] Inference pipelines ready (multi-expert[%d] + shared buffers allocated)\n", MAX_K);
    return ctx;
}
// Reset delta-net and conv GPU state buffers (call at start of new generation)
static void reset_delta_net_state(void) {
    if (!g_metal || !g_metal->delta_net_step) return;
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (g_metal->buf_delta_state[i])
            memset([g_metal->buf_delta_state[i] contents], 0, 64*128*128*sizeof(float));
        if (g_metal->buf_conv_state[i])
            memset([g_metal->buf_conv_state[i] contents], 0, 3*12288*sizeof(float));
    }
}
// Wrap the mmap'd weight file as a Metal buffer (zero-copy on unified memory)
// mmap returns page-aligned addresses, Metal requires the same.
// On Apple Silicon, page size is 16KB.
static void metal_set_weights(MetalCtx *ctx, void *data, size_t size) {
    // Round size up to page boundary (16KB)
    size_t page_size = 16384;
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);
    ctx->wf_buf = [ctx->device newBufferWithBytesNoCopy:data
                                                 length:aligned_size
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (!ctx->wf_buf) {
        fprintf(stderr, "WARNING: Cannot wrap weight file as Metal buffer (size=%.2f GB)\n",
                size / 1e9);
        fprintf(stderr, "  data=%p, aligned_size=%zu -- GPU matmul will fall back to CPU\n",
                data, aligned_size);
    } else {
        printf("[metal] Weight file wrapped as Metal buffer (%.2f GB)\n",
               aligned_size / 1e9);
    }
}
// GPU dequant matvec: out[out_dim] = W_4bit * x[in_dim]
// W_packed, scales, biases are pointers into mmap'd weight file
// x_f32 is CPU float array, result written back to out_f32
//
// We wrap the ENTIRE mmap'd weight file as a single Metal buffer and use
// byte offsets to point each shader argument at the right tensor.
// This avoids per-tensor buffer creation and the page-alignment constraint.
static void gpu_dequant_matvec(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *x_f32, float *out_f32,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    // Copy input to Metal buffer
    memcpy([ctx->buf_input contents], x_f32, in_dim * sizeof(float));
    size_t o_size = (size_t)out_dim * sizeof(float);
    // Compute offsets into the mmap'd weight buffer
    NSUInteger w_off = (NSUInteger)((const char *)W_packed - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales   - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases   - (const char *)[ctx->wf_buf contents]);
    // Ensure output buffer is large enough
    id<MTLBuffer> o_buf = ctx->buf_output;
    if (o_size > [o_buf length]) {
        o_buf = [ctx->device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
    }
    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    // v3 shader uses x_shared[4096], so can only handle in_dim <= 4096
    // For larger in_dim (e.g. o_proj with in_dim=8192), use matvec_fast
    int use_v3 = (in_dim <= 4096);
    [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
    [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
    [enc setBuffer:o_buf        offset:0     atIndex:4];
    [enc setBytes:&out_dim      length:4     atIndex:5];
    [enc setBytes:&in_dim       length:4     atIndex:6];
    [enc setBytes:&group_size   length:4     atIndex:7];
    if (use_v3) {
        // v3: tiled threadgroups, 256 threads, 8 rows per TG
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        // fast: one threadgroup per output row, 64 threads per TG
        NSUInteger tg_size = 64;
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
    // Copy result back
    memcpy(out_f32, [o_buf contents], o_size);
}
// Wrapper: use GPU if available and weight buffer is set, CPU otherwise
static void fast_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    if (g_metal && g_metal->wf_buf) {
        gpu_dequant_matvec(g_metal, W, scales, biases, x, out,
                           (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)group_size);
    } else {
        cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
    }
}
// ============================================================================
// Batched GPU matmul: encode N independent matmuls sharing the same input
// into ONE command buffer, reducing dispatch overhead by N-1 round-trips.
// ============================================================================
typedef struct {
    const void *W;           // packed weights (pointer into mmap'd file)
    const void *scales;      // scales (pointer into mmap'd file)
    const void *biases;      // biases (pointer into mmap'd file)
    float *out_cpu;          // CPU output pointer (result copied here after GPU finishes)
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    int batch_slot;          // which batch_out[slot] to use for GPU output
} BatchMatvecSpec;
// Run N matmuls in a single command buffer. All share the same input vector.
// The input is copied once; all outputs go to preallocated batch_out slots.
static void gpu_batch_matvec(
    MetalCtx *ctx,
    const float *x_f32, uint32_t x_dim,  // shared input
    BatchMatvecSpec *specs, int num_specs
) {
    // Copy input once
    memcpy([ctx->buf_input contents], x_f32, x_dim * sizeof(float));
    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);
        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];
        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
    // Copy results back to CPU
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}
// ============================================================================
// Encode-only variants: add dispatches to an EXISTING command buffer.
// These do NOT commit — the caller batches multiple encode calls into one
// command buffer and commits once, eliminating per-dispatch overhead.
// ============================================================================
// Encode N matmuls into cmdbuf. Input must already be in ctx->buf_input.
static void gpu_encode_batch_matvec(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    BatchMatvecSpec *specs, int num_specs
) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);
        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];
        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }
}
// Copy batch results from GPU buffers back to CPU pointers.
static void gpu_flush_batch_results(MetalCtx *ctx, BatchMatvecSpec *specs, int num_specs) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}
// Encode a single matvec reading from buf_expert_act into buf_expert_out,
// using weight pointers into the mmap'd weight file.
// Used for shared expert down_proj which reads from a different input than
// the attention projections.
static void gpu_encode_dequant_matvec_with_io_bufs(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *scales, const void *biases,
    id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    NSUInteger w_off = (NSUInteger)((const char *)W      - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales  - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases  - (const char *)[ctx->wf_buf contents]);
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    int use_v3 = (in_dim <= 4096);
    [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf offset:b_off atIndex:2];
    [enc setBuffer:in_buf      offset:0     atIndex:3];
    [enc setBuffer:out_buf     offset:0     atIndex:4];
    [enc setBytes:&out_dim     length:4     atIndex:5];
    [enc setBytes:&in_dim      length:4     atIndex:6];
    [enc setBytes:&group_size  length:4     atIndex:7];
    if (use_v3) {
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc endEncoding];
}
// Encode one expert forward using multi-expert slot k.
// Expert data must already be in buf_multi_expert_data[k].
// Input must already be in buf_multi_expert_input.
static void gpu_encode_expert_forward_slot(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k  // slot index
) {
    ExpertLayout layout = active_expert_layout();
    id<MTLComputePipelineState> expert_pipe = g_use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;
    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;
    // gate_proj: data[k] -> gate[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:layout.gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:layout.gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:layout.gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj: data[k] -> up[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:layout.up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:layout.up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:layout.up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU: gate[k], up[k] -> act[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj: act[k] -> out[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:layout.down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:layout.down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:layout.down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}
// Encode one expert forward using explicit data buffer (for double buffering).
// Expert data must already be in data_buf.
// Input must already be in buf_multi_expert_input.
// Uses slot k's gate/up/act/out scratch buffers.
static void gpu_encode_expert_forward_slot_buf(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k,                  // slot index (for gate/up/act/out scratch)
    id<MTLBuffer> data_buf  // expert weight data buffer (from either set A or B)
) {
    ExpertLayout layout = active_expert_layout();
    id<MTLComputePipelineState> expert_pipe = g_use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;
    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;
    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:layout.gate_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:layout.gate_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:layout.gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:layout.up_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:layout.up_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:layout.up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:layout.down_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:layout.down_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:layout.down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}
// Batched expert encoding: encode K experts using 2 encoders per expert
// (gate+up fused, SwiGLU+down fused) + 2 for shared = K*2 + 2 encoders total.
// With K=4: 10 encoders (vs. old 4*K + 2 = 18 with per-operation encoding).
// Each expert gets its own encoder pair for GPU parallelism across experts.
// Within each encoder, gate+up (or SwiGLU+down) are serialized but share
// encoder creation overhead. Net win: fewer encoders, same parallelism.
static void gpu_encode_experts_batched(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int K,                       // number of experts to encode
    const int *valid,            // which experts are valid [MAX_K]
    id<MTLBuffer> __strong *expert_bufs   // per-expert weight data buffers [MAX_K]
) {
    // Select offsets and pipeline based on quantization mode
    ExpertLayout layout = active_expert_layout();
    id<MTLComputePipelineState> expert_pipe = g_use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;
    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;
    // 2-bit: packed_cols = in_dim/16, threadgroups = out_dim/8
    // 4-bit: packed_cols = in_dim/8,  threadgroups = out_dim/8
    // Threadgroup count is the same (based on out_dim), kernel handles packed_cols internally.
    uint32_t gate_up_tgs = (gate_up_out + 7) / 8;
    uint32_t down_tgs    = (down_out + 7) / 8;
    uint32_t swiglu_tgs  = (gate_up_out + 255) / 256;
    // Per-expert: Encoder A (gate+up), Encoder B (SwiGLU+down)
    // Separate encoders per expert enables GPU parallelism across experts.
    // Within each encoder, operations serialize (gate then up, SwiGLU then down).
    for (int k = 0; k < K; k++) {
        if (!valid[k]) continue;
        // Encoder A: gate_proj + up_proj (both read same input, write different outputs)
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            // gate_proj
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:layout.gate_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:layout.gate_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:layout.gate_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // up_proj (same encoder, serialized after gate — shares encoder overhead)
            [enc setBuffer:expert_bufs[k]                  offset:layout.up_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:layout.up_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:layout.up_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // Encoder B: SwiGLU + down_proj (SwiGLU depends on gate+up from Enc A)
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            // SwiGLU
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // down_proj (same encoder, serialized after SwiGLU)
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:layout.down_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:layout.down_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:layout.down_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
            [enc setBytes:&down_out length:4 atIndex:5];
            [enc setBytes:&down_in  length:4 atIndex:6];
            [enc setBytes:&gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }
}
// Encode one expert forward (gate+up+swiglu+down) into cmdbuf.
// Expert data must already be in buf_expert_data.
// Input must already be in buf_expert_input.
__attribute__((unused))
static void gpu_encode_expert_forward(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf
) {
    ExpertLayout layout = active_expert_layout();
    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;
    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data offset:layout.down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:layout.down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:layout.down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}
// Batched wrapper: takes N matmul specs sharing the same input, dispatches
// via GPU batch if available, otherwise falls back to CPU.
static void fast_batch_matvec(
    const float *x, uint32_t x_dim,
    BatchMatvecSpec *specs, int num_specs
) {
    if (g_metal && g_metal->wf_buf) {
        gpu_batch_matvec(g_metal, x, x_dim, specs, num_specs);
    } else {
        for (int i = 0; i < num_specs; i++) {
            BatchMatvecSpec *s = &specs[i];
            cpu_dequant_matvec(s->W, s->scales, s->biases, x, s->out_cpu,
                               s->out_dim, s->in_dim, s->group_size);
        }
    }
}
// ============================================================================
// GPU expert forward: gate+up matvec -> SwiGLU -> down matvec
// All 3 matmuls + activation in a single command buffer submission.
// Expert data is copied into a reusable Metal buffer.
// ============================================================================
// expert_data_already_in_buffer: if true, expert data is already in buf_expert_data
//   (pread'd directly into it), skip the copy.
__attribute__((unused))
static void gpu_expert_forward(
    MetalCtx *ctx,
    const void *expert_data,     // EXPERT_SIZE bytes (may be buf_expert_data contents)
    const float *h_post,         // [HIDDEN_DIM] input
    float *expert_out,           // [HIDDEN_DIM] output
    int expert_data_already_in_buffer
) {
    // Expert layout offsets — select based on quantization mode
    ExpertLayout layout = active_expert_layout();
    id<MTLComputePipelineState> expert_pipe = g_use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;
    // Copy expert weights into Metal buffer only if not already there
    if (!expert_data_already_in_buffer) {
        memcpy([ctx->buf_expert_data contents], expert_data, active_expert_size());
    }
    memcpy([ctx->buf_expert_input contents], h_post, HIDDEN_DIM * sizeof(float));
    uint32_t gate_up_out = MOE_INTERMEDIATE;  // 1024
    uint32_t gate_up_in  = HIDDEN_DIM;        // 4096
    uint32_t down_out    = HIDDEN_DIM;        // 4096
    uint32_t down_in     = MOE_INTERMEDIATE;  // 1024
    uint32_t gs          = GROUP_SIZE;        // 64
    // Build one command buffer with all 4 dispatches:
    // 1. gate_proj matvec (h_post -> gate_out)
    // 2. up_proj matvec (h_post -> up_out)
    // 3. SwiGLU (gate_out, up_out -> act_out)
    // 4. down_proj matvec (act_out -> expert_out)
    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    // --- Dispatch 1: gate_proj [4096] -> [1024] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // --- Dispatch 2: up_proj [4096] -> [1024] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:layout.up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // --- Dispatch 3: SwiGLU(gate, up) -> act ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // --- Dispatch 4: down_proj [1024] -> [4096] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data offset:layout.down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:layout.down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:layout.down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
    // Copy result back to CPU
    memcpy(expert_out, [ctx->buf_expert_out contents], HIDDEN_DIM * sizeof(float));
}
// ============================================================================
// Rotary position embedding (for full attention layers)
// ============================================================================
static void apply_rotary_emb(float *q, float *k, int pos, int num_heads, int num_kv_heads,
                              int head_dim, int rotary_dim) {
    // Apply RoPE to the first rotary_dim dimensions of each head
    // NON-TRADITIONAL (MLX default): pairs are (x[i], x[i + half_dim])
    // where half_dim = rotary_dim / 2
    int half = rotary_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half]  = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half]  = k0 * sin_a + k1 * cos_a;
        }
    }
}
// ============================================================================
// KV Cache for full attention layers
// ============================================================================
typedef struct {
    float *k_cache;  // [max_seq, num_kv_heads * head_dim]
    float *v_cache;  // [max_seq, num_kv_heads * head_dim]
    int len;         // current number of cached entries
} KVCache;
static KVCache *kv_cache_new(void) {
    KVCache *c = calloc(1, sizeof(KVCache));
    c->k_cache = calloc(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    c->v_cache = calloc(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    c->len = 0;
    return c;
}
static void kv_cache_free(KVCache *c) {
    if (c) {
        free(c->k_cache);
        free(c->v_cache);
        free(c);
    }
}
// ============================================================================
// Linear attention state (GatedDeltaNet recurrent state)
// ============================================================================
typedef struct {
    float *conv_state;  // [(kernel_size-1) * conv_dim] for conv1d
    float *ssm_state;   // [num_v_heads, head_v_dim, head_k_dim] recurrent state
} LinearAttnState;
static LinearAttnState *linear_attn_state_new(void) {
    LinearAttnState *s = calloc(1, sizeof(LinearAttnState));
    s->conv_state = calloc((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, sizeof(float));
    s->ssm_state = calloc(LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM, sizeof(float));
    return s;
}
static void linear_attn_state_free(LinearAttnState *s) {
    if (s) {
        free(s->conv_state);
        free(s->ssm_state);
        free(s);
    }
}
// ============================================================================
// Full attention layer forward (single token, incremental)
// ============================================================================
static int fa_debug_count = 0;
static float vec_rms(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum / n);
}
__attribute__((unused))
static void full_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,       // [HIDDEN_DIM] in/out
    KVCache *kv,
    int pos              // position in sequence
) {
    fa_debug_count++;
    int do_debug = 0;  // set to (fa_debug_count <= N) to enable debug
    char name[256];
    float *normed = malloc(HIDDEN_DIM * sizeof(float));
    float *residual = malloc(HIDDEN_DIM * sizeof(float));
    cpu_vec_copy(residual, hidden, HIDDEN_DIM);
    if (do_debug) {
        fprintf(stderr, "[FA-DBG] layer=%d pos=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, pos, vec_rms(hidden, HIDDEN_DIM),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }
    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
    if (do_debug) {
        fprintf(stderr, "[FA-DBG] normed_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(normed, HIDDEN_DIM), normed[0], normed[1], normed[2], normed[3], normed[4]);
    }
    // ---- QKV Projection ----
    int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM;
    int q_dim = NUM_ATTN_HEADS * HEAD_DIM;
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    float *q_proj_out = calloc(q_proj_dim, sizeof(float));
    float *k = calloc(kv_dim, sizeof(float));
    float *v = calloc(kv_dim, sizeof(float));
    // Batch Q/K/V projections into a single GPU command buffer
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    uint32_t *qw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", layer_idx);
    uint16_t *qs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", layer_idx);
    uint16_t *qb = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    uint32_t *kw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", layer_idx);
    uint16_t *ks = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", layer_idx);
    uint16_t *kb = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    uint32_t *vw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", layer_idx);
    uint16_t *vs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", layer_idx);
    uint16_t *vb = get_tensor_ptr(wf, name);
    // Batch Q/K/V into one command buffer (3 dispatches, 1 commit)
    if (qw && qs && qb && kw && ks && kb && vw && vs && vb) {
        BatchMatvecSpec qkv_specs[3] = {
            { qw, qs, qb, q_proj_out, (uint32_t)q_proj_dim, HIDDEN_DIM, GROUP_SIZE, 0 },
            { kw, ks, kb, k,          (uint32_t)kv_dim,     HIDDEN_DIM, GROUP_SIZE, 1 },
            { vw, vs, vb, v,          (uint32_t)kv_dim,     HIDDEN_DIM, GROUP_SIZE, 2 },
        };
        fast_batch_matvec(normed, HIDDEN_DIM, qkv_specs, 3);
    }
    if (do_debug) {
        fprintf(stderr, "[FA-DBG] q_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                q_proj_out[0], q_proj_out[1], q_proj_out[2], q_proj_out[3], q_proj_out[4]);
    }
    float *q = calloc(q_dim, sizeof(float));
    memcpy(q, q_proj_out, q_dim * sizeof(float));
    free(q_proj_out);
    if (do_debug) {
        fprintf(stderr, "[FA-DBG] v_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(v, kv_dim), v[0], v[1], v[2], v[3], v[4]);
    }
    // ---- Q/K RMSNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    uint16_t *qnorm_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    uint16_t *knorm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm_inplace(qnorm_w, q, q_dim, RMS_NORM_EPS);
    cpu_rms_norm_inplace(knorm_w, k, kv_dim, RMS_NORM_EPS);
    // ---- RoPE ----
    apply_rotary_emb(q, k, pos, NUM_ATTN_HEADS, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM);
    // ---- Update KV cache ----
    int cache_pos = kv->len;
    memcpy(kv->k_cache + cache_pos * kv_dim, k, kv_dim * sizeof(float));
    memcpy(kv->v_cache + cache_pos * kv_dim, v, kv_dim * sizeof(float));
    kv->len++;
    // ---- Scaled dot-product attention ----
    // GQA: NUM_ATTN_HEADS=32 heads, NUM_KV_HEADS=2 kv heads
    // Each group of 16 query heads shares 1 kv head
    int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float *attn_out = calloc(q_dim, sizeof(float));
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = q + h * HEAD_DIM;
        // Compute attention scores for all cached positions
        float *scores = malloc(kv->len * sizeof(float));
        for (int p = 0; p < kv->len; p++) {
            float *kp = kv->k_cache + p * kv_dim + kv_h * HEAD_DIM;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += qh[d] * kp[d];
            }
            scores[p] = dot * scale;
        }
        // Softmax
        cpu_softmax(scores, kv->len);
        // Weighted sum of values
        float *oh = attn_out + h * HEAD_DIM;
        for (int p = 0; p < kv->len; p++) {
            float *vp = kv->v_cache + p * kv_dim + kv_h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                oh[d] += scores[p] * vp[d];
            }
        }
        free(scores);
    }
    // ---- Output projection ----
    float *attn_projected = calloc(HIDDEN_DIM, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    uint32_t *ow = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", layer_idx);
    uint16_t *os_ptr = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", layer_idx);
    uint16_t *ob = get_tensor_ptr(wf, name);
    if (ow && os_ptr && ob) fast_dequant_matvec(ow, os_ptr, ob, attn_out, attn_projected, HIDDEN_DIM, q_dim, GROUP_SIZE);
    if (do_debug) {
        fprintf(stderr, "[FA-DBG] attn_out_rms=%.6f o_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(attn_out, q_dim),
                attn_projected[0], attn_projected[1], attn_projected[2], attn_projected[3], attn_projected[4]);
    }
    // ---- Residual connection ----
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = residual[i] + attn_projected[i];
    }
    if (do_debug) {
        fprintf(stderr, "[FA-DBG] AFTER layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, HIDDEN_DIM),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }
    free(normed);
    free(residual);
    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(attn_projected);
}
// ============================================================================
// Linear attention layer forward (GatedDeltaNet, single token, incremental)
// ============================================================================
// RMS norm without weights (just normalize)
static void cpu_rms_norm_bare(const float *x, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms;
}
// RMSNormGated: out = rms_norm(x) * silu(z)
static void cpu_rms_norm_gated(const float *x, const float *z, const uint16_t *w_bf16,
                                float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        float w = bf16_to_f32(w_bf16[i]);
        float silu_z = z[i] / (1.0f + expf(-z[i]));
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}
__attribute__((unused))
static void linear_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [HIDDEN_DIM] in/out
    LinearAttnState *state
) {
    // If bypass is enabled, just pass through (identity)
    if (linear_attn_bypass) {
        (void)wf; (void)layer_idx; (void)state;
        return;
    }
    static int la_debug_count = 0;
    la_debug_count++;
    int la_debug = 0;  // set to (la_debug_count <= N) to enable debug
    if (la_debug) {
        fprintf(stderr, "[LA-DBG] layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, HIDDEN_DIM),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }
    char name[256];
    float *normed = malloc(HIDDEN_DIM * sizeof(float));
    float *residual = malloc(HIDDEN_DIM * sizeof(float));
    cpu_vec_copy(residual, hidden, HIDDEN_DIM);
    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
    // ---- Batch QKV + Z + B + A projections (4 matmuls, 1 command buffer) ----
    int qkv_dim = LINEAR_CONV_DIM;  // 12288
    float *qkv = calloc(qkv_dim, sizeof(float));
    int z_dim = LINEAR_TOTAL_VALUE;  // 8192
    float *z = calloc(z_dim, sizeof(float));
    float *beta = calloc(LINEAR_NUM_V_HEADS, sizeof(float));
    float *alpha = calloc(LINEAR_NUM_V_HEADS, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    uint32_t *qkv_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", layer_idx);
    uint16_t *qkv_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", layer_idx);
    uint16_t *qkv_b = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", layer_idx);
    uint32_t *z_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", layer_idx);
    uint16_t *z_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", layer_idx);
    uint16_t *z_b = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", layer_idx);
    uint32_t *b_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", layer_idx);
    uint16_t *b_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", layer_idx);
    uint16_t *b_b = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", layer_idx);
    uint32_t *a_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", layer_idx);
    uint16_t *a_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", layer_idx);
    uint16_t *a_b = get_tensor_ptr(wf, name);
    if (qkv_w && qkv_s && qkv_b && z_w && z_s && z_b &&
        b_w && b_s && b_b && a_w && a_s && a_b) {
        BatchMatvecSpec la_specs[4] = {
            { qkv_w, qkv_s, qkv_b, qkv,   (uint32_t)qkv_dim,         HIDDEN_DIM, GROUP_SIZE, 0 },
            { z_w,   z_s,   z_b,   z,      (uint32_t)z_dim,           HIDDEN_DIM, GROUP_SIZE, 1 },
            { b_w,   b_s,   b_b,   beta,   (uint32_t)LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, 2 },
            { a_w,   a_s,   a_b,   alpha,  (uint32_t)LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, 3 },
        };
        fast_batch_matvec(normed, HIDDEN_DIM, la_specs, 4);
    }
    // ---- Conv1d step ----
    // conv_state holds last (kernel_size-1) inputs for each of the conv_dim channels
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", layer_idx);
    uint16_t *conv_w = get_tensor_ptr(wf, name);
    float *conv_out = calloc(qkv_dim, sizeof(float));
    if (conv_w) {
        cpu_conv1d_step(state->conv_state, qkv, conv_w, conv_out,
                        qkv_dim, CONV_KERNEL_SIZE);
    }
    // Update conv state: shift left, append new input
    memmove(state->conv_state, state->conv_state + qkv_dim,
            (CONV_KERNEL_SIZE - 2) * qkv_dim * sizeof(float));
    memcpy(state->conv_state + (CONV_KERNEL_SIZE - 2) * qkv_dim, qkv,
           qkv_dim * sizeof(float));
    // ---- Split conv_out into q, k, v ----
    // q: [num_k_heads * head_k_dim] = [2048]
    // k: [num_k_heads * head_k_dim] = [2048]
    // v: [num_v_heads * head_v_dim] = [8192]
    float *lin_q = conv_out;  // first LINEAR_TOTAL_KEY elements
    float *lin_k = conv_out + LINEAR_TOTAL_KEY;  // next LINEAR_TOTAL_KEY
    float *lin_v = conv_out + 2 * LINEAR_TOTAL_KEY;  // rest = LINEAR_TOTAL_VALUE
    // ---- RMS normalize q and k (bare, no weights) ----
    // q: scale = key_dim^(-0.5), normalize per head then scale by key_dim^(-1.0)
    // Actually from the code:
    //   inv_scale = k.shape[-1] ** -0.5 = head_k_dim^(-0.5) = 128^(-0.5)
    //   q = (inv_scale**2) * rms_norm(q) = (1/128) * rms_norm(q)
    //   k = inv_scale * rms_norm(k) = (1/sqrt(128)) * rms_norm(k)
    float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
    for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
        float *qh = lin_q + h * LINEAR_KEY_DIM;
        cpu_rms_norm_bare(qh, qh, LINEAR_KEY_DIM, 1e-6f);
        float q_scale = inv_scale * inv_scale;  // inv_scale^2 = 1/head_k_dim
        for (int d = 0; d < LINEAR_KEY_DIM; d++) qh[d] *= q_scale;
    }
    for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
        float *kh = lin_k + h * LINEAR_KEY_DIM;
        cpu_rms_norm_bare(kh, kh, LINEAR_KEY_DIM, 1e-6f);
        for (int d = 0; d < LINEAR_KEY_DIM; d++) kh[d] *= inv_scale;
    }
    // ---- Gated delta net recurrence ----
    // From gated_delta.py:
    //   g = exp(-exp(A_log) * softplus(a + dt_bias))   -- per-head decay
    //   beta_gate = sigmoid(b)                          -- per-head beta (NO dt_bias)
    //   For each v_head:
    //     state = state * g                             -- decay
    //     kv_mem = sum(state * k, axis=key_dim)         -- predict v from state
    //     delta = (v - kv_mem) * beta_gate              -- error signal
    //     state = state + outer(delta, k)               -- update state
    //     output = sum(state * q, axis=key_dim)         -- read from state
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", layer_idx);
    float *A_log = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", layer_idx);
    uint16_t *dt_bias_bf16 = get_tensor_ptr(wf, name);
    float *out_values = calloc(LINEAR_TOTAL_VALUE, sizeof(float));  // [num_v_heads * head_v_dim]
    int k_heads_per_v = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;  // 64/16 = 4
    // Precompute per-head decay (g) and beta
    float g_decay[LINEAR_NUM_V_HEADS];
    float beta_gate[LINEAR_NUM_V_HEADS];
    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        // g = exp(-exp(A_log) * softplus(a + dt_bias))
        float a_val = alpha[vh];
        float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
        float A_val = A_log ? expf(A_log[vh]) : 1.0f;
        float softplus_val = logf(1.0f + expf(a_val + dt_b));  // softplus(a + dt_bias)
        g_decay[vh] = expf(-A_val * softplus_val);
        // beta = sigmoid(b)  (just b, NO dt_bias)
        beta_gate[vh] = cpu_sigmoid(beta[vh]);
    }
    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        int kh = vh / k_heads_per_v;  // which k head this v head maps to
        float g = g_decay[vh];
        float b_gate = beta_gate[vh];
        // state is [head_v_dim, head_k_dim]
        float *S = state->ssm_state + vh * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
        float *v_h = lin_v + vh * LINEAR_VALUE_DIM;
        float *k_h = lin_k + kh * LINEAR_KEY_DIM;
        // Step 1: Decay state
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                S[vi * LINEAR_KEY_DIM + ki] *= g;
            }
        }
        // Step 2: Compute kv_mem[vi] = sum_ki(S[vi,ki] * k[ki])
        // Then delta[vi] = (v[vi] - kv_mem[vi]) * beta
        // Then state[vi,ki] += k[ki] * delta[vi]
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            float kv_mem = 0.0f;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                kv_mem += S[vi * LINEAR_KEY_DIM + ki] * k_h[ki];
            }
            float delta = (v_h[vi] - kv_mem) * b_gate;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                S[vi * LINEAR_KEY_DIM + ki] += k_h[ki] * delta;
            }
        }
        // Step 3: Output: y[vi] = sum_ki(S[vi,ki] * q[ki])
        float *q_h = lin_q + kh * LINEAR_KEY_DIM;
        float *o_h = out_values + vh * LINEAR_VALUE_DIM;
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            float sum = 0.0f;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                sum += S[vi * LINEAR_KEY_DIM + ki] * q_h[ki];
            }
            o_h[vi] = sum;
        }
    }
    // ---- RMSNormGated: out = rms_norm(out_values_per_head) * silu(z_per_head) * weight ----
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", layer_idx);
    uint16_t *gated_norm_w = get_tensor_ptr(wf, name);
    float *gated_out = calloc(LINEAR_TOTAL_VALUE, sizeof(float));
    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        float *oh = out_values + vh * LINEAR_VALUE_DIM;
        float *zh = z + vh * LINEAR_VALUE_DIM;
        float *gh = gated_out + vh * LINEAR_VALUE_DIM;
        if (gated_norm_w) {
            cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, LINEAR_VALUE_DIM, RMS_NORM_EPS);
        } else {
            memcpy(gh, oh, LINEAR_VALUE_DIM * sizeof(float));
        }
    }
    // ---- Output projection: [value_dim=8192] -> [hidden_dim=4096] ----
    float *attn_out = calloc(HIDDEN_DIM, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", layer_idx);
    uint32_t *out_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", layer_idx);
    uint16_t *out_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", layer_idx);
    uint16_t *out_b = get_tensor_ptr(wf, name);
    if (out_w && out_s && out_b) {
        fast_dequant_matvec(out_w, out_s, out_b, gated_out, attn_out, HIDDEN_DIM,
                            LINEAR_TOTAL_VALUE, GROUP_SIZE);
    }
    // ---- Residual ----
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = residual[i] + attn_out[i];
    }
    if (la_debug) {
        fprintf(stderr, "[LA-DBG] AFTER layer=%d out_proj_rms=%.6f gated_rms=%.6f hidden_rms=%.6f\n",
                layer_idx, vec_rms(attn_out, HIDDEN_DIM),
                vec_rms(gated_out, LINEAR_TOTAL_VALUE),
                vec_rms(hidden, HIDDEN_DIM));
    }
    free(normed);
    free(residual);
    free(qkv);
    free(z);
    free(beta);
    free(alpha);
    free(conv_out);
    free(out_values);
    free(gated_out);
    free(attn_out);
}
// ============================================================================
// MoE forward (routing + expert computation + shared expert)
// ============================================================================
static int moe_debug_count = 0;
__attribute__((unused))
static void moe_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,         // [HIDDEN_DIM] in/out
    const char *model_path __attribute__((unused)),
    int K,                 // number of active experts (e.g. 4)
    int packed_fd          // fd for this layer's packed expert file (-1 if not available)
) {
    moe_debug_count++;
    int moe_debug = 0;  // set to (moe_debug_count <= N) to enable debug
    int moe_dump = 0;
    char name[256];
    float *h_post = malloc(HIDDEN_DIM * sizeof(float));
    float *h_mid = malloc(HIDDEN_DIM * sizeof(float));
    cpu_vec_copy(h_mid, hidden, HIDDEN_DIM);
    // ---- Post-attention LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, h_post, HIDDEN_DIM, RMS_NORM_EPS);
    // ---- Batch routing gate + shared expert gate/up + shared_expert_gate (4 matmuls, 1 commit) ----
    float *gate_scores = calloc(NUM_EXPERTS, sizeof(float));
    float *shared_gate = calloc(SHARED_INTERMEDIATE, sizeof(float));
    float *shared_up = calloc(SHARED_INTERMEDIATE, sizeof(float));
    float shared_gate_score = 0.0f;
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", layer_idx);
    uint32_t *gate_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", layer_idx);
    uint16_t *gate_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", layer_idx);
    uint16_t *gate_b = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
    uint32_t *sgw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
    uint16_t *sgs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
    uint16_t *sgb = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
    uint32_t *suw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
    uint16_t *sus = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
    uint16_t *sub = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    uint32_t *seg_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", layer_idx);
    uint16_t *seg_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", layer_idx);
    uint16_t *seg_b = get_tensor_ptr(wf, name);
    // All 4 matmuls share h_post as input -- batch into one command buffer
    if (gate_w && gate_s && gate_b && sgw && sgs && sgb &&
        suw && sus && sub && seg_w && seg_s && seg_b) {
        BatchMatvecSpec moe_specs[4] = {
            { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)NUM_EXPERTS,        HIDDEN_DIM, GROUP_SIZE, 0 },
            { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 1 },
            { suw,    sus,    sub,    shared_up,           (uint32_t)SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 2 },
            { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            HIDDEN_DIM, GROUP_SIZE, 3 },
        };
        fast_batch_matvec(h_post, HIDDEN_DIM, moe_specs, 4);
    }
    // Softmax routing scores
    cpu_softmax(gate_scores, NUM_EXPERTS);
    // Top-K expert selection
    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, NUM_EXPERTS, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] routing: K=%d experts=[", K);
        for (int k = 0; k < K; k++) fprintf(stderr, "%d(%.4f)%s", expert_indices[k], expert_weights[k], k<K-1?",":"");
        fprintf(stderr, "]\n");
    }
    // ---- Routed expert computation ----
    float *moe_out = calloc(HIDDEN_DIM, sizeof(float));
    if (packed_fd >= 0) {
        float *expert_out = malloc(HIDDEN_DIM * sizeof(float));
        size_t esz = active_expert_size();
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset = (off_t)eidx * esz;
            if (g_metal && g_metal->buf_expert_data) {
                // GPU path: pread directly into Metal buffer, run gate+up+swiglu+down on GPU
                void *expert_buf_ptr = [g_metal->buf_expert_data contents];
                ssize_t nread = pread(packed_fd, expert_buf_ptr, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    continue;
                }
                gpu_expert_forward(g_metal, expert_buf_ptr, h_post, expert_out, 1 /*already in buffer*/);
            } else {
                // CPU fallback
                void *expert_data = malloc(esz);
                ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    free(expert_data);
                    continue;
                }
                uint32_t *gw = (uint32_t *)expert_data;
                uint16_t *gs_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? GATE_S_OFF_2 : 2097152));
                uint16_t *gb_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? GATE_B_OFF_2 : 2228224));
                uint32_t *uw = (uint32_t *)((char *)expert_data + (g_use_2bit ? UP_W_OFF_2 : 2359296));
                uint16_t *us_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? UP_S_OFF_2 : 4456448));
                uint16_t *ub_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? UP_B_OFF_2 : 4587520));
                uint32_t *dw = (uint32_t *)((char *)expert_data + (g_use_2bit ? DOWN_W_OFF_2 : 4718592));
                uint16_t *ds_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? DOWN_S_OFF_2 : 6815744));
                uint16_t *db_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? DOWN_B_OFF_2 : 6946816));
                float *gate_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *up_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *act_out = malloc(MOE_INTERMEDIATE * sizeof(float));
                cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                                   MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
                cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                                   MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
                cpu_swiglu(gate_proj_out, up_proj_out, act_out, MOE_INTERMEDIATE);
                cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out,
                                   HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE);
                free(gate_proj_out);
                free(up_proj_out);
                free(act_out);
                free(expert_data);
            }
            // Accumulate weighted
            if (moe_dump) {
                fprintf(stderr, "[MOE-DUMP] expert[%d] out_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        eidx, vec_rms(expert_out, HIDDEN_DIM),
                        expert_out[0], expert_out[1], expert_out[2], expert_out[3], expert_out[4]);
            }
            cpu_vec_madd(moe_out, expert_out, expert_weights[k], HIDDEN_DIM);
        }
        free(expert_out);
    }
    // ---- Shared expert SwiGLU (gate_proj + up_proj already computed above) ----
    float *shared_out = calloc(HIDDEN_DIM, sizeof(float));
    float *shared_act = calloc(SHARED_INTERMEDIATE, sizeof(float));
    cpu_swiglu(shared_gate, shared_up, shared_act, SHARED_INTERMEDIATE);
    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] layer=%d h_post_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(h_post, HIDDEN_DIM), h_post[0], h_post[1], h_post[2], h_post[3], h_post[4]);
        fprintf(stderr, "[MOE-DUMP] gate_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_gate, SHARED_INTERMEDIATE),
                shared_gate[0], shared_gate[1], shared_gate[2], shared_gate[3], shared_gate[4]);
        fprintf(stderr, "[MOE-DUMP] up_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_up, SHARED_INTERMEDIATE),
                shared_up[0], shared_up[1], shared_up[2], shared_up[3], shared_up[4]);
        fprintf(stderr, "[MOE-DUMP] swiglu_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_act, SHARED_INTERMEDIATE),
                shared_act[0], shared_act[1], shared_act[2], shared_act[3], shared_act[4]);
    }
    // shared_expert down_proj (separate dispatch — different input than h_post)
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
    uint32_t *sdw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
    uint16_t *sds = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);
    uint16_t *sdb = get_tensor_ptr(wf, name);
    if (sdw && sds && sdb) {
        fast_dequant_matvec(sdw, sds, sdb, shared_act, shared_out, HIDDEN_DIM,
                            SHARED_INTERMEDIATE, GROUP_SIZE);
    }
    // ---- Shared expert gate (sigmoid) -- already computed above ----
    float shared_weight = cpu_sigmoid(shared_gate_score);
    // Scale shared expert output
    for (int i = 0; i < HIDDEN_DIM; i++) {
        shared_out[i] *= shared_weight;
    }
    // ---- Combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }
    if (moe_debug) {
        fprintf(stderr, "[MOE-DBG] layer=%d h_mid_rms=%.4f moe_rms=%.4f shared_rms=%.4f shared_gate=%.4f hidden_rms=%.4f\n",
                layer_idx, vec_rms(h_mid, HIDDEN_DIM), vec_rms(moe_out, HIDDEN_DIM),
                vec_rms(shared_out, HIDDEN_DIM), shared_weight,
                vec_rms(hidden, HIDDEN_DIM));
    }
    free(h_post);
    free(h_mid);
    free(gate_scores);
    free(moe_out);
    free(shared_out);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
}
// ============================================================================
// Embedding lookup (4-bit quantized)
// ============================================================================
static void embed_lookup(WeightFile *wf, int token_id, float *out) {
    // Embedding: weight[vocab_size, hidden_dim/8] (U32), scales[vocab_size, groups], biases[vocab_size, groups]
    // For embedding lookup, we just need one row.
    // But the embedding is quantized: each row has hidden_dim/8 uint32 values (packed 4-bit)
    // plus scales and biases per group
    TensorInfo *w_info = get_tensor_info(wf, "model.embed_tokens.weight");
    TensorInfo *s_info = get_tensor_info(wf, "model.embed_tokens.scales");
    TensorInfo *b_info = get_tensor_info(wf, "model.embed_tokens.biases");
    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: embedding tensors not found\n");
        memset(out, 0, HIDDEN_DIM * sizeof(float));
        return;
    }
    // w shape: [248320, 512] U32 -> each row has 512 uint32 = 4096 packed 4-bit values
    int packed_cols = w_info->shape[1];  // 512
    int num_groups = s_info->shape[1];   // 64
    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);
    const uint32_t *w_row = W + (size_t)token_id * packed_cols;
    const uint16_t *s_row = S + (size_t)token_id * num_groups;
    const uint16_t *b_row = B + (size_t)token_id * num_groups;
    int group_size = HIDDEN_DIM / num_groups;  // 4096/64 = 64
    int packed_per_group = group_size / 8;     // 8
    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias = bf16_to_f32(b_row[g]);
        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base = g * group_size + p * 8;
            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = (float)nibble * scale + bias;
            }
        }
    }
}
// ============================================================================
// LM head (logits projection)
// ============================================================================
static void lm_head_forward(WeightFile *wf, const float *hidden, float *logits) {
    // lm_head: [hidden_dim=4096] -> [vocab_size=248320]
    // This is a HUGE matmul. For 248320 output dims, it will be slow on CPU.
    // Optimization: only compute top candidates
    TensorInfo *w_info = get_tensor_info(wf, "lm_head.weight");
    TensorInfo *s_info = get_tensor_info(wf, "lm_head.scales");
    TensorInfo *b_info = get_tensor_info(wf, "lm_head.biases");
    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: lm_head tensors not found\n");
        return;
    }
    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);
    // Full matmul — use GPU if available (248320 output rows!)
    fast_dequant_matvec(W, S, B, hidden, logits, VOCAB_SIZE, HIDDEN_DIM, GROUP_SIZE);
}
// ============================================================================
// Parallel I/O infrastructure for expert pread (from proven main.m pattern)
// ============================================================================
#define NUM_IO_THREADS 4  // 4 threads for K=4 experts (one per expert)
typedef struct {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
    ssize_t result;
    int err;
    const void *mmap_base;  // if non-NULL, memcpy from mmap instead of pread
    // LZ4 compression fields (set by caller when reading compressed experts)
    void *lz4_comp_buf;     // if non-NULL: pread into this, then LZ4 decompress into dst
    uint32_t lz4_comp_size; // compressed size to read from disk
} InferPreadTask;
static inline void infer_pread_task_init(InferPreadTask *task) {
    memset(task, 0, sizeof(*task));
}
typedef struct {
    InferPreadTask *tasks;
    int num_tasks;
    int thread_id;
} InferPreadThreadArg;
static void *infer_pread_thread_fn(void *arg) {
    InferPreadThreadArg *ta = (InferPreadThreadArg *)arg;
    for (int i = ta->thread_id; i < ta->num_tasks; i += NUM_IO_THREADS) {
        InferPreadTask *t = &ta->tasks[i];
        t->err = 0;
        t->result = pread(t->fd, t->dst, t->size, t->offset);
        if (t->result < 0) t->err = errno;
    }
    return NULL;
}
// ============================================================================
// Persistent I/O Thread Pool — eliminates pthread_create/join per layer
// ============================================================================
typedef struct {
    pthread_t threads[NUM_IO_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    InferPreadTask *tasks;
    int num_tasks;
    int tasks_completed;
    int generation;          // incremented each dispatch — workers wait for new gen
    volatile int shutdown;
} IOThreadPool;
static IOThreadPool g_io_pool;
static int g_io_pool_initialized = 0;
static void *io_pool_worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    int my_gen = 0;
    pthread_mutex_lock(&g_io_pool.mutex);
    while (1) {
        while (g_io_pool.generation == my_gen && !g_io_pool.shutdown)
            pthread_cond_wait(&g_io_pool.work_ready, &g_io_pool.mutex);
        if (g_io_pool.shutdown) break;
        my_gen = g_io_pool.generation;
        // Snapshot work for this generation
        int num_tasks = g_io_pool.num_tasks;
        InferPreadTask *tasks = g_io_pool.tasks;
        pthread_mutex_unlock(&g_io_pool.mutex);
        // Process assigned tasks (stride by thread count)
        for (int i = tid; i < num_tasks; i += NUM_IO_THREADS) {
            InferPreadTask *t = &tasks[i];
            t->err = 0;
            if (t->lz4_comp_buf && t->lz4_comp_size > 0) {
                // LZ4 path: read compressed from SSD, decompress into dst
                ssize_t nr = pread(t->fd, t->lz4_comp_buf, t->lz4_comp_size, t->offset);
                if (nr < 0) t->err = errno;
                if (nr == (ssize_t)t->lz4_comp_size) {
                    size_t dec = compression_decode_buffer(
                        t->dst, t->size, t->lz4_comp_buf, t->lz4_comp_size,
                        NULL, COMPRESSION_LZ4);
                    t->result = (ssize_t)dec;
                } else {
                    t->result = -1;
                }
            } else {
                t->result = pread(t->fd, t->dst, t->size, t->offset);
                if (t->result < 0) t->err = errno;
            }
        }
        pthread_mutex_lock(&g_io_pool.mutex);
        g_io_pool.tasks_completed++;
        if (g_io_pool.tasks_completed == NUM_IO_THREADS)
            pthread_cond_signal(&g_io_pool.work_done);
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
    return NULL;
}
static void io_pool_init(void) {
    if (g_io_pool_initialized) return;
    pthread_mutex_init(&g_io_pool.mutex, NULL);
    pthread_cond_init(&g_io_pool.work_ready, NULL);
    pthread_cond_init(&g_io_pool.work_done, NULL);
    g_io_pool.shutdown = 0;
    g_io_pool.generation = 0;
    g_io_pool.tasks = NULL;
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_create(&g_io_pool.threads[i], NULL, io_pool_worker, (void*)(intptr_t)i);
    g_io_pool_initialized = 1;
}
static dispatch_queue_t g_io_gcd_queue = NULL;
static void io_pool_dispatch(InferPreadTask *tasks, int num_tasks) {
    if (num_tasks == 0) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.tasks = tasks;
    g_io_pool.num_tasks = num_tasks;
    g_io_pool.tasks_completed = 0;
    g_io_pool.generation++;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    while (g_io_pool.tasks_completed < NUM_IO_THREADS) {
        pthread_cond_wait(&g_io_pool.work_done, &g_io_pool.mutex);
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
}
// ---- Async expert pread pipeline ----
// Starts pread on background GCD threads immediately after routing.
// The pread overlaps with shared expert prep + next layer's CMD1+attn+CMD2.
// Wait for completion right before CMD3 needs the expert data.
typedef struct {
    InferPreadTask tasks[MAX_K];
    int num_tasks;
    int valid[MAX_K];
    dispatch_group_t group;
    int active;
} AsyncPreadState;
static AsyncPreadState g_async_pread = {0};
static void async_pread_wait(void);
static void async_pread_start(int packed_fd, int *expert_indices, int K,
                               id<MTLBuffer> __strong *dst_bufs, const void *mmap_base) {
    size_t esz = active_expert_size();
    if (g_async_pread.active) {
        // The async state is a single global scratch slot; never overwrite it while
        // background reads are still touching the previous task array.
        async_pread_wait();
    }
    g_async_pread.num_tasks = K;
    g_async_pread.active = 1;
    if (!g_async_pread.group) g_async_pread.group = dispatch_group_create();
    for (int k = 0; k < K; k++) {
        infer_pread_task_init(&g_async_pread.tasks[k]);
        g_async_pread.tasks[k].fd = packed_fd;
        g_async_pread.tasks[k].dst = [dst_bufs[k] contents];
        g_async_pread.tasks[k].offset = (off_t)expert_indices[k] * esz;
        g_async_pread.tasks[k].size = esz;
    }
    // Fire off parallel preads on GCD — returns immediately
    static dispatch_queue_t io_q = NULL;
    if (!io_q) io_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int k = 0; k < K; k++) {
        InferPreadTask *t = &g_async_pread.tasks[k];
        dispatch_group_async(g_async_pread.group, io_q, ^{
            t->result = pread(t->fd, t->dst, t->size, t->offset);
            if (t->result < 0) t->err = errno;
        });
    }
}
static void async_pread_wait(void) {
    if (!g_async_pread.active) return;
    dispatch_group_wait(g_async_pread.group, DISPATCH_TIME_FOREVER);
    for (int k = 0; k < g_async_pread.num_tasks; k++) {
        g_async_pread.valid[k] = (g_async_pread.tasks[k].result == (ssize_t)active_expert_size());
    }
    g_async_pread.active = 0;
}
static void io_pool_shutdown(void) {
    if (!g_io_pool_initialized) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.shutdown = 1;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    pthread_mutex_unlock(&g_io_pool.mutex);
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_join(g_io_pool.threads[i], NULL);
    pthread_mutex_destroy(&g_io_pool.mutex);
    pthread_cond_destroy(&g_io_pool.work_ready);
    pthread_cond_destroy(&g_io_pool.work_done);
    g_io_pool_initialized = 0;
}
// Parallel pread of K experts into Metal buffers using pthreads.
// Returns number of successfully loaded experts, sets valid[] flags.
static int parallel_pread_experts(
    int packed_fd,
    int *expert_indices,
    int K,
    int *valid,  // [MAX_K] output: 1 if expert loaded successfully
    const void *mmap_base  // mmap'd layer file (NULL to use pread)
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        infer_pread_task_init(&tasks[k]);
        tasks[k].fd = packed_fd;
        tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
        tasks[k].offset = (off_t)expert_indices[k] * esz;
        tasks[k].size = esz;
        tasks[k].mmap_base = mmap_base;
    }
    io_pool_dispatch(tasks, K);
    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)esz);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread fd=%d: %zd/%zu errno=%d (%s)\n",
                    expert_indices[k], tasks[k].fd, tasks[k].result, esz,
                    tasks[k].err, tasks[k].err ? strerror(tasks[k].err) : "short read");
        }
    }
    return loaded;
}
// ============================================================================
// Parallel pread into explicit buffer set (for double buffering).
// Same as parallel_pread_experts but reads into caller-specified MTLBuffers.
// ============================================================================
static int parallel_pread_experts_into(
    int packed_fd,
    int *expert_indices,
    int K,
    id<MTLBuffer> __strong *dst_bufs,  // target Metal buffers (set A or B)
    int *valid  // [MAX_K] output: 1 if expert loaded successfully
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        infer_pread_task_init(&tasks[k]);
        tasks[k].fd = packed_fd;
        tasks[k].dst = [dst_bufs[k] contents];
        tasks[k].offset = (off_t)expert_indices[k] * esz;
        tasks[k].size = esz;
    }
    io_pool_dispatch(tasks, K);
    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)esz);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread fd=%d: %zd/%zu errno=%d (%s)\n",
                    expert_indices[k], tasks[k].fd, tasks[k].result, esz,
                    tasks[k].err, tasks[k].err ? strerror(tasks[k].err) : "short read");
        }
    }
    return loaded;
}
// ============================================================================
// Expert LRU Cache: keeps recently-used expert Metal buffers in GPU memory.
//
// Key: (layer_idx, expert_idx) -> Metal buffer containing 7.08MB expert data.
// On cache HIT:  skip pread entirely, use the cached Metal buffer for GPU dispatch.
// On cache MISS: pread into a new/evicted Metal buffer, insert into cache.
// LRU eviction:  when cache is full, evict the least recently used entry.
//
// Memory budget: 2000 entries * 7.08MB = 14.2GB. With 5.5GB non-expert weights
// + 14.2GB cache = 19.7GB total. Fits in 48GB with room for OS.
//
// Unlike Python/MLX where LRU caching caused Metal heap pressure and slower
// mx.eval(), here Metal buffers ARE the cache -- no conversion overhead.
// ============================================================================
typedef struct {
    int layer_idx;
    int expert_idx;
    id<MTLBuffer> buffer;    // Metal buffer holding EXPERT_SIZE bytes
    uint64_t last_used;      // monotonic counter for LRU ordering
} ExpertCacheEntry;
typedef struct {
    ExpertCacheEntry *entries;
    int max_entries;
    int num_entries;
    int used_entries;
    int entry_idx[NUM_LAYERS][NUM_EXPERTS];
    uint64_t access_counter; // monotonic, incremented on every access
    id<MTLDevice> device;    // for allocating new Metal buffers
    // Stats
    uint64_t hits;
    uint64_t misses;
} ExpertLRUCache;
static ExpertLRUCache *g_expert_cache = NULL;
static ExpertLRUCache *g_prefill_chunk_cache = NULL;
static int g_prefill_chunk_cache_active = 0;
static ExpertLRUCache *g_prefill_microbatch_cache = NULL;
// Speculative early routing stats
static uint64_t g_spec_route_attempts = 0;   // total speculative routing attempts
static uint64_t g_spec_route_hits = 0;        // correctly predicted experts (found in cache at real routing time)
static uint64_t g_spec_route_preloads = 0;    // async preloads initiated (cache misses at speculation time)
// ---- Temporal prediction pipeline ----
// Stores previous token's expert routing per layer. On the next token,
// predicted experts are preloaded into buf_multi_expert_data_B during CMD1_wait
// idle time. After routing, hits use buf_B, misses sync-pread into buf_A.
// Different from previous failed speculative attempts:
//   - Loads into scratch buffers (no cache pollution)
//   - Uses CMD1_wait idle time (no additional CPU cost)
//   - Only sync-preads misses (not all K experts)
static int g_pred_experts[NUM_LAYERS][MAX_K];      // previous token's expert indices per layer
static int g_pred_count[NUM_LAYERS];               // how many experts stored per layer
static int g_pred_valid = 0;                       // 1 after first token completes (predictions available)
// g_pred_enabled, g_pred_hits, g_pred_misses, g_pred_layers declared near timing (line ~163)
static ExpertLRUCache *expert_cache_new(id<MTLDevice> device, int max_entries) {
    ExpertLRUCache *cache = calloc(1, sizeof(ExpertLRUCache));
    cache->entries = calloc(max_entries, sizeof(ExpertCacheEntry));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->device = device;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            cache->entry_idx[l][e] = -1;
        }
    }
    // Pre-allocate ALL Metal buffers at startup (avoids allocation overhead at runtime)
    size_t esz = active_expert_size();
    double t_prealloc = now_ms();
    for (int i = 0; i < max_entries; i++) {
        cache->entries[i].buffer = [device newBufferWithLength:esz
                                                      options:MTLResourceStorageModeShared];
        cache->entries[i].layer_idx = -1;
        cache->entries[i].expert_idx = -1;
        cache->entries[i].last_used = 0;
        if (!cache->entries[i].buffer) {
            fprintf(stderr, "WARNING: expert_cache: pre-alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
    }
    cache->num_entries = max_entries; // All slots pre-allocated (but empty keys)
    printf("[expert_cache] Initialized: max_entries=%d (%.1f GB budget), pre-alloc %.0f ms\n",
           max_entries, (double)max_entries * esz / 1e9, now_ms() - t_prealloc);
    return cache;
}
static void expert_cache_free(ExpertLRUCache *cache) {
    if (!cache) return;
    printf("[expert_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    // Metal buffers released by ARC when entries are freed
    free(cache->entries);
    free(cache);
}
static void expert_cache_reset(ExpertLRUCache *cache) {
    if (!cache) return;
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            cache->entry_idx[l][e] = -1;
        }
    }
    for (int i = 0; i < cache->num_entries; i++) {
        cache->entries[i].layer_idx = -1;
        cache->entries[i].expert_idx = -1;
        cache->entries[i].last_used = 0;
    }
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
}
static inline ExpertLRUCache *active_expert_lru_cache(void) {
    if (g_prefill_chunk_cache_active && g_prefill_chunk_cache) return g_prefill_chunk_cache;
    return g_expert_cache;
}
// Lookup: returns the cached Metal buffer if found, otherwise NULL.
// On hit, updates the LRU timestamp.
static id<MTLBuffer> expert_cache_lookup(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    int idx = cache->entry_idx[layer_idx][expert_idx];
    if (idx >= 0) {
        cache->entries[idx].last_used = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer_idx, expert_idx);
        return cache->entries[idx].buffer;
    }
    cache->misses++;
    cache_telemetry_miss(layer_idx, expert_idx);
    return nil;
}
// Insert: adds a new entry. If the cache is full, evicts the LRU entry.
// Returns the Metal buffer to pread into (either newly allocated or evicted+reused).
static id<MTLBuffer> expert_cache_insert(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    id<MTLBuffer> buf = nil;
    int existing = cache->entry_idx[layer_idx][expert_idx];
    if (existing >= 0) {
        cache->entries[existing].last_used = ++cache->access_counter;
        return cache->entries[existing].buffer;
    }
    // Find a slot: first try an unused slot (layer_idx == -1), then LRU evict
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }
    if (target >= 0) {
        // Unused pre-allocated slot
        buf = cache->entries[target].buffer;
        cache->entries[target].layer_idx = layer_idx;
        cache->entries[target].expert_idx = expert_idx;
        cache->entries[target].last_used = ++cache->access_counter;
        cache->entry_idx[layer_idx][expert_idx] = target;
        return buf;
    }
    // Cache full: find LRU entry (smallest last_used)
    int lru_idx = 0;
    uint64_t min_used = cache->entries[0].last_used;
    for (int i = 1; i < cache->num_entries; i++) {
        if (cache->entries[i].last_used < min_used) {
            min_used = cache->entries[i].last_used;
            lru_idx = i;
        }
    }
    // Reuse the evicted entry's Metal buffer (same size, no realloc needed)
    int old_layer = cache->entries[lru_idx].layer_idx;
    int old_expert = cache->entries[lru_idx].expert_idx;
    cache_telemetry_evict(old_layer, old_expert);
    if (old_layer >= 0 && old_expert >= 0) {
        cache->entry_idx[old_layer][old_expert] = -1;
    }
    buf = cache->entries[lru_idx].buffer;
    cache->entries[lru_idx].layer_idx = layer_idx;
    cache->entries[lru_idx].expert_idx = expert_idx;
    cache->entries[lru_idx].last_used = ++cache->access_counter;
    cache->entry_idx[layer_idx][expert_idx] = lru_idx;
    return buf;
}
// ============================================================================
// Malloc-based expert frequency cache.
// Stores expert data in regular malloc'd memory (not Metal buffers) to avoid
// GPU memory pressure. On hit, memcpy to Metal scratch buffer. Much larger
// capacity than Metal buffer LRU cache at the cost of one memcpy per hit.
// ============================================================================
typedef struct {
    void **data;           // [max_entries] page-aligned malloc'd EXPERT_SIZE buffers
    id<MTLBuffer> __strong *metal_bufs;  // [max_entries] zero-copy Metal buffer wrappers
    int *layer_idx;        // [max_entries] layer index for each entry
    int *expert_idx;       // [max_entries] expert index for each entry
    uint64_t *last_used;   // [max_entries] monotonic counter for LRU
    int max_entries;
    int num_entries;
    int used_entries;
    int entry_idx[NUM_LAYERS][NUM_EXPERTS];
    uint64_t access_counter;
    uint64_t hits;
    uint64_t misses;
} MallocExpertCache;
static MallocExpertCache *g_malloc_cache = NULL;
static MallocExpertCache *malloc_cache_init(int max_entries, id<MTLDevice> device) {
    MallocExpertCache *cache = calloc(1, sizeof(MallocExpertCache));
    cache->data = calloc(max_entries, sizeof(void *));
    cache->metal_bufs = (__strong id<MTLBuffer> *)calloc(max_entries, sizeof(id<MTLBuffer>));
    cache->layer_idx = calloc(max_entries, sizeof(int));
    cache->expert_idx = calloc(max_entries, sizeof(int));
    cache->last_used = calloc(max_entries, sizeof(uint64_t));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            cache->entry_idx[l][e] = -1;
        }
    }
    size_t esz = active_expert_size();
    printf("[malloc_cache] Initializing: %d entries (%.1f GB) with zero-copy Metal wrappers\n",
           max_entries, (double)max_entries * esz / 1e9);
    double t_start = now_ms();
    size_t page_size = (size_t)getpagesize();
    // Round expert size up to page boundary for newBufferWithBytesNoCopy
    size_t aligned_size = (esz + page_size - 1) & ~(page_size - 1);
    for (int i = 0; i < max_entries; i++) {
        // Page-aligned allocation for zero-copy Metal buffer
        void *buf = NULL;
        if (posix_memalign(&buf, page_size, aligned_size) != 0 || !buf) {
            fprintf(stderr, "WARNING: malloc_cache: alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
        memset(buf, 0, aligned_size);
        cache->data[i] = buf;
        // Create zero-copy Metal buffer wrapping the malloc'd memory
        // nil deallocator = Metal doesn't free the memory
        cache->metal_bufs[i] = [device newBufferWithBytesNoCopy:buf
                                                         length:aligned_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        cache->layer_idx[i] = -1;
        cache->expert_idx[i] = -1;
        cache->last_used[i] = 0;
    }
    cache->num_entries = max_entries;
    printf("[malloc_cache] Pre-allocated %d entries in %.0f ms\n",
           max_entries, now_ms() - t_start);
    return cache;
}
// Lookup: returns Metal buffer wrapping cached data, or nil. Zero-copy dispatch.
static id<MTLBuffer> malloc_cache_lookup(MallocExpertCache *cache, int layer, int expert) {
    int idx = cache->entry_idx[layer][expert];
    if (idx >= 0) {
        cache->last_used[idx] = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer, expert);
        return cache->metal_bufs[idx];
    }
    cache->misses++;
    cache_telemetry_miss(layer, expert);
    return nil;
}
// Insert: evict LRU if needed, return entry index for pread target.
// Returns the Metal buffer for this entry (caller should pread into cache->data[idx]).
static id<MTLBuffer> malloc_cache_insert(MallocExpertCache *cache, int layer, int expert, int *out_idx) {
    int existing = cache->entry_idx[layer][expert];
    if (existing >= 0) {
        cache->last_used[existing] = ++cache->access_counter;
        if (out_idx) *out_idx = existing;
        return cache->metal_bufs[existing];
    }
    // Find a free slot (layer_idx == -1) or evict LRU
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }
    if (target < 0) {
        // Cache full: evict entry with smallest last_used
        target = 0;
        uint64_t min_used = cache->last_used[0];
        for (int i = 1; i < cache->num_entries; i++) {
            if (cache->last_used[i] < min_used) {
                min_used = cache->last_used[i];
                target = i;
            }
        }
        cache_telemetry_evict(cache->layer_idx[target], cache->expert_idx[target]);
        if (cache->layer_idx[target] >= 0 && cache->expert_idx[target] >= 0) {
            cache->entry_idx[cache->layer_idx[target]][cache->expert_idx[target]] = -1;
        }
    }
    cache->layer_idx[target] = layer;
    cache->expert_idx[target] = expert;
    cache->last_used[target] = ++cache->access_counter;
    cache->entry_idx[layer][expert] = target;
    if (out_idx) *out_idx = target;
    return cache->metal_bufs[target];
}
static void malloc_cache_free(MallocExpertCache *cache) {
    if (!cache) return;
    printf("[malloc_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    for (int i = 0; i < cache->num_entries; i++) {
        cache->metal_bufs[i] = nil;  // release Metal buffer wrapper
        free(cache->data[i]);
    }
    free(cache->data);
    free(cache->metal_bufs);
    free(cache->layer_idx);
    free(cache->expert_idx);
    free(cache->last_used);
    free(cache);
}
// ============================================================================
// Background prefetch thread for double-buffered expert I/O (from main.m).
// Runs pread on a background thread while main thread does GPU compute.
// Uses pure C I/O plan to avoid ARC issues across threads.
// ============================================================================
typedef struct {
    void *dst[MAX_K];       // raw pointers from [buf contents] (no ARC)
    off_t offset[MAX_K];    // file offsets per expert
    int K;                  // number of experts
    int fd;                 // file descriptor for this layer
    int valid[MAX_K];       // output: 1 if pread succeeded
    int loaded;             // output: count of successfully loaded experts
} InferIOPlan;
typedef struct {
    InferIOPlan plan;       // pre-built I/O plan (pure C, no ARC)
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int start;              // signal: set to 1 to start prefetch
    int done;               // signal: set to 1 when prefetch complete
    int shutdown;           // signal: set to 1 to exit thread
} InferPrefetchCtx;
static void *infer_prefetch_thread_fn(void *arg) {
    InferPrefetchCtx *pf = (InferPrefetchCtx *)arg;
    while (1) {
        pthread_mutex_lock(&pf->mutex);
        while (!pf->start && !pf->shutdown) {
            pthread_cond_wait(&pf->cond, &pf->mutex);
        }
        if (pf->shutdown) {
            pthread_mutex_unlock(&pf->mutex);
            break;
        }
        pf->start = 0;
        pthread_mutex_unlock(&pf->mutex);
        // Execute parallel pread (pure C, no ARC objects)
        size_t esz = active_expert_size();
        InferIOPlan *plan = &pf->plan;
        InferPreadTask tasks[MAX_K];
        for (int k = 0; k < plan->K; k++) {
            infer_pread_task_init(&tasks[k]);
            tasks[k].fd = plan->fd;
            tasks[k].dst = plan->dst[k];
            tasks[k].offset = plan->offset[k];
            tasks[k].size = esz;
        }
        io_pool_dispatch(tasks, plan->K);
        plan->loaded = 0;
        for (int k = 0; k < plan->K; k++) {
            plan->valid[k] = (tasks[k].result == (ssize_t)esz);
            if (plan->valid[k]) plan->loaded++;
        }
        // Signal completion
        pthread_mutex_lock(&pf->mutex);
        pf->done = 1;
        pthread_cond_signal(&pf->cond);
        pthread_mutex_unlock(&pf->mutex);
    }
    return NULL;
}
// Build I/O plan on main thread (ARC-safe: extracts void* from id<MTLBuffer>),
// then signal background prefetch thread.
static void infer_prefetch_start(InferPrefetchCtx *pf, int packed_fd,
                                  int *expert_indices, int K,
                                  id<MTLBuffer> __strong *dst_bufs) {
    pthread_mutex_lock(&pf->mutex);
    size_t esz = active_expert_size();
    InferIOPlan *plan = &pf->plan;
    plan->fd = packed_fd;
    plan->K = K;
    for (int k = 0; k < K; k++) {
        plan->dst[k] = [dst_bufs[k] contents];
        plan->offset[k] = (off_t)expert_indices[k] * esz;
        plan->valid[k] = 0;
    }
    plan->loaded = 0;
    pf->done = 0;
    pf->start = 1;
    pthread_cond_signal(&pf->cond);
    pthread_mutex_unlock(&pf->mutex);
}
// Wait for background prefetch to complete. Returns number of loaded experts.
// Copies valid[] flags into caller's array.
static int infer_prefetch_wait(InferPrefetchCtx *pf, int *valid_out, int K) {
    pthread_mutex_lock(&pf->mutex);
    while (!pf->done) {
        pthread_cond_wait(&pf->cond, &pf->mutex);
    }
    int loaded = pf->plan.loaded;
    for (int k = 0; k < K; k++) {
        valid_out[k] = pf->plan.valid[k];
    }
    pthread_mutex_unlock(&pf->mutex);
    return loaded;
}
static InferPrefetchCtx *g_prefetch = NULL;
static pthread_t g_prefetch_tid;
static void infer_prefetch_init(void) {
    if (g_prefetch) return;
    g_prefetch = calloc(1, sizeof(InferPrefetchCtx));
    pthread_mutex_init(&g_prefetch->mutex, NULL);
    pthread_cond_init(&g_prefetch->cond, NULL);
    g_prefetch->shutdown = 0;
    pthread_create(&g_prefetch_tid, NULL, infer_prefetch_thread_fn, g_prefetch);
}
static void infer_prefetch_shutdown(void) {
    if (!g_prefetch) return;
    pthread_mutex_lock(&g_prefetch->mutex);
    g_prefetch->shutdown = 1;
    pthread_cond_signal(&g_prefetch->cond);
    pthread_mutex_unlock(&g_prefetch->mutex);
    pthread_join(g_prefetch_tid, NULL);
    pthread_mutex_destroy(&g_prefetch->mutex);
    pthread_cond_destroy(&g_prefetch->cond);
    free(g_prefetch);
    g_prefetch = NULL;
}
// ============================================================================
// Per-layer weight pointer cache — built once, eliminates 40+ snprintf+lookup
// per layer per token. With 60 layers and 15 tokens = 36,000 lookups saved.
// ============================================================================
typedef struct {
    // Input/post-attention layer norms
    uint16_t *input_norm_w;
    uint16_t *post_attn_norm_w;
    // Full attention weights (non-NULL only for full attention layers)
    uint32_t *q_w; uint16_t *q_s, *q_b;
    uint32_t *k_w; uint16_t *k_s, *k_b;
    uint32_t *v_w; uint16_t *v_s, *v_b;
    uint32_t *o_w; uint16_t *o_s, *o_b;
    uint16_t *q_norm_w, *k_norm_w;
    // Linear attention weights (non-NULL only for linear attention layers)
    uint32_t *qkv_w; uint16_t *qkv_s, *qkv_b;
    uint32_t *z_w;   uint16_t *z_s, *z_b;
    uint32_t *b_w;   uint16_t *b_s, *b_b;
    uint32_t *a_w;   uint16_t *a_s, *a_b;
    uint16_t *conv1d_w;
    float *A_log;
    uint16_t *dt_bias;
    uint16_t *gated_norm_w;
    uint32_t *out_proj_w; uint16_t *out_proj_s, *out_proj_b;
    // MoE routing + shared expert weights
    uint32_t *gate_w; uint16_t *gate_s, *gate_b;
    float *gate_bias_corr;
    uint32_t *sg_w;   uint16_t *sg_s, *sg_b;   // shared gate_proj
    uint32_t *su_w;   uint16_t *su_s, *su_b;   // shared up_proj
    uint32_t *sd_w;   uint16_t *sd_s, *sd_b;   // shared down_proj
    uint32_t *seg_w;  uint16_t *seg_s, *seg_b; // shared_expert_gate
} LayerWeightCache;
static LayerWeightCache layer_cache[NUM_LAYERS];
static int layer_cache_built = 0;
static void build_layer_cache(WeightFile *wf) {
    if (layer_cache_built) return;
    char name[256];
    for (int i = 0; i < NUM_LAYERS; i++) {
        LayerWeightCache *lc = &layer_cache[i];
        int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
        // Norms
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
        lc->input_norm_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
        lc->post_attn_norm_w = get_tensor_ptr(wf, name);
        if (is_full) {
            // Full attention
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
            lc->q_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", i);
            lc->q_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", i);
            lc->q_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
            lc->k_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", i);
            lc->k_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", i);
            lc->k_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
            lc->v_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", i);
            lc->v_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", i);
            lc->v_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
            lc->o_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", i);
            lc->o_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", i);
            lc->o_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", i);
            lc->q_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", i);
            lc->k_norm_w = get_tensor_ptr(wf, name);
        } else {
            // Linear attention
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            lc->qkv_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", i);
            lc->qkv_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", i);
            lc->qkv_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", i);
            lc->z_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", i);
            lc->z_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", i);
            lc->z_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", i);
            lc->b_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", i);
            lc->b_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", i);
            lc->b_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", i);
            lc->a_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", i);
            lc->a_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", i);
            lc->a_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", i);
            lc->conv1d_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", i);
            lc->A_log = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", i);
            lc->dt_bias = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", i);
            lc->gated_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", i);
            lc->out_proj_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", i);
            lc->out_proj_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", i);
            lc->out_proj_b = get_tensor_ptr(wf, name);
        }
        // MoE router weights
        snprintf(name, sizeof(name), "model.layers.%d.block_sparse_moe.gate.weight", i);
        lc->gate_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.block_sparse_moe.gate.scales", i);
        lc->gate_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.block_sparse_moe.gate.biases", i);
        lc->gate_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.block_sparse_moe.e_score_correction_bias", i);
        lc->gate_bias_corr = get_tensor_ptr(wf, name);
        // MiniMax M2.7 has no shared expert path. Leave the legacy shared
        // expert cache pointers unset so we do not probe Qwen-only tensor names.
    }
    layer_cache_built = 1;
    printf("[cache] Pre-computed weight pointers for %d layers\n", NUM_LAYERS);
}
// ============================================================================
// Deferred expert state: holds state for async GPU expert compute.
// GPU experts are submitted async (commit without wait), and the wait+combine
// happens at the start of the NEXT layer. This overlaps ~1ms of GPU expert
// compute with the next layer's attention+routing CPU/GPU work.
// ============================================================================
typedef struct {
    int active;                         // 1 if there's a deferred GPU expert to wait for
    int gpu_combined;                   // 1 if CMD3 includes combine+residual+norm on GPU
                                        // (next layer can skip deferred_wait+finalize+input_norm
                                        //  and submit CMD1 immediately -- buf_input is ready)
    id<MTLCommandBuffer> cmd_experts;   // the async command buffer (committed but not waited)
    float expert_weights[MAX_K];        // routing weights for weighted accumulation
    int valid[MAX_K];                   // which experts loaded successfully
    int actual_K;                       // number of experts
    float h_mid[HIDDEN_DIM];            // saved h_mid for final combine
    float shared_gate_score;            // saved shared expert gate score
    float *hidden;                      // pointer to hidden state (for writing final result)
    int layer_idx;                      // which layer produced this deferred state
} DeferredExpertState;
static DeferredExpertState g_deferred = { .active = 0 };
// Wait for the deferred GPU expert command buffer to complete.
// Split from finalize so timing can be measured independently.
static void wait_deferred_experts_gpu(void) {
    if (!g_deferred.active) return;
    [g_deferred.cmd_experts waitUntilCompleted];
}
// CPU readback + accumulate + combine after GPU is done.
// Must be called after wait_deferred_experts_gpu().
// When gpu_combined=1, the GPU already computed the combine+residual+norm
// in CMD3, so we just need to read back the hidden state from buf_moe_hidden.
static void finalize_deferred_experts(void) {
    if (!g_deferred.active) return;
    float dump_moe_out[HIDDEN_DIM];
    int want_dump = dump_is_active();
    if (want_dump) memset(dump_moe_out, 0, sizeof(dump_moe_out));
    if (g_deferred.gpu_combined) {
        if (want_dump) {
            for (int k = 0; k < g_deferred.actual_K; k++) {
                if (!g_deferred.valid[k]) continue;
                float *expert_result = (float *)[g_metal->buf_multi_expert_out[k] contents];
                cpu_vec_madd(dump_moe_out, expert_result, g_deferred.expert_weights[k], HIDDEN_DIM);
            }
        }
        // GPU-side combine: hidden state is already in buf_moe_hidden.
        // buf_input already has the normalized input for the next layer's CMD1.
        // Just read back hidden (needed for the residual connection in future layers).
        memcpy(g_deferred.hidden, [g_metal->buf_moe_hidden contents],
               HIDDEN_DIM * sizeof(float));
    } else {
        // CPU-side combine (original path)
        // Read back and accumulate routed expert outputs
        float moe_out[HIDDEN_DIM];
        memset(moe_out, 0, sizeof(moe_out));
        for (int k = 0; k < g_deferred.actual_K; k++) {
            if (!g_deferred.valid[k]) continue;
            float *expert_result = (float *)[g_metal->buf_multi_expert_out[k] contents];
            cpu_vec_madd(moe_out, expert_result, g_deferred.expert_weights[k], HIDDEN_DIM);
        }
        if (want_dump) memcpy(dump_moe_out, moe_out, sizeof(dump_moe_out));
        // MiniMax has no shared expert path.
        for (int i = 0; i < HIDDEN_DIM; i++) {
            g_deferred.hidden[i] = g_deferred.h_mid[i] + moe_out[i];
        }
    }
    if (want_dump) {
        dump_write_layer_f32(g_deferred.layer_idx, "moe_out", dump_moe_out, HIDDEN_DIM);
        dump_write_layer_f32(g_deferred.layer_idx, "hidden_out", g_deferred.hidden, HIDDEN_DIM);
    }
    prefill_capture_write_hidden_out(g_deferred.layer_idx, g_deferred.hidden);
    g_deferred.active = 0;
    g_deferred.gpu_combined = 0;
    g_deferred.cmd_experts = nil;
}
// Complete the deferred GPU expert compute: wait for GPU, read back, accumulate, combine.
// Must be called before the next layer modifies static scratch buffers.
static void complete_deferred_experts(void) {
    wait_deferred_experts_gpu();
    finalize_deferred_experts();
}
// Discard the deferred GPU expert result: wait for GPU to finish (for buffer safety)
// but skip the CPU readback/combine. Used during prefill for intermediate tokens
// where the hidden state will be immediately overwritten by the next token's embedding.
// This saves ~0.1-0.2ms per prefill token (avoids unnecessary memcpy + combine work).
static void discard_deferred_experts(void) {
    wait_deferred_experts_gpu();
    // Clear deferred state without reading back results
    if (g_deferred.active) {
        g_deferred.active = 0;
        g_deferred.gpu_combined = 0;
        g_deferred.cmd_experts = nil;
    }
}
// ============================================================================
// Fused layer forward: GPU/CPU overlap + deferred expert pipeline
//
// Pipeline per layer (3 cmd buffers, GPU-side combine in CMD3):
//
//   FAST PATH (when previous CMD3 did GPU-side combine):
//     CMD1: submit immediately (buf_input already populated by CMD3(N-1))
//     WAIT: CMD1 complete (implies CMD3(N-1) also done, queue is serial)
//     CPU:  finalize deferred (read back hidden from buf_moe_hidden)
//
//   SLOW PATH (first layer, or last layer's CMD3 without GPU combine):
//     [DEFERRED] Wait for PREVIOUS layer's CMD3 (if any) + CPU combine
//     CPU:  input_norm(hidden) -> normed -> buf_input
//     CMD1: attention projections (commit)
//     WAIT: CMD1 complete
//
//   Then (both paths):
//     CPU:  attention compute (RoPE/softmax/delta-net)
//     CMD2: o_proj + residual + norm + routing + shared expert projs (8 encoders, 1 commit)
//     WAIT: CMD2 complete
//     CPU:  softmax + top-K routing
//     I/O:  parallel pread K experts (4 pthreads)
//     CMD3: K expert forwards + shared SwiGLU + shared down
//           + moe_combine_residual + rms_norm -> buf_input (ASYNC commit, NO wait)
//     RETURN: GPU experts + combine running async
//
// GPU-side combine eliminates the 0.83ms deferred_wait + CPU combine + input_norm
// at the start of each layer, allowing CMD1 to be submitted immediately.
//
// Key optimizations:
//   1. Parallel pread (4 threads) instead of sequential: ~4x I/O speedup
//   2. o_proj fused into CMD2 with routing (saves 1 commit+wait)
//   3. Deferred CMD3 (expert GPU compute overlapped with next layer)
//   4. GPU-side combine in CMD3 (eliminates CPU deferred_wait + combine + norm)
// ============================================================================
// Static scratch buffers — allocated once, reused across all 60 layers per token.
// Eliminates ~20 malloc/free per layer = ~1200 alloc/free per token.
static float *s_normed    = NULL;   // [HIDDEN_DIM]
static float *s_residual  = NULL;   // [HIDDEN_DIM]
static float *s_attn_proj = NULL;   // [HIDDEN_DIM]
static float *s_h_post    = NULL;   // [HIDDEN_DIM]
static float *s_h_mid     = NULL;   // [HIDDEN_DIM]
static float *s_gate_scores = NULL; // [NUM_EXPERTS]
static float *s_spec_gate_scores = NULL; // [NUM_EXPERTS] speculative routing scratch
static int s_spec_indices[MAX_K];         // speculative routing predicted expert indices
static int s_spec_count = 0;              // number of speculative predictions this layer
static float *s_shared_gate = NULL; // [SHARED_INTERMEDIATE]
static float *s_shared_up  = NULL;  // [SHARED_INTERMEDIATE]
static float *s_moe_out   = NULL;   // [HIDDEN_DIM]
static float *s_shared_out = NULL;  // [HIDDEN_DIM]
// Full attention scratch
static float *s_q_proj_out = NULL;  // [NUM_ATTN_HEADS * HEAD_DIM * 2]
static float *s_k_proj_out = NULL;  // [NUM_KV_HEADS * HEAD_DIM]
static float *s_v_proj_out = NULL;  // [NUM_KV_HEADS * HEAD_DIM]
static float *s_q         = NULL;   // [NUM_ATTN_HEADS * HEAD_DIM]
static float *s_attn_out  = NULL;   // [NUM_ATTN_HEADS * HEAD_DIM]
// Linear attention scratch
static float *s_qkv_proj_out = NULL;   // [LINEAR_CONV_DIM]
static float *s_z_proj_out   = NULL;   // [LINEAR_TOTAL_VALUE]
static float *s_beta_proj_out = NULL;  // [LINEAR_NUM_V_HEADS]
static float *s_alpha_proj_out = NULL; // [LINEAR_NUM_V_HEADS]
static float *s_conv_out  = NULL;   // [LINEAR_CONV_DIM]
static float *s_out_vals  = NULL;   // [LINEAR_TOTAL_VALUE]
static float *s_gated_out = NULL;   // [LINEAR_TOTAL_VALUE]
static void init_layer_scratch(void) {
    if (s_normed) return;  // already initialized
    s_normed     = calloc(HIDDEN_DIM, sizeof(float));
    s_residual   = calloc(HIDDEN_DIM, sizeof(float));
    s_attn_proj  = calloc(HIDDEN_DIM, sizeof(float));
    s_h_post     = calloc(HIDDEN_DIM, sizeof(float));
    s_h_mid      = calloc(HIDDEN_DIM, sizeof(float));
    s_gate_scores = calloc(NUM_EXPERTS, sizeof(float));
    s_spec_gate_scores = calloc(NUM_EXPERTS, sizeof(float));
    s_shared_gate = calloc(SHARED_INTERMEDIATE, sizeof(float));
    s_shared_up  = calloc(SHARED_INTERMEDIATE, sizeof(float));
    s_moe_out    = calloc(HIDDEN_DIM, sizeof(float));
    s_shared_out = calloc(HIDDEN_DIM, sizeof(float));
    s_q_proj_out = calloc(NUM_ATTN_HEADS * HEAD_DIM * 2, sizeof(float));
    s_k_proj_out = calloc(NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    s_v_proj_out = calloc(NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    s_q          = calloc(NUM_ATTN_HEADS * HEAD_DIM, sizeof(float));
    s_attn_out   = calloc(NUM_ATTN_HEADS * HEAD_DIM, sizeof(float));
    s_qkv_proj_out = calloc(LINEAR_CONV_DIM, sizeof(float));
    s_z_proj_out   = calloc(LINEAR_TOTAL_VALUE, sizeof(float));
    s_beta_proj_out = calloc(LINEAR_NUM_V_HEADS, sizeof(float));
    s_alpha_proj_out = calloc(LINEAR_NUM_V_HEADS, sizeof(float));
    s_conv_out   = calloc(LINEAR_CONV_DIM, sizeof(float));
    s_out_vals   = calloc(LINEAR_TOTAL_VALUE, sizeof(float));
    s_gated_out  = calloc(LINEAR_TOTAL_VALUE, sizeof(float));
}
static void fused_layer_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [HIDDEN_DIM] in/out
    KVCache *kv,             // non-NULL for full attention layers
    LinearAttnState *la_state, // non-NULL for linear attention layers
    int pos,                 // position for RoPE
    const void *mmap_base,   // mmap'd layer file (NULL if not available)
    int K,                   // number of active experts
    int packed_fd            // fd for packed expert file
) {
    double t_layer_start = 0, t0 = 0, t1 = 0;
    if (g_timing_enabled) { t_layer_start = now_ms(); }
    int pred_started = 0;  // set to 1 if we started prediction preads during CMD1_wait
    init_layer_scratch();
    if (!layer_cache_built) build_layer_cache(wf);
    LayerWeightCache *lc = &layer_cache[layer_idx];
    int is_full = (kv != NULL);
    // =====================================================================
    // PHASE 1: Deferred completion + CMD1 (attention projections)
    // =====================================================================
    // ---- Prepare attention projection specs (doesn't depend on hidden) ----
    int num_attn_specs = 0;
    BatchMatvecSpec attn_specs[5];
    float *q_proj_out = NULL, *k_out = NULL, *v_out = NULL;
    float *qkv_out = NULL, *z_out = NULL, *beta_out = NULL, *alpha_out = NULL;
    if (is_full) {
        int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM;
        int kv_dim = NUM_KV_HEADS * HEAD_DIM;
        q_proj_out = s_q_proj_out;
        k_out = s_k_proj_out;
        v_out = s_v_proj_out;
        if (lc->q_w && lc->q_s && lc->q_b && lc->k_w && lc->k_s && lc->k_b &&
            lc->v_w && lc->v_s && lc->v_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->q_w, lc->q_s, lc->q_b, q_proj_out, (uint32_t)q_proj_dim, HIDDEN_DIM, GROUP_SIZE, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->k_w, lc->k_s, lc->k_b, k_out,      (uint32_t)kv_dim,     HIDDEN_DIM, GROUP_SIZE, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->v_w, lc->v_s, lc->v_b, v_out,      (uint32_t)kv_dim,     HIDDEN_DIM, GROUP_SIZE, 2 };
            num_attn_specs = 3;
        }
    } else {
        int qkv_dim = LINEAR_CONV_DIM;
        int z_dim = LINEAR_TOTAL_VALUE;
        qkv_out = s_qkv_proj_out;
        z_out = s_z_proj_out;
        beta_out = s_beta_proj_out;
        alpha_out = s_alpha_proj_out;
        if (lc->qkv_w && lc->qkv_s && lc->qkv_b && lc->z_w && lc->z_s && lc->z_b &&
            lc->b_w && lc->b_s && lc->b_b && lc->a_w && lc->a_s && lc->a_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->qkv_w, lc->qkv_s, lc->qkv_b, qkv_out,   (uint32_t)qkv_dim,            HIDDEN_DIM, GROUP_SIZE, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->z_w,   lc->z_s,   lc->z_b,   z_out,      (uint32_t)z_dim,              HIDDEN_DIM, GROUP_SIZE, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->b_w,   lc->b_s,   lc->b_b,   beta_out,   (uint32_t)LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, 2 };
            attn_specs[3] = (BatchMatvecSpec){ lc->a_w,   lc->a_s,   lc->a_b,   alpha_out,  (uint32_t)LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, 3 };
            num_attn_specs = 4;
        }
    }
    // ---- Deferred completion + CMD1 (sequential) ----
    float *normed = s_normed;
    float *residual = s_residual;
    id<MTLCommandBuffer> cmd1 = nil;
    int gpu_linear_attn = 0;  // set to 1 if GPU handles entire linear attention pipeline
    // Pre-compute linear_layer_idx for GPU linear attention encoding in CMD1
    int linear_layer_idx = -1;
    if (!is_full) {
        linear_layer_idx = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL;
    }
    // Can we run the full linear attention pipeline on GPU in CMD1?
    int can_gpu_linear = (gpu_linear_attn_enabled &&
                          !is_full && g_metal && g_metal->delta_net_step &&
                          g_metal->conv1d_step && g_metal->rms_norm_qk &&
                          g_metal->compute_decay_beta && g_metal->gated_rms_norm &&
                          g_metal->wf_buf &&
                          linear_layer_idx >= 0 && linear_layer_idx < NUM_LINEAR_LAYERS &&
                          lc->conv1d_w && lc->A_log && lc->dt_bias && lc->gated_norm_w &&
                          !linear_attn_bypass);
    // Check if previous layer's CMD3 already computed combine+residual+norm on GPU.
    // If so, buf_input already contains the normalized input for this layer's CMD1.
    // We can submit CMD1 immediately — the GPU queue serializes CMD3(N-1) then CMD1(N).
    int prev_gpu_combined = (g_deferred.active && g_deferred.gpu_combined);
    if (prev_gpu_combined && g_metal && g_metal->wf_buf && num_attn_specs > 0) {
        // ---- FAST PATH: GPU-combined previous CMD3 ----
        // buf_input already has the normalized hidden state from CMD3(N-1).
        // Submit CMD1 immediately — GPU runs CMD3(N-1) then CMD1(N) back-to-back.
        if (g_timing_enabled) { t0 = now_ms(); }
        cmd1 = [g_metal->queue commandBuffer];
        gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);
        // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
        if (can_gpu_linear && num_attn_specs == 4) {
            // batch_out[0]=qkv(12288), [1]=z(8192), [2]=beta(64), [3]=alpha(64)
            uint32_t conv_dim = LINEAR_CONV_DIM;
            NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[g_metal->wf_buf contents]);
            // Enc L1: conv1d_step — input=batch_out[0], weights=conv1d_w, state=buf_conv_state, output=buf_conv_output
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->conv1d_step];
                [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:g_metal->batch_out[0]    offset:0            atIndex:1]; // qkv projection output
                [enc setBuffer:g_metal->wf_buf          offset:conv_w_off   atIndex:2]; // conv weights (bf16)
                [enc setBuffer:g_metal->buf_conv_output offset:0            atIndex:3]; // conv output
                [enc setBytes:&conv_dim length:4 atIndex:4];
                uint32_t tgs = (conv_dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc L2: rms_norm_qk — normalize q and k in conv_output in-place
            {
                uint32_t key_dim = LINEAR_KEY_DIM;  // 128
                float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->rms_norm_qk];
                [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];  // q at offset 0
                [enc setBuffer:g_metal->buf_conv_output offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:1];  // k at offset 2048 floats
                [enc setBytes:&key_dim   length:4 atIndex:2];
                [enc setBytes:&inv_scale length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_K_HEADS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(LINEAR_KEY_DIM, 1, 1)];
                [enc endEncoding];
            }
            // Enc L3: compute_decay_beta — alpha=batch_out[3], beta=batch_out[2], A_log+dt_bias from wf_buf
            {
                NSUInteger a_log_off   = (NSUInteger)((const char *)lc->A_log   - (const char *)[g_metal->wf_buf contents]);
                NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias  - (const char *)[g_metal->wf_buf contents]);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->compute_decay_beta];
                [enc setBuffer:g_metal->batch_out[3]       offset:0          atIndex:0]; // alpha
                [enc setBuffer:g_metal->batch_out[2]       offset:0          atIndex:1]; // beta
                [enc setBuffer:g_metal->wf_buf             offset:a_log_off  atIndex:2]; // A_log
                [enc setBuffer:g_metal->wf_buf             offset:dt_bias_off atIndex:3]; // dt_bias (bf16)
                [enc setBuffer:g_metal->buf_delta_g_decay  offset:0          atIndex:4]; // g_decay output
                [enc setBuffer:g_metal->buf_delta_beta     offset:0          atIndex:5]; // beta_gate output
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)];
                [enc endEncoding];
            }
            // Enc L4: gated_delta_net_step — the main recurrence
            {
                uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;  // 4
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->delta_net_step];
                [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0]; // persistent state
                [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:1]; // q (first 2048 floats)
                [enc setBuffer:g_metal->buf_conv_output offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:2]; // k (next 2048)
                [enc setBuffer:g_metal->buf_conv_output offset:2 * LINEAR_TOTAL_KEY * sizeof(float) atIndex:3]; // v (next 8192)
                [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6]; // output [8192]
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
            }
            // Enc L5: gated_rms_norm — normalize+gate delta-net output -> batch_out[6] for CMD2 o_proj
            {
                NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[g_metal->wf_buf contents]);
                uint32_t value_dim = LINEAR_VALUE_DIM;  // 128
                float eps = RMS_NORM_EPS;
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->gated_rms_norm];
                [enc setBuffer:g_metal->buf_delta_output offset:0          atIndex:0]; // values [8192]
                [enc setBuffer:g_metal->batch_out[1]     offset:0          atIndex:1]; // z (z projection output) [8192]
                [enc setBuffer:g_metal->wf_buf           offset:gnorm_w_off atIndex:2]; // weight (bf16)
                [enc setBuffer:g_metal->batch_out[6]     offset:0          atIndex:3]; // output -> batch_out[6] for CMD2
                [enc setBytes:&value_dim length:4 atIndex:4];
                [enc setBytes:&eps       length:4 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
                [enc endEncoding];
            }
            gpu_linear_attn = 1;
        }
        [cmd1 commit];
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }
        // Wait for CMD1 (implies CMD3(N-1) also done, since queue is serial)
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd1 waitUntilCompleted];
        if (!gpu_linear_attn) {
            gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }
        // Now CMD3(N-1) is done. Read back hidden state from GPU.
        if (g_timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts();  // reads buf_moe_hidden -> hidden
        // Start predicted expert preads AFTER CMD1_wait.
        // CMD3(N-1) is guaranteed done (serial queue), so buf_B is safe to overwrite.
        // Predictions overlap with CPU attn + CMD2 + routing (~0.6ms head start).
        // Predicted experts that hit page cache (same as previous token) complete in ~0.1ms.
        if (g_pred_enabled && g_pred_generating && g_pred_valid && packed_fd >= 0 &&
            !g_malloc_cache && !g_expert_cache &&
            g_metal->buf_multi_expert_data_B[0] && g_pred_count[layer_idx] > 0) {
            async_pread_start(packed_fd, g_pred_experts[layer_idx],
                              g_pred_count[layer_idx],
                              g_metal->buf_multi_expert_data_B, mmap_base);
            pred_started = 1;
        }
        // Set up residual for CMD2 (residual = hidden before this layer's attention)
        cpu_vec_copy(residual, hidden, HIDDEN_DIM);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }
        // No input_norm needed — CMD3 already computed it into buf_input.
        // normed is only needed if speculative routing is enabled (currently disabled).
        // Skip the readback to avoid unnecessary overhead.
    } else {
        // ---- ORIGINAL PATH: CPU deferred completion + input norm ----
        // Complete deferred experts from previous layer
        if (g_timing_enabled) { t0 = now_ms(); }
        wait_deferred_experts_gpu();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_wait += t1 - t0; }
        if (g_timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }
        // Input norm
        if (g_timing_enabled) { t0 = now_ms(); }
        cpu_vec_copy(residual, hidden, HIDDEN_DIM);
        cpu_rms_norm(hidden, lc->input_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.input_norm += t1 - t0; }
        // Submit CMD1: attention projections
        if (g_timing_enabled) { t0 = now_ms(); }
        if (g_metal && g_metal->wf_buf && num_attn_specs > 0) {
            memcpy([g_metal->buf_input contents], normed, HIDDEN_DIM * sizeof(float));
            cmd1 = [g_metal->queue commandBuffer];
            gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);
            // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
            if (can_gpu_linear && num_attn_specs == 4) {
                uint32_t conv_dim = LINEAR_CONV_DIM;
                NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[g_metal->wf_buf contents]);
                // Enc L1: conv1d_step
                {
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->conv1d_step];
                    [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:g_metal->batch_out[0]    offset:0            atIndex:1];
                    [enc setBuffer:g_metal->wf_buf          offset:conv_w_off   atIndex:2];
                    [enc setBuffer:g_metal->buf_conv_output offset:0            atIndex:3];
                    [enc setBytes:&conv_dim length:4 atIndex:4];
                    uint32_t tgs = (conv_dim + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                // Enc L2: rms_norm_qk
                {
                    uint32_t key_dim = LINEAR_KEY_DIM;
                    float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->rms_norm_qk];
                    [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_conv_output offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:1];
                    [enc setBytes:&key_dim   length:4 atIndex:2];
                    [enc setBytes:&inv_scale length:4 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_K_HEADS, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(LINEAR_KEY_DIM, 1, 1)];
                    [enc endEncoding];
                }
                // Enc L3: compute_decay_beta
                {
                    NSUInteger a_log_off   = (NSUInteger)((const char *)lc->A_log   - (const char *)[g_metal->wf_buf contents]);
                    NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias  - (const char *)[g_metal->wf_buf contents]);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->compute_decay_beta];
                    [enc setBuffer:g_metal->batch_out[3]       offset:0          atIndex:0];
                    [enc setBuffer:g_metal->batch_out[2]       offset:0          atIndex:1];
                    [enc setBuffer:g_metal->wf_buf             offset:a_log_off  atIndex:2];
                    [enc setBuffer:g_metal->wf_buf             offset:dt_bias_off atIndex:3];
                    [enc setBuffer:g_metal->buf_delta_g_decay  offset:0          atIndex:4];
                    [enc setBuffer:g_metal->buf_delta_beta     offset:0          atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)];
                    [enc endEncoding];
                }
                // Enc L4: gated_delta_net_step
                {
                    uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->delta_net_step];
                    [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_conv_output offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:2];
                    [enc setBuffer:g_metal->buf_conv_output offset:2 * LINEAR_TOTAL_KEY * sizeof(float) atIndex:3];
                    [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                    [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                    [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
                    [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding];
                }
                // Enc L5: gated_rms_norm -> batch_out[6]
                {
                    NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[g_metal->wf_buf contents]);
                    uint32_t value_dim = LINEAR_VALUE_DIM;
                    float eps = RMS_NORM_EPS;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->gated_rms_norm];
                    [enc setBuffer:g_metal->buf_delta_output offset:0          atIndex:0];
                    [enc setBuffer:g_metal->batch_out[1]     offset:0          atIndex:1];
                    [enc setBuffer:g_metal->wf_buf           offset:gnorm_w_off atIndex:2];
                    [enc setBuffer:g_metal->batch_out[6]     offset:0          atIndex:3];
                    [enc setBytes:&value_dim length:4 atIndex:4];
                    [enc setBytes:&eps       length:4 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
                    [enc endEncoding];
                }
                gpu_linear_attn = 1;
            }
            [cmd1 commit];
        } else {
            for (int i = 0; i < num_attn_specs; i++) {
                BatchMatvecSpec *s = &attn_specs[i];
                cpu_dequant_matvec(s->W, s->scales, s->biases, normed, s->out_cpu,
                                   s->out_dim, s->in_dim, s->group_size);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }
        // Wait for CMD1
        if (g_timing_enabled) { t0 = now_ms(); }
        if (cmd1) {
            [cmd1 waitUntilCompleted];
            if (!gpu_linear_attn) {
                gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }
    }
    if (dump_is_active()) {
        if (prev_gpu_combined && g_metal && g_metal->buf_input) {
            memcpy(normed, [g_metal->buf_input contents], HIDDEN_DIM * sizeof(float));
        }
        dump_write_layer_f32(layer_idx, "hidden_in", residual, HIDDEN_DIM);
        dump_write_layer_f32(layer_idx, "normed_in", normed, HIDDEN_DIM);
    }
    // =====================================================================
    // SPECULATIVE EARLY ROUTING — overlap expert I/O with CPU attention
    // =====================================================================
    // Compute approximate routing using the PRE-attention normed hidden state.
    // The real routing (in CMD2/PHASE 3) uses the POST-attention state, so this
    // is an approximation. Fire off async pread for predicted cache misses via
    // dispatch_group so the I/O runs concurrently with CPU attention compute.
    // After CPU attention, we wait for the group to finish. When the real routing
    // happens later, predicted experts are already in the LRU cache as hits.
    dispatch_group_t spec_group = NULL;
    int spec_preload_count = 0;
    int spec_routing_enabled = 0;  // DISABLED: cache pollution + overhead makes it slower
    if (g_timing_enabled) { t0 = now_ms(); }
    s_spec_count = 0;
    ExpertLRUCache *spec_cache = active_expert_lru_cache();
    if (spec_routing_enabled && (spec_cache || g_malloc_cache) && packed_fd >= 0 && lc->gate_w) {
        float *spec_scores = s_spec_gate_scores;
        memset(spec_scores, 0, NUM_EXPERTS * sizeof(float));
        // Gate projection matvec on pre-attention normed input (CPU, ~0.1ms for 512x4096)
        cpu_dequant_matvec(lc->gate_w, lc->gate_s, lc->gate_b,
                           normed, spec_scores,
                           NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE);
        cpu_softmax(spec_scores, NUM_EXPERTS);
        int spec_K = (K > MAX_K) ? MAX_K : K;
        float spec_weights[MAX_K];
        cpu_topk(spec_scores, NUM_EXPERTS, spec_K, s_spec_indices, spec_weights);
        s_spec_count = spec_K;
        g_spec_route_attempts += spec_K;
        // Initialize GCD queue if needed
        if (!g_io_gcd_queue)
            g_io_gcd_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        // Check cache for each predicted expert, start async I/O for misses
        size_t spec_esz = active_expert_size();
        if (g_malloc_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, eidx);
                if (!cached) {
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, eidx, &cidx);
                    if (buf && cidx >= 0) {
                        int fd_copy = packed_fd;
                        void *dst = g_malloc_cache->data[cidx];
                        off_t offset = (off_t)eidx * spec_esz;
                        size_t sz = spec_esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        } else if (spec_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = expert_cache_lookup(spec_cache, layer_idx, eidx);
                if (!cached) {
                    id<MTLBuffer> buf = expert_cache_insert(spec_cache, layer_idx, eidx);
                    if (buf) {
                        int fd_copy = packed_fd;
                        void *dst = [buf contents];
                        off_t offset = (off_t)eidx * spec_esz;
                        size_t sz = spec_esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        }
    }
    (void)spec_preload_count;  // tracked via g_spec_route_preloads
    if (g_timing_enabled) { t1 = now_ms(); g_timing.spec_route += t1 - t0; }
    // =====================================================================
    // PHASE 2: CPU attention compute
    // =====================================================================
    if (g_timing_enabled) { t0 = now_ms(); }
    float *attn_projected = s_attn_proj;
    memset(attn_projected, 0, HIDDEN_DIM * sizeof(float));
    // Pre-lookup o_proj / out_proj weights (used after attention compute)
    // These are looked up NOW to avoid repeated snprintf later.
    uint32_t *oproj_w = NULL;
    uint16_t *oproj_s = NULL, *oproj_b = NULL;
    int oproj_in_dim = 0;
    if (is_full) {
        oproj_w = lc->o_w; oproj_s = lc->o_s; oproj_b = lc->o_b;
        oproj_in_dim = NUM_ATTN_HEADS * HEAD_DIM;
    } else if (!linear_attn_bypass) {
        oproj_w = lc->out_proj_w; oproj_s = lc->out_proj_s; oproj_b = lc->out_proj_b;
        oproj_in_dim = LINEAR_TOTAL_VALUE;
    }
    // All MoE weight pointers from cache (zero snprintf overhead)
    uint32_t *gate_w = lc->gate_w;
    uint16_t *gate_s = lc->gate_s;
    uint16_t *gate_b = lc->gate_b;
    // ---- CPU attention compute (produces attn_out for o_proj) ----
    float *attn_out_for_oproj = NULL;
    if (is_full) {
        // ---- Full attention CPU compute ----
        int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM;
        int q_dim = NUM_ATTN_HEADS * HEAD_DIM;
        int kv_dim = NUM_KV_HEADS * HEAD_DIM;
        (void)q_proj_dim;
        float *q = s_q;
        memcpy(q, q_proj_out, q_dim * sizeof(float));
        if (dump_is_active()) {
            dump_write_layer_f32(layer_idx, "q_proj", q_proj_out, q_dim);
            dump_write_layer_f32(layer_idx, "k_proj", k_out, kv_dim);
            dump_write_layer_f32(layer_idx, "v_proj", v_out, kv_dim);
        }
        // MiniMax applies RMSNorm over the full flattened Q and K vectors.
        cpu_rms_norm_inplace(lc->q_norm_w, q, q_dim, RMS_NORM_EPS);
        cpu_rms_norm_inplace(lc->k_norm_w, k_out, kv_dim, RMS_NORM_EPS);
        if (dump_is_active()) {
            dump_write_layer_f32(layer_idx, "q_norm", q, q_dim);
            dump_write_layer_f32(layer_idx, "k_norm", k_out, kv_dim);
        }
        // RoPE
        apply_rotary_emb(q, k_out, pos, NUM_ATTN_HEADS, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM);
        if (dump_is_active()) {
            dump_write_layer_f32(layer_idx, "q_rope", q, q_dim);
            dump_write_layer_f32(layer_idx, "k_rope", k_out, kv_dim);
        }
        // Update KV cache (CPU + GPU mirror)
        int cache_pos = kv->len;
        memcpy(kv->k_cache + cache_pos * kv_dim, k_out, kv_dim * sizeof(float));
        memcpy(kv->v_cache + cache_pos * kv_dim, v_out, kv_dim * sizeof(float));
        int fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1;
        if (gpu_full_attn_enabled &&
            g_metal && g_metal->attn_scores_pipe &&
            fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS &&
            g_metal->buf_kv_k[fa_idx] && g_metal->buf_kv_v[fa_idx]) {
            memcpy((float *)[g_metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
                   k_out, kv_dim * sizeof(float));
            memcpy((float *)[g_metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
                   v_out, kv_dim * sizeof(float));
        }
        kv->len++;
        // Scaled dot-product attention (GQA) — GPU or CPU
        int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        float *attn_out = s_attn_out;
        memset(attn_out, 0, q_dim * sizeof(float));
        // GPU attention: defer dispatches to CMD2 (fused into single cmd buffer).
        // During prompt prefill we allow GPU attention from token 0 to cut TTFT.
        // Decode keeps the old seq_len>=32 threshold unless explicitly forced.
        int gpu_attn_ready = (gpu_full_attn_enabled &&
                              g_metal && g_metal->attn_scores_pipe &&
                              fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS &&
                              ((g_force_gpu_prefill_attn && kv->len > 0) || kv->len >= 32) &&
                              kv->len < GPU_KV_SEQ);
        if (gpu_attn_ready) {
            // Copy Q to GPU; attention dispatches will be in CMD2
            memcpy([g_metal->buf_attn_q contents], q, q_dim * sizeof(float));
            // attn_out_for_oproj will be set to NULL below — CMD2 reads buf_attn_out
        } else {
            // CPU fallback
            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                int kv_h = h / heads_per_kv;
                float *qh = q + h * HEAD_DIM;
                float *scores = malloc(kv->len * sizeof(float));
                for (int p = 0; p < kv->len; p++) {
                    float *kp = kv->k_cache + p * kv_dim + kv_h * HEAD_DIM;
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kp[d];
                    scores[p] = dot * scale;
                }
                cpu_softmax(scores, kv->len);
                float *oh = attn_out + h * HEAD_DIM;
                for (int p = 0; p < kv->len; p++) {
                    float *vp = kv->v_cache + p * kv_dim + kv_h * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++) oh[d] += scores[p] * vp[d];
                }
                free(scores);
            }
        }
        if (gpu_attn_ready) {
            attn_out_for_oproj = NULL;  // signal CMD2 to use GPU buf_attn_out
        } else {
            attn_out_for_oproj = attn_out;
            prefill_capture_write_attn_out(layer_idx, attn_out, q_dim);
            if (dump_is_active()) {
                dump_write_layer_f32(layer_idx, "attn_out", attn_out, q_dim);
            }
        }
        // q_proj_out, k_out, v_out, q, attn_out are static scratch.
    } else if (gpu_linear_attn) {
        // ---- GPU linear attention: already computed in CMD1 ----
        // batch_out[6] already contains gated_rms_norm output (8192 floats)
        // Set a non-NULL sentinel so CMD2 enters fused path, but skip the memcpy
        static float gpu_linear_sentinel;
        attn_out_for_oproj = &gpu_linear_sentinel;
    } else {
        // ---- Linear attention CPU compute ----
        if (!linear_attn_bypass) {
            int qkv_dim = LINEAR_CONV_DIM;
            // Conv1d step
            uint16_t *conv_w = lc->conv1d_w;
            float *conv_out = s_conv_out;
            memset(conv_out, 0, qkv_dim * sizeof(float));
            if (conv_w) {
                cpu_conv1d_step(la_state->conv_state, qkv_out, conv_w, conv_out,
                                qkv_dim, CONV_KERNEL_SIZE);
            }
            // Update conv state
            memmove(la_state->conv_state, la_state->conv_state + qkv_dim,
                    (CONV_KERNEL_SIZE - 2) * qkv_dim * sizeof(float));
            memcpy(la_state->conv_state + (CONV_KERNEL_SIZE - 2) * qkv_dim, qkv_out,
                   qkv_dim * sizeof(float));
            // Split into q, k, v
            float *lin_q = conv_out;
            float *lin_k = conv_out + LINEAR_TOTAL_KEY;
            float *lin_v = conv_out + 2 * LINEAR_TOTAL_KEY;
            // RMS normalize q and k
            float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
            for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
                float *qh = lin_q + h * LINEAR_KEY_DIM;
                cpu_rms_norm_bare(qh, qh, LINEAR_KEY_DIM, 1e-6f);
                float q_scale = inv_scale * inv_scale;
                for (int d = 0; d < LINEAR_KEY_DIM; d++) qh[d] *= q_scale;
            }
            for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
                float *kh = lin_k + h * LINEAR_KEY_DIM;
                cpu_rms_norm_bare(kh, kh, LINEAR_KEY_DIM, 1e-6f);
                for (int d = 0; d < LINEAR_KEY_DIM; d++) kh[d] *= inv_scale;
            }
            // Gated delta net recurrence
            float *A_log = lc->A_log;
            uint16_t *dt_bias_bf16 = lc->dt_bias;
            float *out_values = s_out_vals;
            memset(out_values, 0, LINEAR_TOTAL_VALUE * sizeof(float));
            int k_heads_per_v = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
            float g_decay[LINEAR_NUM_V_HEADS];
            float beta_gate_arr[LINEAR_NUM_V_HEADS];
            for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
                float a_val = alpha_out[vh];
                float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
                float A_val = A_log ? expf(A_log[vh]) : 1.0f;
                float softplus_val = logf(1.0f + expf(a_val + dt_b));
                g_decay[vh] = expf(-A_val * softplus_val);
                beta_gate_arr[vh] = cpu_sigmoid(beta_out[vh]);
            }
            // Compute linear_layer_idx: count of non-full-attention layers before this one.
            // Full attention at (layer_idx+1) % 4 == 0, i.e. layers 3,7,11,...
            // linear_layer_idx = layer_idx - number_of_full_layers_at_or_before
            //                  = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL
            int linear_layer_idx = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL;
            // GPU delta-net path (falls back to CPU if pipeline unavailable)
            if (g_metal && g_metal->delta_net_step &&
                linear_layer_idx >= 0 && linear_layer_idx < NUM_LINEAR_LAYERS) {
                // Upload CPU-computed data to GPU scratch buffers
                memcpy([g_metal->buf_delta_q contents], lin_q, LINEAR_TOTAL_KEY * sizeof(float));
                memcpy([g_metal->buf_delta_k contents], lin_k, LINEAR_TOTAL_KEY * sizeof(float));
                memcpy([g_metal->buf_delta_v contents], lin_v, LINEAR_TOTAL_VALUE * sizeof(float));
                memcpy([g_metal->buf_delta_g_decay contents], g_decay, LINEAR_NUM_V_HEADS * sizeof(float));
                memcpy([g_metal->buf_delta_beta contents], beta_gate_arr, LINEAR_NUM_V_HEADS * sizeof(float));
                id<MTLCommandBuffer> cmd_dn = [g_metal->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd_dn computeCommandEncoder];
                [enc setComputePipelineState:g_metal->delta_net_step];
                [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_delta_q       offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_delta_k       offset:0 atIndex:2];
                [enc setBuffer:g_metal->buf_delta_v       offset:0 atIndex:3];
                [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
                uint32_t khpv = (uint32_t)k_heads_per_v;
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cmd_dn commit];
                [cmd_dn waitUntilCompleted];
                // Read back GPU result
                memcpy(out_values, [g_metal->buf_delta_output contents], LINEAR_TOTAL_VALUE * sizeof(float));
            } else {
                // CPU delta-net with Accelerate BLAS
                for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
                    int kh = vh / k_heads_per_v;
                    float g = g_decay[vh];
                    float b_gate = beta_gate_arr[vh];
                    float *S = la_state->ssm_state + vh * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
                    float *v_h = lin_v + vh * LINEAR_VALUE_DIM;
                    float *k_h = lin_k + kh * LINEAR_KEY_DIM;
                    // Step 1: Decay S *= g (BLAS sscal on entire state matrix)
                    cblas_sscal(LINEAR_VALUE_DIM * LINEAR_KEY_DIM, g, S, 1);
                    // Step 2: kv_mem = S @ k (each row dot k)
                    // S is [VALUE_DIM x KEY_DIM] row-major, k is [KEY_DIM]
                    // kv_mem[vi] = sum_ki(S[vi,ki] * k[ki]) = matrix-vector: S @ k
                    float kv_mem_vec[LINEAR_VALUE_DIM];
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                                1.0f, S, LINEAR_KEY_DIM, k_h, 1,
                                0.0f, kv_mem_vec, 1);
                    // Step 3: delta = (v - kv_mem) * beta, then rank-1 update S += k * delta^T
                    // delta[vi] = (v[vi] - kv_mem[vi]) * beta
                    float delta_vec[LINEAR_VALUE_DIM];
                    for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
                        delta_vec[vi] = (v_h[vi] - kv_mem_vec[vi]) * b_gate;
                    }
                    // S += delta @ k^T (rank-1 update: sger)
                    // S[vi,ki] += delta[vi] * k[ki]
                    cblas_sger(CblasRowMajor, LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                               1.0f, delta_vec, 1, k_h, 1, S, LINEAR_KEY_DIM);
                    // Step 4: output = S @ q (matrix-vector multiply)
                    float *q_h = lin_q + kh * LINEAR_KEY_DIM;
                    float *o_h = out_values + vh * LINEAR_VALUE_DIM;
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                                1.0f, S, LINEAR_KEY_DIM, q_h, 1,
                                0.0f, o_h, 1);
                }
            }
            // RMSNormGated
            uint16_t *gated_norm_w = lc->gated_norm_w;
            float *gated_out = s_gated_out;
            memset(gated_out, 0, LINEAR_TOTAL_VALUE * sizeof(float));
            for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
                float *oh = out_values + vh * LINEAR_VALUE_DIM;
                float *zh = z_out + vh * LINEAR_VALUE_DIM;
                float *gh = gated_out + vh * LINEAR_VALUE_DIM;
                if (gated_norm_w) {
                    cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, LINEAR_VALUE_DIM, RMS_NORM_EPS);
                } else {
                    memcpy(gh, oh, LINEAR_VALUE_DIM * sizeof(float));
                }
            }
            attn_out_for_oproj = gated_out;
            // conv_out, out_values are static — no free needed
            // gated_out is static — freed/released after CMD2 submission below
        }
        // else: linear_attn_bypass — attn_projected stays zero
        // qkv_out, z_out, beta_out, alpha_out are static scratch.
    }
    // =====================================================================
    // PHASE 3: FULLY FUSED CMD2 — o_proj + residual + norm + routing (1 cmd buffer)
    //   Eliminates 1 GPU round-trip vs old 2-buffer approach.
    //   GPU handles residual_add + rms_norm between o_proj and routing,
    //   so no CPU intervention is needed. 8 encoders, 1 commit+wait.
    //   Buffer flow: batch_out[6]->buf_output->buf_h_mid->buf_input->batch_out[0-3]
    // =====================================================================
    if (g_timing_enabled) { t1 = now_ms(); g_timing.cpu_attn += t1 - t0; }
    // Wait for speculative expert I/O to complete (overlapped with CPU attention)
    if (spec_group) {
        dispatch_group_wait(spec_group, DISPATCH_TIME_FOREVER);
        spec_group = NULL;  // ARC releases the group
    }
    if (g_timing_enabled) { t0 = now_ms(); }
    float *h_post = s_h_post;
    float *h_mid = s_h_mid;
    float *gate_scores = s_gate_scores;
    memset(gate_scores, 0, NUM_EXPERTS * sizeof(float));
    float *shared_gate = s_shared_gate;
    memset(shared_gate, 0, SHARED_INTERMEDIATE * sizeof(float));
    float *shared_up = s_shared_up;
    memset(shared_up, 0, SHARED_INTERMEDIATE * sizeof(float));
    float shared_gate_score = 0.0f;
    int have_moe_weights = (gate_w && gate_s && gate_b);
    // gpu_attn_fuse: attention dispatches fused into CMD2 (full-attn layers only).
    // Prefill can force this path from token 0 to reduce TTFT; decode keeps the
    // old short-sequence CPU threshold unless the caller explicitly overrides it.
    int gpu_attn_fuse = (gpu_full_attn_enabled &&
                         is_full && !attn_out_for_oproj && g_metal && g_metal->attn_scores_pipe
                         && kv && ((g_force_gpu_prefill_attn && kv->len > 0) || kv->len >= 32)
                         && kv->len < GPU_KV_SEQ);
    if ((attn_out_for_oproj || gpu_attn_fuse) && oproj_w && oproj_s && oproj_b &&
        g_metal && g_metal->wf_buf && have_moe_weights &&
        g_metal->residual_add && g_metal->rms_norm_sum &&
        g_metal->rms_norm_apply_bf16 && lc->post_attn_norm_w) {
        // ---- FULLY FUSED CMD2 ----
        // For GPU attention (full-attn layers): attention dispatches are prepended,
        //   o_proj reads from buf_attn_out instead of batch_out[6].
        // For CPU attention / linear attn: o_proj reads from batch_out[6] as before.
        //
        // GPU attn path (11 encoders):
        //   Enc 1-3: attn_scores + softmax + values -> buf_attn_out
        //   Enc 4:   o_proj (buf_attn_out -> buf_output)
        //   Enc 5-7: residual + norm -> buf_input
        //   Enc 8-11: routing + shared expert
        //
        // CPU attn path (8 encoders, unchanged):
        //   Enc 1:   o_proj (batch_out[6] -> buf_output)
        //   Enc 2-4: residual + norm -> buf_input
        //   Enc 5-8: routing + shared expert
        if (!gpu_attn_fuse && !gpu_linear_attn) {
            // CPU/linear attn: copy attention output to GPU input buffer
            memcpy([g_metal->batch_out[6] contents], attn_out_for_oproj,
                   oproj_in_dim * sizeof(float));
        }
        // gpu_linear_attn: batch_out[6] already has the result from CMD1 gated_rms_norm
        // Copy residual into GPU buffer for residual_add kernel
        memcpy([g_metal->buf_residual contents], residual, HIDDEN_DIM * sizeof(float));
        attn_out_for_oproj = NULL;
        id<MTLCommandBuffer> cmd_fused = [g_metal->queue commandBuffer];
        // ---- GPU attention dispatches (only for full-attn layers with GPU path) ----
        if (gpu_attn_fuse) {
            int fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1;
            int kv_dim = NUM_KV_HEADS * HEAD_DIM;
            int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            uint32_t hd = HEAD_DIM;
            uint32_t kvd = (uint32_t)kv_dim;
            uint32_t sl = (uint32_t)kv->len;
            uint32_t seq_stride = GPU_KV_SEQ;
            uint32_t hpkv = (uint32_t)heads_per_kv;
            // Enc A1: attn_scores_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_scores_pipe];
                [enc setBuffer:g_metal->buf_attn_q          offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]    offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_attn_scores     offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&scale     length:4 atIndex:7];
                [enc setBytes:&hpkv      length:4 atIndex:8];
                [enc setBytes:&sl        length:4 atIndex:9];
                uint32_t total_tgs = sl * NUM_ATTN_HEADS;
                [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A2: attn_softmax_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_softmax_pipe];
                [enc setBuffer:g_metal->buf_attn_scores offset:0 atIndex:0];
                [enc setBytes:&sl         length:4 atIndex:1];
                [enc setBytes:&seq_stride  length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(NUM_ATTN_HEADS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A3: attn_values_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_values_pipe];
                [enc setBuffer:g_metal->buf_attn_scores   offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]  offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_attn_out      offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&hpkv      length:4 atIndex:7];
                uint32_t total_threads = HEAD_DIM * NUM_ATTN_HEADS;
                uint32_t tgs = (total_threads + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }
        // ---- o_proj matvec ----
        {
            NSUInteger w_off = (NSUInteger)((const char *)oproj_w - (const char *)[g_metal->wf_buf contents]);
            NSUInteger s_off = (NSUInteger)((const char *)oproj_s - (const char *)[g_metal->wf_buf contents]);
            NSUInteger b_off = (NSUInteger)((const char *)oproj_b - (const char *)[g_metal->wf_buf contents]);
            // For GPU attention: o_proj reads from buf_attn_out
            // For CPU attention: o_proj reads from batch_out[6]
            id<MTLBuffer> oproj_input = gpu_attn_fuse ? g_metal->buf_attn_out : g_metal->batch_out[6];
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t o_out_dim = HIDDEN_DIM;
            uint32_t o_in_dim = (uint32_t)oproj_in_dim;
            uint32_t o_gs = GROUP_SIZE;
            [enc setComputePipelineState:g_metal->matvec_fast];
            [enc setBuffer:g_metal->wf_buf  offset:w_off atIndex:0];
            [enc setBuffer:g_metal->wf_buf  offset:s_off atIndex:1];
            [enc setBuffer:g_metal->wf_buf  offset:b_off atIndex:2];
            [enc setBuffer:oproj_input      offset:0    atIndex:3];
            [enc setBuffer:g_metal->buf_output offset:0 atIndex:4];
            [enc setBytes:&o_out_dim  length:4 atIndex:5];
            [enc setBytes:&o_in_dim   length:4 atIndex:6];
            [enc setBytes:&o_gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(o_out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }
        // ---- Enc 2: residual_add (buf_output + buf_residual -> buf_h_mid) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = HIDDEN_DIM;
            [enc setComputePipelineState:g_metal->residual_add];
            [enc setBuffer:g_metal->buf_residual offset:0 atIndex:0];  // a = residual
            [enc setBuffer:g_metal->buf_output   offset:0 atIndex:1];  // b = o_proj result
            [enc setBuffer:g_metal->buf_h_mid    offset:0 atIndex:2];  // out = h_mid
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // ---- Enc 3: rms_norm_sum_sq (buf_h_mid -> buf_sum_sq) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = HIDDEN_DIM;
            [enc setComputePipelineState:g_metal->rms_norm_sum];
            [enc setBuffer:g_metal->buf_h_mid  offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_sum_sq offset:0 atIndex:1];
            [enc setBytes:&dim length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // ---- Enc 4: rms_norm_apply_bf16 (buf_h_mid + norm_w -> buf_input) ----
        {
            NSUInteger norm_off = (NSUInteger)((const char *)lc->post_attn_norm_w -
                                               (const char *)[g_metal->wf_buf contents]);
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = HIDDEN_DIM;
            float eps = RMS_NORM_EPS;
            [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
            [enc setBuffer:g_metal->buf_h_mid  offset:0       atIndex:0];  // x
            [enc setBuffer:g_metal->wf_buf     offset:norm_off atIndex:1]; // weight (bf16)
            [enc setBuffer:g_metal->buf_sum_sq offset:0       atIndex:2];  // sum_sq
            [enc setBuffer:g_metal->buf_input  offset:0       atIndex:3];  // out = h_post
            [enc setBytes:&dim length:4 atIndex:4];
            [enc setBytes:&eps length:4 atIndex:5];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // ---- Enc 5: routing gate projection (read buf_input) ----
        BatchMatvecSpec moe_specs[1] = {
            { gate_w, gate_s, gate_b, gate_scores, (uint32_t)NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE, 0 },
        };
        // buf_input already contains h_post from Enc 4 output -- no memcpy needed
        gpu_encode_batch_matvec(g_metal, cmd_fused, moe_specs, 1);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }
        // ---- Single commit+wait for all 8 encoders ----
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd_fused commit];
        [cmd_fused waitUntilCompleted];
        // Read back results
        gpu_flush_batch_results(g_metal, moe_specs, 1);
        if (gpu_attn_fuse) {
            prefill_capture_write_attn_out(layer_idx,
                                           (const float *)[g_metal->buf_attn_out contents],
                                           NUM_ATTN_HEADS * HEAD_DIM);
        }
        // Read h_mid from GPU buffer (needed for final combine)
        memcpy(h_mid, [g_metal->buf_h_mid contents], HIDDEN_DIM * sizeof(float));
        // Read h_post from buf_input (needed for expert input)
        memcpy(h_post, [g_metal->buf_input contents], HIDDEN_DIM * sizeof(float));
        prefill_capture_write_h_mid(layer_idx, h_mid);
        prefill_capture_write_h_post(layer_idx, h_post);
        // Update hidden state to h_mid (= residual + o_proj)
        memcpy(hidden, h_mid, HIDDEN_DIM * sizeof(float));
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_wait += t1 - t0; }
    } else {
        // ---- Non-fused fallback path ----
        // O projection
        if (attn_out_for_oproj && oproj_w && oproj_s && oproj_b) {
            fast_dequant_matvec(oproj_w, oproj_s, oproj_b, attn_out_for_oproj,
                                attn_projected, HIDDEN_DIM, oproj_in_dim, GROUP_SIZE);
        }
        // attn_out_for_oproj is static — no free needed
        attn_out_for_oproj = NULL;
        // Residual connection
        for (int i = 0; i < HIDDEN_DIM; i++) {
            hidden[i] = residual[i] + attn_projected[i];
        }
        // attn_projected, normed, residual are static — no free needed
        cpu_vec_copy(h_mid, hidden, HIDDEN_DIM);
        prefill_capture_write_h_mid(layer_idx, h_mid);
        // Post-attention norm
        cpu_rms_norm(hidden, lc->post_attn_norm_w, h_post, HIDDEN_DIM, RMS_NORM_EPS);
        prefill_capture_write_h_post(layer_idx, h_post);
        // Routing gate
        if (have_moe_weights) {
            BatchMatvecSpec moe_specs[1] = {
                { gate_w, gate_s, gate_b, gate_scores, (uint32_t)NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE, 0 },
            };
            fast_batch_matvec(h_post, HIDDEN_DIM, moe_specs, 1);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }
    }
    if (dump_is_active()) {
        for (int i = 0; i < HIDDEN_DIM; i++) {
            attn_projected[i] = h_mid[i] - residual[i];
        }
        dump_write_layer_f32(layer_idx, "o_proj", attn_projected, HIDDEN_DIM);
        dump_write_layer_f32(layer_idx, "h_mid", h_mid, HIDDEN_DIM);
        dump_write_layer_f32(layer_idx, "h_post", h_post, HIDDEN_DIM);
        dump_write_layer_f32(layer_idx, "gate_logits", gate_scores, NUM_EXPERTS);
    }
    // ---- MiniMax sigmoid routing + top-K (CPU) ----
    if (g_timing_enabled) { t0 = now_ms(); }
    int expert_indices[64];
    float expert_weights[64];
    cpu_route_minimax(gate_scores, lc->gate_bias_corr, NUM_EXPERTS, K,
                      s_spec_gate_scores, expert_indices, expert_weights);
    int routed_K = (K > MAX_K) ? MAX_K : K;
    prefill_microbatch_scaffold_note_route(layer_idx, expert_indices, routed_K);
    if (g_prefill_route_capture.enabled) {
        memcpy(g_prefill_route_capture.h_mid, h_mid, HIDDEN_DIM * sizeof(float));
        memcpy(g_prefill_route_capture.h_post, h_post, HIDDEN_DIM * sizeof(float));
        memcpy(g_prefill_route_capture.expert_indices, expert_indices, routed_K * sizeof(int));
        memcpy(g_prefill_route_capture.expert_weights, expert_weights, routed_K * sizeof(float));
        g_prefill_route_capture.actual_k = routed_K;
        memcpy(hidden, h_mid, HIDDEN_DIM * sizeof(float));
        return;
    }
    if (g_freq_tracking) {
        for (int k = 0; k < routed_K; k++) {
            g_expert_freq[layer_idx][expert_indices[k]]++;
        }
        if (layer_idx == 0) g_freq_total_tokens++;
    }
    if (dump_is_active()) {
        int dump_K = routed_K;
        dump_write_layer_f32(layer_idx, "gate_sigmoid", gate_scores, NUM_EXPERTS);
        dump_write_layer_f32(layer_idx, "gate_choice_scores", s_spec_gate_scores, NUM_EXPERTS);
        dump_write_layer_i32(layer_idx, "expert_indices", (const int32_t *)expert_indices, dump_K);
        dump_write_layer_f32(layer_idx, "expert_weights", expert_weights, dump_K);
    }
    // Track speculative routing prediction accuracy
    if (s_spec_count > 0) {
        int cmp_K = routed_K;
        for (int s = 0; s < s_spec_count; s++) {
            for (int r = 0; r < cmp_K; r++) {
                if (s_spec_indices[s] == expert_indices[r]) {
                    g_spec_route_hits++;
                    break;
                }
            }
        }
    }
    if (g_timing_enabled) { t1 = now_ms(); g_timing.routing_cpu += t1 - t0; }
    // Log routing data for predictor training
    if (g_routing_log) {
        int32_t li = layer_idx;
        int32_t ki = (K > MAX_K) ? MAX_K : K;
        fwrite(&li, sizeof(int32_t), 1, g_routing_log);
        fwrite(&ki, sizeof(int32_t), 1, g_routing_log);
        fwrite(hidden, sizeof(float), HIDDEN_DIM, g_routing_log);
        fwrite(expert_indices, sizeof(int32_t), ki, g_routing_log);
        g_routing_log_samples++;
    }
    // ---- Parallel pread + GPU experts ----
    if (g_timing_enabled) { t0 = now_ms(); }
    float *moe_out = s_moe_out;
    memset(moe_out, 0, HIDDEN_DIM * sizeof(float));
    float *shared_out = s_shared_out;
    memset(shared_out, 0, HIDDEN_DIM * sizeof(float));
    int actual_K = routed_K;
    if (packed_fd >= 0 && g_metal && g_metal->buf_multi_expert_data[0]) {
        ExpertLRUCache *active_cache = active_expert_lru_cache();
        // GPU multi-expert path with LRU cache + parallel I/O:
        // For each expert:
        //   - Cache HIT:  dispatch directly from cached Metal buffer (skip pread)
        //   - Cache MISS: pread into cache buffer, then dispatch from it
        // Falls back to original parallel_pread_experts when cache is disabled.
        int valid[MAX_K];
        id<MTLBuffer> expert_bufs[MAX_K];  // buffer to dispatch from per expert
        if (g_malloc_cache) {
            // ---- Malloc cache path (zero-copy Metal buffer wrappers) ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];
            int miss_cache_idx[MAX_K];  // cache entry index for each miss
            int num_misses = 0;
            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: zero-copy dispatch directly from cache buffer
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert entry (get buffer to pread into)
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, expert_indices[k], &cidx);
                    expert_bufs[k] = buf;
                    miss_indices[num_misses] = k;
                    miss_cache_idx[num_misses] = cidx;
                    num_misses++;
                    valid[k] = 0;
                }
            }
            // Phase 2: parallel pread misses directly into cache buffers (zero-copy)
            if (num_misses > 0) {
                size_t esz = active_expert_size();
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    int cidx = miss_cache_idx[m];
                    infer_pread_task_init(&tasks[m]);
                    // Cache population must read from the active layer file.
                    // Avoid tiered/cold-fd helpers here: cold fds are disabled and may be -1.
                    tasks[m].fd = packed_fd;
                    tasks[m].dst = g_malloc_cache->data[cidx];
                    tasks[m].offset = (off_t)expert_indices[k] * esz;
                    tasks[m].size = esz;
                    tasks[m].mmap_base = NULL;  // always pread for cache population
                }
                io_pool_dispatch(tasks, num_misses);
                // Mark valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)esz);
                    if (!valid[k]) {
                        fprintf(stderr,
                                "WARNING: layer %d expert %d pread fd=%d: %zd/%zu errno=%d (%s)\n",
                                layer_idx, expert_indices[k], tasks[m].fd, tasks[m].result, esz,
                                tasks[m].err,
                                tasks[m].err ? strerror(tasks[m].err) : "short read");
                    }
                }
            }
        } else if (active_cache) {
            // ---- Metal buffer LRU cache path ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];       // indices into expert_indices[] for misses
            id<MTLBuffer> miss_bufs[MAX_K]; // cache buffers to pread into
            int num_misses = 0;
            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = expert_cache_lookup(active_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: use this buffer directly for GPU dispatch
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert into cache (allocates or evicts), will pread below
                    id<MTLBuffer> buf = expert_cache_insert(active_cache, layer_idx, expert_indices[k]);
                    if (buf) {
                        expert_bufs[k] = buf;
                        miss_indices[num_misses] = k;
                        miss_bufs[num_misses] = buf;
                        num_misses++;
                        valid[k] = 0;  // not yet loaded
                    } else {
                        expert_bufs[k] = nil;
                        valid[k] = 0;
                    }
                }
            }
            // Phase 2: parallel pread all cache misses
            if (num_misses > 0) {
                size_t esz = active_expert_size();
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    infer_pread_task_init(&tasks[m]);
                    // Cache population must read from the active layer file.
                    // Avoid tiered/cold-fd helpers here: cold fds are disabled and may be -1.
                    tasks[m].fd = packed_fd;
                    tasks[m].dst = [miss_bufs[m] contents];
                    tasks[m].offset = (off_t)expert_indices[k] * esz;
                    tasks[m].size = esz;
                    tasks[m].mmap_base = mmap_base;
                }
                io_pool_dispatch(tasks, num_misses);
                // Mark successfully loaded misses as valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)esz);
                    if (!valid[k]) {
                        fprintf(stderr,
                                "WARNING: layer %d expert %d pread fd=%d: %zd/%zu errno=%d (%s)\n",
                                layer_idx, expert_indices[k], tasks[m].fd, tasks[m].result, esz,
                                tasks[m].err,
                                tasks[m].err ? strerror(tasks[m].err) : "short read");
                    }
                }
            }
        } else if (pred_started) {
            // ---- Prediction path: predicted experts already loading into buf_B ----
            // Wait for predicted preads (they've had ~1.6ms: CMD1_wait + attn + CMD2 + routing)
            async_pread_wait();
            g_pred_layers++;
            // Match predictions against actual routing
            int miss_ei[MAX_K];       // actual expert indices for misses
            int miss_k_slots[MAX_K];  // which k-slot each miss maps to
            int miss_count = 0;
            int hit_count = 0;
            for (int k = 0; k < actual_K; k++) {
                int found = 0;
                for (int p = 0; p < g_pred_count[layer_idx]; p++) {
                    if (expert_indices[k] == g_pred_experts[layer_idx][p] &&
                        g_async_pread.valid[p]) {
                        // Hit! This expert was pre-loaded into buf_B[p]
                        expert_bufs[k] = g_metal->buf_multi_expert_data_B[p];
                        valid[k] = 1;
                        found = 1;
                        hit_count++;
                        break;
                    }
                }
                if (!found) {
                    miss_ei[miss_count] = expert_indices[k];
                    miss_k_slots[miss_count] = k;
                    expert_bufs[k] = g_metal->buf_multi_expert_data[k];
                    miss_count++;
                }
            }
            g_pred_hits += hit_count;
            g_pred_misses += miss_count;
            // Parallel sync-pread misses into buf_A
            if (miss_count > 0) {
                InferPreadTask tasks[MAX_K];
                size_t esz = active_expert_size();
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    infer_pread_task_init(&tasks[m]);
                    tasks[m].fd = packed_fd;
                    tasks[m].dst = [g_metal->buf_multi_expert_data[k] contents];
                    tasks[m].offset = (off_t)miss_ei[m] * esz;
                    tasks[m].size = esz;
                }
                io_pool_dispatch(tasks, miss_count);
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    valid[k] = (tasks[m].result == (ssize_t)active_expert_size());
                }
            }
        } else if (g_use_lz4 && g_lz4_index[layer_idx]) {
            // ---- LZ4 compressed path: read compressed + decompress via io_pool ----
            size_t esz = active_expert_size();
            InferPreadTask tasks[MAX_K];
            for (int k = 0; k < actual_K; k++) {
                LZ4IndexEntry *ie = &g_lz4_index[layer_idx][expert_indices[k]];
                infer_pread_task_init(&tasks[k]);
                tasks[k].fd = packed_fd;
                tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
                tasks[k].offset = ie->offset;
                tasks[k].size = esz;
                tasks[k].mmap_base = NULL;
                tasks[k].lz4_comp_buf = g_lz4_comp_bufs[k];
                tasks[k].lz4_comp_size = ie->comp_size;
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
            io_pool_dispatch(tasks, actual_K);
            for (int k = 0; k < actual_K; k++) {
                valid[k] = (tasks[k].result == (ssize_t)esz);
            }
        } else {
            // ---- No cache, no prediction, no LZ4: ASYNC parallel pread ----
            async_pread_start(packed_fd, expert_indices, actual_K,
                              g_metal->buf_multi_expert_data, mmap_base);
            for (int k = 0; k < actual_K; k++) {
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
        }
        // Expert prep
        memcpy([g_metal->buf_multi_expert_input contents], h_post, HIDDEN_DIM * sizeof(float));
        memset([g_metal->buf_shared_out contents], 0, HIDDEN_DIM * sizeof(float));
        // Wait for non-prediction async pread to complete
        if (!pred_started && g_async_pread.active) {
            async_pread_wait();
            for (int k = 0; k < actual_K; k++) {
                valid[k] = g_async_pread.valid[k];
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.expert_io += t1 - t0; }
        // Store this layer's routing for next token's temporal prediction.
        // MUST happen AFTER the prediction hit check above (which reads g_pred_experts).
        if (g_pred_enabled && g_pred_generating) {
            for (int k = 0; k < actual_K; k++) {
                g_pred_experts[layer_idx][k] = expert_indices[k];
            }
            g_pred_count[layer_idx] = actual_K;
            if (layer_idx == NUM_LAYERS - 1) {
                g_pred_valid = 1;
            }
        }
        if (g_timing_enabled) { t0 = now_ms(); }
        // Step 3: encode all routed experts into one command buffer.
        id<MTLCommandBuffer> cmd_experts = [g_metal->queue commandBuffer];
        gpu_encode_experts_batched(g_metal, cmd_experts, actual_K, valid, expert_bufs);
        // Step 4: GPU-side combine + residual + norm (if not last layer)
        // Appends dispatches to CMD3 so the next layer's CMD1 can submit immediately
        // without waiting for CMD3 to complete + CPU readback.
        //
        // For non-last layers with the combine pipeline available:
        //   Enc C1: moe_combine_residual (expert_outs + h_mid + shared_out -> buf_moe_hidden)
        //   Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
        //   Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_layer_norm_w -> buf_input)
        //
        // This makes CMD3 self-contained: it produces buf_input for the next layer's CMD1.
        // The next layer skips deferred_wait + finalize + input_norm entirely at layer start.
        int gpu_combine = (g_metal->moe_combine_residual &&
                           g_metal->rms_norm_sum &&
                           g_metal->rms_norm_apply_bf16 &&
                           g_metal->wf_buf &&
                           layer_idx < NUM_LAYERS - 1 &&
                           layer_cache[layer_idx + 1].input_norm_w != NULL);
        if (gpu_combine) {
            // Copy h_mid from buf_h_mid (populated by CMD2) — it's still valid on GPU.
            // h_mid is already in buf_h_mid from CMD2's residual_add dispatch.
            // Prepare combine params: expert_weights[0..K-1], no shared expert.
            {
                float *params = (float *)[g_metal->buf_combine_params contents];
                // Zero all 10 slots first (unused experts get weight=0)
                memset(params, 0, 10 * sizeof(float));
                for (int k = 0; k < actual_K; k++) {
                    params[k] = valid[k] ? expert_weights[k] : 0.0f;
                }
            }
            // Enc C1: moe_combine_residual
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                [enc setComputePipelineState:g_metal->moe_combine_residual];
                [enc setBuffer:g_metal->buf_h_mid         offset:0 atIndex:0];   // h_mid
                [enc setBuffer:g_metal->buf_shared_out    offset:0 atIndex:1];   // shared_out
                [enc setBuffer:g_metal->buf_moe_hidden    offset:0 atIndex:2];   // output: hidden
                // Bind all 8 expert output buffers (unused ones have weight=0 in params)
                for (int k = 0; k < MAX_K; k++) {
                    [enc setBuffer:g_metal->buf_multi_expert_out[k] offset:0 atIndex:(3 + k)];
                }
                [enc setBuffer:g_metal->buf_combine_params offset:0 atIndex:11]; // params
                uint32_t dim = HIDDEN_DIM;
                uint32_t k_val = (uint32_t)actual_K;
                [enc setBytes:&dim   length:4 atIndex:12];
                [enc setBytes:&k_val length:4 atIndex:13];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = HIDDEN_DIM;
                [enc setComputePipelineState:g_metal->rms_norm_sum];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0 atIndex:1];
                [enc setBytes:&dim length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_norm_w -> buf_input)
            {
                uint16_t *next_norm_w = layer_cache[layer_idx + 1].input_norm_w;
                NSUInteger norm_off = (NSUInteger)((const char *)next_norm_w -
                                                   (const char *)[g_metal->wf_buf contents]);
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = HIDDEN_DIM;
                float eps = RMS_NORM_EPS;
                [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0       atIndex:0]; // x
                [enc setBuffer:g_metal->wf_buf          offset:norm_off atIndex:1]; // weight (bf16)
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0       atIndex:2]; // sum_sq
                [enc setBuffer:g_metal->buf_input       offset:0       atIndex:3]; // out = normed
                [enc setBytes:&dim length:4 atIndex:4];
                [enc setBytes:&eps length:4 atIndex:5];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }
        // DEFERRED commit — submit async, don't wait.
        [cmd_experts commit];
        if (g_timing_enabled) {
            t1 = now_ms();
            g_timing.cmd3_encode += t1 - t0;
            g_timing.count++;
            g_timing.total += t1 - t_layer_start;
        }
        // Save state for deferred completion
        g_deferred.active = 1;
        g_deferred.gpu_combined = gpu_combine;
        g_deferred.cmd_experts = cmd_experts;
        g_deferred.actual_K = actual_K;
        g_deferred.shared_gate_score = shared_gate_score;
        g_deferred.hidden = hidden;
        g_deferred.layer_idx = layer_idx;
        if (!gpu_combine) {
            // Only need to save h_mid for CPU-side combine path
            memcpy(g_deferred.h_mid, h_mid, HIDDEN_DIM * sizeof(float));
        }
        for (int k = 0; k < actual_K; k++) {
            g_deferred.expert_weights[k] = expert_weights[k];
            g_deferred.valid[k] = valid[k];
        }
        // Return immediately — GPU experts are running async.
        // The next call to fused_layer_forward() or complete_deferred_experts()
        // will wait for the GPU and apply the final combine.
        return;
    } else if (packed_fd >= 0) {
        // CPU fallback for experts
        size_t esz = active_expert_size();
        float *expert_out_cpu = malloc(HIDDEN_DIM * sizeof(float));
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset = (off_t)eidx * esz;
            void *expert_data = malloc(esz);
            ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
            if (nread != (ssize_t)esz) {
                fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                        layer_idx, eidx, nread, esz);
                free(expert_data);
                continue;
            }
            ExpertLayout layout = active_expert_layout();
            uint32_t *gw = (uint32_t *)((char *)expert_data + layout.gate_w_off);
            uint16_t *gs_p = (uint16_t *)((char *)expert_data + layout.gate_s_off);
            uint16_t *gb_p = (uint16_t *)((char *)expert_data + layout.gate_b_off);
            uint32_t *uw = (uint32_t *)((char *)expert_data + layout.up_w_off);
            uint16_t *us_p = (uint16_t *)((char *)expert_data + layout.up_s_off);
            uint16_t *ub_p = (uint16_t *)((char *)expert_data + layout.up_b_off);
            uint32_t *dw = (uint32_t *)((char *)expert_data + layout.down_w_off);
            uint16_t *ds_p = (uint16_t *)((char *)expert_data + layout.down_s_off);
            uint16_t *db_p = (uint16_t *)((char *)expert_data + layout.down_b_off);
            float *gate_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
            float *up_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
            float *act_out = malloc(MOE_INTERMEDIATE * sizeof(float));
            cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                               MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
            cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                               MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
            cpu_swiglu(gate_proj_out, up_proj_out, act_out, MOE_INTERMEDIATE);
            cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out_cpu,
                               HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE);
            free(gate_proj_out);
            free(up_proj_out);
            free(act_out);
            free(expert_data);
            cpu_vec_madd(moe_out, expert_out_cpu, expert_weights[k], HIDDEN_DIM);
        }
        free(expert_out_cpu);
    }
    // ---- Final combine: hidden = h_mid + moe_out ----
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = h_mid[i] + moe_out[i];
    }
    if (dump_is_active()) {
        dump_write_layer_f32(layer_idx, "moe_out", moe_out, HIDDEN_DIM);
        dump_write_layer_f32(layer_idx, "hidden_out", hidden, HIDDEN_DIM);
    }
    prefill_capture_write_hidden_out(layer_idx, hidden);
    if (g_timing_enabled) {
        t1 = now_ms();
        g_timing.cmd3_encode += t1 - t0;  // includes CPU expert compute for non-GPU paths
        g_timing.count++;
        g_timing.total += t1 - t_layer_start;
    }
    // h_post, h_mid, gate_scores, moe_out, shared_out, shared_gate, shared_up
    // are all static scratch buffers — no free needed.
}
// ============================================================================
// Main inference loop
// ============================================================================
// ============================================================================
// Expert frequency analysis (--freq)
// ============================================================================
static int freq_cmp_desc(const void *a, const void *b) {
    return *(const int *)b - *(const int *)a;
}
static void freq_print_analysis(int K) {
    if (!g_freq_tracking || g_freq_total_tokens == 0) return;
    int total_activations_per_layer = g_freq_total_tokens * K;
    fprintf(stderr, "\n=== Expert Frequency Analysis ===\n");
    fprintf(stderr, "Tokens tracked: %d, K=%d, activations/layer=%d\n\n",
            g_freq_total_tokens, K, total_activations_per_layer);
    // Per-layer analysis
    int experts_for_80_total = 0;  // sum across layers for overall estimate
    for (int l = 0; l < NUM_LAYERS; l++) {
        // Count unique experts and sort frequencies descending
        int sorted[NUM_EXPERTS];
        memcpy(sorted, g_expert_freq[l], NUM_EXPERTS * sizeof(int));
        qsort(sorted, NUM_EXPERTS, sizeof(int), freq_cmp_desc);
        int unique = 0;
        for (int e = 0; e < NUM_EXPERTS; e++) {
            if (sorted[e] > 0) unique++;
        }
        // Compute cumulative coverage thresholds
        int cum = 0;
        int top10_cov = 0, top30_cov = 0, top60_cov = 0;
        int n_for_50 = 0, n_for_80 = 0, n_for_90 = 0;
        for (int e = 0; e < NUM_EXPERTS; e++) {
            cum += sorted[e];
            if (e == 9)  top10_cov = cum;
            if (e == 29) top30_cov = cum;
            if (e == 59) top60_cov = cum;
            if (n_for_50 == 0 && cum * 100 >= total_activations_per_layer * 50)
                n_for_50 = e + 1;
            if (n_for_80 == 0 && cum * 100 >= total_activations_per_layer * 80)
                n_for_80 = e + 1;
            if (n_for_90 == 0 && cum * 100 >= total_activations_per_layer * 90)
                n_for_90 = e + 1;
        }
        double pct10 = 100.0 * top10_cov / total_activations_per_layer;
        double pct30 = 100.0 * top30_cov / total_activations_per_layer;
        double pct60 = 100.0 * top60_cov / total_activations_per_layer;
        fprintf(stderr, "Layer %2d: %3d unique experts, "
                "top-10 cover %.0f%%, top-30 cover %.0f%%, top-60 cover %.0f%% "
                "(50%%@%d, 80%%@%d, 90%%@%d)\n",
                l, unique, pct10, pct30, pct60, n_for_50, n_for_80, n_for_90);
        experts_for_80_total += n_for_80;
    }
    // Overall summary: average experts needed for 80% across all layers
    double avg_experts_80 = (double)experts_for_80_total / NUM_LAYERS;
    // Expert size in GB: each expert is active_expert_size() bytes
    double expert_gb = (double)active_expert_size() / (1024.0 * 1024.0 * 1024.0);
    double total_pin_gb = avg_experts_80 * NUM_LAYERS * expert_gb;
    fprintf(stderr, "\n--- Overall Summary ---\n");
    fprintf(stderr, "To achieve 80%% hit rate across all layers, need %d experts pinned "
            "(avg %.0f/layer, %.2f GB)\n",
            experts_for_80_total, avg_experts_80, total_pin_gb);
    fprintf(stderr, "Expert size: %zu bytes (%.3f MB), %d layers x %d experts = %d total\n",
            active_expert_size(), (double)active_expert_size() / (1024.0 * 1024.0),
            NUM_LAYERS, NUM_EXPERTS, NUM_LAYERS * NUM_EXPERTS);
}
#ifndef CHAT_MODE
// ============================================================================
// HTTP Serve Mode — OpenAI-compatible /v1/chat/completions (SSE streaming)
// ============================================================================
// Read exactly n bytes from fd, returns 0 on success, -1 on error/EOF
static int read_exact(int fd, char *buf, int n) {
    int got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) return -1;
        got += (int)r;
    }
    return 0;
}
// Read HTTP request into buf (up to bufsz-1). Returns total bytes read, or -1.
// Reads headers, then Content-Length body if present.
static int read_http_request(int fd, char *buf, int bufsz) {
    int total = 0;
    // Read until we find \r\n\r\n (end of headers)
    while (total < bufsz - 1) {
        ssize_t r = read(fd, buf + total, 1);
        if (r <= 0) return -1;
        total++;
        if (total >= 4 &&
            buf[total-4] == '\r' && buf[total-3] == '\n' &&
            buf[total-2] == '\r' && buf[total-1] == '\n') {
            break;
        }
    }
    buf[total] = '\0';
    // Find Content-Length
    const char *cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        int content_len = atoi(cl + 15);
        if (content_len > 0 && total + content_len < bufsz - 1) {
            if (read_exact(fd, buf + total, content_len) < 0) return -1;
            total += content_len;
            buf[total] = '\0';
        }
    }
    return total;
}
// Extract tools JSON from request and build a compact canonical MiniMax tool catalog.
// We keep only the fields that materially help routing: name plus terse parameter
// signatures. This is intentionally much smaller than raw JSON Schema because the
// current prefill path is still token-serial and large tool catalogs are otherwise
// prohibitively expensive to ingest.
#define SERVE_MAX_TOOLS 32
#define SERVE_MAX_TOOL_PARAMS 64
#define SERVE_BASE_SNAPSHOT_SLOTS 4

typedef struct {
    char name[128];
    char type[32];
    int required;
} ServeToolParam;

typedef struct {
    char name[128];
    char description[256];
    int param_count;
    ServeToolParam params[SERVE_MAX_TOOL_PARAMS];
} ServeCompactTool;

typedef struct {
    char *prompt_text;
    uint64_t hash;
    int tool_count;
    int total_params;
    size_t prompt_bytes;
} ServeToolCatalog;

typedef enum {
    SERVE_TURN_INVALID = 0,
    SERVE_TURN_NEW_SESSION = 1,
    SERVE_TURN_CONTINUATION = 2,
    SERVE_TURN_TOOL_RESPONSE = 3,
} ServeTurnKind;

typedef struct {
    int has_session_id;
    char session_id[64];
    int max_tokens;
    char last_role[32];
    char *content;
    int has_tools_field;
    int tools_ignored_for_live_session;
    ServeToolCatalog tools;
    ServeTurnKind turn_kind;
    int is_continuation;
    double normalize_ms;
    double tool_parse_ms;
} ServeRequest;

typedef struct {
    float *k_snapshot;
    float *v_snapshot;
    size_t bytes;
    int len;
} KVSnapshot;

typedef struct {
    int valid;
    uint64_t system_hash;
    uint64_t tool_hash;
    int prompt_len;
    uint64_t last_used_tick;
    KVSnapshot kv[NUM_LAYERS];
    float *la_conv[NUM_LAYERS];
    float *la_ssm[NUM_LAYERS];
    void *gpu_delta[NUM_LINEAR_LAYERS];
    void *gpu_conv[NUM_LINEAR_LAYERS];
} ServeBaseSnapshot;

typedef struct {
    double embed_ms;
    double transformer_ms;
    int chunk_cache_used;
    int chunk_tokens;
    int chunk_count;
    int chunk_cache_entries;
    uint64_t chunk_cache_hits;
    uint64_t chunk_cache_misses;
    int microbatch_scaffold_used;
    int microbatch_chunk_tokens;
    int microbatch_chunk_count;
    int microbatch_max_unique_experts;
    uint64_t microbatch_total_routes;
    uint64_t microbatch_total_unique_routes;
    double microbatch_avg_unique_per_layer_chunk;
    double microbatch_dedup_ratio;
    int microbatch_exec_used;
    int microbatch_cache_entries;
} ServePrefillTiming;

typedef struct {
    int valid;
    int prompt_tokens;
    int pos_after;
    int next_token;
    double embed_ms;
    double transformer_ms;
    double lm_ms;
    int microbatch_exec_used;
    int microbatch_chunk_tokens;
    int microbatch_chunk_count;
    int microbatch_cache_entries;
    float *layer_attn_out;    // NUM_LAYERS * HIDDEN_DIM
    float *layer_h_mid;       // NUM_LAYERS * HIDDEN_DIM
    float *layer_h_post;      // NUM_LAYERS * HIDDEN_DIM
    float *layer_hidden_out;  // NUM_LAYERS * HIDDEN_DIM
    float *final_hidden;      // HIDDEN_DIM
    float *logits;            // VOCAB_SIZE
} PrefillParityStageResult;

typedef struct {
    int has_base;
    int has_delta;
    PrefillParityStageResult base;
    PrefillParityStageResult delta;
} PrefillParityRunResult;

static const char *skip_json_string(const char *p) {
    if (*p != '"') return p;
    p++;
    while (*p) {
        if (*p == '\\') { p += 2; continue; }
        if (*p == '"') return p + 1;
        p++;
    }
    return p;
}

static const char *skip_json_ws(const char *p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return p;
}

static const char *find_matching_json_block(const char *p, char open_ch, char close_ch) {
    if (!p || *p != open_ch) return NULL;
    int depth = 0;
    while (*p) {
        if (*p == '"') { p = skip_json_string(p); continue; }
        if (*p == open_ch) { depth++; p++; continue; }
        if (*p == close_ch) {
            depth--;
            if (depth == 0) return p;
            p++;
            continue;
        }
        p++;
    }
    return NULL;
}

static int copy_json_string(const char *p, char *out, size_t out_size, const char **out_end) {
    if (!p || *p != '"' || !out || out_size == 0) return 0;
    p++;
    size_t w = 0;
    while (*p) {
        if (*p == '\\' && *(p + 1)) {
            p++;
            char c = *p;
            switch (c) {
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                default: break;
            }
            if (w + 1 < out_size) out[w++] = c;
            p++;
            continue;
        }
        if (*p == '"') {
            out[w] = '\0';
            if (out_end) *out_end = p + 1;
            return 1;
        }
        if (w + 1 < out_size) out[w++] = *p;
        p++;
    }
    out[w] = '\0';
    return 0;
}

static void compact_copy_text(const char *src, char *dst, size_t dst_size) {
    if (!dst || dst_size == 0) return;
    dst[0] = '\0';
    if (!src) return;
    size_t w = 0;
    int last_space = 1;
    for (const char *p = src; *p && w + 1 < dst_size; p++) {
        char c = *p;
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        if (c == ' ') {
            if (last_space) continue;
            last_space = 1;
            dst[w++] = ' ';
            continue;
        }
        last_space = 0;
        dst[w++] = c;
    }
    while (w > 0 && dst[w - 1] == ' ') w--;
    dst[w] = '\0';
}

static int json_find_top_level_field(const char *obj, const char *field, const char **value_start) {
    if (!obj || *obj != '{' || !field) return 0;
    int depth = 0;
    for (const char *p = obj; *p; ) {
        if (*p == '"') {
            char key[128];
            const char *after = NULL;
            if (!copy_json_string(p, key, sizeof(key), &after)) return 0;
            const char *q = skip_json_ws(after);
            if (depth == 1 && *q == ':' && strcmp(key, field) == 0) {
                if (value_start) *value_start = skip_json_ws(q + 1);
                return 1;
            }
            p = after;
            continue;
        }
        if (*p == '{') depth++;
        else if (*p == '}') depth--;
        p++;
    }
    return 0;
}

static int json_extract_top_level_string_field(const char *obj, const char *field, char *out, size_t out_size) {
    const char *value = NULL;
    if (!json_find_top_level_field(obj, field, &value)) return 0;
    if (!value || *value != '"') return 0;
    return copy_json_string(value, out, out_size, NULL);
}

static void normalize_simple_type(const char *src, char *dst, size_t dst_size) {
    if (!dst || dst_size == 0) return;
    if (!src || !src[0]) {
        snprintf(dst, dst_size, "string");
        return;
    }
    if (strcmp(src, "integer") == 0 || strcmp(src, "number") == 0 ||
        strcmp(src, "boolean") == 0 || strcmp(src, "array") == 0 ||
        strcmp(src, "object") == 0 || strcmp(src, "string") == 0) {
        snprintf(dst, dst_size, "%s", src);
    } else {
        snprintf(dst, dst_size, "string");
    }
}

static int required_name_contains(char required_names[SERVE_MAX_TOOL_PARAMS][128], int required_count, const char *name) {
    for (int i = 0; i < required_count; i++) {
        if (strcmp(required_names[i], name) == 0) return 1;
    }
    return 0;
}

static void parse_required_names(const char *required_arr, char required_names[SERVE_MAX_TOOL_PARAMS][128], int *required_count) {
    *required_count = 0;
    if (!required_arr || *required_arr != '[') return;
    for (const char *p = required_arr + 1; *p && *p != ']'; ) {
        if (*p == '"') {
            if (*required_count >= SERVE_MAX_TOOL_PARAMS) break;
            const char *after = NULL;
            if (copy_json_string(p, required_names[*required_count], sizeof(required_names[*required_count]), &after)) {
                (*required_count)++;
                p = after;
                continue;
            }
            break;
        }
        p++;
    }
}

static void parse_compact_tool(const char *func_obj, ServeCompactTool *tool) {
    memset(tool, 0, sizeof(*tool));
    json_extract_top_level_string_field(func_obj, "name", tool->name, sizeof(tool->name));
    char raw_desc[512] = {0};
    if (json_extract_top_level_string_field(func_obj, "description", raw_desc, sizeof(raw_desc))) {
        compact_copy_text(raw_desc, tool->description, sizeof(tool->description));
    }
    const char *params_obj = NULL;
    if (!json_find_top_level_field(func_obj, "parameters", &params_obj) || !params_obj || *params_obj != '{') {
        return;
    }
    const char *required_arr = NULL;
    char required_names[SERVE_MAX_TOOL_PARAMS][128];
    int required_count = 0;
    if (json_find_top_level_field(params_obj, "required", &required_arr) && required_arr && *required_arr == '[') {
        parse_required_names(required_arr, required_names, &required_count);
    }
    const char *properties_obj = NULL;
    if (!json_find_top_level_field(params_obj, "properties", &properties_obj) || !properties_obj || *properties_obj != '{') {
        return;
    }
    int depth = 0;
    for (const char *p = properties_obj; *p; ) {
        if (*p == '"') {
            char prop_name[128];
            const char *after = NULL;
            if (!copy_json_string(p, prop_name, sizeof(prop_name), &after)) break;
            const char *q = skip_json_ws(after);
            if (depth == 1 && *q == ':' && tool->param_count < SERVE_MAX_TOOL_PARAMS) {
                const char *value = skip_json_ws(q + 1);
                if (*value == '{') {
                    const char *value_end = find_matching_json_block(value, '{', '}');
                    if (!value_end) break;
                    size_t span_len = (size_t)(value_end - value + 1);
                    char *prop_obj = malloc(span_len + 1);
                    if (!prop_obj) break;
                    strncpy(prop_obj, value, span_len);
                    prop_obj[span_len] = '\0';
                    ServeToolParam *param = &tool->params[tool->param_count++];
                    snprintf(param->name, sizeof(param->name), "%s", prop_name);
                    char raw_type[64] = {0};
                    if (json_extract_top_level_string_field(prop_obj, "type", raw_type, sizeof(raw_type))) {
                        normalize_simple_type(raw_type, param->type, sizeof(param->type));
                    } else {
                        snprintf(param->type, sizeof(param->type), "string");
                    }
                    param->required = required_name_contains(required_names, required_count, prop_name);
                    free(prop_obj);
                    p = value_end + 1;
                    continue;
                }
            }
            p = after;
            continue;
        }
        if (*p == '{') depth++;
        else if (*p == '}') {
            depth--;
            if (depth == 0) break;
        }
        p++;
    }
}

static int serve_tool_cmp(const void *a, const void *b) {
    const ServeCompactTool *ta = (const ServeCompactTool *)a;
    const ServeCompactTool *tb = (const ServeCompactTool *)b;
    return strcmp(ta->name, tb->name);
}

static int serve_tool_param_cmp(const void *a, const void *b) {
    const ServeToolParam *pa = (const ServeToolParam *)a;
    const ServeToolParam *pb = (const ServeToolParam *)b;
    if (pa->required != pb->required) return pb->required - pa->required;
    return strcmp(pa->name, pb->name);
}

static uint64_t fnv1a64_str(const char *s) {
    uint64_t h = 1469598103934665603ull;
    for (; s && *s; s++) {
        h ^= (uint8_t)*s;
        h *= 1099511628211ull;
    }
    return h;
}

static void serve_tool_catalog_free(ServeToolCatalog *catalog) {
    if (!catalog) return;
    free(catalog->prompt_text);
    catalog->prompt_text = NULL;
    catalog->hash = 0;
    catalog->tool_count = 0;
    catalog->total_params = 0;
    catalog->prompt_bytes = 0;
}

static ServeToolCatalog build_tool_catalog_for_minimax(const char *buf) {
    ServeToolCatalog catalog;
    memset(&catalog, 0, sizeof(catalog));
    const char *tools_start = strstr(buf, "\"tools\"");
    if (!tools_start) return catalog;
    const char *arr_start = strchr(tools_start, '[');
    if (!arr_start) return catalog;
    const char *arr_end = find_matching_json_block(arr_start, '[', ']');
    if (!arr_end) return catalog;
    ServeCompactTool tools[SERVE_MAX_TOOLS];
    memset(tools, 0, sizeof(tools));
    int tool_count = 0;
    const char *obj_start = arr_start + 1;
    while (*obj_start && obj_start < arr_end) {
        obj_start = skip_json_ws(obj_start);
        if (*obj_start == ',') { obj_start++; continue; }
        if (*obj_start == ']' || *obj_start == '\0') break;
        if (*obj_start != '{') { obj_start++; continue; }
        const char *obj_end = find_matching_json_block(obj_start, '{', '}');
        if (!obj_end) break;
        size_t obj_len = (size_t)(obj_end - obj_start + 1);
        char *obj_copy = malloc(obj_len + 1);
        if (!obj_copy) break;
        strncpy(obj_copy, obj_start, obj_len);
        obj_copy[obj_len] = '\0';
        const char *func_val = NULL;
        if (tool_count < SERVE_MAX_TOOLS &&
            json_find_top_level_field(obj_copy, "function", &func_val) &&
            func_val && *func_val == '{') {
            const char *func_end = find_matching_json_block(func_val, '{', '}');
            if (func_end) {
                size_t flen = (size_t)(func_end - func_val + 1);
                char *func_copy = malloc(flen + 1);
                if (func_copy) {
                    strncpy(func_copy, func_val, flen);
                    func_copy[flen] = '\0';
                    parse_compact_tool(func_copy, &tools[tool_count]);
                    if (tools[tool_count].name[0]) {
                        tool_count++;
                    }
                    free(func_copy);
                }
            }
        }
        free(obj_copy);
        obj_start = obj_end + 1;
    }
    if (tool_count == 0) return catalog;
    qsort(tools, tool_count, sizeof(tools[0]), serve_tool_cmp);
    int total_params = 0;
    for (int i = 0; i < tool_count; i++) {
        if (tools[i].param_count > 1) {
            qsort(tools[i].params, tools[i].param_count, sizeof(tools[i].params[0]), serve_tool_param_cmp);
        }
        total_params += tools[i].param_count;
    }
    size_t result_cap = 1024 + (size_t)tool_count * 192 + (size_t)total_params * 48;
    if (result_cap < 2048) result_cap = 2048;
    char *result = malloc(result_cap);
    if (!result) return catalog;
    int offset = 0;
    offset += snprintf(result + offset, result_cap - offset,
        "\n\n# Tools\n"
        "You may call one or more tools to assist with the user query.\n"
        "Here are the tools available in compact signature format.\n"
        "Signature markers: ! required, [] array, {} object, ? boolean, # number.\n\n"
        "<tools>\n");
    for (int i = 0; i < tool_count; i++) {
        const ServeCompactTool *tool = &tools[i];
        offset += snprintf(result + offset, result_cap - offset, "<tool>%s", tool->name);
        if (tool->param_count > 0) {
            offset += snprintf(result + offset, result_cap - offset, "(");
            for (int j = 0; j < tool->param_count; j++) {
                const ServeToolParam *param = &tool->params[j];
                const char *marker = "";
                if (strcmp(param->type, "array") == 0) marker = "[]";
                else if (strcmp(param->type, "object") == 0) marker = "{}";
                else if (strcmp(param->type, "boolean") == 0) marker = "?";
                else if (strcmp(param->type, "integer") == 0 || strcmp(param->type, "number") == 0) marker = "#";
                offset += snprintf(result + offset, result_cap - offset,
                    "%s%s%s%s",
                    (j == 0) ? "" : ",",
                    param->name,
                    marker,
                    param->required ? "!" : "");
            }
            offset += snprintf(result + offset, result_cap - offset, ")");
        }
        offset += snprintf(result + offset, result_cap - offset, "</tool>\n");
    }
    offset += snprintf(result + offset, result_cap - offset,
        "</tools>\n\n"
        "When making tool calls, use XML format to invoke tools and pass parameters:\n"
        "<minimax:tool_call>\n"
        "<invoke name=\"tool-name\">\n"
        "<parameter name=\"param-key\">param-value</parameter>\n"
        "</invoke>\n"
        "</minimax:tool_call>");
    if (offset < 0) offset = 0;
    if ((size_t)offset >= result_cap) offset = (int)result_cap - 1;
    result[offset] = '\0';
    catalog.prompt_text = result;
    catalog.hash = fnv1a64_str(result);
    catalog.tool_count = tool_count;
    catalog.total_params = total_params;
    catalog.prompt_bytes = strlen(result);
    fprintf(stderr, "[serve] parsed tool count: %d params=%d bytes=%zu hash=%016llx\n",
            tool_count, total_params, catalog.prompt_bytes, (unsigned long long)catalog.hash);
    return catalog;
}
// Extract the last "content" value from an OpenAI messages array.
// Minimal JSON parsing: find last "content":" and extract the string value.
// Returns pointer into buf (null-terminated in place), or NULL.
static char *extract_last_content(char *buf) {
    char *last = NULL;
    char *p = buf;
    for (;;) {
        p = strstr(p, "\"content\"");
        if (!p) break;
        p += 9; // skip "content"
        // Skip whitespace and colon
        while (*p == ' ' || *p == '\t' || *p == ':') p++;
        if (*p == '"') {
            p++; // skip opening quote
            last = p;
            // Find closing quote (handle escapes)
            while (*p && !(*p == '"' && *(p-1) != '\\')) p++;
        }
    }
    if (last) {
        // Null-terminate the content string (overwrite closing quote)
        char *end = last;
        while (*end && !(*end == '"' && (end == last || *(end-1) != '\\'))) end++;
        *end = '\0';
        // Unescape \\n -> \n, \\" -> ", \\\\ -> backslash inline
        char *r = last, *w = last;
        while (*r) {
            if (*r == '\\' && *(r+1)) {
                r++;
                switch (*r) {
                    case 'n':  *w++ = '\n'; r++; break;
                    case 't':  *w++ = '\t'; r++; break;
                    case '"':  *w++ = '"';  r++; break;
                    case '\\': *w++ = '\\'; r++; break;
                    default:   *w++ = '\\'; *w++ = *r++; break;
                }
            } else {
                *w++ = *r++;
            }
        }
        *w = '\0';
    }
    return last;
}

static int extract_last_role(const char *buf, char *out_role, int out_size) {
    const char *p = buf;
    const char *last_role_start = NULL;
    const char *last_role_end = NULL;
    while (*p) {
        if (*p == '"') {
            const char *after_quote = p + 1;
            const char *close = after_quote;
            while (*close && *close != '"') { if (*close == '\\') close++; close++; }
            if (*close == '"') {
                int kw_len = (int)(close - after_quote);
                if (kw_len == 4 && strncmp(after_quote, "role", 4) == 0) {
                    const char *kp = close + 1;
                    while (*kp == ' ' || *kp == '\t' || *kp == ':') kp++;
                    if (*kp == '"') {
                        kp++;
                        const char *vend = kp;
                        while (*vend && *vend != '"') vend++;
                        if (*vend == '"') {
                            last_role_start = kp;
                            last_role_end = vend;
                        }
                    }
                }
                p = close + 1;
                continue;
            }
        }
        p++;
    }
    if (!last_role_start) return 0;
    int rlen = (int)(last_role_end - last_role_start);
    if (rlen >= out_size) rlen = out_size - 1;
    strncpy(out_role, last_role_start, rlen);
    out_role[rlen] = '\0';
    return 1;
}

// Extract "max_tokens" or "max_completion_tokens" from JSON body. Returns value or default.
static int extract_max_tokens(const char *buf, int default_val) {
    const char *p = strstr(buf, "\"max_completion_tokens\"");
    if (!p) p = strstr(buf, "\"max_tokens\"");
    if (!p) return default_val;
    p = strchr(p, ':');
    if (!p) return default_val;
    return atoi(p + 1);
}
// Save a conversation turn to ~/.flash-moe/sessions/<session_id>.jsonl
// Shared data store with the chat client.
static void server_save_turn(const char *session_id, const char *role, const char *content) {
    if (!session_id || !session_id[0] || !content) return;
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    char dir[1024], path[1024];
    snprintf(dir, sizeof(dir), "%s/.flash-moe/sessions", home);
    mkdir(dir, 0755);
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/.flash-moe", home);
    mkdir(parent, 0755);
    mkdir(dir, 0755);
    snprintf(path, sizeof(path), "%s/%s.jsonl", dir, session_id);
    FILE *f = fopen(path, "a");
    if (!f) return;
    // JSON-escape content
    size_t clen = strlen(content);
    char *escaped = malloc(clen * 2 + 1);
    int j = 0;
    for (size_t i = 0; i < clen; i++) {
        switch (content[i]) {
            case '"': escaped[j++]='\\'; escaped[j++]='"'; break;
            case '\\': escaped[j++]='\\'; escaped[j++]='\\'; break;
            case '\n': escaped[j++]='\\'; escaped[j++]='n'; break;
            case '\r': escaped[j++]='\\'; escaped[j++]='r'; break;
            case '\t': escaped[j++]='\\'; escaped[j++]='t'; break;
            default: escaped[j++]=content[i]; break;
        }
    }
    escaped[j] = 0;
    fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
    free(escaped);
    fclose(f);
}
// Extract "session_id" string from JSON body. Copies into out_buf (max out_size).
// Returns 1 if found, 0 if missing.
static int extract_session_id(const char *buf, char *out_buf, int out_size) {
    const char *p = strstr(buf, "\"session_id\"");
    if (!p) return 0;
    p += 12; // skip "session_id"
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    if (*p != '"') return 0;
    p++; // skip opening quote
    int i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        out_buf[i++] = *p++;
    }
    out_buf[i] = '\0';
    return i > 0 ? 1 : 0;
}

#define TOOL_RESPONSE_COMPACT_CHARS 320
#define TOOL_RESPONSE_INLINE_LINES 10
#define TOOL_RESPONSE_HEAD_LINES 12
#define TOOL_RESPONSE_TAIL_LINES 8

static int count_text_lines(const char *text) {
    if (!text || !text[0]) return 0;
    int lines = 1;
    for (const char *p = text; *p; p++) {
        if (*p == '\n') lines++;
    }
    return lines;
}

static size_t copy_line_window(const char *text, int start_line, int max_lines, char *out, size_t out_size) {
    if (!text || !out || out_size == 0 || max_lines <= 0) return 0;
    out[0] = '\0';
    int current_line = 0;
    int copied_lines = 0;
    size_t w = 0;
    const char *p = text;
    while (*p && current_line < start_line) {
        if (*p == '\n') current_line++;
        p++;
    }
    while (*p && copied_lines < max_lines && w + 1 < out_size) {
        out[w++] = *p;
        if (*p == '\n') copied_lines++;
        p++;
    }
    out[w] = '\0';
    return w;
}

static char *compact_tool_response_content(const char *content) {
    if (!content) return NULL;
    size_t len = strlen(content);
    int lines = count_text_lines(content);
    if (len <= TOOL_RESPONSE_COMPACT_CHARS && lines <= TOOL_RESPONSE_INLINE_LINES) {
        return strdup(content);
    }

    if (lines <= 1) {
        size_t keep = 240;
        size_t out_cap = keep + 64;
        char *out = malloc(out_cap);
        if (!out) return strdup(content);
        snprintf(out, out_cap, "[tool output trimmed] %.240s%s",
                 content, len > keep ? " ..." : "");
        if (strlen(out) >= len) {
            free(out);
            return strdup(content);
        }
        return out;
    }

    char head[1024];
    char tail[768];
    int head_lines = (lines <= 16) ? 8 : TOOL_RESPONSE_HEAD_LINES;
    int tail_lines = (lines > 24) ? TOOL_RESPONSE_TAIL_LINES : 0;
    if (lines <= head_lines) {
        head_lines = lines;
        tail_lines = 0;
    }
    copy_line_window(content, 0, head_lines, head, sizeof(head));
    if (tail_lines > 0) {
        int tail_start = lines - tail_lines;
        if (tail_start < head_lines) tail_start = head_lines;
        copy_line_window(content, tail_start, tail_lines, tail, sizeof(tail));
    } else {
        tail[0] = '\0';
    }

    size_t out_cap = strlen(head) + strlen(tail) + 128;
    char *out = malloc(out_cap);
    if (!out) return strdup(content);
    if (tail[0]) {
        snprintf(out, out_cap,
                 "[tool output trimmed: %d lines]\n%s...\n%s",
                 lines, head, tail);
    } else {
        snprintf(out, out_cap,
                 "[tool output trimmed: %d lines]\n%s",
                 lines, head);
    }
    if (strlen(out) >= len) {
        free(out);
        return strdup(content);
    }
    return out;
}

static void serve_request_free(ServeRequest *req) {
    if (!req) return;
    free(req->content);
    req->content = NULL;
    serve_tool_catalog_free(&req->tools);
}

static int normalize_serve_request(const char *body, const char *active_session_id, int active_hot_session,
                                   ServeRequest *req, char *err_buf, size_t err_buf_size) {
    memset(req, 0, sizeof(*req));
    double t_norm = now_ms();
    req->max_tokens = extract_max_tokens(body, 8192);
    if (req->max_tokens > 32768) req->max_tokens = 32768;
    req->has_tools_field = strstr(body, "\"tools\"") != NULL;
    req->has_session_id = extract_session_id(body, req->session_id, sizeof(req->session_id));
    if (!extract_last_role(body, req->last_role, sizeof(req->last_role))) {
        snprintf(req->last_role, sizeof(req->last_role), "user");
    }
    int active_match = (req->has_session_id &&
                        active_session_id && active_session_id[0] &&
                        strcmp(req->session_id, active_session_id) == 0);
    req->is_continuation = active_match;
    req->tools_ignored_for_live_session = active_match && req->has_tools_field;
    double t_tools = now_ms();
    if (!active_match) {
        req->tools = build_tool_catalog_for_minimax(body);
    }
    req->tool_parse_ms = now_ms() - t_tools;
    char *body_copy = strdup(body);
    if (!body_copy) {
        snprintf(err_buf, err_buf_size, "{\"error\":{\"message\":\"request copy failed\",\"type\":\"server_error\"}}\n");
        return 0;
    }
    char *content = extract_last_content(body_copy);
    if (!content || !content[0]) {
        free(body_copy);
        snprintf(err_buf, err_buf_size, "{\"error\":{\"message\":\"no content in messages\",\"type\":\"invalid_request_error\"}}\n");
        return 0;
    }
    req->content = strdup(content);
    free(body_copy);
    if (!req->content) {
        snprintf(err_buf, err_buf_size, "{\"error\":{\"message\":\"content copy failed\",\"type\":\"server_error\"}}\n");
        return 0;
    }
    if (strcmp(req->last_role, "tool") == 0) {
        char *compacted = compact_tool_response_content(req->content);
        if (compacted) {
            if (strcmp(compacted, req->content) != 0) {
                fprintf(stderr, "[serve] compacted tool response: %zu -> %zu chars\n",
                        strlen(req->content), strlen(compacted));
            }
            free(req->content);
            req->content = compacted;
        }
    }
    if (strcmp(req->last_role, "user") == 0) {
        req->turn_kind = active_match ? SERVE_TURN_CONTINUATION : SERVE_TURN_NEW_SESSION;
    } else if (strcmp(req->last_role, "tool") == 0) {
        if (!req->has_session_id && !active_hot_session) {
            snprintf(err_buf, err_buf_size,
                     "{\"error\":{\"message\":\"tool responses require session_id\",\"type\":\"invalid_request_error\"}}\n");
            return 0;
        }
        if (req->has_session_id && !active_match) {
            snprintf(err_buf, err_buf_size,
                     "{\"error\":{\"message\":\"tool response session_id does not match the active hot session\",\"type\":\"invalid_request_error\"}}\n");
            return 0;
        }
        req->is_continuation = active_match || (!req->has_session_id && active_hot_session);
        req->turn_kind = SERVE_TURN_TOOL_RESPONSE;
    } else {
        snprintf(err_buf, err_buf_size,
                 "{\"error\":{\"message\":\"unsupported last message role; expected user or tool\",\"type\":\"invalid_request_error\"}}\n");
        return 0;
    }
    req->normalize_ms = now_ms() - t_norm;
    return 1;
}
// Write a full HTTP response string to fd
static void http_write(int fd, const char *data, int len) {
    int sent = 0;
    while (sent < len) {
        ssize_t w = write(fd, data + sent, len - sent);
        if (w <= 0) break;
        sent += (int)w;
    }
}
static void http_write_str(int fd, const char *s) {
    http_write(fd, s, (int)strlen(s));
}

static void dump_request_body_to_dir(const char *dir, unsigned long long seq, const char *body) {
    if (!dir || !dir[0] || !body) return;
    if (dump_ensure_dir(dir) != 0) return;
    char path[1400];
    snprintf(path, sizeof(path), "%s/request_%04llu.json", dir, seq);
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "WARNING: failed to open request dump %s: %s\n", path, strerror(errno));
        return;
    }
    fwrite(body, 1, strlen(body), f);
    fclose(f);
    fprintf(stderr, "[serve] dumped request body to %s\n", path);
}

static void serve_log_prefill_chunk_cache(const char *label, const ServePrefillTiming *timing) {
    if (!label || !timing || !timing->chunk_cache_used) return;
    fprintf(stderr,
            "[serve] %s prefill chunk-cache: chunk=%d chunks=%d entries=%d hits=%llu misses=%llu\n",
            label,
            timing->chunk_tokens,
            timing->chunk_count,
            timing->chunk_cache_entries,
            (unsigned long long)timing->chunk_cache_hits,
            (unsigned long long)timing->chunk_cache_misses);
}

static void serve_log_prefill_microbatch_scaffold(const char *label, const ServePrefillTiming *timing) {
    if (!label || !timing || !timing->microbatch_scaffold_used) return;
    if (timing->microbatch_exec_used) {
        fprintf(stderr,
                "[serve] %s microbatch exec: chunk=%d chunks=%d avg_unique/layer=%.1f max_unique=%d dedup=%.1f%% routes=%llu unique=%llu cache_entries=%d\n",
                label,
                timing->microbatch_chunk_tokens,
                timing->microbatch_chunk_count,
                timing->microbatch_avg_unique_per_layer_chunk,
                timing->microbatch_max_unique_experts,
                100.0 * timing->microbatch_dedup_ratio,
                (unsigned long long)timing->microbatch_total_routes,
                (unsigned long long)timing->microbatch_total_unique_routes,
                timing->microbatch_cache_entries);
    } else {
        fprintf(stderr,
                "[serve] %s microbatch scaffold: chunk=%d chunks=%d avg_unique/layer=%.1f max_unique=%d dedup=%.1f%% routes=%llu unique=%llu\n",
                label,
                timing->microbatch_chunk_tokens,
                timing->microbatch_chunk_count,
                timing->microbatch_avg_unique_per_layer_chunk,
                timing->microbatch_max_unique_experts,
                100.0 * timing->microbatch_dedup_ratio,
                (unsigned long long)timing->microbatch_total_routes,
                (unsigned long long)timing->microbatch_total_unique_routes);
    }
}
// ============================================================================
// Tool call state machine for OpenAI-compatible tool_calls output
// ============================================================================
typedef struct {
    int in_tool_call;           // Inside <minimax:tool_call>...</minimax:tool_call>
    char tool_name[256];        // Extracted function name
    char tool_args[4096];       // Extracted arguments
    int tool_args_len;
    int got_name;               // Have we seen "name=..." yet
} ToolCallState;
// Send an SSE chunk with a token delta (regular content)
// Returns 0 on success, -1 if client disconnected
static int sse_send_delta(int fd, const char *request_id, const char *token_text) {
    char chunk[4096];
    // Escape the token text for JSON
    char escaped[2048];
    char *w = escaped;
    for (const char *r = token_text; *r && w < escaped + sizeof(escaped) - 8; r++) {
        switch (*r) {
            case '"':  *w++ = '\\'; *w++ = '"';  break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n';  break;
            case '\r': *w++ = '\\'; *w++ = 'r';  break;
            case '\t': *w++ = '\\'; *w++ = 't';  break;
            default:   *w++ = *r; break;
        }
    }
    *w = '\0';
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        request_id, escaped);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}
// Send an SSE chunk formatted as an OpenAI tool call
static int sse_send_tool_call(int fd, const char *request_id, int call_idx, const char *tool_name, const char *raw_args) {
    char chunk[8192];
    char arguments_json_str[4096];
    const char *trim_args = raw_args;
    while (*trim_args == ' ' || *trim_args == '\n') trim_args++;
    if (*trim_args == '{') {
        char *w = arguments_json_str;
        for (const char *r = trim_args; *r && w < arguments_json_str + sizeof(arguments_json_str) - 8; r++) {
            if (*r == '"') { *w++ = '\\'; *w++ = '"'; }
            else if (*r == '\\') { *w++ = '\\'; *w++ = '\\'; }
            else if (*r == '\n') { *w++ = '\\'; *w++ = 'n'; }
            else if (*r == '\r') { *w++ = '\\'; *w++ = 'r'; }
            else if (*r == '\t') { *w++ = '\\'; *w++ = 't'; }
            else { *w++ = *r; }
        }
        *w = '\0';
    } else {
        char escaped_raw[4096];
        char *w = escaped_raw;
        for (const char *r = trim_args; *r && w < escaped_raw + sizeof(escaped_raw) - 8; r++) {
            if (*r == '"') { *w++ = '\\'; *w++ = '"'; }
            else if (*r == '\\') { *w++ = '\\'; *w++ = '\\'; }
            else if (*r == '\n') { *w++ = '\\'; *w++ = 'n'; }
            else if (*r == '\r') { *w++ = '\\'; *w++ = 'r'; }
            else if (*r == '\t') { *w++ = '\\'; *w++ = 't'; }
            else { *w++ = *r; }
        }
        *w = '\0';
        snprintf(arguments_json_str, sizeof(arguments_json_str), "{\\\"command\\\": \\\"%s\\\"}", escaped_raw);
    }
    char call_id[32];
    snprintf(call_id, sizeof(call_id), "call_%d", call_idx + 1);
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{"
        "\"tool_calls\":[{\"index\":%d,\"id\":\"%s\",\"type\":\"function\","
        "\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}"
        "]},\"finish_reason\":null}]}\n\n",
        request_id, call_idx, call_id, tool_name, arguments_json_str);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}
static void sse_send_done(int fd, const char *request_id, const char *finish_reason) {
    char chunk[1024];
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n"
        "data: [DONE]\n\n",
        request_id, finish_reason);
    http_write(fd, chunk, n);
}
static const char *SSE_HEADERS =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "\r\n";
static const char *CORS_RESPONSE =
    "HTTP/1.1 204 No Content\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
    "Access-Control-Max-Age: 86400\r\n"
    "\r\n";
// Tokenize a user turn (system prompt already cached in KV).
// MiniMax M2.7: uses special tokens ]~b]user and [e~[
static PromptTokens *tokenize_user_turn(const char *user_content) {
    // Format: ]~b]user\n<content>[e~[\n]~b]ai\n<think>\n
    const char *prefix = "]~b]user\n";
    const char *suffix = "[e~[\n]~b]ai\n<think>\n";
    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}
// Tokenize a continuation turn for session caching.
// MiniMax M2.7: uses special tokens ]~b]user and [e~[
static PromptTokens *tokenize_continuation_turn(const char *user_content) {
    // Previous assistant turn ended with [e~[\n
    // Format: ]~b]user\n<content>[e~[\n]~b]ai\n<think>\n
    const char *prefix = "]~b]user\n";
    const char *suffix = "[e~[\n]~b]ai\n<think>\n";
    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

static PromptTokens *tokenize_tool_response_turn(const char *tool_content) {
    const char *prefix = "]~b]tool\n<response>";
    const char *suffix = "</response>[e~[\n]~b]ai\n६\n";
    size_t prompt_len = strlen(prefix) + strlen(tool_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, tool_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

static PromptTokens *tokenize_tool_then_user_turn(const char *tool_content, const char *user_content) {
    const char *p1 = "]~b]tool\n<response>";
    const char *p2 = "</response>[e~[\n]~b]user\n";
    const char *suffix = "[e~[\n]~b]ai\n६\n";
    size_t prompt_len = strlen(p1) + strlen(tool_content) + strlen(p2) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s%s%s", p1, tool_content, p2, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Load custom system prompt from ~/.flash-moe/system.md, or use default
static char *load_system_prompt(void) {
    const char *home = getenv("HOME");
    if (home) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/.flash-moe/system.md", home);
        FILE *f = fopen(path, "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = malloc(sz + 1);
            size_t n = fread(buf, 1, sz, f);
            buf[n] = 0;
            fclose(f);
            fprintf(stderr, "[serve] Loaded custom system prompt from %s (%ld bytes)\n", path, sz);
            return buf;
        }
    }
    // MiniMax M2.7 default identity
    return strdup("You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.");
}
// Tokenize a full chat message (system prompt + user turn) for first-time use.
static PromptTokens *tokenize_chat_message(const char *user_content, const char *tools_section) {
    static char *sys_prompt_text = NULL;
    if (!sys_prompt_text) sys_prompt_text = load_system_prompt();
    // MiniMax M2.7 format:
    // ]~!b[]~b]system\n<system_prompt>[tools][e~[\n]~b]user\n<user_content>[e~[\n]~b]ai\n<think>\n
    size_t sys_len = strlen(sys_prompt_text);
    size_t tools_len = tools_section ? strlen(tools_section) : 0;
    size_t user_len = strlen(user_content);
    size_t total = 100 + sys_len + tools_len + 50 + user_len;  // padding for special tokens
    char *prompt = malloc(total);
    if (!prompt) return NULL;
    if (tools_section && tools_section[0]) {
        snprintf(prompt, total, "]~!b[]~b]system\n%s%s[e~[\n]~b]user\n%s[e~[\n]~b]ai\n<think>\n",
                 sys_prompt_text, tools_section, user_content);
    } else {
        snprintf(prompt, total, "]~!b[]~b]system\n%s[e~[\n]~b]user\n%s[e~[\n]~b]ai\n<think>\n",
                 sys_prompt_text, user_content);
    }
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}
// Keep old signature for backward compat (unused but prevents compiler warning)
__attribute__((unused))
static PromptTokens *tokenize_chat_message_old(const char *user_content) {
    const char *prefix =
        "System: You are a helpful assistant.\n\nUser: ";
    const char *suffix = "\n\nAssistant: ";
    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

static void write_http_json_error(int fd, int status_code, const char *status_text, const char *json_body) {
    char header[512];
    int body_len = (int)strlen(json_body);
    int n = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "Content-Length: %d\r\n"
        "\r\n",
        status_code, status_text, body_len);
    http_write(fd, header, n);
    http_write(fd, json_body, body_len);
}

static void serve_reset_runtime_state(void **layer_states, KVCache **kv_caches) {
    size_t conv_state_size = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
    size_t ssm_state_size = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (kv_caches[i]) kv_caches[i]->len = 0;
        if (layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            memset(s->conv_state, 0, conv_state_size);
            memset(s->ssm_state, 0, ssm_state_size);
        }
    }
    reset_delta_net_state();
}

static void serve_snapshot_clear(ServeBaseSnapshot *snap) {
    if (!snap) return;
    for (int i = 0; i < NUM_LAYERS; i++) {
        free(snap->kv[i].k_snapshot);
        free(snap->kv[i].v_snapshot);
        snap->kv[i].k_snapshot = NULL;
        snap->kv[i].v_snapshot = NULL;
        snap->kv[i].bytes = 0;
        snap->kv[i].len = 0;
        free(snap->la_conv[i]);
        free(snap->la_ssm[i]);
        snap->la_conv[i] = NULL;
        snap->la_ssm[i] = NULL;
    }
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        free(snap->gpu_delta[i]);
        free(snap->gpu_conv[i]);
        snap->gpu_delta[i] = NULL;
        snap->gpu_conv[i] = NULL;
    }
    snap->valid = 0;
    snap->system_hash = 0;
    snap->tool_hash = 0;
    snap->prompt_len = 0;
    snap->last_used_tick = 0;
}

static void serve_snapshot_capture(ServeBaseSnapshot *snap,
                                   uint64_t system_hash, uint64_t tool_hash, int pos,
                                   void **layer_states, KVCache **kv_caches) {
    serve_snapshot_clear(snap);
    size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;
    size_t conv_state_size = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
    size_t ssm_state_size = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (kv_caches[i] && kv_caches[i]->len > 0) {
            size_t sz = (size_t)kv_caches[i]->len * kv_dim * sizeof(float);
            snap->kv[i].k_snapshot = malloc(sz);
            snap->kv[i].v_snapshot = malloc(sz);
            if (snap->kv[i].k_snapshot && snap->kv[i].v_snapshot) {
                memcpy(snap->kv[i].k_snapshot, kv_caches[i]->k_cache, sz);
                memcpy(snap->kv[i].v_snapshot, kv_caches[i]->v_cache, sz);
                snap->kv[i].bytes = sz;
                snap->kv[i].len = kv_caches[i]->len;
            }
        }
        if (layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            snap->la_conv[i] = malloc(conv_state_size);
            snap->la_ssm[i] = malloc(ssm_state_size);
            if (snap->la_conv[i]) memcpy(snap->la_conv[i], s->conv_state, conv_state_size);
            if (snap->la_ssm[i]) memcpy(snap->la_ssm[i], s->ssm_state, ssm_state_size);
        }
    }
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
            if (g_metal->buf_delta_state[i]) {
                size_t sz = 64 * 128 * 128 * sizeof(float);
                snap->gpu_delta[i] = malloc(sz);
                if (snap->gpu_delta[i]) memcpy(snap->gpu_delta[i], [g_metal->buf_delta_state[i] contents], sz);
            }
            if (g_metal->buf_conv_state[i]) {
                size_t sz = 3 * 12288 * sizeof(float);
                snap->gpu_conv[i] = malloc(sz);
                if (snap->gpu_conv[i]) memcpy(snap->gpu_conv[i], [g_metal->buf_conv_state[i] contents], sz);
            }
        }
    }
    snap->valid = 1;
    snap->system_hash = system_hash;
    snap->tool_hash = tool_hash;
    snap->prompt_len = pos;
}

static void serve_snapshot_restore(const ServeBaseSnapshot *snap,
                                   void **layer_states, KVCache **kv_caches) {
    size_t conv_state_size = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
    size_t ssm_state_size = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (kv_caches[i]) {
            if (snap->kv[i].k_snapshot && snap->kv[i].v_snapshot && snap->kv[i].bytes > 0) {
                memcpy(kv_caches[i]->k_cache, snap->kv[i].k_snapshot, snap->kv[i].bytes);
                memcpy(kv_caches[i]->v_cache, snap->kv[i].v_snapshot, snap->kv[i].bytes);
                kv_caches[i]->len = snap->kv[i].len;
                if (gpu_full_attn_enabled && g_metal) {
                    int fa_idx = (i + 1) / FULL_ATTN_INTERVAL - 1;
                    if (fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS &&
                        g_metal->buf_kv_k[fa_idx] && g_metal->buf_kv_v[fa_idx]) {
                        memcpy([g_metal->buf_kv_k[fa_idx] contents], snap->kv[i].k_snapshot, snap->kv[i].bytes);
                        memcpy([g_metal->buf_kv_v[fa_idx] contents], snap->kv[i].v_snapshot, snap->kv[i].bytes);
                    }
                }
            } else {
                kv_caches[i]->len = 0;
            }
        }
        if (layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            if (snap->la_conv[i]) memcpy(s->conv_state, snap->la_conv[i], conv_state_size);
            else memset(s->conv_state, 0, conv_state_size);
            if (snap->la_ssm[i]) memcpy(s->ssm_state, snap->la_ssm[i], ssm_state_size);
            else memset(s->ssm_state, 0, ssm_state_size);
        }
    }
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
            if (snap->gpu_delta[i] && g_metal->buf_delta_state[i]) {
                memcpy([g_metal->buf_delta_state[i] contents], snap->gpu_delta[i], 64 * 128 * 128 * sizeof(float));
            }
            if (snap->gpu_conv[i] && g_metal->buf_conv_state[i]) {
                memcpy([g_metal->buf_conv_state[i] contents], snap->gpu_conv[i], 3 * 12288 * sizeof(float));
            }
        }
    } else {
        reset_delta_net_state();
    }
}

static ServeBaseSnapshot *serve_find_snapshot(ServeBaseSnapshot *snapshots,
                                              uint64_t system_hash, uint64_t tool_hash) {
    for (int i = 0; i < SERVE_BASE_SNAPSHOT_SLOTS; i++) {
        if (snapshots[i].valid &&
            snapshots[i].system_hash == system_hash &&
            snapshots[i].tool_hash == tool_hash) {
            return &snapshots[i];
        }
    }
    return NULL;
}

static ServeBaseSnapshot *serve_pick_snapshot_slot(ServeBaseSnapshot *snapshots) {
    ServeBaseSnapshot *victim = &snapshots[0];
    for (int i = 0; i < SERVE_BASE_SNAPSHOT_SLOTS; i++) {
        if (!snapshots[i].valid) return &snapshots[i];
        if (snapshots[i].last_used_tick < victim->last_used_tick) {
            victim = &snapshots[i];
        }
    }
    return victim;
}

static int serve_run_prefill_serial(WeightFile *wf,
                                    PromptTokens *pt,
                                    void **layer_states, KVCache **kv_caches,
                                    void **layer_mmaps, int *layer_fds,
                                    float *hidden, int K, int *pos_io,
                                    ServePrefillTiming *timing);
static void sync_cpu_to_gpu_delta_state_serve(void **layer_states);

static int prefill_microbatch_effective_cache_entries(void) {
    if (g_prefill_microbatch_cache_entries > 0) return g_prefill_microbatch_cache_entries;
    if (g_prefill_microbatch_tokens > 0) return g_prefill_microbatch_tokens * MAX_K;
    return 0;
}

static int prefill_microbatch_ensure_cache(void) {
    int want_entries = prefill_microbatch_effective_cache_entries();
    if (want_entries <= 0 || !g_metal) return 0;
    if (g_prefill_microbatch_cache && g_prefill_microbatch_cache->max_entries == want_entries) {
        expert_cache_reset(g_prefill_microbatch_cache);
        return 1;
    }
    if (g_prefill_microbatch_cache) {
        expert_cache_free(g_prefill_microbatch_cache);
        g_prefill_microbatch_cache = NULL;
    }
    g_prefill_microbatch_cache = expert_cache_new(g_metal->device, want_entries);
    return g_prefill_microbatch_cache != NULL;
}

typedef struct {
    id<MTLBuffer> expert_buf;
    NSUInteger input_offset;
    NSUInteger gate_offset;
    NSUInteger up_offset;
    NSUInteger act_offset;
    NSUInteger out_offset;
} PrefillMicrobatchDispatch;

static int prefill_microbatch_ensure_gpu_scratch(int chunk_tokens) {
    if (!g_metal || chunk_tokens <= 0) return 0;
    int routed_slots = chunk_tokens * MAX_K;
    if (g_prefill_microbatch_gpu.inputs &&
        g_prefill_microbatch_gpu.chunk_tokens_capacity >= chunk_tokens &&
        g_prefill_microbatch_gpu.routed_slots_capacity >= routed_slots) {
        return 1;
    }
    g_prefill_microbatch_gpu.inputs = nil;
    g_prefill_microbatch_gpu.gate = nil;
    g_prefill_microbatch_gpu.up = nil;
    g_prefill_microbatch_gpu.act = nil;
    g_prefill_microbatch_gpu.out = nil;
    g_prefill_microbatch_gpu.chunk_tokens_capacity = 0;
    g_prefill_microbatch_gpu.routed_slots_capacity = 0;

    g_prefill_microbatch_gpu.inputs =
        [g_metal->device newBufferWithLength:(size_t)chunk_tokens * HIDDEN_DIM * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    g_prefill_microbatch_gpu.gate =
        [g_metal->device newBufferWithLength:(size_t)routed_slots * MOE_INTERMEDIATE * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    g_prefill_microbatch_gpu.up =
        [g_metal->device newBufferWithLength:(size_t)routed_slots * MOE_INTERMEDIATE * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    g_prefill_microbatch_gpu.act =
        [g_metal->device newBufferWithLength:(size_t)routed_slots * MOE_INTERMEDIATE * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    g_prefill_microbatch_gpu.out =
        [g_metal->device newBufferWithLength:(size_t)routed_slots * HIDDEN_DIM * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    if (!g_prefill_microbatch_gpu.inputs || !g_prefill_microbatch_gpu.gate ||
        !g_prefill_microbatch_gpu.up || !g_prefill_microbatch_gpu.act ||
        !g_prefill_microbatch_gpu.out) {
        return 0;
    }
    g_prefill_microbatch_gpu.chunk_tokens_capacity = chunk_tokens;
    g_prefill_microbatch_gpu.routed_slots_capacity = routed_slots;
    return 1;
}

static void gpu_encode_expert_dispatches_chunked(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int dispatch_count,
    const PrefillMicrobatchDispatch *dispatches,
    id<MTLBuffer> input_buf,
    id<MTLBuffer> gate_buf,
    id<MTLBuffer> up_buf,
    id<MTLBuffer> act_buf,
    id<MTLBuffer> out_buf
) {
    if (!ctx || !cmdbuf || dispatch_count <= 0 || !dispatches) return;
    ExpertLayout layout = active_expert_layout();
    id<MTLComputePipelineState> expert_pipe = g_use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;
    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;
    uint32_t gate_up_tgs = (gate_up_out + 7) / 8;
    uint32_t down_tgs    = (down_out + 7) / 8;
    uint32_t swiglu_tgs  = (gate_up_out + 255) / 256;

    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        for (int i = 0; i < dispatch_count; i++) {
            const PrefillMicrobatchDispatch *d = &dispatches[i];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:d->expert_buf offset:layout.gate_w_off atIndex:0];
            [enc setBuffer:d->expert_buf offset:layout.gate_s_off atIndex:1];
            [enc setBuffer:d->expert_buf offset:layout.gate_b_off atIndex:2];
            [enc setBuffer:input_buf     offset:d->input_offset   atIndex:3];
            [enc setBuffer:gate_buf      offset:d->gate_offset    atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:d->expert_buf offset:layout.up_w_off atIndex:0];
            [enc setBuffer:d->expert_buf offset:layout.up_s_off atIndex:1];
            [enc setBuffer:d->expert_buf offset:layout.up_b_off atIndex:2];
            [enc setBuffer:up_buf        offset:d->up_offset     atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        for (int i = 0; i < dispatch_count; i++) {
            const PrefillMicrobatchDispatch *d = &dispatches[i];
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:gate_buf offset:d->gate_offset atIndex:0];
            [enc setBuffer:up_buf   offset:d->up_offset   atIndex:1];
            [enc setBuffer:act_buf  offset:d->act_offset  atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:d->expert_buf offset:layout.down_w_off atIndex:0];
            [enc setBuffer:d->expert_buf offset:layout.down_s_off atIndex:1];
            [enc setBuffer:d->expert_buf offset:layout.down_b_off atIndex:2];
            [enc setBuffer:act_buf       offset:d->act_offset     atIndex:3];
            [enc setBuffer:out_buf       offset:d->out_offset     atIndex:4];
            [enc setBytes:&down_out length:4 atIndex:5];
            [enc setBytes:&down_in  length:4 atIndex:6];
            [enc setBytes:&gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
    }
}

static int prefill_microbatch_collect_routes_for_layer(WeightFile *wf,
                                                       int layer,
                                                       int chunk_count,
                                                       int pos_base,
                                                       float *hidden_chunk,
                                                       void **layer_states, KVCache **kv_caches,
                                                       void **layer_mmaps, int *layer_fds,
                                                       int K,
                                                       float *h_mid_chunk,
                                                       float *h_post_chunk,
                                                       int *expert_indices_chunk,
                                                       float *expert_weights_chunk,
                                                       int *actual_k_chunk) {
    for (int token_idx = 0; token_idx < chunk_count; token_idx++) {
        float *hidden = hidden_chunk + (size_t)token_idx * HIDDEN_DIM;
        float *h_mid = h_mid_chunk + (size_t)token_idx * HIDDEN_DIM;
        float *h_post = h_post_chunk + (size_t)token_idx * HIDDEN_DIM;
        int *expert_indices = expert_indices_chunk + (size_t)token_idx * MAX_K;
        float *expert_weights = expert_weights_chunk + (size_t)token_idx * MAX_K;
        memset(expert_indices, 0, MAX_K * sizeof(int));
        memset(expert_weights, 0, MAX_K * sizeof(float));
        prefill_route_capture_enable(h_mid, h_post, expert_indices, expert_weights);
        int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
        fused_layer_forward(wf, layer, hidden,
                            is_full ? kv_caches[layer] : NULL,
                            is_full ? NULL : layer_states[layer],
                            pos_base + token_idx,
                            layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                            K, layer_fds[layer]);
        actual_k_chunk[token_idx] = g_prefill_route_capture.actual_k;
        prefill_route_capture_disable();
    }
    return 0;
}

static int prefill_microbatch_preload_unique_experts(int layer,
                                                     int chunk_count,
                                                     const int *expert_indices_chunk,
                                                     const int *actual_k_chunk,
                                                     int packed_fd,
                                                     const void *mmap_base,
                                                     int *unique_count_out) {
    if (!g_prefill_microbatch_cache || packed_fd < 0) return 0;
    int max_routes = chunk_count * MAX_K;
    int *unique_ids = malloc((size_t)max_routes * sizeof(int));
    if (!unique_ids) return -1;
    int unique_count = 0;
    uint8_t seen[NUM_EXPERTS];
    memset(seen, 0, sizeof(seen));
    for (int token_idx = 0; token_idx < chunk_count; token_idx++) {
        const int *expert_indices = expert_indices_chunk + (size_t)token_idx * MAX_K;
        int actual_k = actual_k_chunk[token_idx];
        for (int k = 0; k < actual_k; k++) {
            int expert_idx = expert_indices[k];
            if (expert_idx < 0 || expert_idx >= NUM_EXPERTS) continue;
            if (!seen[expert_idx]) {
                seen[expert_idx] = 1;
                unique_ids[unique_count++] = expert_idx;
            }
        }
    }
    if (unique_count_out) *unique_count_out = unique_count;
    if (unique_count == 0) {
        free(unique_ids);
        return 0;
    }
    if (unique_count > g_prefill_microbatch_cache->max_entries) {
        free(unique_ids);
        return -2;
    }
    size_t esz = active_expert_size();
    InferPreadTask *tasks = calloc((size_t)unique_count, sizeof(InferPreadTask));
    if (!tasks) {
        free(unique_ids);
        return -1;
    }
    for (int i = 0; i < unique_count; i++) {
        int expert_idx = unique_ids[i];
        id<MTLBuffer> buf = expert_cache_insert(g_prefill_microbatch_cache, layer, expert_idx);
        infer_pread_task_init(&tasks[i]);
        tasks[i].fd = packed_fd;
        tasks[i].dst = [buf contents];
        tasks[i].offset = (off_t)expert_idx * esz;
        tasks[i].size = esz;
        tasks[i].mmap_base = mmap_base;
    }
    io_pool_dispatch(tasks, unique_count);
    for (int i = 0; i < unique_count; i++) {
        if (tasks[i].result != (ssize_t)esz) {
            free(tasks);
            free(unique_ids);
            return -1;
        }
    }
    free(tasks);
    free(unique_ids);
    return 0;
}

static int prefill_microbatch_apply_routed_experts_for_layer(int layer,
                                                             int chunk_count,
                                                             float *hidden_chunk,
                                                             const float *h_mid_chunk,
                                                             const float *h_post_chunk,
                                                             const int *expert_indices_chunk,
                                                             const float *expert_weights_chunk,
                                                             const int *actual_k_chunk) {
    if (!g_metal || !g_prefill_microbatch_cache) return -1;
    float *expert_out = malloc(HIDDEN_DIM * sizeof(float));
    float *moe_out = calloc(HIDDEN_DIM, sizeof(float));
    if (!expert_out || !moe_out) {
        free(expert_out);
        free(moe_out);
        return -1;
    }

    // Keep the stable per-selection GPU apply path here for serve requests.
    // The layer-major executor already saves substantial TTFT by deduplicating
    // expert loads across the chunk; the more aggressive chunk-level batched
    // expert dispatch path remains parity-clean but has been unstable in live
    // serve mode and is therefore kept disabled until it is hardened.
    for (int token_idx = 0; token_idx < chunk_count; token_idx++) {
        const float *h_mid = h_mid_chunk + (size_t)token_idx * HIDDEN_DIM;
        const float *h_post = h_post_chunk + (size_t)token_idx * HIDDEN_DIM;
        const int *expert_indices = expert_indices_chunk + (size_t)token_idx * MAX_K;
        const float *expert_weights = expert_weights_chunk + (size_t)token_idx * MAX_K;
        int actual_k = actual_k_chunk[token_idx];
        memset(moe_out, 0, HIDDEN_DIM * sizeof(float));
        for (int k = 0; k < actual_k; k++) {
            id<MTLBuffer> expert_buf =
                expert_cache_lookup(g_prefill_microbatch_cache, layer, expert_indices[k]);
            if (!expert_buf) {
                free(expert_out);
                free(moe_out);
                return -1;
            }
            gpu_expert_forward(g_metal, [expert_buf contents], h_post, expert_out, 0);
            cpu_vec_madd(moe_out, expert_out, expert_weights[k], HIDDEN_DIM);
        }
        float *hidden = hidden_chunk + (size_t)token_idx * HIDDEN_DIM;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            hidden[i] = h_mid[i] + moe_out[i];
        }
    }

    free(expert_out);
    free(moe_out);
    return 0;
}

static int serve_run_prefill_microbatch_exec(WeightFile *wf,
                                             PromptTokens *pt,
                                             void **layer_states, KVCache **kv_caches,
                                             void **layer_mmaps, int *layer_fds,
                                             float *hidden, int K, int *pos_io,
                                             ServePrefillTiming *timing) {
    if (timing) memset(timing, 0, sizeof(*timing));
    if (!pt || pt->count <= 0) return 0;
    if (!g_metal || g_malloc_cache || g_expert_cache ||
        g_prefill_microbatch_tokens <= 1 || pt->count < g_prefill_microbatch_tokens) {
        return serve_run_prefill_serial(wf, pt, layer_states, kv_caches,
                                        layer_mmaps, layer_fds, hidden, K, pos_io, timing);
    }
    if (!prefill_microbatch_ensure_cache()) {
        return serve_run_prefill_serial(wf, pt, layer_states, kv_caches,
                                        layer_mmaps, layer_fds, hidden, K, pos_io, timing);
    }
    float *embed_batch = malloc((size_t)pt->count * HIDDEN_DIM * sizeof(float));
    if (!embed_batch) return -1;
    double t_embed = now_ms();
    for (int i = 0; i < pt->count; i++) {
        embed_lookup(wf, pt->ids[i], embed_batch + (size_t)i * HIDDEN_DIM);
    }
    if (timing) timing->embed_ms = now_ms() - t_embed;
    int chunk_tokens = g_prefill_microbatch_tokens;
    int cache_entries = prefill_microbatch_effective_cache_entries();
    float *hidden_chunk = malloc((size_t)chunk_tokens * HIDDEN_DIM * sizeof(float));
    float *h_mid_chunk = malloc((size_t)chunk_tokens * HIDDEN_DIM * sizeof(float));
    float *h_post_chunk = malloc((size_t)chunk_tokens * HIDDEN_DIM * sizeof(float));
    int *expert_indices_chunk = malloc((size_t)chunk_tokens * MAX_K * sizeof(int));
    float *expert_weights_chunk = malloc((size_t)chunk_tokens * MAX_K * sizeof(float));
    int *actual_k_chunk = malloc((size_t)chunk_tokens * sizeof(int));
    if (!hidden_chunk || !h_mid_chunk || !h_post_chunk ||
        !expert_indices_chunk || !expert_weights_chunk || !actual_k_chunk) {
        free(embed_batch);
        free(hidden_chunk); free(h_mid_chunk); free(h_post_chunk);
        free(expert_indices_chunk); free(expert_weights_chunk); free(actual_k_chunk);
        return -1;
    }
    prefill_microbatch_scaffold_begin(chunk_tokens);
    double t_transformer = now_ms();
    int prev_force = g_force_gpu_prefill_attn;
    g_force_gpu_prefill_attn = 1;
    int pos = *pos_io;
    int last_chunk_count = 0;
    for (int chunk_start = 0; chunk_start < pt->count; chunk_start += chunk_tokens) {
        int chunk_count = pt->count - chunk_start;
        if (chunk_count > chunk_tokens) chunk_count = chunk_tokens;
        last_chunk_count = chunk_count;
        int is_final_chunk = (chunk_start + chunk_count) >= pt->count;
        if (is_final_chunk) {
            prefill_capture_activate();
        }
        prefill_microbatch_scaffold_begin_chunk();
        for (int i = 0; i < chunk_count; i++) {
            cache_telemetry_note_token();
            memcpy(hidden_chunk + (size_t)i * HIDDEN_DIM,
                   embed_batch + (size_t)(chunk_start + i) * HIDDEN_DIM,
                   HIDDEN_DIM * sizeof(float));
        }
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            if (prefill_microbatch_collect_routes_for_layer(wf, layer, chunk_count, pos,
                                                            hidden_chunk,
                                                            layer_states, kv_caches,
                                                            layer_mmaps, layer_fds,
                                                            K,
                                                            h_mid_chunk, h_post_chunk,
                                                            expert_indices_chunk,
                                                            expert_weights_chunk,
                                                            actual_k_chunk) != 0) {
                prefill_route_capture_disable();
                prefill_microbatch_scaffold_end_chunk();
                prefill_microbatch_scaffold_disable();
                free(embed_batch);
                free(hidden_chunk); free(h_mid_chunk); free(h_post_chunk);
                free(expert_indices_chunk); free(expert_weights_chunk); free(actual_k_chunk);
                g_force_gpu_prefill_attn = prev_force;
                return -1;
            }
            expert_cache_reset(g_prefill_microbatch_cache);
            int unique_count = 0;
            int preload_rc = prefill_microbatch_preload_unique_experts(
                layer, chunk_count, expert_indices_chunk, actual_k_chunk,
                layer_fds[layer],
                layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                &unique_count);
            if (preload_rc != 0) {
                prefill_microbatch_scaffold_end_chunk();
                prefill_microbatch_scaffold_disable();
                free(embed_batch);
                free(hidden_chunk); free(h_mid_chunk); free(h_post_chunk);
                free(expert_indices_chunk); free(expert_weights_chunk); free(actual_k_chunk);
                g_force_gpu_prefill_attn = prev_force;
                fprintf(stderr,
                        "[serve] ERROR: microbatch executor preload failed at layer=%d rc=%d\n",
                        layer, preload_rc);
                return -1;
            }
            if (prefill_microbatch_apply_routed_experts_for_layer(layer, chunk_count,
                                                                  hidden_chunk,
                                                                  h_mid_chunk, h_post_chunk,
                                                                  expert_indices_chunk,
                                                                  expert_weights_chunk,
                                                                  actual_k_chunk) != 0) {
                prefill_microbatch_scaffold_end_chunk();
                prefill_microbatch_scaffold_disable();
                free(embed_batch);
                free(hidden_chunk); free(h_mid_chunk); free(h_post_chunk);
                free(expert_indices_chunk); free(expert_weights_chunk); free(actual_k_chunk);
                g_force_gpu_prefill_attn = prev_force;
                fprintf(stderr,
                        "[serve] ERROR: microbatch executor expert apply failed at layer=%d\n",
                        layer);
                return -1;
            }
            if (is_final_chunk) {
                prefill_capture_write_hidden_out(
                    layer,
                    hidden_chunk + (size_t)(chunk_count - 1) * HIDDEN_DIM);
            }
        }
        prefill_microbatch_scaffold_end_chunk();
        if (is_final_chunk) {
            prefill_capture_finish();
        }
        pos += chunk_count;
    }
    memcpy(hidden, hidden_chunk + (size_t)(last_chunk_count - 1) * HIDDEN_DIM,
           HIDDEN_DIM * sizeof(float));
    *pos_io = pos;
    g_force_gpu_prefill_attn = prev_force;
    if (timing) {
        timing->transformer_ms = now_ms() - t_transformer;
        timing->microbatch_scaffold_used = 1;
        timing->microbatch_exec_used = 1;
        timing->microbatch_chunk_tokens = chunk_tokens;
        timing->microbatch_chunk_count = (int)g_prefill_microbatch.chunk_count;
        timing->microbatch_max_unique_experts = g_prefill_microbatch.max_unique_experts;
        timing->microbatch_total_routes = g_prefill_microbatch.total_routes;
        timing->microbatch_total_unique_routes = g_prefill_microbatch.total_unique_routes;
        timing->microbatch_avg_unique_per_layer_chunk =
            g_prefill_microbatch.total_layer_chunks > 0
                ? (double)g_prefill_microbatch.total_unique_routes / (double)g_prefill_microbatch.total_layer_chunks
                : 0.0;
        timing->microbatch_dedup_ratio =
            g_prefill_microbatch.total_routes > 0
                ? 1.0 - ((double)g_prefill_microbatch.total_unique_routes / (double)g_prefill_microbatch.total_routes)
                : 0.0;
        timing->microbatch_cache_entries = cache_entries;
    }
    prefill_microbatch_scaffold_disable();
    free(embed_batch);
    free(hidden_chunk); free(h_mid_chunk); free(h_post_chunk);
    free(expert_indices_chunk); free(expert_weights_chunk); free(actual_k_chunk);
    return 0;
}

static int serve_run_prefill_serial(WeightFile *wf,
                                    PromptTokens *pt,
                                    void **layer_states, KVCache **kv_caches,
                                    void **layer_mmaps, int *layer_fds,
                                    float *hidden, int K, int *pos_io,
                                    ServePrefillTiming *timing) {
    if (timing) memset(timing, 0, sizeof(*timing));
    if (!pt || pt->count <= 0) return 0;
    float *embed_batch = NULL;
    int use_chunk_cache = 0;
    int chunk_cache_chunks = 0;
    int use_microbatch_scaffold = 0;
    int microbatch_chunk_open = 0;
    if (pt->count > 1) {
        double t_embed = now_ms();
        embed_batch = malloc((size_t)pt->count * HIDDEN_DIM * sizeof(float));
        if (!embed_batch) return -1;
        for (int i = 0; i < pt->count; i++) {
            embed_lookup(wf, pt->ids[i], embed_batch + (size_t)i * HIDDEN_DIM);
        }
        if (timing) timing->embed_ms = now_ms() - t_embed;
    }
    if (g_prefill_microbatch_tokens > 0 &&
        pt->count >= g_prefill_microbatch_tokens) {
        prefill_microbatch_scaffold_begin(g_prefill_microbatch_tokens);
        use_microbatch_scaffold = g_prefill_microbatch.enabled;
        if (timing && use_microbatch_scaffold) {
            timing->microbatch_scaffold_used = 1;
            timing->microbatch_chunk_tokens = g_prefill_microbatch_tokens;
        }
    }
    if (!g_malloc_cache && !g_expert_cache && g_metal &&
        g_prefill_chunk_tokens > 0 &&
        g_prefill_chunk_cache_entries > 0 &&
        pt->count >= g_prefill_chunk_tokens) {
        if (!g_prefill_chunk_cache) {
            g_prefill_chunk_cache = expert_cache_new(g_metal->device, g_prefill_chunk_cache_entries);
        }
        if (g_prefill_chunk_cache) {
            expert_cache_reset(g_prefill_chunk_cache);
            g_prefill_chunk_cache_active = 1;
            use_chunk_cache = 1;
            if (timing) {
                timing->chunk_cache_used = 1;
                timing->chunk_tokens = g_prefill_chunk_tokens;
                timing->chunk_cache_entries = g_prefill_chunk_cache_entries;
            }
        }
    }
    double t_transformer = now_ms();
    int prev_force = g_force_gpu_prefill_attn;
    g_force_gpu_prefill_attn = 1;
    for (int i = 0; i < pt->count - 1; i++) {
        if (use_microbatch_scaffold && !microbatch_chunk_open) {
            prefill_microbatch_scaffold_begin_chunk();
            microbatch_chunk_open = 1;
        }
        if (use_chunk_cache && (i % g_prefill_chunk_tokens) == 0) {
            expert_cache_reset(g_prefill_chunk_cache);
            chunk_cache_chunks++;
        }
        cache_telemetry_note_token();
        if (embed_batch) {
            memcpy(hidden, embed_batch + (size_t)i * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
        } else {
            embed_lookup(wf, pt->ids[i], hidden);
        }
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            fused_layer_forward(wf, layer, hidden,
                                is_full ? kv_caches[layer] : NULL,
                                is_full ? NULL : layer_states[layer],
                                *pos_io,
                                layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                K, layer_fds[layer]);
        }
        discard_deferred_experts();
        (*pos_io)++;
        if (use_microbatch_scaffold &&
            (((i + 1) % g_prefill_microbatch_tokens) == 0)) {
            prefill_microbatch_scaffold_end_chunk();
            microbatch_chunk_open = 0;
        }
    }
    if (use_chunk_cache && (((pt->count - 1) % g_prefill_chunk_tokens) == 0)) {
        expert_cache_reset(g_prefill_chunk_cache);
        chunk_cache_chunks++;
    }
    if (use_microbatch_scaffold && !microbatch_chunk_open) {
        prefill_microbatch_scaffold_begin_chunk();
        microbatch_chunk_open = 1;
    }
    cache_telemetry_note_token();
    prefill_capture_activate();
    if (embed_batch) {
        memcpy(hidden, embed_batch + (size_t)(pt->count - 1) * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
    } else {
        embed_lookup(wf, pt->ids[0], hidden);
    }
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
        fused_layer_forward(wf, layer, hidden,
                            is_full ? kv_caches[layer] : NULL,
                            is_full ? NULL : layer_states[layer],
                            *pos_io,
                            layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                            K, layer_fds[layer]);
    }
    complete_deferred_experts();
    prefill_capture_finish();
    (*pos_io)++;
    if (use_microbatch_scaffold && microbatch_chunk_open) {
        prefill_microbatch_scaffold_end_chunk();
        microbatch_chunk_open = 0;
    }
    g_force_gpu_prefill_attn = prev_force;
    g_prefill_chunk_cache_active = 0;
    if (timing) timing->transformer_ms = now_ms() - t_transformer;
    if (timing && use_chunk_cache) {
        timing->chunk_count = chunk_cache_chunks;
        timing->chunk_cache_hits = g_prefill_chunk_cache->hits;
        timing->chunk_cache_misses = g_prefill_chunk_cache->misses;
    }
    if (timing && use_microbatch_scaffold) {
        timing->microbatch_chunk_count = (int)g_prefill_microbatch.chunk_count;
        timing->microbatch_max_unique_experts = g_prefill_microbatch.max_unique_experts;
        timing->microbatch_total_routes = g_prefill_microbatch.total_routes;
        timing->microbatch_total_unique_routes = g_prefill_microbatch.total_unique_routes;
        timing->microbatch_avg_unique_per_layer_chunk =
            g_prefill_microbatch.total_layer_chunks > 0
                ? (double)g_prefill_microbatch.total_unique_routes / (double)g_prefill_microbatch.total_layer_chunks
                : 0.0;
        timing->microbatch_dedup_ratio =
            g_prefill_microbatch.total_routes > 0
                ? 1.0 - ((double)g_prefill_microbatch.total_unique_routes / (double)g_prefill_microbatch.total_routes)
                : 0.0;
    }
    prefill_microbatch_scaffold_disable();
    free(embed_batch);
    return 0;
}

static int serve_run_prefill(WeightFile *wf,
                             PromptTokens *pt,
                             void **layer_states, KVCache **kv_caches,
                             void **layer_mmaps, int *layer_fds,
                             float *hidden, int K, int *pos_io,
                             ServePrefillTiming *timing) {
    if (g_prefill_microbatch_exec &&
        g_prefill_microbatch_tokens > 1 &&
        pt && pt->count >= g_prefill_microbatch_tokens) {
        return serve_run_prefill_microbatch_exec(wf, pt, layer_states, kv_caches,
                                                 layer_mmaps, layer_fds, hidden, K, pos_io, timing);
    }
    return serve_run_prefill_serial(wf, pt, layer_states, kv_caches,
                                    layer_mmaps, layer_fds, hidden, K, pos_io, timing);
}

static char *read_text_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t nr = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[nr] = '\0';
    if (size_out) *size_out = nr;
    return buf;
}

static void prefill_parity_stage_free(PrefillParityStageResult *stage) {
    if (!stage) return;
    free(stage->layer_attn_out);
    free(stage->layer_h_mid);
    free(stage->layer_h_post);
    free(stage->layer_hidden_out);
    free(stage->final_hidden);
    free(stage->logits);
    memset(stage, 0, sizeof(*stage));
}

static int prefill_parity_stage_alloc(PrefillParityStageResult *stage) {
    memset(stage, 0, sizeof(*stage));
    stage->layer_attn_out = calloc((size_t)NUM_LAYERS * HIDDEN_DIM, sizeof(float));
    stage->layer_h_mid = calloc((size_t)NUM_LAYERS * HIDDEN_DIM, sizeof(float));
    stage->layer_h_post = calloc((size_t)NUM_LAYERS * HIDDEN_DIM, sizeof(float));
    stage->layer_hidden_out = calloc((size_t)NUM_LAYERS * HIDDEN_DIM, sizeof(float));
    stage->final_hidden = calloc(HIDDEN_DIM, sizeof(float));
    stage->logits = calloc(VOCAB_SIZE, sizeof(float));
    if (!stage->layer_attn_out || !stage->layer_h_mid || !stage->layer_h_post ||
        !stage->layer_hidden_out || !stage->final_hidden || !stage->logits) {
        prefill_parity_stage_free(stage);
        return 0;
    }
    return 1;
}

static void prefill_parity_run_free(PrefillParityRunResult *run) {
    if (!run) return;
    prefill_parity_stage_free(&run->base);
    prefill_parity_stage_free(&run->delta);
    memset(run, 0, sizeof(*run));
}

static void diff_stats_f32(const float *a, const float *b, size_t count,
                           double *out_max_abs, double *out_rms, size_t *out_max_idx) {
    double max_abs = 0.0;
    double sum_sq = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < count; i++) {
        double d = (double)a[i] - (double)b[i];
        double ad = fabs(d);
        if (ad > max_abs) {
            max_abs = ad;
            max_idx = i;
        }
        sum_sq += d * d;
    }
    if (out_max_abs) *out_max_abs = max_abs;
    if (out_rms) *out_rms = count > 0 ? sqrt(sum_sq / (double)count) : 0.0;
    if (out_max_idx) *out_max_idx = max_idx;
}

static void find_first_and_worst_layer_diff(const float *a_layers, const float *b_layers,
                                            int *out_first_layer, double *out_first_max,
                                            int *out_worst_layer, double *out_worst_max,
                                            double *out_worst_rms) {
    int first_div_layer = -1;
    int worst_layer = -1;
    double first_div_max = 0.0;
    double worst_layer_max = 0.0;
    double worst_layer_rms = 0.0;
    const double layer_warn = 1e-3;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        double layer_max = 0.0, layer_rms = 0.0;
        diff_stats_f32(a_layers + (size_t)layer * HIDDEN_DIM,
                       b_layers + (size_t)layer * HIDDEN_DIM,
                       HIDDEN_DIM, &layer_max, &layer_rms, NULL);
        if (first_div_layer < 0 && layer_max > layer_warn) {
            first_div_layer = layer;
            first_div_max = layer_max;
        }
        if (worst_layer < 0 || layer_max > worst_layer_max) {
            worst_layer = layer;
            worst_layer_max = layer_max;
            worst_layer_rms = layer_rms;
        }
    }
    if (out_first_layer) *out_first_layer = first_div_layer;
    if (out_first_max) *out_first_max = first_div_max;
    if (out_worst_layer) *out_worst_layer = worst_layer;
    if (out_worst_max) *out_worst_max = worst_layer_max;
    if (out_worst_rms) *out_worst_rms = worst_layer_rms;
}

static int run_prefill_parity_stage(WeightFile *wf,
                                    PromptTokens *pt,
                                    void **layer_states, KVCache **kv_caches,
                                    void **layer_mmaps, int *layer_fds,
                                    float *hidden, uint16_t *final_norm_w,
                                    int K, int *pos_io, int use_gpu_prefill,
                                    PrefillParityStageResult *out) {
    if (!pt || pt->count <= 0 || !out) return -1;
    if (!prefill_parity_stage_alloc(out)) return -1;
    ServePrefillTiming timing = {0};
    int prev_gpu_full_attn = gpu_full_attn_enabled;
    gpu_full_attn_enabled = use_gpu_prefill ? 1 : 0;
    prefill_capture_set_storage(out->layer_hidden_out,
                                out->layer_attn_out,
                                out->layer_h_mid,
                                out->layer_h_post);
    int rc = serve_run_prefill(wf, pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                               hidden, K, pos_io, &timing);
    prefill_capture_disable();
    gpu_full_attn_enabled = prev_gpu_full_attn;
    if (rc != 0) {
        prefill_parity_stage_free(out);
        return -1;
    }
    out->prompt_tokens = pt->count;
    out->pos_after = *pos_io;
    out->embed_ms = timing.embed_ms;
    out->transformer_ms = timing.transformer_ms;
    out->microbatch_exec_used = timing.microbatch_exec_used;
    out->microbatch_chunk_tokens = timing.microbatch_chunk_tokens;
    out->microbatch_chunk_count = timing.microbatch_chunk_count;
    out->microbatch_cache_entries = timing.microbatch_cache_entries;
    if (final_norm_w) {
        cpu_rms_norm(hidden, final_norm_w, out->final_hidden, HIDDEN_DIM, RMS_NORM_EPS);
    } else {
        memcpy(out->final_hidden, hidden, HIDDEN_DIM * sizeof(float));
    }
    double t_lm = now_ms();
    lm_head_forward(wf, out->final_hidden, out->logits);
    out->lm_ms = now_ms() - t_lm;
    out->next_token = cpu_argmax(out->logits, VOCAB_SIZE);
    out->valid = 1;
    return 0;
}

static void report_prefill_compare_stage(const char *prefix,
                                         const char *stage_label,
                                         const char *lhs_label,
                                         const PrefillParityStageResult *lhs_stage,
                                         const char *rhs_label,
                                         const PrefillParityStageResult *rhs_stage,
                                         Vocabulary *vocab) {
    if (!prefix || !stage_label || !lhs_label || !rhs_label ||
        !lhs_stage || !rhs_stage || !lhs_stage->valid || !rhs_stage->valid) return;
    double hidden_max = 0.0, hidden_rms = 0.0;
    double logits_max = 0.0, logits_rms = 0.0;
    size_t logits_max_idx = 0;
    diff_stats_f32(lhs_stage->final_hidden, rhs_stage->final_hidden, HIDDEN_DIM,
                   &hidden_max, &hidden_rms, NULL);
    diff_stats_f32(lhs_stage->logits, rhs_stage->logits, VOCAB_SIZE,
                   &logits_max, &logits_rms, &logits_max_idx);
    int attn_first = -1, attn_worst = -1;
    int hmid_first = -1, hmid_worst = -1;
    int hpost_first = -1, hpost_worst = -1;
    int hidden_first = -1, hidden_worst = -1;
    double attn_first_max = 0.0, attn_worst_max = 0.0, attn_worst_rms = 0.0;
    double hmid_first_max = 0.0, hmid_worst_max = 0.0, hmid_worst_rms = 0.0;
    double hpost_first_max = 0.0, hpost_worst_max = 0.0, hpost_worst_rms = 0.0;
    double hidden_first_max = 0.0, hidden_worst_max = 0.0, hidden_worst_rms = 0.0;
    find_first_and_worst_layer_diff(lhs_stage->layer_attn_out, rhs_stage->layer_attn_out,
                                    &attn_first, &attn_first_max,
                                    &attn_worst, &attn_worst_max, &attn_worst_rms);
    find_first_and_worst_layer_diff(lhs_stage->layer_h_mid, rhs_stage->layer_h_mid,
                                    &hmid_first, &hmid_first_max,
                                    &hmid_worst, &hmid_worst_max, &hmid_worst_rms);
    find_first_and_worst_layer_diff(lhs_stage->layer_h_post, rhs_stage->layer_h_post,
                                    &hpost_first, &hpost_first_max,
                                    &hpost_worst, &hpost_worst_max, &hpost_worst_rms);
    find_first_and_worst_layer_diff(lhs_stage->layer_hidden_out, rhs_stage->layer_hidden_out,
                                    &hidden_first, &hidden_first_max,
                                    &hidden_worst, &hidden_worst_max, &hidden_worst_rms);
    char lhs_suffix[96] = {0};
    char rhs_suffix[96] = {0};
    if (lhs_stage->microbatch_exec_used) {
        snprintf(lhs_suffix, sizeof(lhs_suffix),
                 " microbatch=yes chunk=%d chunks=%d cache=%d",
                 lhs_stage->microbatch_chunk_tokens,
                 lhs_stage->microbatch_chunk_count,
                 lhs_stage->microbatch_cache_entries);
    }
    if (rhs_stage->microbatch_exec_used) {
        snprintf(rhs_suffix, sizeof(rhs_suffix),
                 " microbatch=yes chunk=%d chunks=%d cache=%d",
                 rhs_stage->microbatch_chunk_tokens,
                 rhs_stage->microbatch_chunk_count,
                 rhs_stage->microbatch_cache_entries);
    }
    fprintf(stderr,
            "[%s] %s %s: prompt=%d pos=%d next=%d (\"%s\") embed=%.0fms transformer=%.0fms lm=%.0fms%s\n",
            prefix, stage_label, lhs_label,
            lhs_stage->prompt_tokens, lhs_stage->pos_after,
            lhs_stage->next_token, decode_token(vocab, lhs_stage->next_token),
            lhs_stage->embed_ms, lhs_stage->transformer_ms, lhs_stage->lm_ms,
            lhs_suffix);
    fprintf(stderr,
            "[%s] %s %s: prompt=%d pos=%d next=%d (\"%s\") embed=%.0fms transformer=%.0fms lm=%.0fms%s\n",
            prefix, stage_label, rhs_label,
            rhs_stage->prompt_tokens, rhs_stage->pos_after,
            rhs_stage->next_token, decode_token(vocab, rhs_stage->next_token),
            rhs_stage->embed_ms, rhs_stage->transformer_ms, rhs_stage->lm_ms,
            rhs_suffix);
    fprintf(stderr,
            "[%s] %s diff: next_match=%s final_hidden max_abs=%.6g rms=%.6g logits max_abs=%.6g rms=%.6g max_logit_idx=%zu\n",
            prefix, stage_label,
            lhs_stage->next_token == rhs_stage->next_token ? "yes" : "no",
            hidden_max, hidden_rms,
            logits_max, logits_rms,
            logits_max_idx);
    fprintf(stderr,
            "[%s] %s stages: attn_out first=%d max=%.6g worst=%d/%.6g rms=%.6g | h_mid first=%d max=%.6g worst=%d/%.6g rms=%.6g\n",
            prefix, stage_label,
            attn_first, attn_first_max, attn_worst, attn_worst_max, attn_worst_rms,
            hmid_first, hmid_first_max, hmid_worst, hmid_worst_max, hmid_worst_rms);
    fprintf(stderr,
            "[%s] %s stages: h_post first=%d max=%.6g worst=%d/%.6g rms=%.6g | hidden_out first=%d max=%.6g worst=%d/%.6g rms=%.6g\n",
            prefix, stage_label,
            hpost_first, hpost_first_max, hpost_worst, hpost_worst_max, hpost_worst_rms,
            hidden_first, hidden_first_max, hidden_worst, hidden_worst_max, hidden_worst_rms);
}

static void report_prefill_parity_stage(const char *label,
                                        const PrefillParityStageResult *cpu_stage,
                                        const PrefillParityStageResult *gpu_stage,
                                        Vocabulary *vocab) {
    report_prefill_compare_stage("prefill-parity", label, "cpu", cpu_stage, "gpu", gpu_stage, vocab);
}

static int run_prefill_parity_request(const char *body_path,
                                      WeightFile *wf, Vocabulary *vocab,
                                      void **layer_states, KVCache **kv_caches,
                                      void **layer_mmaps, int *layer_fds,
                                      float *hidden, uint16_t *final_norm_w, int K) {
    if (!body_path || !body_path[0]) {
        fprintf(stderr, "ERROR: --prefill-parity-request requires a JSON request body path\n");
        return 1;
    }
    if (!g_metal || !g_metal->attn_scores_pipe) {
        fprintf(stderr, "ERROR: GPU full-attention parity requires Metal attention support\n");
        return 1;
    }
    size_t body_size = 0;
    char *body = read_text_file(body_path, &body_size);
    if (!body) {
        fprintf(stderr, "ERROR: failed to read parity request body: %s\n", body_path);
        return 1;
    }
    ServeRequest req;
    char err_json[512];
    if (!normalize_serve_request(body, "", 0, &req, err_json, sizeof(err_json))) {
        fprintf(stderr, "[prefill-parity] invalid request: %s", err_json);
        free(body);
        serve_request_free(&req);
        return 1;
    }
    fprintf(stderr,
            "[prefill-parity] request=%s bytes=%zu turn=%s tools=%d tool_bytes=%zu content_chars=%zu\n",
            body_path, body_size,
            req.turn_kind == SERVE_TURN_TOOL_RESPONSE ? "tool_response" :
            (req.turn_kind == SERVE_TURN_CONTINUATION ? "continuation" : "new_session"),
            req.tools.tool_count, req.tools.prompt_bytes, strlen(req.content));

    PromptTokens *base_pt = NULL;
    PromptTokens *delta_pt = NULL;
    if (req.turn_kind == SERVE_TURN_NEW_SESSION) {
        base_pt = tokenize_chat_message("", req.tools.prompt_text);
        if (!base_pt) {
            fprintf(stderr, "ERROR: failed to tokenize base parity prompt\n");
            free(body);
            serve_request_free(&req);
            return 1;
        }
    }
    if (req.turn_kind == SERVE_TURN_TOOL_RESPONSE) {
        delta_pt = tokenize_tool_response_turn(req.content);
    } else if (req.turn_kind == SERVE_TURN_CONTINUATION) {
        delta_pt = tokenize_continuation_turn(req.content);
    } else {
        delta_pt = tokenize_user_turn(req.content);
    }
    if (!delta_pt) {
        fprintf(stderr, "ERROR: failed to tokenize delta parity prompt\n");
        if (base_pt) { free(base_pt->ids); free(base_pt); }
        free(body);
        serve_request_free(&req);
        return 1;
    }

    PrefillParityRunResult cpu_run = {0};
    PrefillParityRunResult gpu_run = {0};
    int rc = 0;

    serve_reset_runtime_state(layer_states, kv_caches);
    int pos = 0;
    if (base_pt) {
        cpu_run.has_base = 1;
        if (run_prefill_parity_stage(wf, base_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                     hidden, final_norm_w, K, &pos, 0, &cpu_run.base) != 0) {
            fprintf(stderr, "ERROR: CPU base parity prefill failed\n");
            rc = 1;
            goto cleanup;
        }
    }
    cpu_run.has_delta = 1;
    if (run_prefill_parity_stage(wf, delta_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                 hidden, final_norm_w, K, &pos, 0, &cpu_run.delta) != 0) {
        fprintf(stderr, "ERROR: CPU delta parity prefill failed\n");
        rc = 1;
        goto cleanup;
    }

    serve_reset_runtime_state(layer_states, kv_caches);
    pos = 0;
    if (base_pt) {
        gpu_run.has_base = 1;
        if (run_prefill_parity_stage(wf, base_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                     hidden, final_norm_w, K, &pos, 1, &gpu_run.base) != 0) {
            fprintf(stderr, "ERROR: GPU base parity prefill failed\n");
            rc = 1;
            goto cleanup;
        }
    }
    gpu_run.has_delta = 1;
    if (run_prefill_parity_stage(wf, delta_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                 hidden, final_norm_w, K, &pos, 1, &gpu_run.delta) != 0) {
        fprintf(stderr, "ERROR: GPU delta parity prefill failed\n");
        rc = 1;
        goto cleanup;
    }

    if (cpu_run.has_base && gpu_run.has_base) {
        report_prefill_parity_stage("base", &cpu_run.base, &gpu_run.base, vocab);
    }
    report_prefill_parity_stage("delta", &cpu_run.delta, &gpu_run.delta, vocab);

cleanup:
    prefill_parity_run_free(&cpu_run);
    prefill_parity_run_free(&gpu_run);
    if (base_pt) { free(base_pt->ids); free(base_pt); }
    if (delta_pt) { free(delta_pt->ids); free(delta_pt); }
    free(body);
    serve_request_free(&req);
    return rc;
}

static int run_prefill_microbatch_parity_request(const char *body_path,
                                                 WeightFile *wf, Vocabulary *vocab,
                                                 void **layer_states, KVCache **kv_caches,
                                                 void **layer_mmaps, int *layer_fds,
                                                 float *hidden, uint16_t *final_norm_w, int K) {
    if (!body_path || !body_path[0]) {
        fprintf(stderr, "ERROR: --prefill-microbatch-parity-request requires a JSON request body path\n");
        return 1;
    }
    if (!g_metal || !g_metal->attn_scores_pipe) {
        fprintf(stderr, "ERROR: microbatch parity requires Metal full-attention support\n");
        return 1;
    }
    if (g_malloc_cache || g_expert_cache) {
        fprintf(stderr, "ERROR: microbatch parity must run without global expert caches enabled\n");
        return 1;
    }
    size_t body_size = 0;
    char *body = read_text_file(body_path, &body_size);
    if (!body) {
        fprintf(stderr, "ERROR: failed to read microbatch parity request body: %s\n", body_path);
        return 1;
    }
    ServeRequest req;
    char err_json[512];
    if (!normalize_serve_request(body, "", 0, &req, err_json, sizeof(err_json))) {
        fprintf(stderr, "[microbatch-parity] invalid request: %s", err_json);
        free(body);
        serve_request_free(&req);
        return 1;
    }
    fprintf(stderr,
            "[microbatch-parity] request=%s bytes=%zu turn=%s tools=%d tool_bytes=%zu content_chars=%zu\n",
            body_path, body_size,
            req.turn_kind == SERVE_TURN_TOOL_RESPONSE ? "tool_response" :
            (req.turn_kind == SERVE_TURN_CONTINUATION ? "continuation" : "new_session"),
            req.tools.tool_count, req.tools.prompt_bytes, strlen(req.content));

    PromptTokens *base_pt = NULL;
    PromptTokens *delta_pt = NULL;
    if (req.turn_kind == SERVE_TURN_NEW_SESSION) {
        base_pt = tokenize_chat_message("", req.tools.prompt_text);
        if (!base_pt) {
            fprintf(stderr, "ERROR: failed to tokenize base microbatch parity prompt\n");
            free(body);
            serve_request_free(&req);
            return 1;
        }
    }
    if (req.turn_kind == SERVE_TURN_TOOL_RESPONSE) {
        delta_pt = tokenize_tool_response_turn(req.content);
    } else if (req.turn_kind == SERVE_TURN_CONTINUATION) {
        delta_pt = tokenize_continuation_turn(req.content);
    } else {
        delta_pt = tokenize_user_turn(req.content);
    }
    if (!delta_pt) {
        fprintf(stderr, "ERROR: failed to tokenize delta microbatch parity prompt\n");
        if (base_pt) { free(base_pt->ids); free(base_pt); }
        free(body);
        serve_request_free(&req);
        return 1;
    }

    int prev_microbatch_exec = g_prefill_microbatch_exec;
    int prev_microbatch_tokens = g_prefill_microbatch_tokens;
    int prev_gpu_full_attn = gpu_full_attn_enabled;
    if (g_prefill_microbatch_tokens <= 0) g_prefill_microbatch_tokens = 32;

    PrefillParityRunResult serial_run = {0};
    PrefillParityRunResult micro_run = {0};
    int rc = 0;
    int pos = 0;
    int use_gpu_prefill = prev_gpu_full_attn ? 1 : 0;

    g_prefill_microbatch_exec = 0;
    serve_reset_runtime_state(layer_states, kv_caches);
    if (base_pt) {
        serial_run.has_base = 1;
        if (run_prefill_parity_stage(wf, base_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                     hidden, final_norm_w, K, &pos, use_gpu_prefill, &serial_run.base) != 0) {
            fprintf(stderr, "ERROR: serial base parity prefill failed\n");
            rc = 1;
            goto cleanup;
        }
    }
    serial_run.has_delta = 1;
    if (run_prefill_parity_stage(wf, delta_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                 hidden, final_norm_w, K, &pos, use_gpu_prefill, &serial_run.delta) != 0) {
        fprintf(stderr, "ERROR: serial delta parity prefill failed\n");
        rc = 1;
        goto cleanup;
    }

    g_prefill_microbatch_exec = 1;
    serve_reset_runtime_state(layer_states, kv_caches);
    pos = 0;
    if (base_pt) {
        micro_run.has_base = 1;
        if (run_prefill_parity_stage(wf, base_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                     hidden, final_norm_w, K, &pos, use_gpu_prefill, &micro_run.base) != 0) {
            fprintf(stderr, "ERROR: microbatch base parity prefill failed\n");
            rc = 1;
            goto cleanup;
        }
    }
    micro_run.has_delta = 1;
    if (run_prefill_parity_stage(wf, delta_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                 hidden, final_norm_w, K, &pos, use_gpu_prefill, &micro_run.delta) != 0) {
        fprintf(stderr, "ERROR: microbatch delta parity prefill failed\n");
        rc = 1;
        goto cleanup;
    }

    if (serial_run.has_base && micro_run.has_base) {
        report_prefill_compare_stage("microbatch-parity", "base",
                                     "serial", &serial_run.base,
                                     "micro", &micro_run.base,
                                     vocab);
    }
    report_prefill_compare_stage("microbatch-parity", "delta",
                                 "serial", &serial_run.delta,
                                 "micro", &micro_run.delta,
                                 vocab);

cleanup:
    g_prefill_microbatch_exec = prev_microbatch_exec;
    g_prefill_microbatch_tokens = prev_microbatch_tokens;
    gpu_full_attn_enabled = prev_gpu_full_attn;
    prefill_parity_run_free(&serial_run);
    prefill_parity_run_free(&micro_run);
    if (base_pt) { free(base_pt->ids); free(base_pt); }
    if (delta_pt) { free(delta_pt->ids); free(delta_pt); }
    free(body);
    serve_request_free(&req);
    return rc;
}

static int run_serve_replay_requests(int request_count,
                                     const char **request_paths,
                                     WeightFile *wf, Vocabulary *vocab,
                                     void **layer_states, KVCache **kv_caches,
                                     void **layer_mmaps, int *layer_fds,
                                     float *hidden, float *logits,
                                     uint16_t *final_norm_w, int K) {
    if (request_count <= 0 || !request_paths) {
        fprintf(stderr, "ERROR: --serve-replay-request requires at least one request body path\n");
        return 1;
    }

    ServeBaseSnapshot base_snapshots[SERVE_BASE_SNAPSHOT_SLOTS];
    memset(base_snapshots, 0, sizeof(base_snapshots));
    uint64_t snapshot_clock = 1;
    char *system_prompt_for_hash = load_system_prompt();
    uint64_t system_hash = fnv1a64_str(system_prompt_for_hash ? system_prompt_for_hash : "");
    free(system_prompt_for_hash);

    fprintf(stderr, "[serve-replay] Replaying %d request(s)\n", request_count);
    fprintf(stderr, "[serve-replay] Pre-caching base system snapshot...\n");
    serve_reset_runtime_state(layer_states, kv_caches);
    PromptTokens *sys_pt = tokenize_chat_message("", NULL);
    if (!sys_pt) {
        fprintf(stderr, "[serve-replay] ERROR: failed to tokenize base system snapshot\n");
        return 1;
    }
    int sys_pos = 0;
    ServePrefillTiming sys_prefill = {0};
    if (g_prefill_microbatch_exec) {
        fprintf(stderr,
                "[serve-replay] Base system snapshot forcing serial prefill for safety (experimental executor remains enabled for requests)\n");
    }
    if (serve_run_prefill_serial(wf, sys_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                 hidden, K, &sys_pos, &sys_prefill) != 0) {
        fprintf(stderr, "[serve-replay] ERROR: failed to build base system snapshot\n");
        free(sys_pt->ids);
        free(sys_pt);
        return 1;
    }
    sync_cpu_to_gpu_delta_state_serve(layer_states);
    serve_snapshot_capture(&base_snapshots[0], system_hash, 0, sys_pos, layer_states, kv_caches);
    base_snapshots[0].last_used_tick = snapshot_clock++;
    fprintf(stderr,
            "[serve-replay] Base system snapshot cached: prompt=%d tokens embed=%.0fms transformer=%.0fms\n",
            sys_pos, sys_prefill.embed_ms, sys_prefill.transformer_ms);
    serve_log_prefill_chunk_cache("replay-base-system", &sys_prefill);
    serve_log_prefill_microbatch_scaffold("replay-base-system", &sys_prefill);
    free(sys_pt->ids);
    free(sys_pt);

    char active_session_id[64] = {0};
    int session_pos = 0;
    int active_hot_session = 0;
    float *final_hidden = calloc(HIDDEN_DIM, sizeof(float));
    if (!final_hidden) {
        fprintf(stderr, "[serve-replay] ERROR: failed to allocate final hidden buffer\n");
        for (int i = 0; i < SERVE_BASE_SNAPSHOT_SLOTS; i++) serve_snapshot_clear(&base_snapshots[i]);
        return 1;
    }

    int rc = 0;
    for (int req_idx = 0; req_idx < request_count; req_idx++) {
        size_t body_size = 0;
        char *body = read_text_file(request_paths[req_idx], &body_size);
        if (!body) {
            fprintf(stderr, "[serve-replay] ERROR: failed to read request body: %s\n",
                    request_paths[req_idx]);
            rc = 1;
            break;
        }
        ServeRequest req;
        char err_json[512];
        if (!normalize_serve_request(body, active_session_id, active_hot_session, &req, err_json, sizeof(err_json))) {
            fprintf(stderr, "[serve-replay] invalid request %s: %s",
                    request_paths[req_idx], err_json);
            free(body);
            serve_request_free(&req);
            rc = 1;
            break;
        }

        char request_id[64];
        snprintf(request_id, sizeof(request_id), "replay-%d", req_idx + 1);
        const char *turn_label =
            (req.turn_kind == SERVE_TURN_TOOL_RESPONSE) ? "tool_response" :
            (req.turn_kind == SERVE_TURN_CONTINUATION) ? "continuation" : "new_session";
        fprintf(stderr,
                "[serve-replay] %s file=%s bytes=%zu role=%s turn=%s content=%zu chars, session=%s%s tools=%d bytes=%zu%s\n",
                request_id, request_paths[req_idx], body_size,
                req.last_role, turn_label, strlen(req.content),
                req.has_session_id ? req.session_id : "(none)",
                req.is_continuation ? " [CONTINUE]" : " [NEW]",
                req.tools.tool_count,
                req.tools.prompt_bytes,
                req.tools_ignored_for_live_session ? " [tools ignored: hot session]" : "");

        double t_snapshot = now_ms();
        int pos = 0;
        if (req.turn_kind == SERVE_TURN_CONTINUATION || req.turn_kind == SERVE_TURN_TOOL_RESPONSE) {
            pos = session_pos;
        } else {
            uint64_t tool_hash = req.tools.tool_count > 0 ? req.tools.hash : 0;
            ServeBaseSnapshot *base_snapshot = serve_find_snapshot(base_snapshots, system_hash, tool_hash);
            int live_base_pos = 0;
            int live_base_valid = 0;
            if (!base_snapshot) {
                base_snapshot = serve_pick_snapshot_slot(base_snapshots);
                serve_reset_runtime_state(layer_states, kv_caches);
                PromptTokens *base_pt = tokenize_chat_message("", req.tools.prompt_text);
                if (!base_pt) {
                    fprintf(stderr, "[serve-replay] %s ERROR: base snapshot tokenization failed\n", request_id);
                    free(body);
                    serve_request_free(&req);
                    rc = 1;
                    break;
                }
                int base_pos = 0;
                ServePrefillTiming base_timing = {0};
                if (serve_run_prefill(wf, base_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                      hidden, K, &base_pos, &base_timing) != 0) {
                    fprintf(stderr, "[serve-replay] %s ERROR: base snapshot prefill failed\n", request_id);
                    free(base_pt->ids);
                    free(base_pt);
                    free(body);
                    serve_request_free(&req);
                    rc = 1;
                    break;
                }
                live_base_pos = base_pos;
                live_base_valid = 1;
                sync_cpu_to_gpu_delta_state_serve(layer_states);
                int snapshot_persisted = 0;
                if (!base_timing.microbatch_exec_used) {
                    serve_snapshot_capture(base_snapshot, system_hash, tool_hash, base_pos, layer_states, kv_caches);
                    base_snapshot->last_used_tick = snapshot_clock++;
                    snapshot_persisted = 1;
                } else {
                    fprintf(stderr,
                            "[serve-replay] %s microbatch base snapshot left uncached for safety (experimental executor)\n",
                            request_id);
                }
                fprintf(stderr,
                        "[serve-replay] %s built base snapshot tools=%d bytes=%zu hash=%016llx prompt=%d tokens embed=%.0fms transformer=%.0fms%s\n",
                        request_id, req.tools.tool_count, req.tools.prompt_bytes,
                        (unsigned long long)tool_hash,
                        base_pos, base_timing.embed_ms, base_timing.transformer_ms,
                        snapshot_persisted ? "" : " [uncached]");
                serve_log_prefill_chunk_cache(request_id, &base_timing);
                serve_log_prefill_microbatch_scaffold(request_id, &base_timing);
                free(base_pt->ids);
                free(base_pt);
            } else {
                serve_snapshot_restore(base_snapshot, layer_states, kv_caches);
                base_snapshot->last_used_tick = snapshot_clock++;
            }
            pos = live_base_valid ? live_base_pos : base_snapshot->prompt_len;
            if (req.has_session_id) {
                strncpy(active_session_id, req.session_id, sizeof(active_session_id) - 1);
                active_session_id[sizeof(active_session_id) - 1] = '\0';
            } else {
                active_session_id[0] = '\0';
            }
            active_hot_session = 1;
        }
        double snapshot_ms = now_ms() - t_snapshot;

        PromptTokens *pt = NULL;
        if (req.turn_kind == SERVE_TURN_TOOL_RESPONSE) {
            pt = tokenize_tool_response_turn(req.content);
        } else if (req.turn_kind == SERVE_TURN_CONTINUATION) {
            pt = tokenize_continuation_turn(req.content);
        } else {
            pt = tokenize_user_turn(req.content);
        }
        if (!pt) {
            fprintf(stderr, "[serve-replay] %s ERROR: request delta tokenization failed\n", request_id);
            free(body);
            serve_request_free(&req);
            rc = 1;
            break;
        }
        fprintf(stderr, "[serve-replay] %s delta_prompt=%d tokens\n", request_id, pt->count);
        ServePrefillTiming req_prefill = {0};
        if (serve_run_prefill(wf, pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                              hidden, K, &pos, &req_prefill) != 0) {
            fprintf(stderr, "[serve-replay] %s ERROR: request prefill failed\n", request_id);
            free(pt->ids);
            free(pt);
            free(body);
            serve_request_free(&req);
            rc = 1;
            break;
        }
        fprintf(stderr,
                "[serve-replay] %s timings normalize=%.0fms tool_parse=%.0fms snapshot=%.0fms embed=%.0fms transformer=%.0fms\n",
                request_id,
                req.normalize_ms,
                req.tool_parse_ms,
                snapshot_ms,
                req_prefill.embed_ms,
                req_prefill.transformer_ms);
        serve_log_prefill_chunk_cache(request_id, &req_prefill);
        serve_log_prefill_microbatch_scaffold(request_id, &req_prefill);

        if (final_norm_w) {
            cpu_rms_norm(hidden, final_norm_w, final_hidden, HIDDEN_DIM, RMS_NORM_EPS);
        } else {
            memcpy(final_hidden, hidden, HIDDEN_DIM * sizeof(float));
        }
        double t_lm = now_ms();
        lm_head_forward(wf, final_hidden, logits);
        double lm_ms = now_ms() - t_lm;
        int next_token = cpu_argmax(logits, VOCAB_SIZE);
        fprintf(stderr,
                "[serve-replay] %s lm_head=%.0fms next=%d (\"%s\")\n",
                request_id, lm_ms, next_token, decode_token(vocab, next_token));

        session_pos = pos;
        fprintf(stderr, "[serve-replay] %s session_pos=%d (session=%s)\n",
                request_id, session_pos, active_session_id[0] ? active_session_id : "(none)");

        free(pt->ids);
        free(pt);
        free(body);
        serve_request_free(&req);
    }

    free(final_hidden);
    for (int i = 0; i < SERVE_BASE_SNAPSHOT_SLOTS; i++) {
        serve_snapshot_clear(&base_snapshots[i]);
    }
    return rc;
}
// The main serve loop. Model state must already be initialized.
// Sync CPU linear attention state → GPU buffers
static void sync_cpu_to_gpu_delta_state_serve(void **layer_states) {
    if (!g_metal || !g_metal->delta_net_step || !layer_states) return;
    int li = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        if ((i + 1) % FULL_ATTN_INTERVAL == 0) continue;
        if (!layer_states[i]) { li++; continue; }
        LinearAttnState *la = (LinearAttnState *)layer_states[i];
        if (li < NUM_LINEAR_LAYERS) {
            if (g_metal->buf_delta_state[li] && la->ssm_state)
                memcpy([g_metal->buf_delta_state[li] contents], la->ssm_state,
                       LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float));
            if (g_metal->buf_conv_state[li] && la->conv_state)
                memcpy([g_metal->buf_conv_state[li] contents], la->conv_state,
                       (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float));
        }
        li++;
    }
}
static void serve_loop(
    int port,
    WeightFile *wf, Vocabulary *vocab,
    void **layer_states, KVCache **kv_caches,
    void **layer_mmaps, int *layer_fds,
    float *hidden, float *logits,
    uint16_t *final_norm_w, int K,
    const char *dump_request_dir)
{
    // Ignore SIGPIPE (client disconnect mid-write)
    signal(SIGPIPE, SIG_IGN);
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("listen"); close(server_fd); return;
    }
    printf("[serve] Listening on http://0.0.0.0:%d\n", port);
    printf("[serve] Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health\n");
    fflush(stdout);
    static uint64_t req_counter = 0;
    static uint64_t request_dump_counter = 0;
    ServeBaseSnapshot base_snapshots[SERVE_BASE_SNAPSHOT_SLOTS];
    memset(base_snapshots, 0, sizeof(base_snapshots));
    uint64_t snapshot_clock = 1;
    char *system_prompt_for_hash = load_system_prompt();
    uint64_t system_hash = fnv1a64_str(system_prompt_for_hash ? system_prompt_for_hash : "");
    free(system_prompt_for_hash);
    fprintf(stderr, "[serve] Pre-caching base system snapshot...\n");
    serve_reset_runtime_state(layer_states, kv_caches);
    PromptTokens *sys_pt = tokenize_chat_message("", NULL);
    if (!sys_pt) {
        fprintf(stderr, "[serve] ERROR: failed to tokenize base system snapshot\n");
        close(server_fd);
        return;
    }
    int sys_pos = 0;
    ServePrefillTiming sys_prefill = {0};
    if (g_prefill_microbatch_exec) {
        fprintf(stderr,
                "[serve] Base system snapshot forcing serial prefill for safety (experimental executor remains enabled for requests)\n");
    }
    if (serve_run_prefill_serial(wf, sys_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                 hidden, K, &sys_pos, &sys_prefill) != 0) {
        fprintf(stderr, "[serve] ERROR: failed to build base system snapshot\n");
        free(sys_pt->ids);
        free(sys_pt);
        close(server_fd);
        return;
    }
    sync_cpu_to_gpu_delta_state_serve(layer_states);
    serve_snapshot_capture(&base_snapshots[0], system_hash, 0, sys_pos, layer_states, kv_caches);
    base_snapshots[0].last_used_tick = snapshot_clock++;
    fprintf(stderr,
            "[serve] Base system snapshot cached: prompt=%d tokens embed=%.0fms transformer=%.0fms\n",
            sys_pos, sys_prefill.embed_ms, sys_prefill.transformer_ms);
    serve_log_prefill_chunk_cache("base-system", &sys_prefill);
    serve_log_prefill_microbatch_scaffold("base-system", &sys_prefill);
    free(sys_pt->ids);
    free(sys_pt);
    // ---- Session state: track one active conversation session ----
    // This phase keeps a single hot in-process session and small reusable base snapshots.
    char active_session_id[64] = {0};
    int session_pos = 0;  // RoPE position after last generation for the active session
    int active_hot_session = 0;
    for (;;) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) { perror("accept"); continue; }
        // Read HTTP request
        char *reqbuf = malloc(1024 * 1024); // 1MB max request
        int reqlen = read_http_request(client_fd, reqbuf, 1024 * 1024);
        if (reqlen <= 0) { free(reqbuf); close(client_fd); continue; }
        // Parse method and path from first line
        char method[16] = {0}, path[256] = {0};
        sscanf(reqbuf, "%15s %255s", method, path);
        // Handle CORS preflight
        if (strcmp(method, "OPTIONS") == 0) {
            http_write_str(client_fd, CORS_RESPONSE);
            free(reqbuf); close(client_fd);
            continue;
        }
        // GET /health
        if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
            const char *resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"status\":\"ok\",\"model\":\"minimax-m2.7\"}\n";
            http_write_str(client_fd, resp);
            free(reqbuf); close(client_fd);
            continue;
        }
        // GET /v1/models
        if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
            const char *resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"object\":\"list\",\"data\":[{\"id\":\"minimax-m2.7\","
                "\"object\":\"model\",\"owned_by\":\"local\"}]}\n";
            http_write_str(client_fd, resp);
            free(reqbuf); close(client_fd);
            continue;
        }
        // POST /v1/chat/completions
        if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0) {
            // Find body (after \r\n\r\n)
            char *body = strstr(reqbuf, "\r\n\r\n");
            if (!body) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no body\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            body += 4;
            if (dump_request_dir && dump_request_dir[0]) {
                dump_request_body_to_dir(dump_request_dir, ++request_dump_counter, body);
            }
            ServeRequest req;
            char err_json[512];
            if (!normalize_serve_request(body, active_session_id, active_hot_session, &req, err_json, sizeof(err_json))) {
                write_http_json_error(client_fd, 400, "Bad Request", err_json);
                serve_request_free(&req);
                free(reqbuf);
                close(client_fd);
                continue;
            }
            char request_id[64];
            snprintf(request_id, sizeof(request_id), "chatcmpl-%llu", ++req_counter);
            const char *turn_label =
                (req.turn_kind == SERVE_TURN_TOOL_RESPONSE) ? "tool_response" :
                (req.turn_kind == SERVE_TURN_CONTINUATION) ? "continuation" : "new_session";
            fprintf(stderr,
                    "[serve] %s role=%s turn=%s content=%zu chars, max_tokens=%d, session=%s%s tools=%d bytes=%zu%s\n",
                    request_id, req.last_role, turn_label, strlen(req.content), req.max_tokens,
                    req.has_session_id ? req.session_id : "(none)",
                    req.is_continuation ? " [CONTINUE]" : " [NEW]",
                    req.tools.tool_count,
                    req.tools.prompt_bytes,
                    req.tools_ignored_for_live_session ? " [tools ignored: hot session]" : "");
            double t_snapshot = now_ms();
            int pos = 0;
            if (req.turn_kind == SERVE_TURN_CONTINUATION || req.turn_kind == SERVE_TURN_TOOL_RESPONSE) {
                pos = session_pos;
            } else {
                uint64_t tool_hash = req.tools.tool_count > 0 ? req.tools.hash : 0;
                ServeBaseSnapshot *base_snapshot = serve_find_snapshot(base_snapshots, system_hash, tool_hash);
                int live_base_pos = 0;
                int live_base_valid = 0;
                if (!base_snapshot) {
                    base_snapshot = serve_pick_snapshot_slot(base_snapshots);
                    serve_reset_runtime_state(layer_states, kv_caches);
                    PromptTokens *base_pt = tokenize_chat_message("", req.tools.prompt_text);
                    if (!base_pt) {
                        write_http_json_error(client_fd, 500, "Internal Server Error",
                                              "{\"error\":{\"message\":\"base snapshot tokenization failed\",\"type\":\"server_error\"}}\n");
                        serve_request_free(&req);
                        free(reqbuf);
                        close(client_fd);
                        continue;
                    }
                    int base_pos = 0;
                    ServePrefillTiming base_timing = {0};
                    if (serve_run_prefill(wf, base_pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                          hidden, K, &base_pos, &base_timing) != 0) {
                        write_http_json_error(client_fd, 500, "Internal Server Error",
                                              "{\"error\":{\"message\":\"base snapshot prefill failed\",\"type\":\"server_error\"}}\n");
                        free(base_pt->ids);
                        free(base_pt);
                        serve_request_free(&req);
                        free(reqbuf);
                        close(client_fd);
                        continue;
                    }
                    live_base_pos = base_pos;
                    live_base_valid = 1;
                    sync_cpu_to_gpu_delta_state_serve(layer_states);
                    int snapshot_persisted = 0;
                    if (!base_timing.microbatch_exec_used) {
                        serve_snapshot_capture(base_snapshot, system_hash, tool_hash, base_pos, layer_states, kv_caches);
                        base_snapshot->last_used_tick = snapshot_clock++;
                        snapshot_persisted = 1;
                    } else {
                        fprintf(stderr,
                                "[serve] %s microbatch base snapshot left uncached for safety (experimental executor)\n",
                                request_id);
                    }
                    fprintf(stderr,
                            "[serve] %s built base snapshot tools=%d bytes=%zu hash=%016llx prompt=%d tokens embed=%.0fms transformer=%.0fms%s\n",
                            request_id, req.tools.tool_count, req.tools.prompt_bytes,
                            (unsigned long long)tool_hash,
                            base_pos, base_timing.embed_ms, base_timing.transformer_ms,
                            snapshot_persisted ? "" : " [uncached]");
                    serve_log_prefill_chunk_cache(request_id, &base_timing);
                    serve_log_prefill_microbatch_scaffold(request_id, &base_timing);
                    free(base_pt->ids);
                    free(base_pt);
                } else {
                    serve_snapshot_restore(base_snapshot, layer_states, kv_caches);
                    base_snapshot->last_used_tick = snapshot_clock++;
                }
                pos = live_base_valid ? live_base_pos : base_snapshot->prompt_len;
                if (req.has_session_id) {
                    strncpy(active_session_id, req.session_id, sizeof(active_session_id) - 1);
                    active_session_id[sizeof(active_session_id) - 1] = '\0';
                } else {
                    active_session_id[0] = '\0';
                }
                active_hot_session = 1;
            }
            double snapshot_ms = now_ms() - t_snapshot;
            // ---- Tokenize request delta ----
            PromptTokens *pt = NULL;
            if (req.turn_kind == SERVE_TURN_TOOL_RESPONSE) {
                pt = tokenize_tool_response_turn(req.content);
            } else if (req.turn_kind == SERVE_TURN_CONTINUATION) {
                pt = tokenize_continuation_turn(req.content);
            } else {
                pt = tokenize_user_turn(req.content);
            }
            if (!pt) {
                write_http_json_error(client_fd, 500, "Internal Server Error",
                                      "{\"error\":{\"message\":\"request delta tokenization failed\",\"type\":\"server_error\"}}\n");
                serve_request_free(&req);
                free(reqbuf);
                close(client_fd);
                continue;
            }
            fprintf(stderr, "[serve] %s delta_prompt=%d tokens\n", request_id, pt->count);
            if (g_cache_telemetry_enabled) cache_telemetry_reset();
            // ---- Send SSE headers ----
            http_write_str(client_fd, SSE_HEADERS);
            // ---- Batch prefill ----
            ServePrefillTiming req_prefill = {0};
            if (serve_run_prefill(wf, pt, layer_states, kv_caches, layer_mmaps, layer_fds,
                                  hidden, K, &pos, &req_prefill) != 0) {
                fprintf(stderr, "[serve] %s ERROR: request prefill failed\n", request_id);
                serve_request_free(&req);
                free(pt->ids);
                free(pt);
                free(reqbuf);
                close(client_fd);
                continue;
            }
            fprintf(stderr,
                    "[serve] %s timings normalize=%.0fms tool_parse=%.0fms snapshot=%.0fms embed=%.0fms transformer=%.0fms\n",
                    request_id,
                    req.normalize_ms,
                    req.tool_parse_ms,
                    snapshot_ms,
                    req_prefill.embed_ms,
                    req_prefill.transformer_ms);
            serve_log_prefill_chunk_cache(request_id, &req_prefill);
            serve_log_prefill_microbatch_scaffold(request_id, &req_prefill);
            // ---- Final norm + LM head for first token ----
            if (final_norm_w) {
                float *normed = malloc(HIDDEN_DIM * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
                memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
                free(normed);
            }
            double t_lm = now_ms();
            lm_head_forward(wf, hidden, logits);
            double lm_ms = now_ms() - t_lm;
            int next_token = cpu_argmax(logits, VOCAB_SIZE);
            fprintf(stderr, "[serve] %s lm_head=%.0fms\n", request_id, lm_ms);
            // ---- Auto-regressive generation with SSE streaming ----
            if (g_pred_enabled) {
                g_pred_generating = 1;
                g_pred_valid = 0;
            }
            double t_gen = now_ms();
            int gen_count = 0;
            int in_think = 0;
            int think_tokens = 0;
            // Tool call state machine variables
            int in_tool_call = 0;
            char tool_buffer[8192] = {0};
            int tool_buf_len = 0;
            int tool_call_count = 0;
            int had_tool_call = 0;
            // Accumulate response for session persistence
            char *gen_response = calloc(1, 256 * 1024);
            int gen_resp_len = 0;
            for (int gen = 0; gen < req.max_tokens; gen++) {
                if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) {
                    // Feed EOS through the model so session state includes it
                    cache_telemetry_note_token();
                    embed_lookup(wf, next_token, hidden);
                    for (int layer = 0; layer < NUM_LAYERS; layer++) {
                        int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                        fused_layer_forward(wf, layer, hidden,
                                            is_full ? kv_caches[layer] : NULL,
                                            is_full ? NULL : layer_states[layer],
                                            pos,
                                            layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                            K, layer_fds[layer]);
                    }
                    discard_deferred_experts();
                    pos++;
                    break;
                }
                // Think budget enforcement
                if (next_token == THINK_START_TOKEN) in_think = 1;
                if (next_token == THINK_END_TOKEN) in_think = 0;
                if (in_think) {
                    think_tokens++;
                    if (g_think_budget > 0 && think_tokens >= g_think_budget) {
                        next_token = THINK_END_TOKEN;  // force end thinking
                        in_think = 0;
                    }
                }
                // After tool calls: stop if next token isn't another tool call or EOS
                if (had_tool_call && next_token != 200052 && next_token != EOS_TOKEN_1 && next_token != EOS_TOKEN_2) {
                    break;
                }
                // Tool call state machine
                if (next_token == 200052) {  // <minimax:tool_call>
                    in_tool_call = 1;
                    tool_buf_len = 0;
                    tool_buffer[0] = '\0';
                    goto compute_next;
                }
                if (next_token == 200053) {  // </minimax:tool_call>
                    in_tool_call = 0;
                    // Parse ALL <invoke> tags in the buffer (model may emit multiple tool calls)
                    char *invoke_search = tool_buffer;
                    while ((invoke_search = strstr(invoke_search, "<invoke name=\"")) != NULL) {
                        char tool_name[256] = "bash";
                        char json_args[4096] = "";
                        char *name_start = invoke_search + 14;  // length of `<invoke name="`
                        char *name_end = strchr(name_start, '"');
                        char *args_content = tool_buffer;
                        if (name_end) {
                            int nlen = (int)(name_end - name_start);
                            if (nlen > 255) nlen = 255;
                            strncpy(tool_name, name_start, nlen);
                            tool_name[nlen] = '\0';
                            args_content = strchr(name_end, '>');
                            if (args_content) args_content++;
                            else args_content = name_end + 1;
                        }
                        // Find the end of this <invoke>...</invoke> block
                        char *invoke_end = strstr(args_content, "</invoke>");
                        char *next_invoke_start = NULL;
                        if (invoke_end) {
                            // Temporarily null-terminate for parameter parsing
                            *invoke_end = '\0';
                            next_invoke_start = invoke_end + 9;  // skip "</invoke>"
                        }
                        // Parse <parameter name="key">value</parameter> tags into JSON
                        char *param_start = args_content;
                        int json_pos = 1;
                        json_args[0] = '{';
                        while ((param_start = strstr(param_start, "<parameter name=\"")) != NULL && json_pos < 4090) {
                            char *pname = param_start + 17;
                            char *pend = strchr(pname, '"');
                            if (!pend) break;
                            char pname_buf[128];
                            int pname_len = (int)(pend - pname);
                            if (pname_len > 127) pname_len = 127;
                            strncpy(pname_buf, pname, pname_len);
                            pname_buf[pname_len] = '\0';
                            char *pvalue = strchr(pend, '>');
                            if (!pvalue) break;
                            pvalue++;
                            char *pvalue_end = strstr(pvalue, "</parameter>");
                            if (!pvalue_end) break;
                            char pvalue_buf[2048];
                            int pvalue_len = (int)(pvalue_end - pvalue);
                            if (pvalue_len > 2047) pvalue_len = 2047;
                            strncpy(pvalue_buf, pvalue, pvalue_len);
                            pvalue_buf[pvalue_len] = '\0';
                            char *trim_start = pvalue_buf;
                            while (*trim_start == ' ' || *trim_start == '\n' || *trim_start == '\r' || *trim_start == '\t') trim_start++;
                            char *trim_end = trim_start + strlen(trim_start);
                            while (trim_end > trim_start &&
                                   (trim_end[-1] == ' ' || trim_end[-1] == '\n' ||
                                    trim_end[-1] == '\r' || trim_end[-1] == '\t')) {
                                trim_end--;
                            }
                            *trim_end = '\0';
                            char escaped_value[4096];
                            char *ew = escaped_value;
                            for (const char *ev = trim_start; *ev && ew < escaped_value + sizeof(escaped_value) - 8; ev++) {
                                if (*ev == '"') { *ew++ = '\\'; *ew++ = '"'; }
                                else if (*ev == '\\') { *ew++ = '\\'; *ew++ = '\\'; }
                                else if (*ev == '\n') { *ew++ = '\\'; *ew++ = 'n'; }
                                else if (*ev == '\r') { *ew++ = '\\'; *ew++ = 'r'; }
                                else if (*ev == '\t') { *ew++ = '\\'; *ew++ = 't'; }
                                else { *ew++ = *ev; }
                            }
                            *ew = '\0';
                            int is_number = 0, is_boolean = 0, is_null = 0, is_json = 0;
                            size_t trimmed_len = strlen(trim_start);
                            if (trimmed_len > 0) {
                                const char *v = trim_start;
                                if (*v == '{' || *v == '[' || *v == '"') is_json = 1;
                                if (*v == '-' || (*v >= '0' && *v <= '9')) {
                                    is_number = 1;
                                    const char *np = v;
                                    if (*np == '-') np++;
                                    int has_dot = 0;
                                    while (*np && np < trim_start + trimmed_len) {
                                        if (*np == '.' && !has_dot) { has_dot = 1; np++; continue; }
                                        if (*np >= '0' && *np <= '9') { np++; continue; }
                                        is_number = 0; break;
                                    }
                                    if (np != trim_start + trimmed_len) is_number = 0;
                                }
                                if (trimmed_len == 4 && strncmp(trim_start, "true", 4) == 0) is_boolean = 1;
                                if (trimmed_len == 5 && strncmp(trim_start, "false", 5) == 0) is_boolean = 1;
                                if (trimmed_len == 4 && strncmp(trim_start, "null", 4) == 0) is_null = 1;
                            }
                            if (json_pos > 1) json_args[json_pos++] = ',';
                            if (is_json || is_number || is_boolean || is_null) {
                                json_pos += snprintf(json_args + json_pos, 4096 - json_pos, "\"%s\":%s", pname_buf, trim_start);
                            } else {
                                json_pos += snprintf(json_args + json_pos, 4096 - json_pos, "\"%s\":\"%s\"", pname_buf, escaped_value);
                            }
                            param_start = pvalue_end + 12;
                        }
                        json_args[json_pos] = '}';
                        json_args[json_pos + 1] = '\0';
                        fprintf(stderr, "[TOOL] name=\"%s\" args=%s\n", tool_name, json_args);
                        if (sse_send_tool_call(client_fd, request_id, tool_call_count, tool_name, json_args) < 0) {
                            fprintf(stderr, "[serve] %s client disconnected during tool call\n", request_id);
                            break;
                        }
                        tool_call_count++;
                        had_tool_call = 1;
                        // Restore the </invoke> tag if we null-terminated it
                        if (invoke_end) *invoke_end = '<';
                        invoke_search = next_invoke_start ? next_invoke_start : tool_buffer + tool_buf_len;
                    }
                    goto compute_next;
                }
                const char *tok_str = decode_token(vocab, next_token);
                if (in_tool_call) {
                    // Buffer the tokens quietly
                    if (tok_str && tool_buf_len + (int)strlen(tok_str) < (int)sizeof(tool_buffer) - 1) {
                        strcpy(tool_buffer + tool_buf_len, tok_str);
                        tool_buf_len += (int)strlen(tok_str);
                    }
                    goto compute_next;
                }
                // Accumulate non-thinking response for session persistence
                if (!in_think && tok_str && gen_resp_len + (int)strlen(tok_str) < 256*1024 - 1) {
                    int tlen = (int)strlen(tok_str);
                    memcpy(gen_response + gen_resp_len, tok_str, tlen);
                    gen_resp_len += tlen;
                    gen_response[gen_resp_len] = 0;
                }
                if (sse_send_delta(client_fd, request_id, tok_str) < 0) {
                    fprintf(stderr, "[serve] %s client disconnected, stopping generation\n", request_id);
                    break;
                }
            compute_next:
                gen_count++;
                // Generate next
                cache_telemetry_note_token();
                embed_lookup(wf, next_token, hidden);
                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                complete_deferred_experts();
                pos++;
                if (final_norm_w) {
                    float *normed = malloc(HIDDEN_DIM * sizeof(float));
                    cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
                    memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
                    free(normed);
                }
                lm_head_forward(wf, hidden, logits);
                next_token = cpu_argmax(logits, VOCAB_SIZE);
            }
            sse_send_done(client_fd, request_id, had_tool_call ? "tool_calls" : "stop");
            // ---- Save session state ----
            free(gen_response);
            // The KV caches + linear attention state already contain this conversation.
            // Just record the position so the next request can continue from here.
            session_pos = pos;
            active_hot_session = 1;
            fprintf(stderr, "[serve] %s session_pos=%d (session=%s)\n",
                    request_id, session_pos,
                    active_session_id[0] ? active_session_id : "(none)");
            double gen_ms = now_ms() - t_gen;
            fprintf(stderr, "[serve] %s generated=%d tokens in %.0fms (%.2f tok/s)\n",
                    request_id, gen_count, gen_ms,
                    gen_count > 0 ? gen_count * 1000.0 / gen_ms : 0.0);
            if (g_expert_cache) {
                cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
            } else if (g_malloc_cache) {
                cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
            }
            serve_request_free(&req);
            free(pt->ids);
            free(pt);
            free(reqbuf);
            close(client_fd);
            continue;
        }
        // Unknown endpoint
        const char *resp404 =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n"
            "{\"error\":\"not found\"}\n";
        http_write_str(client_fd, resp404);
        free(reqbuf);
        close(client_fd);
    }
}
// ============================================================================
static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH         Model path\n");
    printf("  --weights PATH       model_weights.bin path\n");
    printf("  --manifest PATH      model_weights.json path\n");
    printf("  --vocab PATH         vocab.bin path\n");
    printf("  --tokenizer PATH     tokenizer.bin path\n");
    printf("  --dump-dir PATH      Dump last prefill token intermediates\n");
    printf("  --dump-request-dir PATH  Dump raw /v1/chat/completions request bodies in serve mode\n");
    printf("  --prompt-tokens PATH prompt_tokens.bin path\n");
    printf("  --prompt TEXT        Prompt text (requires tokenizer.bin)\n");
    printf("  --tokens N           Max tokens to generate (default: 20)\n");
    printf("  --k N                Active experts per layer (default: 4)\n");
    printf("  --cache-entries N    Expert LRU cache size (default: 2500, 0 = disabled)\n");
    printf("  --malloc-cache N     Malloc expert cache entries (e.g., 2581 = 17GB for 80%% hit)\n");
    printf("  --cpu-linear         Legacy flag (MiniMax uses full attention only)\n");
    printf("  --timing             Enable per-layer timing breakdown\n");
    printf("  --freq               Enable expert frequency tracking + analysis\n");
    printf("  --cache-telemetry    Report cold vs eviction misses and reuse distance\n");
    printf("  --2bit               Reserved for a future MiniMax q2 runtime path\n");
    printf("  --gpu-linear         Legacy flag (MiniMax uses full attention only)\n");
    printf("  --cpu-full-attn      Force CPU full attention (disable GPU-assisted prefill attention)\n");
    printf("  --auto-cache         Enable automatic malloc expert cache in serve mode\n");
    printf("  --prefill-chunk N    Experimental prompt-local expert cache chunk size (default: 0=off)\n");
    printf("  --prefill-chunk-cache N  Experimental chunk-cache expert entries (default: 0=off)\n");
    printf("  --prefill-microbatch N   Scaffold: collect per-chunk layer routing/dedup stats during prefill\n");
    printf("  --prefill-microbatch-exec  Experimental layer-major microbatch prefill executor\n");
    printf("  --prefill-microbatch-cache N  Expert cache entries for the microbatch executor (default: chunk*K)\n");
    printf("  --prefill-parity-request PATH  Compare CPU vs GPU serve-style prefill on request JSON body\n");
    printf("  --prefill-microbatch-parity-request PATH  Compare serial vs microbatch serve-style prefill on request JSON body\n");
    printf("  --serve-replay-request PATH  Replay a dumped serve request body locally (repeatable)\n");
    printf("  --predict            Enable temporal expert prediction (prefetch during CMD1_wait)\n");
    printf("  --collect-routing F  Log routing data to binary file F (for predictor training)\n");
    printf("  --think-budget N     Max thinking tokens before force </think> (default: 2048, 0=unlimited)\n");
    printf("  --serve PORT         Run HTTP server (OpenAI-compatible API)\n");
    printf("  --help               This message\n");
}
int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = MODEL_PATH_DEFAULT;
        const char *weights_path = NULL;
        const char *manifest_path = NULL;
        const char *vocab_path = NULL;
        const char *tokenizer_path = NULL;
        const char *dump_dir = NULL;
        const char *dump_request_dir = NULL;
        const char *prompt_tokens_path = NULL;
        const char *prompt_text = NULL;
        const char *prefill_parity_request_path = NULL;
        const char *prefill_microbatch_parity_request_path = NULL;
        const char *serve_replay_request_paths[32] = {0};
        int serve_replay_request_count = 0;
        int max_tokens = 20;
        int K = 4;
        int cache_entries = 0;  // default 0: trust OS page cache (38% faster than Metal LRU)
        int malloc_cache_entries = 0;  // 0 = disabled (override with --malloc-cache)
        int serve_port = 0;  // 0 = disabled, >0 = HTTP serve mode
        int serve_auto_cache_enabled = 0;
        static struct option long_options[] = {
            {"model",         required_argument, 0, 'm'},
            {"weights",       required_argument, 0, 'w'},
            {"manifest",      required_argument, 0, 'j'},
            {"vocab",         required_argument, 0, 'v'},
            {"tokenizer",     required_argument, 0, 'y'},
            {"dump-dir",      required_argument, 0, 'U'},
            {"dump-request-dir", required_argument, 0, 'W'},
            {"prompt-tokens", required_argument, 0, 'p'},
            {"prompt",        required_argument, 0, 'P'},
            {"tokens",        required_argument, 0, 't'},
            {"k",             required_argument, 0, 'k'},
            {"cache-entries",  required_argument, 0, 'C'},
            {"malloc-cache",   required_argument, 0, 'M'},
            {"cpu-linear",    no_argument,       0, 'L'},
            {"skip-linear",   no_argument,       0, 'S'},
            {"timing",        no_argument,       0, 'T'},
            {"freq",          no_argument,       0, 'F'},
            {"cache-telemetry", no_argument,     0, 'E'},
            {"2bit",          no_argument,       0, '2'},
            {"gpu-linear",    no_argument,       0, 'G'},
            {"cpu-full-attn", no_argument,       0, 'A'},
            {"auto-cache",    no_argument,       0, 'Q'},
            {"prefill-chunk", required_argument, 0, 'I'},
            {"prefill-chunk-cache", required_argument, 0, 'J'},
            {"prefill-microbatch", required_argument, 0, 'Y'},
            {"prefill-microbatch-exec", no_argument, 0, 'e'},
            {"prefill-microbatch-cache", required_argument, 0, 'c'},
            {"prefill-parity-request", required_argument, 0, 'X'},
            {"prefill-microbatch-parity-request", required_argument, 0, 'V'},
            {"serve-replay-request", required_argument, 0, 'r'},
            {"think-budget",  required_argument, 0, 'B'},
            {"serve",         required_argument, 0, 'R'},
            {"predict",       no_argument,       0, 'D'},
            {"collect-routing", required_argument, 0, 'Z'},
            {"help",          no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };
        int c;
        while ((c = getopt_long(argc, argv, "m:w:j:v:y:U:W:p:P:X:V:r:t:k:C:M:R:B:I:J:Y:c:eLSTFE2GAQh", long_options, NULL)) != -1) {
            switch (c) {
                case 'm': model_path = optarg; break;
                case 'w': weights_path = optarg; break;
                case 'j': manifest_path = optarg; break;
                case 'v': vocab_path = optarg; break;
                case 'y': tokenizer_path = optarg; break;
                case 'U': dump_dir = optarg; break;
                case 'W': dump_request_dir = optarg; break;
                case 'p': prompt_tokens_path = optarg; break;
                case 'P': prompt_text = optarg; break;
                case 'X': prefill_parity_request_path = optarg; break;
                case 'V': prefill_microbatch_parity_request_path = optarg; break;
                case 'r':
                    if (serve_replay_request_count < (int)(sizeof(serve_replay_request_paths) / sizeof(serve_replay_request_paths[0]))) {
                        serve_replay_request_paths[serve_replay_request_count++] = optarg;
                    } else {
                        fprintf(stderr, "ERROR: too many --serve-replay-request arguments (max %zu)\n",
                                sizeof(serve_replay_request_paths) / sizeof(serve_replay_request_paths[0]));
                        return 1;
                    }
                    break;
                case 't': max_tokens = atoi(optarg); break;
                case 'k': K = atoi(optarg); break;
                case 'C': cache_entries = atoi(optarg); break;
                case 'M': malloc_cache_entries = atoi(optarg); break;
                case 'L': gpu_linear_attn_enabled = 0; break;
                case 'S': linear_attn_bypass = 1; break;
                case 'T': g_timing_enabled = 1; break;
                case 'F': g_freq_tracking = 1; break;
                case 'E': g_cache_telemetry_enabled = 1; break;
                case '2': g_use_2bit = 1; break;
                case 'G': gpu_linear_attn_enabled = 1; break;
                case 'A': gpu_full_attn_enabled = 0; break;
                case 'Q': serve_auto_cache_enabled = 1; break;
                case 'I': g_prefill_chunk_tokens = atoi(optarg); break;
                case 'J': g_prefill_chunk_cache_entries = atoi(optarg); break;
                case 'Y': g_prefill_microbatch_tokens = atoi(optarg); break;
                case 'c': g_prefill_microbatch_cache_entries = atoi(optarg); break;
                case 'e': g_prefill_microbatch_exec = 1; break;
                case 'D': g_pred_enabled = 1; break;
                case 'Z':
                    g_routing_log = fopen(optarg, "wb");
                    if (!g_routing_log) {
                        fprintf(stderr, "ERROR: cannot open routing log: %s\n", optarg);
                        return 1;
                    }
                    break;
                case 'B': g_think_budget = atoi(optarg); break;
                case 'R': serve_port = atoi(optarg); break;
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }
        if (g_use_2bit) {
            fprintf(stderr, "ERROR: MiniMax q2 experts are not wired into the runtime yet. Use the q4 pack.\n");
            return 1;
        }
        if (g_prefill_microbatch_exec && g_prefill_microbatch_tokens <= 0) {
            g_prefill_microbatch_tokens = 32;
        }
        // Build default paths
        char default_weights[1024], default_manifest[1024], default_vocab[1024], default_tokenizer[1024];
        // Try to find files relative to the executable
        if (!weights_path) {
            snprintf(default_weights, sizeof(default_weights),
                     "%s/flash_moe_dense_q4/model_weights.bin", model_path);
            if (access(default_weights, R_OK) != 0) {
                snprintf(default_weights, sizeof(default_weights),
                         "metal_infer/model_weights.bin");
                if (access(default_weights, R_OK) != 0) {
                    snprintf(default_weights, sizeof(default_weights),
                             "model_weights.bin");
                }
            }
            weights_path = default_weights;
        }
        if (!manifest_path) {
            snprintf(default_manifest, sizeof(default_manifest),
                     "%s/flash_moe_dense_q4/model_weights.json", model_path);
            if (access(default_manifest, R_OK) != 0) {
                snprintf(default_manifest, sizeof(default_manifest),
                         "metal_infer/model_weights.json");
                if (access(default_manifest, R_OK) != 0) {
                    snprintf(default_manifest, sizeof(default_manifest),
                             "model_weights.json");
                }
            }
            manifest_path = default_manifest;
        }
        if (!vocab_path) {
            snprintf(default_vocab, sizeof(default_vocab),
                     "%s/vocab.bin", model_path);
            if (access(default_vocab, R_OK) != 0) {
                snprintf(default_vocab, sizeof(default_vocab),
                         "metal_infer/vocab.bin");
                if (access(default_vocab, R_OK) != 0) {
                    snprintf(default_vocab, sizeof(default_vocab),
                             "vocab.bin");
                }
            }
            vocab_path = default_vocab;
        }
        if (!tokenizer_path) {
            snprintf(default_tokenizer, sizeof(default_tokenizer),
                     "%s/tokenizer.bin", model_path);
            if (access(default_tokenizer, R_OK) != 0) {
                snprintf(default_tokenizer, sizeof(default_tokenizer),
                         "metal_infer/tokenizer.bin");
                if (access(default_tokenizer, R_OK) != 0) {
                    snprintf(default_tokenizer, sizeof(default_tokenizer),
                             "tokenizer.bin");
                }
            }
            tokenizer_path = default_tokenizer;
        }
        set_tokenizer_path(tokenizer_path);
        if (serve_port > 0 && serve_auto_cache_enabled &&
            malloc_cache_entries == 0 && cache_entries == 0) {
            int auto_entries = auto_malloc_cache_entries_for_serve();
            if (auto_entries > 0) {
                malloc_cache_entries = auto_entries;
                fprintf(stderr,
                        "[serve] Auto-enabled malloc cache: %d entries (%.1f GB). Use --malloc-cache N to override.\n",
                        malloc_cache_entries,
                        (double)malloc_cache_entries * active_expert_size() / 1e9);
            }
        }
        // ---- Initialize Metal ----
        g_metal = metal_setup();
        if (!g_metal) {
            fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
        }
        // ---- Initialize persistent I/O thread pool ----
        io_pool_init();
        // ---- Initialize malloc expert cache (if requested) ----
        if (malloc_cache_entries > 0) {
            g_malloc_cache = malloc_cache_init(malloc_cache_entries, g_metal ? g_metal->device : MTLCreateSystemDefaultDevice());
            cache_entries = 0;  // disable Metal LRU cache when malloc cache is active
        }
        // ---- Initialize expert LRU cache ----
        if (cache_entries > 0 && g_metal) {
            g_expert_cache = expert_cache_new(g_metal->device, cache_entries);
        }
        if (g_pred_enabled && (g_malloc_cache || g_expert_cache)) {
            fprintf(stderr, "WARNING: --predict ignored while an expert cache is active; cache reuse is the supported fast path.\n");
            g_pred_enabled = 0;
        }
        printf("=== MiniMax-M2.7 Metal Inference Engine ===\n");
        printf("Model:    %s\n", model_path);
        printf("Weights:  %s\n", weights_path);
        printf("Manifest: %s\n", manifest_path);
        printf("Vocab:    %s\n", vocab_path);
        printf("Tokenizer:%s%s\n", tokenizer_path[0] ? " " : "", tokenizer_path);
        printf("K:        %d experts/layer\n", K);
        printf("Quant:    %s experts (%zu bytes each)\n", g_use_2bit ? "2-bit" : "4-bit", active_expert_size());
        if (gpu_full_attn_enabled && g_metal) {
            printf("Attention: GPU-assisted full attention (prefill token-0 enabled, decode threshold=32)\n");
        } else {
            printf("Attention: CPU full attention\n");
        }
        printf("Tokens:   %d\n", max_tokens);
        if (g_malloc_cache) {
            printf("Cache:    malloc %d entries (%.1f GB)\n",
                   malloc_cache_entries, (double)malloc_cache_entries * active_expert_size() / 1e9);
        } else {
            printf("Cache:    %d entries%s\n", cache_entries,
                   cache_entries > 0 ? "" : " (disabled)");
        }
        if (serve_port > 0) {
            if (!g_malloc_cache && !g_expert_cache &&
                g_prefill_chunk_tokens > 0 && g_prefill_chunk_cache_entries > 0) {
                printf("Prefill:  chunked serve prefill (%d-token chunks, %d-entry local expert cache)\n",
                       g_prefill_chunk_tokens, g_prefill_chunk_cache_entries);
            } else {
                printf("Prefill:  token-serial serve prefill\n");
            }
            if (g_prefill_microbatch_tokens > 0) {
                if (g_prefill_microbatch_exec) {
                    printf("Microbatch: experimental layer-major executor (%d-token chunks, %d-entry expert cache)\n",
                           g_prefill_microbatch_tokens,
                           prefill_microbatch_effective_cache_entries());
                } else {
                    printf("Scaffold: prefill microbatch planning (%d-token chunks, serial executor)\n",
                           g_prefill_microbatch_tokens);
                }
            }
        }
        double t0 = now_ms();
        // ---- Load weights ----
        WeightFile *wf = open_weights(weights_path, manifest_path);
        if (!wf) {
            fprintf(stderr, "ERROR: Failed to load weights\n");
            return 1;
        }
        // Wrap weight file for Metal GPU access
        if (g_metal) {
            metal_set_weights(g_metal, wf->data, wf->size);
        }
        // ---- Load vocabulary ----
        Vocabulary *vocab = load_vocab(vocab_path);
        if (!vocab) {
            fprintf(stderr, "ERROR: Failed to load vocabulary\n");
            return 1;
        }
        // ---- Get prompt tokens (skip in serve mode) ----
        PromptTokens *pt = NULL;
        if (serve_port == 0) {
            if (prompt_text) {
                pt = encode_prompt_text_to_tokens(prompt_text);
                if (!pt) {
                    fprintf(stderr, "ERROR: Failed to encode prompt. Make sure tokenizer.bin is available.\n");
                    return 1;
                }
            } else if (!prompt_tokens_path) {
                pt = encode_prompt_text_to_tokens("Hello, what is");
                if (!pt) {
                    fprintf(stderr, "ERROR: No prompt tokens and tokenizer.bin not found\n");
                    return 1;
                }
            } else {
                pt = load_prompt_tokens(prompt_tokens_path);
            }
            if (!pt) {
                fprintf(stderr, "ERROR: Failed to load prompt tokens from %s\n", prompt_tokens_path);
                return 1;
            }
            printf("[prompt] %d tokens:", pt->count);
            for (int i = 0; i < pt->count && i < 20; i++) {
                printf(" %d", pt->ids[i]);
            }
            printf("\n");
        }
        if (serve_port == 0 && dump_dir && pt && pt->count > 0) {
            dump_begin(dump_dir, pt, pt->count - 1, pt->count - 1, K, model_path);
            if (g_dump.enabled) {
                printf("[dump] Writing last prefill token intermediates to %s\n", dump_dir);
            }
        }
        const char *expert_dir_name = "packed_experts_minimax_q4";
        // ---- Open + mmap packed expert files ----
        // Tiered I/O: two fds per layer file.
        //   layer_fds[i]      = warm fd (page cached) — for experts seen before
        //   layer_fds_cold[i] = cold fd (F_NOCACHE)   — for first-time expert reads
        // Seen-expert bitset tracks which (layer, expert) pairs have been read before.
        // First read goes through cold fd (no page cache pollution).
        // Subsequent reads go through warm fd (page cache hit = 32 GB/s vs 5.5 GB/s).
        int layer_fds[NUM_LAYERS];
        int layer_fds_cold[NUM_LAYERS];
        void *layer_mmaps[NUM_LAYERS];
        size_t layer_mmap_sizes[NUM_LAYERS];
        int expert_layers_available = 0;
        // Reset the global seen-expert bitset
        memset(g_expert_seen, 0, sizeof(g_expert_seen));
        for (int i = 0; i < NUM_LAYERS; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                     expert_dir_name, i);
            layer_fds[i] = open(path, O_RDONLY);
            layer_fds_cold[i] = -1;  // no longer used (trust OS page cache)
            layer_mmaps[i] = MAP_FAILED;
            layer_mmap_sizes[i] = 0;
            if (layer_fds[i] >= 0) {
                expert_layers_available++;
                // Disable readahead: expert reads are random (different offsets per token).
                // Read-ahead prefetches adjacent data we won't use, wasting SSD bandwidth.
                fcntl(layer_fds[i], F_RDAHEAD, 0);
                struct stat st;
                if (fstat(layer_fds[i], &st) == 0 && st.st_size > 0) {
                    layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, layer_fds[i], 0);
                    if (layer_mmaps[i] != MAP_FAILED) {
                        layer_mmap_sizes[i] = st.st_size;
                        // No madvise: kernel default is best.
                        // MADV_RANDOM disables readahead (tested: hurts).
                        // MADV_SEQUENTIAL doesn't reduce I/O fragmentation (tested: no effect).
                        // The kernel fragments 3.9MB preads into ~5.7 disk ops regardless
                        // of hints — this is inherent to the page cache's physical page layout.
                    }
                }
            }
        }
        printf("[experts] %d/%d packed layer files available (mmap'd)\n", expert_layers_available, NUM_LAYERS);
        // ---- LZ4 compressed experts: auto-detect and load ----
        {
            char lz4_probe[1024];
            snprintf(lz4_probe, sizeof(lz4_probe), "%s/packed_experts_minimax_q4_lz4/layer_00.bin", model_path);
            if (!g_use_2bit && access(lz4_probe, R_OK) == 0) {
                int lz4_layers = 0;
                for (int i = 0; i < NUM_LAYERS; i++) {
                    char lz4_path[1024];
                    snprintf(lz4_path, sizeof(lz4_path), "%s/packed_experts_minimax_q4_lz4/layer_%02d.bin", model_path, i);
                    int lz4_fd = open(lz4_path, O_RDONLY);
                    if (lz4_fd >= 0) {
                        // Load index header (512 entries × 16 bytes = 8KB)
                        g_lz4_index[i] = malloc(NUM_EXPERTS * sizeof(LZ4IndexEntry));
                        ssize_t nr = pread(lz4_fd, g_lz4_index[i],
                                           NUM_EXPERTS * sizeof(LZ4IndexEntry), 0);
                        if (nr == NUM_EXPERTS * (ssize_t)sizeof(LZ4IndexEntry)) {
                            // Replace the raw fd with the LZ4 fd
                            close(layer_fds[i]);
                            layer_fds[i] = lz4_fd;
                            fcntl(lz4_fd, F_RDAHEAD, 1);
                            lz4_layers++;
                        } else {
                            free(g_lz4_index[i]);
                            g_lz4_index[i] = NULL;
                            close(lz4_fd);
                        }
                    }
                }
                if (lz4_layers > 0) {
                    g_use_lz4 = 1;
                    // Allocate compressed read buffers (one per expert slot)
                    for (int k = 0; k < MAX_K; k++) {
                        g_lz4_comp_bufs[k] = malloc(EXPERT_SIZE + 4096);
                    }
                    printf("[lz4] %d/%d layers using LZ4 compressed experts\n",
                           lz4_layers, NUM_LAYERS);
                }
            }
        }
        // Wire up tiered I/O globals
        g_layer_fds_cold = layer_fds_cold;
        if (!g_use_lz4)
            printf("[tiered-io] Cold fds (F_NOCACHE) + warm fds (page cached) active\n");
        // Warm page cache hint
        if (expert_layers_available > 0) {
            double t_warm = now_ms();
            for (int i = 0; i < NUM_LAYERS; i++) {
                if (layer_fds[i] >= 0) {
                    char dummy[4096];
                    pread(layer_fds[i], dummy, sizeof(dummy), 0);
                }
            }
            printf("[warmup] Page cache hint: %.1f ms\n", now_ms() - t_warm);
        }
        // ---- Allocate per-layer state ----
        void **layer_states = calloc(NUM_LAYERS, sizeof(void *));
        KVCache **kv_caches = calloc(NUM_LAYERS, sizeof(KVCache *));
        for (int i = 0; i < NUM_LAYERS; i++) {
            int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
            if (is_full) {
                kv_caches[i] = kv_cache_new();
            } else {
                layer_states[i] = linear_attn_state_new();
            }
        }
        double t_init = now_ms();
        printf("[init] Setup: %.1f ms\n\n", t_init - t0);
        // ---- Allocate working buffers ----
        float *hidden = calloc(HIDDEN_DIM, sizeof(float));
        float *logits = calloc(VOCAB_SIZE, sizeof(float));
        uint16_t *final_norm_w = get_tensor_ptr(wf, "model.norm.weight");
        if (prefill_parity_request_path) {
            reset_delta_net_state();
            int rc = run_prefill_parity_request(prefill_parity_request_path, wf, vocab,
                                                layer_states, kv_caches,
                                                (void **)layer_mmaps, layer_fds,
                                                hidden, final_norm_w, K);
            free(hidden);
            free(logits);
            return rc;
        }
        if (prefill_microbatch_parity_request_path) {
            reset_delta_net_state();
            int rc = run_prefill_microbatch_parity_request(prefill_microbatch_parity_request_path, wf, vocab,
                                                           layer_states, kv_caches,
                                                           (void **)layer_mmaps, layer_fds,
                                                           hidden, final_norm_w, K);
            free(hidden);
            free(logits);
            return rc;
        }
        if (serve_replay_request_count > 0) {
            reset_delta_net_state();
            int rc = run_serve_replay_requests(serve_replay_request_count,
                                               serve_replay_request_paths,
                                               wf, vocab,
                                               layer_states, kv_caches,
                                               (void **)layer_mmaps, layer_fds,
                                               hidden, logits, final_norm_w, K);
            free(hidden);
            free(logits);
            return rc;
        }
        // ---- Serve mode: enter HTTP server loop (never returns) ----
        if (serve_port > 0) {
            reset_delta_net_state();
            serve_loop(serve_port, wf, vocab,
                       layer_states, kv_caches,
                       (void **)layer_mmaps, layer_fds,
                       hidden, logits, final_norm_w, K, dump_request_dir);
            // serve_loop never returns, but cleanup just in case
            free(hidden); free(logits);
            return 0;
        }
        // ---- Generate tokens ----
        reset_delta_net_state();  // zero GPU delta-net state before generation
        if (g_cache_telemetry_enabled) cache_telemetry_reset();
        printf("--- Generating %d tokens ---\n", max_tokens);
        int pos = 0;  // position counter for RoPE
        // ---- Batch prefill: pre-embed all prompt tokens ----
        // Embedding all tokens upfront into a batch buffer avoids interleaving
        // embed_lookup with GPU work, and enables the optimized prefill loop below.
        float *embed_batch = NULL;
        if (pt->count > 1) {
            embed_batch = malloc((size_t)pt->count * HIDDEN_DIM * sizeof(float));
            double t_embed = now_ms();
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(wf, pt->ids[i], embed_batch + (size_t)i * HIDDEN_DIM);
            }
            double embed_ms = now_ms() - t_embed;
            printf("  [prefill] batch embed %d tokens: %.1f ms\n", pt->count, embed_ms);
        }
        // ---- Batch prefill loop ----
        // Process all prompt tokens through the model. For intermediate tokens
        // (not the last), we use discard_deferred_experts() which waits for the GPU
        // but skips the CPU readback/combine of the last layer's expert outputs.
        // This is safe because the hidden state from intermediate prefill tokens
        // is immediately overwritten by the next token's embedding — the recurrent
        // state (KV cache, delta-net state) is already updated inside fused_layer_forward.
        if (pt->count > 1) {
            double t_prefill_batch = now_ms();
            double first_tok_ms = 0;
            for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) {
                double t_tok = now_ms();
                // Load pre-embedded token from batch buffer
                cache_telemetry_note_token();
                memcpy(hidden, embed_batch + (size_t)token_idx * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));
                // Run through all 60 transformer layers
                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                // Discard last layer's expert output — hidden will be overwritten
                // by the next token's embedding. Only wait for GPU (buffer safety).
                discard_deferred_experts();
                pos++;
                if (token_idx == 0) {
                    first_tok_ms = now_ms() - t_tok;
                }
            }
            double prefill_batch_ms = now_ms() - t_prefill_batch;
            double avg_ms = (pt->count > 2) ?
                (prefill_batch_ms - first_tok_ms) / (pt->count - 2) : first_tok_ms;
            printf("  [prefill] %d/%d tokens: %.0f ms (first: %.0f ms, rest avg: %.0f ms)\n",
                   pt->count - 1, pt->count, prefill_batch_ms, first_tok_ms, avg_ms);
        }
        // ---- Last prefill token (or single-token prompt) ----
        // This one needs full completion since we need hidden state for logits.
        dump_activate_capture();
        {
            cache_telemetry_note_token();
            if (embed_batch) {
                memcpy(hidden, embed_batch + (size_t)(pt->count - 1) * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));
            } else {
                embed_lookup(wf, pt->ids[0], hidden);
            }
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Full completion — need hidden state for final norm + lm_head
            complete_deferred_experts();
            pos++;
        }
        if (embed_batch) { free(embed_batch); embed_batch = NULL; }
        // ---- Final norm ----
        if (final_norm_w) {
            float *normed = malloc(HIDDEN_DIM * sizeof(float));
            cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
            memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
            free(normed);
        }
        // ---- LM head ----
        double t_lm = now_ms();
        lm_head_forward(wf, hidden, logits);
        double lm_ms = now_ms() - t_lm;
        // ---- Sample first token ----
        int next_token = cpu_argmax(logits, VOCAB_SIZE);
        double ttft_ms = now_ms() - t0;
        dump_write_result(hidden, logits, next_token);
        dump_finish_capture();
        // Debug: show top-5 logits for first token
        {
            // Find top 5 manually
            int top5[5] = {0,0,0,0,0};
            float topv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (int i = 0; i < VOCAB_SIZE; i++) {
                int min_k = 0;
                for (int k = 1; k < 5; k++) if (topv[k] < topv[min_k]) min_k = k;
                if (logits[i] > topv[min_k]) { topv[min_k] = logits[i]; top5[min_k] = i; }
            }
            fprintf(stderr, "[debug] Top 5 logits (next_token=%d):\n", next_token);
            for (int i = 0; i < 5; i++) {
                fprintf(stderr, "  token %d (\"%s\") logit=%.4f\n",
                        top5[i], decode_token(vocab, top5[i]), topv[i]);
            }
            fprintf(stderr, "[debug] hidden rms after final_norm=%.4f, logits rms=%.4f\n",
                    vec_rms(hidden, HIDDEN_DIM), vec_rms(logits, VOCAB_SIZE));
        }
        printf("[ttft] %.0f ms (prefill %d tokens + lm_head %.0f ms)\n",
               ttft_ms, pt->count, lm_ms);
        printf("\n--- Output ---\n");
        printf("%s", decode_token(vocab, next_token));
        fflush(stdout);
        int total_generated = 1;
        int in_think = (next_token == THINK_START_TOKEN) ? 1 : 0;
        int think_tokens = 0;
        // ---- Auto-regressive generation ----
        if (g_timing_enabled) timing_reset();
        if (g_pred_enabled) {
            g_pred_generating = 1;  // enable prediction storage/use during generation
            g_pred_valid = 0;       // reset — first gen token builds predictions
        }
        for (int gen = 1; gen < max_tokens; gen++) {
            double t_gen_start = now_ms();
            // Check EOS
            if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) {
                fprintf(stderr, "\n[eos] Token %d at position %d\n", next_token, gen);
                break;
            }
            // Think budget enforcement
            if (next_token == THINK_START_TOKEN) in_think = 1;
            if (next_token == THINK_END_TOKEN) in_think = 0;
            if (in_think) think_tokens++;
            // Embed the just-generated token (next iteration)
            cache_telemetry_note_token();
            embed_lookup(wf, next_token, hidden);
            // Run 60 layers (fused: 1+K cmd buffers per layer)
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Complete last layer's deferred GPU experts before final norm
            complete_deferred_experts();
            pos++;
            // Final norm
            if (final_norm_w) {
                float *normed = malloc(HIDDEN_DIM * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
                memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
                free(normed);
            }
            // LM head
            lm_head_forward(wf, hidden, logits);
            // Greedy sample
            next_token = cpu_argmax(logits, VOCAB_SIZE);
            // Think budget: force end thinking if over budget
            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = THINK_END_TOKEN;
                in_think = 0;
            }
            total_generated++;
            // Print decoded token
            printf("%s", decode_token(vocab, next_token));
            fflush(stdout);
            double t_gen_end = now_ms();
            double tok_time = t_gen_end - t_gen_start;
            // Print progress to stderr
            fprintf(stderr, "  [gen %d/%d] token_id=%d (%.0f ms, %.2f tok/s)\n",
                    gen, max_tokens, next_token, tok_time, 1000.0 / tok_time);
        }
        if (g_timing_enabled) timing_print();
        printf("\n\n--- Statistics ---\n");
        double total_time = now_ms() - t0;
        printf("Total time:     %.1f s\n", total_time / 1000.0);
        printf("TTFT:           %.0f ms\n", ttft_ms);
        printf("Tokens:         %d generated\n", total_generated);
        if (total_generated > 1) {
            double gen_time = total_time - ttft_ms;
            printf("Generation:     %.1f s (%.2f tok/s)\n",
                   gen_time / 1000.0, (total_generated - 1) * 1000.0 / gen_time);
        }
        printf("Config:         K=%d experts, %d layers\n", K, NUM_LAYERS);
        if (g_expert_cache) {
            uint64_t total = g_expert_cache->hits + g_expert_cache->misses;
            printf("Expert cache:   %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                   g_expert_cache->hits, g_expert_cache->misses,
                   total > 0 ? 100.0 * g_expert_cache->hits / total : 0.0,
                   g_expert_cache->num_entries, g_expert_cache->max_entries);
            cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
        } else if (g_malloc_cache) {
            uint64_t total = g_malloc_cache->hits + g_malloc_cache->misses;
            printf("Expert cache:   malloc %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                   g_malloc_cache->hits, g_malloc_cache->misses,
                   total > 0 ? 100.0 * g_malloc_cache->hits / total : 0.0,
                   g_malloc_cache->num_entries, g_malloc_cache->max_entries);
            cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
        }
        if (g_spec_route_attempts > 0) {
            printf("Spec routing:   %llu attempts, %llu preloads, %llu hits (%.1f%% prediction accuracy)\n",
                   g_spec_route_attempts, g_spec_route_preloads, g_spec_route_hits,
                   g_spec_route_attempts > 0
                       ? 100.0 * g_spec_route_hits / g_spec_route_attempts : 0.0);
        }
        if (g_freq_tracking) freq_print_analysis(K);
        if (g_routing_log) {
            fclose(g_routing_log);
            fprintf(stderr, "[routing] Logged %d samples to routing data file\n",
                    g_routing_log_samples);
            g_routing_log = NULL;
        }
        // ---- Cleanup ----
        io_pool_shutdown();
        if (g_malloc_cache) {
            malloc_cache_free(g_malloc_cache);
            g_malloc_cache = NULL;
        }
        if (g_expert_cache) {
            expert_cache_free(g_expert_cache);
            g_expert_cache = NULL;
        }
        if (g_prefill_chunk_cache) {
            expert_cache_free(g_prefill_chunk_cache);
            g_prefill_chunk_cache = NULL;
        }
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (kv_caches[i]) kv_cache_free(kv_caches[i]);
            if (layer_states[i]) linear_attn_state_free(layer_states[i]);
            if (layer_mmaps[i] != MAP_FAILED) munmap(layer_mmaps[i], layer_mmap_sizes[i]);
            if (layer_fds[i] >= 0) close(layer_fds[i]);
            if (layer_fds_cold[i] >= 0) close(layer_fds_cold[i]);
        }
        free(layer_states);
        free(kv_caches);
        free(hidden);
        free(logits);
        return 0;
    }
}
#endif // CHAT_MODE
