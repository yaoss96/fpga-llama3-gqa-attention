// =====================================================================
// cfg.h — 全局参数(FPT'26 Track B, Llama3-8B attention, ZCU104)
// 所有规模旋钮集中于此;改 LANES/TKV/TQ 即可做资源-性能扫描(Pareto)
// =====================================================================
#pragma once
#include <cstdint>

// ---- Llama3-8B 注意力规格(不可改) ----
constexpr int HQ   = 32;          // query 头数
constexpr int HKV  = 8;           // KV 头数 (GQA)
constexpr int GQA  = HQ / HKV;    // 每 KV 头共享的 Q 头数 = 4
constexpr int HD   = 128;         // head_dim
constexpr int HD2  = HD / 2;      // RoPE 频率数 = 64
constexpr float SCALE     = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float ROPE_THETA = 500000.0f;

// ---- 架构旋钮(资源-性能 Pareto 扫描维度) ----
#ifndef CFG_LANES
#define CFG_LANES 16              // 可经 -DCFG_LANES=N 覆盖(ZCU104 默认 16,Ultra96v2 用 8 以塞进 XCZU3EG)
#endif
constexpr int LANES = CFG_LANES;  // 点积并行 lane 数(每 lane 1 个双打包 DSP = 2 乘法)
#ifndef CFG_NCLUSTER
#define CFG_NCLUSTER 1            // 簇数:N 个 KV 头并行处理(-DCFG_NCLUSTER=N;ZCU104 资源多可 2/4)
#endif
constexpr int N_CLUSTER = CFG_NCLUSTER;
constexpr int HPC       = HKV / N_CLUSTER;   // 每簇处理的 KV 头数
constexpr int TKV   = 64;         // KV tile 行数(片上缓冲)
constexpr int TQ    = 8;          // prefill Q tile 行数(decode 时 q_len=1 自动退化)
constexpr int NPH   = 8;          // 浮点累加相位数(覆盖 fadd 流水延迟,II=1 关键)
constexpr int MAX_SEQ = 8192;     // KV cache 容量(Llama3 上下文)

// ---- 总线/布局 ----
constexpr int WLANES = 8;                 // 每个 128-bit 字含 8 个 bf16
constexpr int HDW    = HD / WLANES;       // 每 head-row 16 字 = 256B(整突发)
using bf16_t = uint16_t;                  // bf16 存储类型(位模式)
struct word_t  { bf16_t lane[WLANES]; };  // 128-bit gmem 字(HLS 自动打包)
struct fword_t { float  f[4]; };          // 128-bit float 字(sincos 表)
constexpr int SCW = HD / 4;               // sincos 每行 32 个 fword(64 cos + 64 sin)

static_assert(HD % LANES == 0, "HD must be divisible by LANES");
static_assert(GQA % 2 == 0, "dual-pack pairs two Q heads");
static_assert(HKV % N_CLUSTER == 0, "HKV must be divisible by N_CLUSTER");
