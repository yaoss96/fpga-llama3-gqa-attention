// =====================================================================
// dsp_pack.h — 单 DSP48E2 双 bf16 乘法(本方案核心创新点)
//
// 原理:bf16 有效数(含隐藏位)仅 8 bit。DSP48E2 = 27x18 乘法器:
//   27 位口装 packed = mq0 + mq1·2^18(26 bit,18-bit 间距留 2 保护位)
//   18 位口装共享操作数 mk(8 bit)
//   积 prod = mk·mq0 + mk·mq1·2^18,且 mk·mq0 ≤ 255² < 2^16 → 两积无重叠
// GQA 中 4 个 Q 头共享同一条 K → 天然成对共享操作数:乘法器 DSP 直接减半。
// 指数加/符号异或/规格化在 fabric(每乘 ~15 LUT)。
// 数值与 bf16_mul 标量路径逐位一致(testbench 强制校验)。
// 若 HLS 工具将该乘法拆开,可用 DSP48E2 原语直接例化兜底(RTL 黑盒)。
// =====================================================================
#pragma once
#include "bf16.h"

inline void bf16_dual_mul(bf16_t shared_k, bf16_t q0, bf16_t q1,
                          float &r0, float &r1) {
#pragma HLS INLINE
    bf16_parts pk = bf16_unpack(shared_k);
    bf16_parts p0 = bf16_unpack(q0);
    bf16_parts p1 = bf16_unpack(q1);

    uint32_t packed = ((uint32_t)p1.m << 18) | p0.m;     // 26 bit → 27 位口
    uint64_t prod   = (uint64_t)packed * (uint64_t)pk.m; // 26x8 → 34 bit,单 DSP
#pragma HLS BIND_OP variable=prod op=mul impl=dsp

    uint32_t mm0 = (uint32_t)(prod & 0xFFFFu);           // bits[15:0]  = mk·mq0
    uint32_t mm1 = (uint32_t)((prod >> 18) & 0xFFFFu);   // bits[33:18] = mk·mq1

    r0 = assemble_prod(pk.s ^ p0.s, pk.e, p0.e, mm0);
    r1 = assemble_prod(pk.s ^ p1.s, pk.e, p1.e, mm1);
}
