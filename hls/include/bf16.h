// =====================================================================
// bf16.h — bf16 位级算术(与打包 DSP 路径数值完全一致)
// 约定:denormal 输入 flush-to-zero(FTZ);注意力内部不产生 inf/nan
// =====================================================================
#pragma once
#include <cstdint>
#include "cfg.h"

union f32_bits { float f; uint32_t u; };

inline float u2f(uint32_t u) { f32_bits b; b.u = u; return b.f; }
inline uint32_t f2u(float f) { f32_bits b; b.f = f; return b.u; }

// bf16 -> fp32:精确(bf16 即 fp32 高 16 位)
inline float bf16_to_f32(bf16_t x) { return u2f((uint32_t)x << 16); }

// fp32 -> bf16:round-to-nearest-even(与 numpy/torch 一致)
inline bf16_t f32_to_bf16(float f) {
    uint32_t u = f2u(f);
    if ((u & 0x7F800000u) == 0x7F800000u) return (bf16_t)(u >> 16); // inf/nan 透传
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (bf16_t)(u >> 16);
}

struct bf16_parts {
    uint32_t s;   // 符号
    int32_t  e;   // 8-bit 指数域
    uint32_t m;   // 8-bit 有效数(含隐藏位);0 表示零/denormal(FTZ)
};

inline bf16_parts bf16_unpack(bf16_t x) {
    bf16_parts p;
    p.s = (x >> 15) & 1u;
    p.e = (int32_t)((x >> 7) & 0xFFu);
    p.m = (p.e == 0) ? 0u : (0x80u | (uint32_t)(x & 0x7Fu));
    return p;
}

// 由 8x8 整数尾数积装配 fp32(精确:16-bit 积 < fp32 24-bit 尾数)
// mm ∈ [2^14, 2^16);E_f32 = ea + eb + top - 127,top = mm 的 bit15
inline float assemble_prod(uint32_t sgn_xor, int32_t ea, int32_t eb, uint32_t mm) {
    if (mm == 0) return 0.0f;
    uint32_t top  = (mm >> 15) & 1u;
    int32_t  E    = ea + eb + (int32_t)top - 127;
    if (E <= 0)   return 0.0f;                      // 下溢 FTZ
    if (E >= 255) return u2f((sgn_xor << 31) | 0x7F800000u); // 上溢(注意力中不会发生)
    uint32_t frac = top ? ((mm & 0x7FFFu) << 8) : ((mm & 0x3FFFu) << 9);
    return u2f((sgn_xor << 31) | ((uint32_t)E << 23) | frac);
}

// 标量 bf16 乘法(整数尾数路径)→ 精确 fp32
inline float bf16_mul(bf16_t a, bf16_t b) {
    bf16_parts pa = bf16_unpack(a), pb = bf16_unpack(b);
    uint32_t mm = pa.m * pb.m;                       // 8x8,单 DSP 可容
    return assemble_prod(pa.s ^ pb.s, pa.e, pb.e, mm);
}
