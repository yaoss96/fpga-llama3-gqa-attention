// =====================================================================
// softmax.h — 在线 softmax 的 exp 单元:exp2 查表 + 指数域移位,零 DSP
//   e^x = 2^(x·log2e),x ≤ 0(在线 softmax 保证 s−m ≤ 0)
//   小数部分查 256 项 ROM,整数部分直接加到 fp32 指数位
//   相对误差 ≤ 2^(1/256)−1 ≈ 0.27%(< bf16 半 ulp 量级,可加插值再降)
// =====================================================================
#pragma once
#include "bf16.h"
#include "exp2_lut.h"

constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -3.0e38f;   // 在线 softmax 的 m 初值(代替 -inf)

inline float exp2_neg(float x) {      // 要求 x <= 0
    if (x <= -126.0f) return 0.0f;    // 下溢截断(也兜住 m 初值路径)
    float t = x * 256.0f;
    int xi = (int)t;
    if ((float)xi > t) xi -= 1;       // floor(负数)
    int n = xi >> 8;                  // 2 的整数次幂(≤0)
    uint32_t fi = (uint32_t)(xi & 0xFF);
    uint32_t u = f2u(EXP2_LUT[fi]);   // ∈ [1,2),指数域 = 127
    u += (uint32_t)(n << 23);         // 指数域直接加 n(x>-126 保证不下溢)
    return u2f(u);
}
