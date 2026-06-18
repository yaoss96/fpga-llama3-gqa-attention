// =====================================================================
// rope.h — RoPE 旋转单元(HF Llama 半分裂约定,θ=500000)
//   out[i]     = x[i]·cos[i] − x[i+64]·sin[i]
//   out[i+64]  = x[i+64]·cos[i] + x[i]·sin[i]      (i < 64)
// sincos 为 fp32,由主机按位置预生成存 DDR,随行流式读取(<0.1% 带宽)
// KV cache 存旋转后的 K(bf16)→ decode 读取路径零 RoPE 开销
// =====================================================================
#pragma once
#include "bf16.h"

inline void rope_rotate(const bf16_t in[HD], const float cs[HD2], const float sn[HD2],
                        bf16_t out[HD], int en) {
ROPE_LOOP:
    for (int i = 0; i < HD2; ++i) {
#pragma HLS PIPELINE II=1
        if (en) {
            float x1 = bf16_to_f32(in[i]);
            float x2 = bf16_to_f32(in[i + HD2]);
            out[i]       = f32_to_bf16(x1 * cs[i] - x2 * sn[i]);
            out[i + HD2] = f32_to_bf16(x2 * cs[i] + x1 * sn[i]);
        } else {
            out[i]       = in[i];
            out[i + HD2] = in[i + HD2];
        }
    }
}
