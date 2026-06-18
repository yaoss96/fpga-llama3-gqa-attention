// =====================================================================
// tb_attn.cpp — C 仿真测试台(csim / g++ 双用)
//  1) 位级单元测试:RNE 舍入、整数路径乘法精确性、双打包≡标量、exp2 误差
//  2) 端到端:prefill(150 token)+ 3 步 decode,对比双精度黄金模型(跨多 KV tile)
//     黄金模型:同布局/同 RoPE 约定(HF 半分裂)/同 GQA 映射,double 计算
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "hls_stream.h"
#include "cfg.h"
#include "bf16.h"
#include "dsp_pack.h"
#include "rope.h"
#include "softmax.h"

void attn_top(word_t*, word_t*, word_t*,                 // kcache, vcache, kvin
              const word_t*, word_t*, const fword_t*,    // qin0, out0, sincos0 (簇0/gmem0)
#if CFG_NCLUSTER >= 2
              const word_t*, word_t*, const fword_t*,    // qin1, out1, sincos1 (簇1/gmem1)
#endif
              hls::stream<word_t>*, hls::stream<word_t>*, int, int, int, int);

// 各簇 q/out/sincos 指向同一缓冲(访问不相交头切片);N=2 时复制一份给簇1
#if CFG_NCLUSTER >= 2
#define QOS_ARGS qin.data(), outb.data(), sincos.data(), qin.data(), outb.data(), sincos.data()
#else
#define QOS_ARGS qin.data(), outb.data(), sincos.data()
#endif

// 按 kernel 消费顺序喂 AXIS 流(模拟 DMA);多簇:头 kvh 归簇 kvh/HPC,喂到 ks[c]/vs[c]
static void feed_kv_streams(const std::vector<word_t>& kc, const std::vector<word_t>& vc,
                            hls::stream<word_t>* ks, hls::stream<word_t>* vs,
                            int q_len, int pos_base) {
    for (int kvh = 0; kvh < HKV; ++kvh) {
        int c = kvh / HPC;
        for (int qt0 = 0; qt0 < q_len; qt0 += TQ) {
            int rows = (q_len - qt0 < TQ) ? (q_len - qt0) : TQ;
            int last_pos = pos_base + qt0 + rows - 1;
            int n_kt = last_pos / TKV + 1;
            for (int kt = 0; kt < n_kt; ++kt) {
                int t0 = kt * TKV;
                int tvalid = (last_pos + 1 - t0 < TKV) ? (last_pos + 1 - t0) : TKV;
                for (int j = 0; j < tvalid; ++j)
                    for (int w = 0; w < HDW; ++w) {
                        int idx = (kvh * MAX_SEQ + (t0 + j)) * HDW + w;
                        ks[c].write(kc[idx]); vs[c].write(vc[idx]);
                    }
            }
        }
    }
}

static std::mt19937 rng(2026);

static bf16_t rand_bf16(double scale = 1.0) {
    std::normal_distribution<double> nd(0.0, scale);
    return f32_to_bf16((float)nd(rng));
}

// ---------------- 单元测试 ----------------
static int test_rne() {
    std::uniform_real_distribution<float> ud(-100.f, 100.f);
    for (int i = 0; i < 200000; ++i) {
        float f = ud(rng);
        bf16_t got = f32_to_bf16(f);
        // 参考:在 bf16 网格上取最近(平局取偶)
        uint32_t u = f2u(f);
        uint32_t lo = u & 0xFFFF0000u, hi = lo + 0x10000u;
        double df = f, dlo = (double)u2f(lo), dhi = (double)u2f(hi);
        uint32_t ref;
        if (fabs(df - dlo) < fabs(dhi - df)) ref = lo;
        else if (fabs(df - dlo) > fabs(dhi - df)) ref = hi;
        else ref = ((lo >> 16) & 1u) ? hi : lo;     // tie → even
        if (got != (bf16_t)(ref >> 16)) {
            printf("[RNE] FAIL f=%g got=%04x ref=%04x\n", f, got, ref >> 16);
            return 1;
        }
    }
    printf("[RNE] PASS (200k)\n");
    return 0;
}

static int test_mul_exact() {
    for (int i = 0; i < 200000; ++i) {
        bf16_t a = rand_bf16(4.0), b = rand_bf16(4.0);
        float ref = bf16_to_f32(a) * bf16_to_f32(b);  // fp32 乘对 8-bit 有效数精确
        float got = bf16_mul(a, b);
        if (f2u(got) != f2u(ref) && !(got == 0.f && fabsf(ref) < 1e-37f)) {
            printf("[MUL] FAIL a=%04x b=%04x got=%g ref=%g\n", a, b, got, ref);
            return 1;
        }
    }
    printf("[MUL] PASS (200k, 与 fp32 精确积逐位一致)\n");
    return 0;
}

static int test_dual() {
    for (int i = 0; i < 200000; ++i) {
        bf16_t k = rand_bf16(2.0), q0 = rand_bf16(2.0), q1 = rand_bf16(2.0);
        float r0, r1;
        bf16_dual_mul(k, q0, q1, r0, r1);
        if (f2u(r0) != f2u(bf16_mul(k, q0)) || f2u(r1) != f2u(bf16_mul(k, q1))) {
            printf("[DUAL] FAIL\n");
            return 1;
        }
    }
    printf("[DUAL] PASS (200k, 双打包 ≡ 标量逐位一致)\n");
    return 0;
}

static int test_exp2() {
    double maxrel = 0;
    for (int i = 0; i < 100000; ++i) {
        std::uniform_real_distribution<float> ud(-30.f, 0.f);
        float x = ud(rng);
        double ref = exp2((double)x), got = (double)exp2_neg(x);
        maxrel = std::max(maxrel, fabs(got - ref) / ref);
    }
    printf("[EXP2] max rel err = %.3e %s\n", maxrel, maxrel < 3e-3 ? "PASS" : "FAIL");
    return maxrel < 3e-3 ? 0 : 1;
}

// ---------------- 端到端 ----------------
static const int PRE = 150, DEC = 3, TOTAL = PRE + DEC;  // 153 > 2×TKV:覆盖跨tile在线重缩放

static void rope_d(const bf16_t in[HD], const float cs[HD2], const float sn[HD2], double out[HD]) {
    for (int i = 0; i < HD2; ++i) {
        double x1 = bf16_to_f32(in[i]), x2 = bf16_to_f32(in[i + HD2]);
        out[i]       = x1 * (double)cs[i] - x2 * (double)sn[i];
        out[i + HD2] = x2 * (double)cs[i] + x1 * (double)sn[i];
    }
}

int main() {
    int fails = test_rne() + test_mul_exact() + test_dual() + test_exp2();
    if (fails) return fails;

    // ---- 输入与 sincos 表 ----
    static bf16_t q[TOTAL][HQ][HD], k[TOTAL][HKV][HD], v[TOTAL][HKV][HD];
    for (int t = 0; t < TOTAL; ++t) {
        for (int h = 0; h < HQ; ++h)
            for (int d = 0; d < HD; ++d) q[t][h][d] = rand_bf16();
        for (int h = 0; h < HKV; ++h)
            for (int d = 0; d < HD; ++d) { k[t][h][d] = rand_bf16(); v[t][h][d] = rand_bf16(); }
    }
    static float csT[TOTAL][HD2], snT[TOTAL][HD2];
    for (int p = 0; p < TOTAL; ++p)
        for (int i = 0; i < HD2; ++i) {
            double ang = (double)p * pow((double)ROPE_THETA, -2.0 * i / HD);
            csT[p][i] = (float)cos(ang);
            snT[p][i] = (float)sin(ang);
        }

    // ---- 设备缓冲 ----
    // Keep testbench host buffers at least as large as the HLS m_axi depths.
    // The cosim wrapC layer dumps the pragma depth, not just the touched range.
    const size_t DEPTH_QO = 131072;
    const size_t DEPTH_KVIN = 65536;
    const size_t DEPTH_SINCOS = 262144;
    std::vector<word_t> kcache((size_t)HKV * MAX_SEQ * HDW), vcache((size_t)HKV * MAX_SEQ * HDW);
    std::vector<word_t> qin(DEPTH_QO), kvin(DEPTH_KVIN);
    std::vector<word_t> outb(DEPTH_QO);
    std::vector<fword_t> sincos(DEPTH_SINCOS);
    for (int p = 0; p < TOTAL; ++p)
        for (int w = 0; w < SCW; ++w)
            for (int e = 0; e < 4; ++e)
                sincos[p * SCW + w].f[e] = (w < SCW / 2) ? csT[p][w * 4 + e] : snT[p][(w - SCW / 2) * 4 + e];

    auto pack_row = [](std::vector<word_t> &buf, int base_w, const bf16_t row[HD]) {
        for (int w = 0; w < HDW; ++w)
            for (int l = 0; l < WLANES; ++l) buf[base_w + w].lane[l] = row[w * WLANES + l];
    };

    static bf16_t outc[TOTAL][HQ][HD];   // 收集所有调用的输出

    // ---- 调用 1:prefill(q_len=PRE, pos_base=0)----
    for (int r = 0; r < PRE; ++r) {
        for (int h = 0; h < HQ; ++h) pack_row(qin, (r * HQ + h) * HDW, q[r][h]);
        for (int h = 0; h < HKV; ++h) {
            pack_row(kvin, ((r * 2 + 0) * HKV + h) * HDW, k[r][h]);
            pack_row(kvin, ((r * 2 + 1) * HKV + h) * HDW, v[r][h]);
        }
    }
    hls::stream<word_t> kstr[N_CLUSTER], vstr[N_CLUSTER];
    attn_top(kcache.data(), vcache.data(), kvin.data(), QOS_ARGS,
             kstr, vstr, PRE, 0, 1, 1);                                // append(写 cache)
    feed_kv_streams(kcache, vcache, kstr, vstr, PRE, 0);               // DMA 模拟:喂 KV 流(分簇)
    attn_top(kcache.data(), vcache.data(), kvin.data(), QOS_ARGS,
             kstr, vstr, PRE, 0, 1, 2);                                // attend(消费流)
    for (int r = 0; r < PRE; ++r)
        for (int h = 0; h < HQ; ++h)
            for (int w = 0; w < HDW; ++w)
                for (int l = 0; l < WLANES; ++l)
                    outc[r][h][w * WLANES + l] = outb[(r * HQ + h) * HDW + w].lane[l];

    // ---- 调用 2..4:decode(q_len=1)----
    for (int s = 0; s < DEC; ++s) {
        const int p = PRE + s;
        for (int h = 0; h < HQ; ++h) pack_row(qin, h * HDW, q[p][h]);
        for (int h = 0; h < HKV; ++h) {
            pack_row(kvin, (0 * HKV + h) * HDW, k[p][h]);
            pack_row(kvin, (1 * HKV + h) * HDW, v[p][h]);
        }
        hls::stream<word_t> kstr[N_CLUSTER], vstr[N_CLUSTER];
        attn_top(kcache.data(), vcache.data(), kvin.data(), QOS_ARGS,
                 kstr, vstr, 1, p, 1, 1);                             // append
        feed_kv_streams(kcache, vcache, kstr, vstr, 1, p);            // DMA 模拟(分簇)
        attn_top(kcache.data(), vcache.data(), kvin.data(), QOS_ARGS,
                 kstr, vstr, 1, p, 1, 2);                             // attend(sincos 按绝对位置)
        for (int h = 0; h < HQ; ++h)
            for (int w = 0; w < HDW; ++w)
                for (int l = 0; l < WLANES; ++l)
                    outc[p][h][w * WLANES + l] = outb[(h)*HDW + w].lane[l];
    }

    // ---- 黄金模型(double)与比对 ----
    double maxrel = 0, sumrel = 0, worst_g = 0, worst_o = 0;
    long cnt = 0;
    double cosmin = 1.0;
    for (int P = 0; P < TOTAL; ++P) {
        for (int hq = 0; hq < HQ; ++hq) {
            const int kvh = hq / GQA;
            double qr[HD];
            rope_d(q[P][hq], csT[P], snT[P], qr);
            std::vector<double> sc(P + 1);
            double mx = -1e300;
            for (int j = 0; j <= P; ++j) {
                double kr[HD];
                rope_d(k[j][kvh], csT[j], snT[j], kr);
                double dot = 0;
                for (int d = 0; d < HD; ++d) dot += qr[d] * kr[d];
                sc[j] = dot / sqrt((double)HD);
                mx = std::max(mx, sc[j]);
            }
            double l = 0;
            for (int j = 0; j <= P; ++j) { sc[j] = exp(sc[j] - mx); l += sc[j]; }
            double og[HD] = {0};
            for (int j = 0; j <= P; ++j)
                for (int d = 0; d < HD; ++d) og[d] += sc[j] * (double)bf16_to_f32(v[j][kvh][d]);
            // 比对:误差按行 RMS 归一化(bf16 验证标准做法,等价 atol+rtol)
            double rms = 0;
            for (int d = 0; d < HD; ++d) { double gd = og[d] / l; rms += gd * gd; }
            rms = sqrt(rms / HD);
            double dotp = 0, n1 = 0, n2 = 0;
            for (int d = 0; d < HD; ++d) {
                double gold = og[d] / l;
                double got = (double)bf16_to_f32(outc[P][hq][d]);
                double rel = fabs(got - gold) / rms;
                if (rel > maxrel) { maxrel = rel; worst_g = gold; worst_o = got; }
                sumrel += rel; ++cnt;
                dotp += gold * got; n1 += gold * gold; n2 += got * got;
            }
            cosmin = std::min(cosmin, dotp / sqrt(n1 * n2));
        }
    }
    printf("[E2E] rows=%d  max_err/rms=%.4f (gold=%.4g got=%.4g)  mean_err/rms=%.5f  min_cos=%.6f\n",
           TOTAL, maxrel, worst_g, worst_o, sumrel / cnt, cosmin);
    const bool pass = maxrel < 0.05 && (sumrel / cnt) < 0.01 && cosmin > 0.999;
    printf("[E2E] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
