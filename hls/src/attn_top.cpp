// =====================================================================
// attn_top.cpp — FPT'26 Track B 注意力加速器顶层(ZCU104 / PYNQ)
//
// 统一 prefill/decode 路径:decode 即 q_len=1 的 prefill(同一份硬件)
//   Phase 0: 新 K/V 行 RoPE 后追加写入 KV cache(K 旋转后缓存)
//   Phase 1: FlashAttention-2 式分块在线 softmax,逐 KV 头处理
//            每 KV 头广播给 4 个 Q 头(GQA 复用),Q 头两两共享
//            双打包 DSP 乘法器(QK^T 与 P·V 均复用该原语)
//
// 吞吐优化(本版):
//   ① m_axi 突发:KV tile 整 tile 连续地址单循环读 → HLS 推断长突发
//   ② DATAFLOW 双缓冲:生产者(突发读→FIFO) || 消费者(计算),隐藏访存延迟
//   ③ 内层点积 PIPELINE II=1(QK/PV 相位累加覆盖 fadd 延迟)
//
// 内存布局(均为 word_t = 8×bf16 = 128bit):
//   kcache/vcache : [HKV][MAX_SEQ][HDW]
//   qin           : [q_len][HQ][HDW]      (未旋转 Q)
//   kvin          : [q_len][2][HKV][HDW]  (未旋转 K, V)
//   out           : [q_len][HQ][HDW]
//   sincos        : [pos][SCW] fword_t    (每行 64 cos + 64 sin, fp32)
// =====================================================================
#include <cstring>
#include "hls_stream.h"
#include "cfg.h"
#include "bf16.h"
#include "dsp_pack.h"
#include "rope.h"
#include "softmax.h"

static inline int idx_cache(int kvh, int t) { return (kvh * MAX_SEQ + t) * HDW; }

// ---------- 行级搬运 ----------
static void load_row(const word_t *src, int base_w, bf16_t row[HD]) {
LOAD_ROW:
    for (int w = 0; w < HDW; ++w) {
#pragma HLS PIPELINE II=1
        word_t wd = src[base_w + w];
        for (int l = 0; l < WLANES; ++l) {
#pragma HLS UNROLL
            row[w * WLANES + l] = wd.lane[l];
        }
    }
}

static void store_row(word_t *dst, int base_w, const bf16_t row[HD]) {
STORE_ROW:
    for (int w = 0; w < HDW; ++w) {
#pragma HLS PIPELINE II=1
        word_t wd;
        for (int l = 0; l < WLANES; ++l) {
#pragma HLS UNROLL
            wd.lane[l] = row[w * WLANES + l];
        }
        dst[base_w + w] = wd;
    }
}

static void load_sincos(const fword_t *sincos, int pos, float cs[HD2], float sn[HD2]) {
LOAD_SC:
    for (int w = 0; w < SCW; ++w) {     // 单次连续突发:前 16 字 cos,后 16 字 sin
#pragma HLS PIPELINE II=1
        fword_t x = sincos[pos * SCW + w];
        for (int k = 0; k < 4; ++k) {
#pragma HLS UNROLL
            if (w < HD2 / 4) cs[w * 4 + k]             = x.f[k];
            else             sn[(w - HD2 / 4) * 4 + k] = x.f[k];
        }
    }
}

// ---------- 求和树 ----------
static float tree_sum(const float v[], int n) {  // n ∈ {NPH, LANES, HD/LANES} 小常数
    float acc = 0.0f;
TREE:
    for (int i = 0; i < n; ++i) {
#pragma HLS UNROLL
        acc += v[i];
    }
    return acc;
}

// ---------- Phase 0:RoPE + KV cache 追加 ----------
static void append_kv(word_t *kcache, word_t *vcache, const word_t *kvin,
                      const fword_t *sincos, int q_len, int pos_base, int rope_en) {
    bf16_t krow[HD], krot[HD], vrow[HD];
#pragma HLS ARRAY_PARTITION variable=krow cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=krot cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=vrow cyclic factor=8
    float cs[HD2], sn[HD2];
#pragma HLS ARRAY_PARTITION variable=cs cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=sn cyclic factor=4
APPEND_R:
    for (int r = 0; r < q_len; ++r) {
        const int pos = pos_base + r;
        load_sincos(sincos, pos, cs, sn);
APPEND_H:
        for (int h = 0; h < HKV; ++h) {
            load_row(kvin, ((r * 2 + 0) * HKV + h) * HDW, krow);
            rope_rotate(krow, cs, sn, krot, rope_en);
            store_row(kcache, idx_cache(h, pos), krot);
            load_row(kvin, ((r * 2 + 1) * HKV + h) * HDW, vrow);
            store_row(vcache, idx_cache(h, pos), vrow);
        }
    }
}

// ---------- Phase 1 消费者:AXIS 流 → tile + 在线 softmax ----------
// KV 读由 BD 里的 AXI DMA 突发搬运,经顶层 AXIS 输入口喂进来(绕开 HLS m_axi 单拍读)
// 主机按消费顺序(逐 KV 头 × 逐 Q tile × 逐 KV tile)用 DMA 把 cache 流给本 kernel
// ---------- 访存↔算力重叠:单个 KV tile 的"装载"与"计算"拆成两个 DATAFLOW 进程 ----------
// loader 弹流填下一 tile 的 k_tile/v_tile,与 compute 算当前 tile 并发;
// k_tile/v_tile 作为进程间数组通道,HLS 自动实现乒乓(PIPO)双缓冲(避开 16KB 结构体流超 4096-bit 限制)
static void load_tile(hls::stream<word_t> &ksf, hls::stream<word_t> &vsf,
                      bf16_t k_tile[TKV][HD], bf16_t v_tile[TKV][HD],
                      int kt, int last_pos) {
#pragma HLS INLINE                       // 内联进 kv_consumer:消除函数边界互连,布线更易(原合并版 50min 布通)
    const int t0 = kt * TKV;
    const int tvalid = ((last_pos + 1 - t0) < TKV) ? (last_pos + 1 - t0) : TKV;
    const int nwords = tvalid * HDW;
POP_K:
    for (int w = 0; w < nwords; ++w) {
#pragma HLS PIPELINE II=1
        word_t wd = ksf.read();
        const int j = w / HDW, ww = w % HDW;
        for (int l = 0; l < WLANES; ++l) {
#pragma HLS UNROLL
            k_tile[j][ww * WLANES + l] = wd.lane[l];
        }
    }
POP_V:
    for (int w = 0; w < nwords; ++w) {
#pragma HLS PIPELINE II=1
        word_t wd = vsf.read();
        const int j = w / HDW, ww = w % HDW;
        for (int l = 0; l < WLANES; ++l) {
#pragma HLS UNROLL
            v_tile[j][ww * WLANES + l] = wd.lane[l];
        }
    }
}

static void compute_tile(const bf16_t k_tile[TKV][HD], const bf16_t v_tile[TKV][HD],
                         const bf16_t q_tile[TQ][GQA][HD],
                         float m_st[TQ][GQA], float l_st[TQ][GQA], float Oacc[TQ][GQA][HD],
                         int kt, int rows, int qt0, int pos_base, int last_pos, int skip_compute) {
#pragma HLS INLINE                       // 内联进 kv_consumer:与 load_tile 合并成单函数,布线更易
    const int t0 = kt * TKV;
    const int tvalid = ((last_pos + 1 - t0) < TKV) ? (last_pos + 1 - t0) : TKV;
    if (skip_compute) {   // profiling:仅测 KV 访存/dataflow,触碰数据防 DCE,跳过算力
        Oacc[0][0][0] += bf16_to_f32(k_tile[tvalid - 1][0]) + bf16_to_f32(v_tile[tvalid - 1][0]);
        return;
    }
ROW_LOOP:
        for (int r = 0; r < rows; ++r) {
            const int rowpos = pos_base + qt0 + r;
            const int jlim = rowpos - t0;             // 因果上界(tile 内,含)
            if (jlim < 0) continue;                   // 整 tile 在未来 → 跳过
            const int jcnt = (jlim + 1 < tvalid) ? (jlim + 1) : tvalid;

PAIR_LOOP:
            for (int gp = 0; gp < GQA / 2; ++gp) {
                const int h0 = 2 * gp, h1 = 2 * gp + 1;
                float s0[TKV], s1[TKV];
                bf16_t p0b[TKV], p1b[TKV];

                // ---- QK^T:双打包 DSP,K 广播给头对 ----
                // 每 j 的各 lane-chunk 部分和先存下,再单独归约 → 写读解耦,无 false 依赖冒险
                // (原来内联归约 + DEPENDENCE false 在 LANES=16 时 HD/LANES 变小,fp 树延迟 > 间距 → 板上读到未就绪值)
                float c0[TKV][HD / LANES], c1[TKV][HD / LANES];
#pragma HLS ARRAY_PARTITION variable=c0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=c1 complete dim=2
QK_J:
                for (int j = 0; j < jcnt; ++j) {
QK_C:
                    for (int c = 0; c < HD / LANES; ++c) {
#pragma HLS PIPELINE II=1
                        float pr0[LANES], pr1[LANES];
                        for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
                            const int d = c * LANES + l;
                            bf16_dual_mul(k_tile[j][d], q_tile[r][h0][d],
                                          q_tile[r][h1][d], pr0[l], pr1[l]);
                        }
                        c0[j][c] = tree_sum(pr0, LANES);
                        c1[j][c] = tree_sum(pr1, LANES);
                    }
                }
QK_RED:
                for (int j = 0; j < jcnt; ++j) {     // 单独归约:c0/c1 全部写完才读,无冒险
#pragma HLS PIPELINE II=1
                    s0[j] = tree_sum(c0[j], HD / LANES) * SCALE;
                    s1[j] = tree_sum(c1[j], HD / LANES) * SCALE;
                }

                // ---- 在线 softmax(每头独立状态)----
                // max 归约用 NPH 相位寄存器:消除标量环路依赖(fcmp 延迟 > II)
                float m0 = m_st[r][h0], m1 = m_st[r][h1];
                float tmx0[NPH], tmx1[NPH];
                for (int p = 0; p < NPH; ++p) { tmx0[p] = m0; tmx1[p] = m1; }
SMAX_MAX:
                for (int j = 0; j < jcnt; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=tmx0 inter distance=8 true
#pragma HLS DEPENDENCE variable=tmx1 inter distance=8 true
                    const int ph = j % NPH;
                    tmx0[ph] = (s0[j] > tmx0[ph]) ? s0[j] : tmx0[ph];
                    tmx1[ph] = (s1[j] > tmx1[ph]) ? s1[j] : tmx1[ph];
                }
                float tm0 = m0, tm1 = m1;
                for (int p = 0; p < NPH; ++p) {
#pragma HLS UNROLL
                    tm0 = (tmx0[p] > tm0) ? tmx0[p] : tm0;
                    tm1 = (tmx1[p] > tm1) ? tmx1[p] : tm1;
                }
                const float a0 = exp2_neg((m0 - tm0) * LOG2E);  // 首 tile: m=NEG_BIG → α=0
                const float a1 = exp2_neg((m1 - tm1) * LOG2E);
                float ps0[NPH], ps1[NPH];
                for (int p = 0; p < NPH; ++p) { ps0[p] = 0.0f; ps1[p] = 0.0f; }
SMAX_EXP:
                for (int j = 0; j < jcnt; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=ps0 inter distance=8 true
#pragma HLS DEPENDENCE variable=ps1 inter distance=8 true
                    const float e0 = exp2_neg((s0[j] - tm0) * LOG2E);
                    const float e1 = exp2_neg((s1[j] - tm1) * LOG2E);
                    ps0[j % NPH] += e0;
                    ps1[j % NPH] += e1;
                    p0b[j] = f32_to_bf16(e0);          // P 量化为 bf16 → 双打包 PV
                    p1b[j] = f32_to_bf16(e1);
                }
                l_st[r][h0] = a0 * l_st[r][h0] + tree_sum(ps0, NPH);
                l_st[r][h1] = a1 * l_st[r][h1] + tree_sum(ps1, NPH);
                m_st[r][h0] = tm0;
                m_st[r][h1] = tm1;

                // ---- P·V:双打包 DSP,V 广播给头对;相位累加保 II=1 ----
PV_C:
                for (int c = 0; c < HD / LANES; ++c) {
                    float rot0[LANES][NPH], rot1[LANES][NPH];
                    for (int l = 0; l < LANES; ++l)
                        for (int p = 0; p < NPH; ++p) { rot0[l][p] = 0.0f; rot1[l][p] = 0.0f; }
PV_J:
                    for (int j = 0; j < jcnt; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=rot0 inter distance=8 true
#pragma HLS DEPENDENCE variable=rot1 inter distance=8 true
                        for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
                            const int d = c * LANES + l;
                            float v0, v1;
                            bf16_dual_mul(v_tile[j][d], p0b[j], p1b[j], v0, v1);
                            rot0[l][j % NPH] += v0;
                            rot1[l][j % NPH] += v1;
                        }
                    }
PV_WB:
                    for (int l = 0; l < LANES; ++l) {
#pragma HLS PIPELINE II=1
                        const int d = c * LANES + l;
                        Oacc[r][h0][d] = a0 * Oacc[r][h0][d] + tree_sum(rot0[l], NPH);
                        Oacc[r][h1][d] = a1 * Oacc[r][h1][d] + tree_sum(rot1[l], NPH);
                    }
                }
            } // PAIR_LOOP
        } // ROW_LOOP
}

// CONS_KT 驱动:单缓冲、无 DATAFLOW → load_tile 与 compute_tile 顺序执行(不重叠)。
// 重叠版实测在硬件上无收益(decode 被主机逐头喂数串行掩盖)且加布线拥塞,故用这个干净的顺序版。
static void kv_consumer(hls::stream<word_t> &ksf, hls::stream<word_t> &vsf,
                        const bf16_t q_tile[TQ][GQA][HD],
                        float m_st[TQ][GQA], float l_st[TQ][GQA], float Oacc[TQ][GQA][HD],
                        int rows, int qt0, int pos_base, int n_kt, int last_pos,
                        int skip_compute) {
    bf16_t k_tile[TKV][HD], v_tile[TKV][HD];
#pragma HLS ARRAY_PARTITION variable=k_tile cyclic factor=LANES dim=2
#pragma HLS ARRAY_PARTITION variable=v_tile cyclic factor=LANES dim=2
CONS_KT:
    for (int kt = 0; kt < n_kt; ++kt) {
        load_tile(ksf, vsf, k_tile, v_tile, kt, last_pos);
        compute_tile(k_tile, v_tile, q_tile, m_st, l_st, Oacc, kt,
                     rows, qt0, pos_base, last_pos, skip_compute);
    }
}

// ---------- Phase 1:单 KV 头 × 单 Q tile 的核心计算 ----------
static void attend_tile(const word_t *qin, word_t *out, const fword_t *sincos,
                        hls::stream<word_t> &k_in, hls::stream<word_t> &v_in,
                        int kvh, int qt0, int rows, int pos_base, int rope_en,
                        int skip_compute) {
    // 非 static:多簇并行时每个簇实例各自一份状态(static 会被并行实例共享 → 冲突)
    bf16_t q_tile[TQ][GQA][HD];
#pragma HLS ARRAY_PARTITION variable=q_tile cyclic factor=LANES dim=3
    float Oacc[TQ][GQA][HD];
#pragma HLS ARRAY_PARTITION variable=Oacc cyclic factor=LANES dim=3
#pragma HLS ARRAY_PARTITION variable=Oacc complete dim=2
    float m_st[TQ][GQA], l_st[TQ][GQA];

    float cs[HD2], sn[HD2];
#pragma HLS ARRAY_PARTITION variable=cs cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=sn cyclic factor=4
    bf16_t raw[HD], rot[HD];
#pragma HLS ARRAY_PARTITION variable=raw cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=rot cyclic factor=8

    // -- Q tile 装载 + RoPE + 状态初始化 --
QLOAD_R:
    for (int r = 0; r < rows; ++r) {
        load_sincos(sincos, pos_base + qt0 + r, cs, sn);
QLOAD_G:
        for (int g = 0; g < GQA; ++g) {
            const int hq = kvh * GQA + g;
            load_row(qin, ((qt0 + r) * HQ + hq) * HDW, raw);
            rope_rotate(raw, cs, sn, rot, rope_en);
            for (int d = 0; d < HD; ++d) {
#pragma HLS PIPELINE II=1
                q_tile[r][g][d] = rot[d];
                Oacc[r][g][d] = 0.0f;
            }
            m_st[r][g] = NEG_BIG;
            l_st[r][g] = 0.0f;
        }
    }

    const int last_pos = pos_base + qt0 + rows - 1;   // tile 内最大可见位置
    const int n_kt = last_pos / TKV + 1;

    // KV 从 AXIS 流(DMA 喂)逐 tile 弹出 + 在线 softmax
    kv_consumer(k_in, v_in, q_tile, m_st, l_st, Oacc, rows, qt0, pos_base, n_kt, last_pos, skip_compute);

    if (skip_compute) return;   // profiling:跳过归一化写回

    // -- 归一化 + 写回 --
WB_R:
    for (int r = 0; r < rows; ++r) {
WB_G:
        for (int g = 0; g < GQA; ++g) {
            const float inv_l = 1.0f / l_st[r][g];     // 因果含自身 → l > 0 恒成立
            bf16_t orow[HD];
#pragma HLS ARRAY_PARTITION variable=orow cyclic factor=8
            for (int d = 0; d < HD; ++d) {
#pragma HLS PIPELINE II=1
                orow[d] = f32_to_bf16(Oacc[r][g][d] * inv_l);
            }
            store_row(out, ((qt0 + r) * HQ + kvh * GQA + g) * HDW, orow);
        }
    }
}

// ---------- 多簇:一个簇串行处理 HPC 个 KV 头(N_CLUSTER 个簇之间并行,各有独立 AXIS 流)----------
static void attend_cluster(const word_t *qin, word_t *out, const fword_t *sincos,
                           hls::stream<word_t> &k_in, hls::stream<word_t> &v_in,
                           int cluster, int q_len, int pos_base, int rope_en, int skip_compute) {
#pragma HLS INLINE OFF      // 各簇保持独立实例:DATAFLOW 才能并发(否则共享 attend_tile → 串行)
CL_H:
    for (int hl = 0; hl < HPC; ++hl) {
        const int kvh = cluster * HPC + hl;
QT_LOOP:
        for (int qt0 = 0; qt0 < q_len; qt0 += TQ) {
            const int rows = ((q_len - qt0) < TQ) ? (q_len - qt0) : TQ;
            attend_tile(qin, out, sincos, k_in, v_in, kvh, qt0, rows, pos_base, rope_en, skip_compute);
        }
    }
}

// ---------- 多簇并行外壳:DATAFLOW 在函数体顶部 + 显式列出各簇调用(canonical 形式)----------
// 关键:DATAFLOW 多进程不能共享同一 m_axi bundle → 每簇用各自 bundle 的 q/out/sincos 指针
static_assert(N_CLUSTER <= 2, "本设计 HP 口仅够 N_CLUSTER<=2(4 DMA + 2 簇 gmem)");
static void attend_all(const word_t *qin0, word_t *out0, const fword_t *sincos0,
#if CFG_NCLUSTER >= 2
                       const word_t *qin1, word_t *out1, const fword_t *sincos1,
#endif
                       hls::stream<word_t> k_in[N_CLUSTER], hls::stream<word_t> v_in[N_CLUSTER],
                       int q_len, int pos_base, int rope_en, int skip_compute) {
#pragma HLS DATAFLOW
    attend_cluster(qin0, out0, sincos0, k_in[0], v_in[0], 0, q_len, pos_base, rope_en, skip_compute);
#if CFG_NCLUSTER >= 2
    attend_cluster(qin1, out1, sincos1, k_in[1], v_in[1], 1, q_len, pos_base, rope_en, skip_compute);
#endif
}

// ---------- 顶层 ----------
// gmem0:append(kcache/vcache/kvin)+ 簇0(qin0/out0/sincos0);gmem1:簇1(qin1/out1/sincos1)
// append(mode=1)与 attend(mode=2)是不同调用 → gmem0 时序上不会被两进程同时访问
void attn_top(word_t *kcache, word_t *vcache, word_t *kvin,
              const word_t *qin0, word_t *out0, const fword_t *sincos0,
#if CFG_NCLUSTER >= 2
              const word_t *qin1, word_t *out1, const fword_t *sincos1,
#endif
              hls::stream<word_t> k_in[N_CLUSTER], hls::stream<word_t> v_in[N_CLUSTER],
              int q_len, int pos_base, int rope_en, int mode) {
#pragma HLS INTERFACE m_axi port=kcache  offset=slave bundle=gmem0 depth=1048576
#pragma HLS INTERFACE m_axi port=vcache  offset=slave bundle=gmem0 depth=1048576
#pragma HLS INTERFACE m_axi port=kvin    offset=slave bundle=gmem0 depth=65536
#pragma HLS INTERFACE m_axi port=qin0    offset=slave bundle=gmem0 depth=131072
#pragma HLS INTERFACE m_axi port=out0    offset=slave bundle=gmem0 depth=131072
#pragma HLS INTERFACE m_axi port=sincos0 offset=slave bundle=gmem0 depth=262144
#if CFG_NCLUSTER >= 2
#pragma HLS INTERFACE m_axi port=qin1    offset=slave bundle=gmem1 depth=131072
#pragma HLS INTERFACE m_axi port=out1    offset=slave bundle=gmem1 depth=131072
#pragma HLS INTERFACE m_axi port=sincos1 offset=slave bundle=gmem1 depth=262144
#endif
#pragma HLS INTERFACE axis port=k_in
#pragma HLS INTERFACE axis port=v_in
#pragma HLS INTERFACE s_axilite port=q_len
#pragma HLS INTERFACE s_axilite port=pos_base
#pragma HLS INTERFACE s_axilite port=rope_en
#pragma HLS INTERFACE s_axilite port=mode
#pragma HLS INTERFACE s_axilite port=return

    const int do_append = mode & 1;
    const int do_attend = (mode >> 1) & 1;
    const int skip_compute = (mode >> 2) & 1;

    // Phase 0:本批 token 的 K/V 旋转并入 cache(decode 的关键步骤)
    if (do_append) append_kv(kcache, vcache, kvin, sincos0, q_len, pos_base, rope_en);

    // Phase 1:N_CLUSTER 个簇并行,每簇处理 HKV/N_CLUSTER 个 KV 头(DATAFLOW 并发实例)
    if (do_attend) attend_all(qin0, out0, sincos0,
#if CFG_NCLUSTER >= 2
                              qin1, out1, sincos1,
#endif
                              k_in, v_in, q_len, pos_base, rope_en, skip_compute);
}
