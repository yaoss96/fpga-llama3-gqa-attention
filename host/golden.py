"""golden.py — numpy 黄金模型与 bf16/sincos 工具(PC 与板上通用)
布局与 HLS kernel 严格一致;RoPE 为 HF Llama 半分裂约定,theta=500000。
依赖: pip install numpy ml_dtypes
"""
import numpy as np

HQ, HKV, GQA, HD, HD2 = 32, 8, 4, 128, 64
ROPE_THETA = 500000.0
SCALE = 1.0 / np.sqrt(HD)

try:                      # 优先用 ml_dtypes(PYNQ 3.0+);否则纯 numpy 回退(PYNQ 2.7 离线板)
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16

    def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
        """fp32 -> bf16 位模式(uint16, RNE,与硬件一致)"""
        return x.astype(np.float32).astype(BF16).view(np.uint16)

    def bf16_bits_to_f32(x: np.ndarray) -> np.ndarray:
        return x.view(BF16).astype(np.float32)

except ImportError:       # 纯 numpy RNE(round-half-to-even),与硬件 bf16.h 同口径
    def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
        return ((u + bias) >> np.uint32(16)).astype(np.uint16)

    def bf16_bits_to_f32(x: np.ndarray) -> np.ndarray:
        u = (np.ascontiguousarray(x, dtype=np.uint16).astype(np.uint32) << np.uint32(16))
        return u.view(np.float32)


def make_sincos(max_pos: int) -> np.ndarray:
    """[max_pos, 128] float32:每行前 64 cos、后 64 sin(与 DDR 布局一致)"""
    inv_freq = ROPE_THETA ** (-np.arange(0, HD2, dtype=np.float64) * 2 / HD)
    ang = np.arange(max_pos, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)


def rope_half(x: np.ndarray, sincos_rows: np.ndarray) -> np.ndarray:
    """x: [..., HD] float32;sincos_rows: [..., 128](cos|sin)。半分裂旋转。"""
    cs, sn = sincos_rows[..., :HD2], sincos_rows[..., HD2:]
    x1, x2 = x[..., :HD2], x[..., HD2:]
    return np.concatenate([x1 * cs - x2 * sn, x2 * cs + x1 * sn], axis=-1)


def sdpa_golden(q, k, v, pos_base, sincos, quantize_inputs=True):
    """逐位置因果 GQA-SDPA 参考(fp32 计算)。
    q: [L, HQ, HD] / k,v: [T, HKV, HD](T = pos_base+L,含历史,未旋转)
    返回 [L, HQ, HD] float32
    """
    L = q.shape[0]
    T = pos_base + L
    if quantize_inputs:  # 模拟 bf16 输入量化
        q = bf16_bits_to_f32(f32_to_bf16_bits(q))
        k = bf16_bits_to_f32(f32_to_bf16_bits(k))
        v = bf16_bits_to_f32(f32_to_bf16_bits(v))
    k_rot = rope_half(k[:T], sincos[:T, None, :])           # [T, HKV, HD]
    out = np.zeros((L, HQ, HD), np.float32)
    for r in range(L):
        P = pos_base + r
        q_rot = rope_half(q[r], sincos[P][None, :])         # [HQ, HD]
        for hq in range(HQ):
            kvh = hq // GQA
            s = k_rot[: P + 1, kvh] @ q_rot[hq] * SCALE     # [P+1]
            p = np.exp(s - s.max()); p /= p.sum()
            out[r, hq] = p @ v[: P + 1, kvh]
    return out


def compare(got_f32, gold_f32, name=""):
    """按行 RMS 归一化误差 + 余弦相似度(与 C 测试台同口径)"""
    g, o = gold_f32.reshape(-1, HD), got_f32.reshape(-1, HD)
    rms = np.sqrt((g ** 2).mean(axis=1, keepdims=True))
    err = np.abs(o - g) / rms
    cos = (g * o).sum(1) / np.sqrt((g ** 2).sum(1) * (o ** 2).sum(1))
    print(f"[{name}] max_err/rms={err.max():.4f}  mean={err.mean():.5f}  min_cos={cos.min():.6f}")
    return err.max() < 0.05 and cos.min() > 0.999
