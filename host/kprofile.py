"""profile.py — mode 差分计时,分解 decode kernel 时间(定位访存 vs 算力 vs 停顿)
mode: bit0=append, bit1=attend, bit2=attend内跳过算力(仅访存/dataflow)
  T_full=3, T_append=1, T_attend=2, T_loadonly=6  →  T_compute ≈ T_attend - T_loadonly
"""
import numpy as np, sys
from driver import AttnAccel
from golden import HQ, HKV, HD
from pynq import Clocks

S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
acc = AttnAccel("attn_top.bit", max_seq=S + 64, max_q=512)
Clocks.fclk0_mhz = float(__import__("os").environ.get("FPT_CLK", "250"))
print("PL clock = %.1f MHz, S = %d" % (Clocks.fclk0_mhz, S), flush=True)

rng = np.random.default_rng(0)
filled = 0
while filled < S:                                   # 用 prefill 灌满 cache 到 S
    n = min(512, S - filled)
    acc.prefill(rng.standard_normal((n, HQ, HD), np.float32),
                rng.standard_normal((n, HKV, HD), np.float32),
                rng.standard_normal((n, HKV, HD), np.float32))
    filled += n
q1 = rng.standard_normal((HQ, HD), np.float32)
k1 = rng.standard_normal((HKV, HD), np.float32)
v1 = rng.standard_normal((HKV, HD), np.float32)
acc.decode_step(q1, k1, v1)                          # 预热:载入 qb/kvb,写 cache[S]
acc.pos = S
pos = acc.pos
print("cache filled, pos=%d. 计时 kernel-only(不含 sync),iters=20:" % pos, flush=True)

res = {}
for name, m in [("append (1)", 1), ("attend (2)", 2), ("attend-loadonly (6)", 6)]:
    t = acc.time_kernel(1, pos, m, iters=20)   # mode2/6 内部会喂 DMA
    res[m] = t
    print("  mode %-22s = %8.3f ms" % (name, t * 1e3), flush=True)

t_app, t_att, t_load = res[1], res[2], res[6]
print("  full = append+attend         = %8.3f ms" % ((t_app + t_att) * 1e3), flush=True)
t_comp = t_att - t_load
print("\n=== 分解(decode, S=%d)===" % S)
print("  append (Phase0)        : %8.3f ms" % (t_app * 1e3))
print("  attend 访存/dataflow    : %8.3f ms  (%.0f%% of attend)" % (t_load * 1e3, 100 * t_load / t_att))
print("  attend 算力(QK+sm+PV)  : %8.3f ms  (%.0f%% of attend)" % (t_comp * 1e3, 100 * t_comp / t_att))
print("  => 瓶颈: %s" % ("算力受限" if t_comp > t_load else "访存/停顿受限"))
print("  per-token: 访存 %.2f us, 算力 %.2f us (8 KV heads 合计)" % (t_load / S * 1e6, t_comp / S * 1e6))
