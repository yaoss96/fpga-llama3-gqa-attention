"""apm.py — APM 扫描 slot0(K 读口)各 metric,定位 decode 访存瓶颈
单 metric counter(MC0),逐 metric 重测。metric(xaxipmon):
  1=读事务数 3=读字节数 5=读总延迟 11=读突发完成数(RLAST) 14=最小读延迟 15=最大读延迟
判定:bytes/txn = 读字节/读事务 → ≥256B 在突发;avg_lat = 读总延迟/读事务(周期)
"""
import sys, time, numpy as np
from driver import AttnAccel
from golden import HQ, HKV, HD
from pynq import Clocks

S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
MODE = int(sys.argv[2]) if len(sys.argv) > 2 else 6
acc = AttnAccel("attn_top.bit", max_seq=S + 64, max_q=512)
Clocks.fclk0_mhz = float(__import__("os").environ.get("FPT_CLK", "250"))
ol = acc.ol
apm = getattr(ol, [k for k in ol.ip_dict if "apm" in k.lower()][0])
mm = apm.mmio
CR, MSR0, MC0 = 0x300, 0x044, 0x100

rng = np.random.default_rng(0)
filled = 0
while filled < S:
    n = min(512, S - filled)
    acc.prefill(rng.standard_normal((n, HQ, HD), np.float32),
                rng.standard_normal((n, HKV, HD), np.float32),
                rng.standard_normal((n, HKV, HD), np.float32))
    filled += n
acc.decode_step(rng.standard_normal((HQ, HD), np.float32),
                rng.standard_normal((HKV, HD), np.float32),
                rng.standard_normal((HKV, HD), np.float32))
acc.pos = S
iters = 20
print("cache pos=%d, mode=%d, S=%d, iters=%d" % (acc.pos, MODE, S, iters), flush=True)

def measure(metric):                       # MC0 监控 slot0 的 metric,返回 (计数, 墙钟)
    mm.write(MSR0, (0 << 5) | metric)      # slot0
    mm.write(CR, 0x2); mm.write(CR, 0x0); mm.write(CR, 0x1)
    t0 = time.perf_counter()
    for _ in range(iters):
        acc._call(1, S, mode=MODE)
    dt = time.perf_counter() - t0
    mm.write(CR, 0x0)
    return mm.read(MC0), dt

labels = {1: "读事务数", 3: "读字节数", 5: "读总延迟(cyc)", 11: "读突发完成RLAST", 14: "最小读延迟", 15: "最大读延迟"}
res = {}
for m in (1, 3, 5, 11, 14, 15):
    v, dt = measure(m)
    res[m] = (v, dt)
    print("  metric %-2d %-16s = %d" % (m, labels[m], v), flush=True)

rb = res[3][0]; rt = res[1][0]; rl = res[5][0]; dt = res[3][1]
print("\n=== K 读口诊断 (mode=%d, S=%d) ===" % (MODE, S))
print("读字节 = %.1f MB,读事务 = %d,读突发(RLAST) = %d" % (rb / 1e6, rt, res[11][0]))
print("bytes/txn = %.1f B  -> %s" % (rb / max(rt, 1),
      "在长突发(瓶颈=DDR延迟/控制器)" if rb / max(rt, 1) >= 128 else "碎突发/单拍(kernel没发出长突发)"))
print("平均读延迟 = %.1f cyc (=%.1f ns @250MHz);min=%d max=%d" %
      (rl / max(rt, 1), rl / max(rt, 1) * 4, res[14][0], res[15][0]))
print("单口 K 读带宽 = %.3f GB/s" % (rb / dt / 1e9))
