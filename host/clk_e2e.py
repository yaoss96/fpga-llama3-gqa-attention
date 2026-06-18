"""clk_e2e.py — 设置 PL 时钟到目标频率(PYNQ 2.7 默认只给 100MHz),再跑 153-token E2E 校验"""
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from pynq import Clocks
from driver import AttnAccel
from golden import sdpa_golden, compare, make_sincos, HQ, HKV, HD

target = float(sys.argv[1]) if len(sys.argv) > 1 else 250.0
acc = AttnAccel("attn_top.bit")          # 加载 overlay
print("fclk0 before:", Clocks.fclk0_mhz, flush=True)
Clocks.fclk0_mhz = target                # 提频
print("fclk0 after :", Clocks.fclk0_mhz, flush=True)

rng = np.random.default_rng(2026)
L, D = 150, 3
q = rng.standard_normal((L + D, HQ, HD), np.float32)
k = rng.standard_normal((L + D, HKV, HD), np.float32)
v = rng.standard_normal((L + D, HKV, HD), np.float32)
got = [acc.prefill(q[:L], k[:L], v[:L])]
for s in range(D):
    got.append(acc.decode_step(q[L + s], k[L + s], v[L + s])[None])
got = np.concatenate(got, axis=0)
gold = sdpa_golden(q, k, v, 0, make_sincos(L + D))
ok = compare(got, gold, "E2E@%gMHz" % Clocks.fclk0_mhz)
print("RESULT:", "PASS" if ok else "FAIL", flush=True)
