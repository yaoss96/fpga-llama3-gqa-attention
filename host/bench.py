"""bench.py — 在板性能测量(论文 profiling 数据源)
输出:decode 延迟-序列长度曲线、有效带宽、prefill 延迟/GFLOPs、功耗(PMBus)
"""
import time
import numpy as np
from driver import AttnAccel
from golden import HQ, HKV, HD

KV_BYTES_PER_TOK = 2 * HKV * HD * 2          # 4096 B(K+V, bf16)


def power_watts():
    try:
        import pynq.pmbus as pmbus
        rails = pmbus.get_rails()
        return sum(r.power.value for r in rails.values() if hasattr(r, "power"))
    except Exception:
        return float("nan")


def bench_decode(acc, seq_points=(512, 1024, 2048, 4096, 8192), iters=20):
    rng = np.random.default_rng(0)
    print(f"{'S':>6} {'ms/tok':>9} {'GB/s有效':>9} {'BW利用%':>8} {'W':>6}")
    for S in seq_points:
        acc.reset()
        # 预填 cache 至 S-1(用 prefill 分批灌入,内容随机)
        filled = 0
        while filled < S - 1:
            n = min(512, S - 1 - filled)
            acc.prefill(rng.standard_normal((n, HQ, HD), dtype=np.float32),
                        rng.standard_normal((n, HKV, HD), dtype=np.float32),
                        rng.standard_normal((n, HKV, HD), dtype=np.float32))
            filled += n
        q1 = rng.standard_normal((HQ, HD), dtype=np.float32)
        k1 = rng.standard_normal((HKV, HD), dtype=np.float32)
        v1 = rng.standard_normal((HKV, HD), dtype=np.float32)
        acc.decode_step(q1, k1, v1)                       # 预热
        acc.pos -= 1
        t0 = time.perf_counter()
        for _ in range(iters):
            acc.decode_step(q1, k1, v1)
            acc.pos -= 1                                  # 固定 S 重复测
        dt = (time.perf_counter() - t0) / iters
        gbs = S * KV_BYTES_PER_TOK / dt / 1e9
        print(f"{S:>6} {dt*1e3:>9.3f} {gbs:>9.2f} {gbs/17.0*100:>7.1f}% {power_watts():>6.2f}")


def bench_prefill(acc, lengths=(256, 512, 1024, 2048), iters=3):
    rng = np.random.default_rng(1)
    print(f"{'L':>6} {'ms':>10} {'GFLOP/s':>9}")
    for L in lengths:
        q = rng.standard_normal((L, HQ, HD), dtype=np.float32)
        k = rng.standard_normal((L, HKV, HD), dtype=np.float32)
        v = rng.standard_normal((L, HKV, HD), dtype=np.float32)
        acc.reset(); acc.prefill(q, k, v)                 # 预热
        t = 0.0
        for _ in range(iters):
            acc.reset()
            t0 = time.perf_counter(); acc.prefill(q, k, v); t += time.perf_counter() - t0
        dt = t / iters
        flops = 2 * 4096 * L * L                          # causal 两个 matmul
        print(f"{L:>6} {dt*1e3:>10.2f} {flops/dt/1e9:>9.1f}")


if __name__ == "__main__":
    acc = AttnAccel()
    try:                                   # PYNQ2.7 默认 PL 时钟仅 100MHz,提到设计目标 250MHz
        from pynq import Clocks
        Clocks.fclk0_mhz = float(__import__("os").environ.get("FPT_CLK", "250"))
        print("PL clock = %.1f MHz" % Clocks.fclk0_mhz)
    except Exception as e:
        print("clock set skipped:", e)
    print("== Decode(内存受限:看带宽利用率)=="); bench_decode(acc)
    print("== Prefill(计算受限:看 GFLOP/s)=="); bench_prefill(acc)
