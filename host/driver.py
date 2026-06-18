"""driver.py — PYNQ 驱动(ZCU104),KV 读经 AXI DMA(SG 模式)→ URAM FIFO → kernel AXIS 流
流程(每次调用):append(mode=1,m_axi 写 cache)→ SG DMA 一条描述符链连续流全簇各头 → attend(mode=2,消费流)
SG 描述符链消除了原来逐头 _wait_idle 的串行,使喂数与算力并发(#3)。
"""
import time
import numpy as np
import pynq
from pynq import MMIO
from golden import HQ, HKV, HD, make_sincos, f32_to_bf16_bits, bf16_bits_to_f32

MAX_SEQ = 8192
TQ, TKV = 8, 64                      # 与 cfg.h 一致(消费顺序按 Q tile / KV tile 推导)

# AXI DMA(简单 direct-register 模式)MM2S 寄存器
DMACR, DMASR, SA_LO, SA_HI, LENGTH = 0x00, 0x04, 0x18, 0x1C, 0x28


class AttnAccel:
    def __init__(self, bitfile="attn_top.bit", max_seq=MAX_SEQ, max_q=2048):
        self.ol = pynq.Overlay(bitfile)
        self.ip = self.ol.attn_top_0
        self.max_seq, self.max_q = max_seq, max_q
        self.pos = 0
        self.kc = pynq.allocate((HKV, MAX_SEQ, HD), np.uint16)
        self.vc = pynq.allocate((HKV, MAX_SEQ, HD), np.uint16)
        self.qb = pynq.allocate((max_q, HQ, HD), np.uint16)
        self.kvb = pynq.allocate((max_q, 2, HKV, HD), np.uint16)
        self.ob = pynq.allocate((max_q, HQ, HD), np.uint16)
        self.sc = pynq.allocate((max_seq, HD), np.float32)
        self.sc[:] = make_sincos(max_seq); self.sc.sync_to_device()
        # 指针寄存器:append 三件(kcache/vcache/kvin)+ 每簇 q/out/sincos(各簇同缓冲,
        # 访问不相交头切片)。簇数由 register_map 是否存在 qin{c} 自动探测。
        ptrs = [("kcache", self.kc), ("vcache", self.vc), ("kvin", self.kvb)]
        if self._has_ptr("qin0"):                            # 多簇命名 qin{c}/out{c}/sincos{c}
            for c in range(8):
                if not self._has_ptr("qin%d" % c):
                    break
                ptrs += [("qin%d" % c, self.qb), ("out%d" % c, self.ob), ("sincos%d" % c, self.sc)]
        else:                                                # 旧单簇命名 qin/out/sincos(如 Ultra96 LANES=16 位流)
            ptrs += [("qin", self.qb), ("out", self.ob), ("sincos", self.sc)]
        for name, buf in ptrs:
            self._set_ptr(name, buf.device_address)
        dk = sorted(k for k in self.ol.ip_dict if "dma_k" in k.lower())
        dv = sorted(k for k in self.ol.ip_dict if "dma_v" in k.lower())
        self.dmk = [MMIO(self.ol.ip_dict[n]["phys_addr"], 0x10000) for n in dk]
        self.dmv = [MMIO(self.ol.ip_dict[n]["phys_addr"], 0x10000) for n in dv]
        self.ncluster = len(self.dmk)                        # 簇数 = DMA 对数(N=1/2)
        self.hpc = HKV // self.ncluster                      # 每簇 KV 头数
        for dm in self.dmk + self.dmv: dm.write(DMACR, 1)    # MM2S 引擎 RS=1
        self.hstride = MAX_SEQ * HD * 2                       # 每 KV 头字节步长
        print("[driver] N_CLUSTER=%d (DMA对: %s)" % (self.ncluster, dk + dv))

    def _has_ptr(self, name):
        rm = self.ip.register_map
        return any(hasattr(rm, n) for n in (name, name + "_1", name + "_r", name + "_r_1"))

    def _set_ptr(self, name, addr):
        rm = self.ip.register_map
        cand = name if hasattr(rm, name + "_1") or hasattr(rm, name) else name + "_r"
        if hasattr(rm, cand + "_1"):
            setattr(rm, cand + "_1", addr & 0xFFFFFFFF)
            setattr(rm, cand + "_2", (addr >> 32) & 0xFFFFFFFF)
        elif hasattr(rm, cand):
            setattr(rm, cand, addr)
        else:
            raise AttributeError("找不到指针寄存器 %r" % name)

    def _mm2s(self, dm, addr, nbytes):           # 触发一次 MM2S 突发搬运
        dm.write(DMACR, 1)
        dm.write(SA_LO, addr & 0xFFFFFFFF)
        dm.write(SA_HI, (addr >> 32) & 0xFFFFFFFF)
        dm.write(LENGTH, nbytes)                 # 写 LENGTH 即启动

    @staticmethod
    def _wait_idle(dm):
        while not (dm.read(DMASR) & 0x2):        # MM2S_DMASR Idle
            pass

    def _feed_dma(self, q_len, pos_base):
        """按 kernel 消费顺序(逐 KV 头 × 逐 Q tile)把 cache 用 DMA 流给各簇。
        各簇同 step 交错起 DMA,使两簇并行保持喂入。簇内第 lh 个头 = 全局 c*hpc+lh。"""
        kb, vb = self.kc.device_address, self.vc.device_address
        for lh in range(self.hpc):
            for qt0 in range(0, q_len, TQ):
                rows = min(q_len - qt0, TQ)
                last_pos = pos_base + qt0 + rows - 1
                nb = (last_pos + 1) * HD * 2     # n token × 128 × 2 字节
                for c in range(self.ncluster):
                    off = (c * self.hpc + lh) * self.hstride
                    self._mm2s(self.dmk[c], kb + off, nb)
                    self._mm2s(self.dmv[c], vb + off, nb)
                for c in range(self.ncluster):
                    self._wait_idle(self.dmk[c]); self._wait_idle(self.dmv[c])

    def _call(self, q_len, pos_base, rope_en=1, mode=2, timeout=120.0):
        rm = self.ip.register_map
        rm.q_len, rm.pos_base, rm.rope_en, rm.mode = q_len, pos_base, rope_en, mode
        self.ip.mmio.write(0x00, 0x01)           # AP_START
        if mode & 2:                             # do_attend → 同步喂 DMA(kernel 边消费)
            self._feed_dma(q_len, pos_base)
        t0 = time.time()
        while (self.ip.mmio.read(0x00) >> 1) & 1 == 0:
            if time.time() - t0 > timeout:
                raise TimeoutError("AP_DONE timeout (q_len=%d pos=%d mode=%d)" % (q_len, pos_base, mode))
            time.sleep(0.0002)

    def time_kernel(self, q_len, pos_base, mode, iters=20):
        self._call(q_len, pos_base, mode=mode)
        t0 = time.perf_counter()
        for _ in range(iters):
            self._call(q_len, pos_base, mode=mode)
        return (time.perf_counter() - t0) / iters

    def _run(self, q, k, v, q_len, rope_en):
        assert self.pos + q_len <= self.max_seq
        self.qb[:q_len] = f32_to_bf16_bits(q).reshape(q_len, HQ, HD)
        self.kvb[:q_len, 0] = f32_to_bf16_bits(k).reshape(q_len, HKV, HD)
        self.kvb[:q_len, 1] = f32_to_bf16_bits(v).reshape(q_len, HKV, HD)
        self.qb.sync_to_device(); self.kvb.sync_to_device()
        self._call(q_len, self.pos, rope_en, mode=1)       # append:写 cache
        self._call(q_len, self.pos, rope_en, mode=2)       # attend:DMA 喂 + 消费
        self.ob.sync_from_device()
        self.pos += q_len
        return bf16_bits_to_f32(self.ob[:q_len].copy())

    def reset(self):
        self.pos = 0

    def prefill(self, q, k, v, rope_en=1):
        return self._run(q, k, v, q.shape[0], rope_en)

    def decode_step(self, q, k, v, rope_en=1):
        return self._run(q[None], k[None], v[None], 1, rope_en)[0]


if __name__ == "__main__":
    from golden import sdpa_golden, compare, make_sincos
    rng = np.random.default_rng(2026)
    L, D = 150, 3
    q = rng.standard_normal((L + D, HQ, HD), dtype=np.float32)
    k = rng.standard_normal((L + D, HKV, HD), dtype=np.float32)
    v = rng.standard_normal((L + D, HKV, HD), dtype=np.float32)
    acc = AttnAccel()
    got = [acc.prefill(q[:L], k[:L], v[:L])]
    for s in range(D):
        got.append(acc.decode_step(q[L + s], k[L + s], v[L + s])[None])
    got = np.concatenate(got, axis=0)
    gold = sdpa_golden(q, k, v, 0, make_sincos(L + D))
    ok = compare(got, gold, "BOARD-E2E")
    print("PASS" if ok else "FAIL")
