# bf16 GQA Attention Accelerator for Llama3-8B on Embedded FPGA

An HLS accelerator for the **grouped-query attention (GQA)** operator of
**Llama3-8B** (RoPE + incrementally-built KV cache), targeting embedded AMD
FPGAs (**ZCU104 / XCZU7EV** and **Ultra96v2 / XCZU3EG**). All tensors are
**bf16** with fp32 internal accumulation; no AI Engine or hard tensor cores are
used, so the design optimises for resource efficiency.

The accelerator covers both inference regimes:

- **Prefill** — process `L` prompt tokens at once (compute-bound, GFLOP/s).
- **Decode** — one new token attending to `S` cached tokens (latency-critical,
  ms/token).

## Key ideas

1. **Burst-restored streaming memory path.** A direct read of the KV cache on
   this PS/DDR path collapses to single-beat 16-byte transfers. The cache is
   instead delivered through AXI4-Stream channels driven by AXI-DMA engines in
   the block design, restoring 256-beat bursts (~8.3× bytes/transaction). The
   kernel only writes new tokens back (append).
2. **GQA-aware arithmetic sharing.** Because one KV head serves four query
   heads, each key/value element is reused across a query head-pair on a single
   multiplier, halving the arithmetic of the score and output products.
3. **Streaming (flash) softmax.** Online softmax with running statistics so no
   full score row is ever materialised, keeping on-chip memory small.
4. **Multi-cluster head parallelism.** KV heads are split across independent
   compute clusters that run concurrently, each with its own path to memory;
   replicating clusters scales the design (near-linear).

## Results (measured on board, ZCU104 @ 300 MHz, `LANES=32`, `N_CLUSTER=2`)

| Decode context `S` | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| ms/token | 3.06 | 3.94 | 5.65 | **9.08** | **16.0** |

Prefill: 10.3 GFLOP/s at `L=2048`. Correctness vs an fp32 NumPy reference:
max relative/RMS error 0.0222, min cosine 0.99998. Post-route resources:
**LUT 90% / FF 47% / DSP 29% / BRAM 40% / URAM 0%**, timing met at 300 MHz.
Differential on-board profiling shows decode is **compute-bound** (compute 64%,
KV-load 31%, append 5%).

Two-board comparison (both measured): Ultra96v2 (`LANES=16`, `N=1`, 250 MHz)
reaches 21.9 ms/token decode at `S=4096`; ZCU104 is ~2.4× faster.

## Repository layout

```
hls/
  include/   bf16, dsp_pack, exp2_lut, rope, softmax, cfg (config knobs)
  src/       attn_top.cpp     — top-level kernel
  tb/        tb_attn.cpp      — C-sim testbench (feeds AXIS like the on-board DMA)
scripts/
  run_hls_zcu104.tcl          — HLS csim + csynth + export IP (xczu7ev, 300 MHz)
  build_vivado_zcu104.tcl     — block design + bitstream (PS + 4 AXI-DMA + 2 clusters)
  run_hls_ultra96.tcl         — Ultra96v2 variant (xczu3eg, 250 MHz)
  build_vivado_ultra96.tcl
host/                         — PYNQ host code (runs on the board)
  driver.py                   — overlay driver (DMA feed, append/attend)
  golden.py                   — fp32 reference + bf16 helpers
  bench.py                    — decode/prefill throughput sweep
  clk_e2e.py                  — set PL clock + 153-token correctness check
  kprofile.py                 — differential profiling (append/load/compute)
  apm.py                      — AXI Performance Monitor readout
```

Configuration knobs (`hls/include/cfg.h`, overridable on the HLS command line):
`CFG_LANES` (lanes), `CFG_NCLUSTER` (head-parallel clusters), plus model
parameters `HQ=32, HKV=8, HD=128, MAX_SEQ=8192`, RoPE `θ=500000`.

## Build

Requires **AMD Vitis HLS + Vivado 2024.2**.

```bash
# 1. HLS: C-sim (vs NumPy golden), C-synth, export IP
vitis_hls -f scripts/run_hls_zcu104.tcl

# 2. Bitstream (two stages: block design, then implementation)
vivado -mode batch -source scripts/build_vivado_zcu104.tcl -tclargs bd
vivado -mode batch -source scripts/build_vivado_zcu104.tcl -tclargs impl
# -> produces attn_top.bit and attn_top.hwh
```

## Run on the board (PYNQ)

Copy `attn_top.bit`, `attn_top.hwh` and `host/*.py` to the board (PYNQ 3.1 on
ZCU104), then, as root with the PYNQ venv Python:

```bash
python clk_e2e.py 300     # set PL clock to 300 MHz + correctness check  -> RESULT: PASS
FPT_CLK=300 python bench.py      # decode (ms/tok) + prefill (GFLOP/s) sweep
FPT_CLK=300 python kprofile.py   # differential profiling
```

Host requirements: `numpy`, `ml_dtypes`, `pynq`.

## License

MIT — see [LICENSE](LICENSE).
