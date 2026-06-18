# run_hls_ultra96.tcl — Vitis HLS for Ultra96v2 (XCZU3EG)
# csynth + IP export only (csim already validated on the ZCU104 flow).
# 缩小到 LANES=8 以塞进 XCZU3EG 的 70k LUT;part 为 Ultra96v2 工业级。
# 用法: vitis_hls -f scripts/run_hls_ultra96.tcl
open_project -reset attn_prj_u96
set_top attn_top
add_files hls/src/attn_top.cpp -cflags "-Ihls/include -std=c++14 -DCFG_LANES=16"
add_files -tb hls/tb/tb_attn.cpp -cflags "-Ihls/include -std=c++14 -DCFG_LANES=16"
open_solution -reset sol1
set_part xczu3eg-sbva484-1-i
create_clock -period 4.0 -name default   ;# 250 MHz(嵌入式小芯片保守起步,时序裕量大)
csynth_design
export_design -format ip_catalog -ipname attn_top -vendor fpt26 -version 1.0
exit
