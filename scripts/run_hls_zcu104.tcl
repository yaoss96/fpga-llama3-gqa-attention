# run_hls_zcu104.tcl — Vitis HLS for ZCU104 (XCZU7EV-2),性能最大化:LANES=32 @300MHz
# 用法: vitis_hls -f scripts/run_hls_zcu104.tcl
open_project -reset attn_prj_zcu104
set_top attn_top
add_files hls/src/attn_top.cpp -cflags "-Ihls/include -std=c++14 -DCFG_LANES=32 -DCFG_NCLUSTER=2"
add_files -tb hls/tb/tb_attn.cpp -cflags "-Ihls/include -std=c++14 -DCFG_LANES=32 -DCFG_NCLUSTER=2"
open_solution -reset sol1
set_part xczu7ev-ffvc1156-2-e
create_clock -period 3.333 -name default   ;# 300 MHz(xczu7ev-2 速度档,时序紧则改 4.0)
csim_design
csynth_design
export_design -format ip_catalog -ipname attn_top -vendor fpt26 -version 1.0
exit
