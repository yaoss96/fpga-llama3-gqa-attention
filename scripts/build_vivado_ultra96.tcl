# build_vivado_ultra96.tcl — Vivado 块设计:Ultra96v2(XCZU3EG)+ attn_top IP → PYNQ 产物
# 前置: 先跑 run_hls_ultra96.tcl 导出 IP(attn_prj_u96/sol1/impl/ip)
# 用法(分两次调用以重置 Tcl slave 解释器计数,避开 zynq_ps SystemC 生成的 slaveNN 上限):
#   vivado -mode batch -source scripts/build_vivado_ultra96.tcl -tclargs bd
#   vivado -mode batch -source scripts/build_vivado_ultra96.tcl -tclargs impl
# (不带参数 = all = 单次跑完;IP 子模块多时可能触发 slave 上限,故默认分两段)
set part   xczu3eg-sbva484-1-i
set clkmhz 250                      ;# 与 HLS 4.0ns(250MHz)一致;小芯片保守
set stage  "all"
if {[llength $argv] > 0} { set stage [lindex $argv 0] }
set proj_xpr ./attn_u96/attn_u96.xpr

if {$stage eq "impl"} {
    open_project $proj_xpr
} else {
create_project attn_u96 ./attn_u96 -part $part -force

# --- Ultra96v2 板级文件(Avnet-tria);动态取最新版本号 ---
set bp [lindex [get_board_parts -quiet -latest_file_version *ultra96v2*] 0]
if {$bp eq ""} { error "未找到 Ultra96v2 板级文件,请确认 board_files 已安装" }
puts "INFO: using board_part = $bp"
set_property board_part $bp [current_project]

set_property ip_repo_paths ./attn_prj_u96/sol1/impl/ip [current_project]
update_ip_catalog

create_bd_design "design_1"

# --- PS:套用 Ultra96v2 板级预设(LPDDR4/MIO),再开 4 个 HP 从口 + HPM0 主口 + PL 时钟 ---
set ps_vlnv [lindex [lsort [get_ipdefs -all xilinx.com:ip:zynq_ultra_ps_e:*]] end]
create_bd_cell -type ip -vlnv $ps_vlnv zynq_ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} [get_bd_cells zynq_ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clkmhz \
] [get_bd_cells zynq_ps]
# 说明: S_AXI_GP2..GP5 即 HP0..HP3_FPD

# --- 加速器 IP(动态取 vlnv,回退到固定名) ---
set acc_vlnv [lindex [get_ipdefs -quiet -all *:*:attn_top:*] 0]
if {$acc_vlnv eq ""} { set acc_vlnv fpt26:hls:attn_top:1.0 }
create_bd_cell -type ip -vlnv $acc_vlnv attn_top_0

# --- 两个 AXI DMA:MM2S 从 DDR 突发读 KV cache → 流进 attn_top k_in/v_in(DMA 保证突发)---
set dma_vlnv [lindex [lsort [get_ipdefs -all xilinx.com:ip:axi_dma:*]] end]
foreach d {dma_k dma_v} {
    create_bd_cell -type ip -vlnv $dma_vlnv $d
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_include_mm2s {1} CONFIG.c_include_s2mm {0} \
        CONFIG.c_m_axi_mm2s_data_width {128} \
        CONFIG.c_m_axis_mm2s_tdata_width {128} \
        CONFIG.c_mm2s_burst_size {256} \
        CONFIG.c_sg_length_width {26} \
    ] [get_bd_cells $d]
}
connect_bd_intf_net [get_bd_intf_pins dma_k/M_AXIS_MM2S] [get_bd_intf_pins attn_top_0/k_in]
connect_bd_intf_net [get_bd_intf_pins dma_v/M_AXIS_MM2S] [get_bd_intf_pins attn_top_0/v_in]

# --- HP 口分配:HP0=dma_k 读, HP1=dma_v 读, HP2=kernel gmem2, HP3=kernel gmem3 ---
foreach {master hp} {dma_k/M_AXI_MM2S S_AXI_HP0_FPD dma_v/M_AXI_MM2S S_AXI_HP1_FPD \
                     attn_top_0/m_axi_gmem2 S_AXI_HP2_FPD attn_top_0/m_axi_gmem3 S_AXI_HP3_FPD} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        [list Master "/$master" Slave "/zynq_ps/$hp" intc_ip "Auto" master_apm "0"] \
        [get_bd_intf_pins zynq_ps/$hp]
}
# 控制口 HPM0 → kernel s_axilite + 两个 DMA S_AXI_LITE
foreach s {attn_top_0/s_axi_control dma_k/S_AXI_LITE dma_v/S_AXI_LITE} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        [list Master "/zynq_ps/M_AXI_HPM0_FPD" Slave "/$s" intc_ip "Auto"] \
        [get_bd_intf_pins $s]
}

# --- APM:监控 dma_k/dma_v 的 MM2S 读(验证 DMA 是否真发长突发)---
set apm_vlnv [lindex [lsort [get_ipdefs -all xilinx.com:ip:axi_perf_mon:*]] end]
create_bd_cell -type ip -vlnv $apm_vlnv apm0
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {2} CONFIG.C_ENABLE_EVENT_COUNT {1} \
    CONFIG.C_ENABLE_EVENT_LOG {0} CONFIG.C_SHOW_AXI_IDS {0} CONFIG.C_SHOW_AXI_LEN {1}] [get_bd_cells apm0]
connect_bd_intf_net [get_bd_intf_pins apm0/SLOT_0_AXI] [get_bd_intf_pins dma_k/M_AXI_MM2S]
connect_bd_intf_net [get_bd_intf_pins apm0/SLOT_1_AXI] [get_bd_intf_pins dma_v/M_AXI_MM2S]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    [list Master "/zynq_ps/M_AXI_HPM0_FPD" Slave "/apm0/S_AXI" intc_ip "Auto"] \
    [get_bd_intf_pins apm0/S_AXI]
set apm_clk  [get_bd_pins zynq_ps/pl_clk0]
set rstcell  [lindex [get_bd_cells -quiet -hier -filter {VLNV =~ xilinx.com:ip:proc_sys_reset:*}] 0]
set apm_rstn [get_bd_pins $rstcell/peripheral_aresetn]
foreach p {core_aclk slot_0_axi_aclk slot_1_axi_aclk} { connect_bd_net $apm_clk [get_bd_pins apm0/$p] }
foreach p {core_aresetn slot_0_axi_aresetn slot_1_axi_aresetn} { connect_bd_net $apm_rstn [get_bd_pins apm0/$p] }

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# 全局综合 + 只生成综合产物:避开逐IP OOC 触发的 SystemC 仿真 wrapper(ttcl)生成失败
set_property synth_checkpoint_mode None [get_files design_1.bd]
generate_target synthesis [get_files design_1.bd]

make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [make_wrapper -files [get_files design_1.bd] -top -fileset sources_1]
set_property top design_1_wrapper [current_fileset]
}  ;# end BD-creation block(stage bd/all)

if {$stage eq "bd"} { puts "===== BD STAGE DONE ====="; exit }

# --- 先跑综合,打印真实(post-synth)利用率作为早期 gate ---
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synth_1 失败,见日志" }
open_run synth_1 -name synth_1
report_utilization -file ./attn_u96/post_synth_util.rpt
puts "===== POST-SYNTH UTILIZATION (真实综合数,非 HLS 预估) ====="
puts [report_utilization -return_string]
puts "===== END POST-SYNTH ====="

# --- 实现 + 出比特流 ---
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "impl_1 未完成(可能超资源/时序),请查看 Vivado 日志与 post_synth_util.rpt"
}
open_run impl_1
puts "===== POST-ROUTE UTILIZATION (板上实际占用) ====="
puts [report_utilization -return_string]
puts "===== END POST-ROUTE ====="
report_utilization   -file ./attn_u96/post_route_util.rpt
report_timing_summary -file ./attn_u96/post_route_timing.rpt -max_paths 5
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "WNS(setup worst slack) = $wns ns"

# --- 收集 PYNQ 产物:attn_top.bit + attn_top.hwh ---
set bitsrc [glob -nocomplain ./attn_u96/attn_u96.runs/impl_1/design_1_wrapper.bit]
set hwhsrc [glob -nocomplain ./attn_u96/attn_u96.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh]
if {$bitsrc eq ""} { set bitsrc [glob ./attn_u96/attn_u96.runs/impl_1/*.bit] }
if {$hwhsrc eq ""} { set hwhsrc [glob ./attn_u96/attn_u96.gen/sources_1/bd/design_1/hw_handoff/*.hwh] }
file copy -force $bitsrc ./attn_top.bit
file copy -force $hwhsrc ./attn_top.hwh
puts "DONE: attn_top.bit / attn_top.hwh 已生成 → 拷到板上与 python/ 同目录"
