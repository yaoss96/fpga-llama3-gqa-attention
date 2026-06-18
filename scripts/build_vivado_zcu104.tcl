# build_vivado_zcu104.tcl — Vivado BD:ZCU104(XCZU7EV-2)+ attn_top IP + 双DMA + APM → PYNQ 产物
# 无 ZCU104 板级文件 → part-only,手动配 PS(PS 已由板上 boot 配好,overlay 只需 HP 口/PL时钟/有效DDR地址映射)
# 用法(分两段,避开 slave 解释器上限):
#   vivado -mode batch -source scripts/build_vivado_zcu104.tcl -tclargs bd
#   vivado -mode batch -source scripts/build_vivado_zcu104.tcl -tclargs impl
set part     xczu7ev-ffvc1156-2-e
set clkmhz   300                  ;# k_tile/v_tile 绑 BRAM 后 LUT 降到 ~76%,300MHz 应可布通
set ncluster 2                    ;# 多簇:N 对 DMA(2N 个),与 -DCFG_NCLUSTER 一致
set stage    "all"
if {[llength $argv] > 0} { set stage [lindex $argv 0] }
set proj_xpr ./attn_zcu104/attn_zcu104.xpr

if {$stage eq "impl"} {
    open_project $proj_xpr
} else {
create_project attn_zcu104 ./attn_zcu104 -part $part -force
set_property ip_repo_paths ./attn_prj_zcu104/sol1/impl/ip [current_project]
update_ip_catalog
create_bd_design "design_1"

# --- PS:part-only 手动配(无板级预设);开 4 HP + HPM0 + PL 时钟,DDR 用 IP 默认(2GB) ---
set ps_vlnv [lindex [lsort [get_ipdefs -all xilinx.com:ip:zynq_ultra_ps_e:*]] end]
create_bd_cell -type ip -vlnv $ps_vlnv zynq_ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__USE__S_AXI_GP1 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} CONFIG.PSU__SAXIGP1__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clkmhz \
] [get_bd_cells zynq_ps]

# --- 加速器 IP ---
set acc_vlnv [lindex [get_ipdefs -quiet -all *:*:attn_top:*] 0]
if {$acc_vlnv eq ""} { set acc_vlnv fpt26:hls:attn_top:1.0 }
create_bd_cell -type ip -vlnv $acc_vlnv attn_top_0

# --- N 对 AXI DMA(简单模式):每簇 K/V 各一,MM2S 从 DDR 突发读 → 直连 kernel k_in_c/v_in_c ---
set dma_vlnv [lindex [lsort [get_ipdefs -all xilinx.com:ip:axi_dma:*]] end]
set dma_masters {}
for {set c 0} {$c < $ncluster} {incr c} {
    foreach kv {k v} {
        set d "dma_${kv}_${c}"
        create_bd_cell -type ip -vlnv $dma_vlnv $d
        set_property -dict [list \
            CONFIG.c_include_sg {0} CONFIG.c_include_mm2s {1} CONFIG.c_include_s2mm {0} \
            CONFIG.c_m_axi_mm2s_data_width {128} CONFIG.c_m_axis_mm2s_tdata_width {128} \
            CONFIG.c_mm2s_burst_size {256} CONFIG.c_sg_length_width {26} \
        ] [get_bd_cells $d]
        connect_bd_intf_net [get_bd_intf_pins $d/M_AXIS_MM2S] [get_bd_intf_pins attn_top_0/${kv}_in_${c}]
        lappend dma_masters $d/M_AXI_MM2S
    }
}

# --- HP 口:2N 个 DMA → HP0..HP3,kernel gmem0/gmem1(各簇 q/out/sincos)→ 相干口 HPC0/HPC1 ---
set hp_list {S_AXI_HP0_FPD S_AXI_HP1_FPD S_AXI_HP2_FPD S_AXI_HP3_FPD}
set i 0
foreach m $dma_masters {
    set hp [lindex $hp_list $i]; incr i
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        [list Master "/$m" Slave "/zynq_ps/$hp" intc_ip "Auto" master_apm "0"] \
        [get_bd_intf_pins zynq_ps/$hp]
}
foreach {master hp} {attn_top_0/m_axi_gmem0 S_AXI_HPC0_FPD attn_top_0/m_axi_gmem1 S_AXI_HPC1_FPD} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        [list Master "/$master" Slave "/zynq_ps/$hp" intc_ip "Auto" master_apm "0"] \
        [get_bd_intf_pins zynq_ps/$hp]
}

# --- 控制口 HPM0 → kernel + 所有 DMA ---
set ctrl_slaves {attn_top_0/s_axi_control}
for {set c 0} {$c < $ncluster} {incr c} {
    lappend ctrl_slaves dma_k_${c}/S_AXI_LITE dma_v_${c}/S_AXI_LITE
}
foreach s $ctrl_slaves {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        [list Master "/zynq_ps/M_AXI_HPM0_FPD" Slave "/$s" intc_ip "Auto"] \
        [get_bd_intf_pins $s]
}

# --- APM:监控簇0/簇1 的 K-DMA MM2S 读(2 槽) ---
set apm_vlnv [lindex [lsort [get_ipdefs -all xilinx.com:ip:axi_perf_mon:*]] end]
create_bd_cell -type ip -vlnv $apm_vlnv apm0
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {2} CONFIG.C_ENABLE_EVENT_COUNT {1} \
    CONFIG.C_ENABLE_EVENT_LOG {0} CONFIG.C_SHOW_AXI_IDS {0} CONFIG.C_SHOW_AXI_LEN {1}] [get_bd_cells apm0]
set apm_slot1 [expr {$ncluster > 1 ? "dma_k_1/M_AXI_MM2S" : "dma_v_0/M_AXI_MM2S"}]
connect_bd_intf_net [get_bd_intf_pins apm0/SLOT_0_AXI] [get_bd_intf_pins dma_k_0/M_AXI_MM2S]
connect_bd_intf_net [get_bd_intf_pins apm0/SLOT_1_AXI] [get_bd_intf_pins $apm_slot1]
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
set_property synth_checkpoint_mode None [get_files design_1.bd]
generate_target synthesis [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [make_wrapper -files [get_files design_1.bd] -top -fileset sources_1]
set_property top design_1_wrapper [current_fileset]
}  ;# end BD

if {$stage eq "bd"} { puts "===== BD STAGE DONE ====="; exit }

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synth_1 失败" }
open_run synth_1 -name synth_1
puts "===== POST-SYNTH UTILIZATION ====="
puts [report_utilization -return_string]
puts "===== END POST-SYNTH ====="
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "impl_1 未完成" }
open_run impl_1
puts "===== POST-ROUTE UTILIZATION ====="
puts [report_utilization -return_string]
puts "===== END POST-ROUTE ====="
report_utilization   -file ./attn_zcu104/post_route_util.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "WNS(setup worst slack) = $wns ns"
set bitsrc [glob ./attn_zcu104/attn_zcu104.runs/impl_1/*.bit]
set hwhsrc [glob ./attn_zcu104/attn_zcu104.gen/sources_1/bd/design_1/hw_handoff/*.hwh]
file copy -force $bitsrc ./attn_top.bit
file copy -force $hwhsrc ./attn_top.hwh
puts "DONE: attn_top.bit / attn_top.hwh 已生成(ZCU104)"
