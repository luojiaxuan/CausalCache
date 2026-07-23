# exp/history-gated-mainline-v1 分支 provenance(冻结于创建时刻)

- base_main_commit: 43468dc
- GUI-Owl snapshot manifest: code/configs/gui_owl_1_5_8b_snapshot.json (sha256 前缀 50b675ec31c5c46d)
- GUI-Odyssey corrected dataset: hyper00 /data02/jaxan/artifacts/sft/ody-sft-v2/
  (37,635 样本;动作分布 click 7226 / swipe 1377 / type 1043 / system_button 979 /
  terminate 1000(自然配比)/ long_press 55;修复:坐标直映射 [0,999]、KEY_HOME/BACK→
  system_button、INCOMPLETE 排除;完整 drop 核算入 V1 正式 manifest 时补齐)
- AndroidWorld development roster: ceiling 15 模板(已用于工程调试,只许 canary/调试)
- AndroidWorld sealed roster: test-25 模板(从未参与任何训练/调试/选择,保持封存)
- OSWorld rosters: 由 OSWorld runner 会话按同规则划定后登记于此
- 关系:main 继续保底路线(platform-specific full-layer + selector);本分支失败不阻塞 main
