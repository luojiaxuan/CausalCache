# data/results/archive 索引

history-gated mainline(2026-07-23 起)之前各实验线的轻量结果目录,整目录平移至此,
每个目录保留自己的 README 与 summary/artifact JSON,内容未改动。时代级摘要与失败原因见
[`docs/archive/README.md`](../../../docs/archive/README.md)。仍在活跃使用的结果留在
`data/results/` 顶层:`hgkv_gate_v1`(正式 gate PASS,s100)、`hgkv_selector_v1`、
`fl75_selector_v1`(旧 full-layer s75 selector 消融)、
`odyssey_zeroshot_closed_loop_v1`(2026-07-24 第三轮裁决)与全部 `osworld_*`。

按时代分组(目录名前缀):

- **v1 探索与 policy 选型(07-14)**:`qwen_policy_*`、`ui_tars_*`、`showui_policy_*`、
  `open_cua_policy_*`、`gui_owl_*`(含 1.5-8B-Think 验证)、`androidworld_environment_smoke`、
  `synthetic_phase0`、`go_no_go_diagnostic_v1*`——backbone coverage gate 全部 NO-GO,
  GUI-Owl 被选为 v2 substrate。
- **Restoration v2(07-15 ~ 07-17)**:`restoration_v2_*`——substrate 筛选、v2.1/v2.2-eager
  标签、192-state expansion 及 forensic/scientific repair、OCR/RGB 与 policy-vision baseline。
- **Gate v1(07-17)**:`gate_v1_*`——formal-58 cache/train 与 fresh-16 evaluation 及修复链。
- **Independent confirm(07-18)**:`independent_*`——confirm-20 `NO_GO` 与失败分解。
- **Spatial audit(07-15)**:`spatial_reference_audit_v1`、`subset_search_ablation_v1`。
- **Set-utility / selector v0(07-18 ~ 07-21)**:`set_utility_*`、
  `guiodyssey_official_split_audit_v1`——labels、learning curve、held-out `NO_GO`、
  contextual v3/v4、decision distillation v2、direct marginal v3 `NO_GO`、selector LoRA、
  budget-deferral 停止;oracle headroom 确认但无 learned selector 稳定胜 recent。
- **闭环探索 / memory ceiling / margin-SFT(07-18 ~ 07-22)**:
  `exploratory_closed_loop_validation12_local_arms_v1`、`androidworld_memory_ceiling_v1`
  (`STORY_DEAD_CAPABILITY_BOUND`)、`success_action_recovery_gate_v1`(冻结 policy 对历史
  图内容不敏感的 U_act 门禁,history-gated 主线的动机证据之一)。

大 artifact 不在本仓库:HF 仓库与 Hyper00/Hyper01 持久路径(含 `PENDING_HF_UPLOAD` 状态)
以各目录 README 与根 README 为准。
