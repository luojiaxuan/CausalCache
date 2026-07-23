# Budget-deferral frozen-candidate evaluation v1

状态：`CANCELLED_BY_USER_BEFORE_SELECTION_OR_TRUTH`。

2026-07-21 用户决定停止本轮 frozen-candidate evaluation，转向 selector-side GUI-Owl LoRA。两个
extraction container 已停止；Hyper00/Hyper01 分别保留 42/33 个可断点恢复的 receipt（约 1.3GB/963MB）。
没有运行 selection，没有读取或挂载 restoration truth，也没有产生 evaluation 结论。

当前 candidate 已在 development truth 上冻结为 B1/B2 recent、B3/B4 Structured DeepSets direct。本阶段
只物化 evaluation features 与 contextual requirements，没有读取或挂载 restoration truth。

- denominator：1,046 states / 100 trajectories；
- 历史暴露切片：all=1,046、state-new=241、trajectory-new=16 states / 6 trajectories；
- feature content SHA256：`619ecf944ea56c0f7c3747b188641f873f77139d7f94aeefb3272f9dc6c3e63f`；
- contextual input content SHA256：`7ecffcc4b3a30c59bfd148dc77e54ae7e7cb401e417e086fde582cfb2c4e2abd`；
- contextual entities：2,492（query=1,046、event=1,446）；
- `evaluation_labels_included=false`、`evaluation_labels_loaded=false`、`label_file_read_count=0`。

Hyper00/Hyper01 各自用本机 128 个 visual-token shards 生成 disjoint feature partition，随后在 Hyper00 exact
merge；canonical feature/contextual roots 分别为：

- `/data02/jaxan/artifacts/causalcache-budget-deferral-eval-features-full-4f09d07`；
- `/data02/jaxan/artifacts/causalcache-budget-deferral-eval-contextual-inputs-4f09d07`。

原 GPU extraction 按
[`causalcache_set_utility_budget_deferral_evaluation_stage_a_v1.json`](../../../../code/configs/causalcache_set_utility_budget_deferral_evaluation_stage_a_v1.json)
冻结为 Hyper00 6×H200 + Hyper01 5×H200、11 个 resumable partitions。该计划保留作历史记录，不再继续执行；
partial cache 仅保留为可恢复临时 artifact，不上传 HF。
