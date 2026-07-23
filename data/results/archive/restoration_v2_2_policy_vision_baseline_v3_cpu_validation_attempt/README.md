# Policy-vision v3 CPU validation attempt

状态：`INVALID_POLICY_VISION_V3_CPU_VALIDATE_STATE_PROJECTION`；formal artifact 状态暂记为
`COMPLETED_PENDING_VERSIONED_CPU_REPLAY_VALIDATION`。

唯一 v3 GPU attempt 已完成 15-state/75-image/31-feature-forward schedule，并原子发布 exact-three artifact。
same-device 与 cross-device replay 的最大 absolute score difference 均为 0；policy、language model、LM head、
generation、gate、matched-NLL、closed-loop 与 confirm/test operation 均为 0。GPU attempt ledger 已永久 claim，
不得重跑。

紧接 GPU run 的 CPU `validate` 在重建 provenance 时失败。生成阶段的 feature record 使用四键 state：

```text
index, role, trajectory_id, state_id
```

published evaluated row 则按既有 output schema 保存七键 state，额外包含
`decision_step_id/candidate_event_step_ids/budget_event_capacity`。`_feature_record_from_evaluated_row` 错把七键
state 原样送入只接受四键 feature-state 的 `validate_feature_worker_provenance`，因此报
`policy-vision row-to-worker provenance drifted`。worker/device/GPU fields 本身没有漂移。

在同一 committed source 上做的 bounded CPU diagnostic 只把 recorded state 投影回冻结四键，其余 row、summary、
immutable labels 和 witness 不变；它逐 byte 重建 `README.md`、`state_scores.jsonl`、`summary.json` 三份文件均
exact match。该诊断证明现有 artifact 含有完整重建信息，但不替代 versioned validation repair 和独立数值审计。
原 diagnostic source 已按 byte-identical SHA256 `84e12ce1...78e4` 保存到
`code/scripts/diagnose_restoration_v2_2_policy_vision_v3_cpu_replay.py`。`failure.json` 同时记录首次 validate 的
operator-reconstructed argv、由 output publish 与 diagnostic source birth time 构成的 UTC bracket，以及
container/Python runtime；由于原命令没有单独 machine log，不把这些 operator records 冒充独立 formal replay。

下一步只能先提交 exact artifact 与本 failure binding，再冻结纯 CPU、reporting-only validation repair。不得修改
三份 artifact bytes，不得重跑 GPU，不得据此提前解锁 gate 或 confirm。
