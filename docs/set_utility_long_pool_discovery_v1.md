# Set Utility long-pool discovery v1

> `SUPERSEDED_BEFORE_EXECUTION`：本 16-shard、13–64 discovery 从未运行，不再是当前数据入口。
> 本文仅保留为历史契约；不得运行下方命令来产生当前 roster。现行流程见
> [full-pool P-1](set_utility_full_pool_inventory_v1.md) 与
> [set utility 实现交接](set_utility_implementation_v1.md)。

## 目的

旧 GUIOdyssey selector 把 `decision_count` 限在 4–12；在已 pin 的前 16 个 shards 中，有 81 条 row
仅在该检查点被归类为 `decision_count_above_maximum`。本 P0 只回答：保持旧 success、terminal、action
parser、image 与 source-id 条件不变时，13–64 decision 的长轨迹实际还剩多少。

这一步是 policy-blind census，不是训练数据 split，也不是 restoration 实验。它不分配 train/dev/holdout，
不选择 query state，不读取 OCR/policy/restoration/gate output，不生成 `D(S)`，更不授权后续 label run。

## 冻结输入与唯一变化

- source 仍为 `cua-lite/GUIOdyssey@ea08072b30e523fb4492e4f4597505879ffcd63b` 的同一 16 个
  byte-pinned parquet shards，共 2,252,923,738 bytes / 212 rows；
- 复用 `independent_reference_gate_v1` 的 canonicalizer 与所有非长度 eligibility；
- discovery 的长度范围固定为 13–64，和旧 4–12 output roles 结构不相交；
- 候选只按新 salt 计算 deterministic SHA，并汇总 `13–16`、`17–24`、`25–64` 三个长度层；
- 结果只含 source identity、transport row、decision count、app/action coverage、hash 与 exclusion counts；
- 为在下一次 freeze 前做 duplicate-safe split，额外记录规范化 instruction + 排序 app labels 的 SHA256 group key，
  但不写入 instruction 原文。该 key 只用于 group overlap audit，不分配任何 split。

P0 完成后才能另立 label Execution-A。Execution-A 必须在任何 teacher output 前冻结 trajectory/group-level
train/dev/untouched 分割、每条轨迹唯一 query state、候选数层、`|S|<=2` label inventory 与 higher-cardinality
transfer 子集。Teacher Execution-B 仍需再单独 commit。

## 执行

本步骤 CPU-only，不需要 GPU preflight。源 parquet 必须先按 pinned manifest 放在 `<SOURCE_ROOT>`：

```bash
PYTHONPATH=code python3 code/scripts/materialize_set_utility_long_pool.py \
  --repository-root . \
  --config code/configs/causalcache_set_utility_long_pool_discovery_v1.json \
  --source-root <SOURCE_ROOT> \
  --output data/manifests/set_utility_long_pool_discovery_v1.json
```

若 output 已存在，runner fail closed；不得覆盖后重跑并静默改变 roster。生成 manifest、轻量汇总和下一步
具体数量应 commit/push `main`。未来 derived trajectories、OCR/features、`D(S)` tables 与 checkpoints 进入
private Hugging Face dataset/model repo，Git 只保存 manifest、config、代码和轻量结果。
