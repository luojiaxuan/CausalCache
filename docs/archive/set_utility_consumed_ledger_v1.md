# Set Utility Consumed Identity Ledger v1

> 状态：canonical ledger 已从 Git-pinned historical manifests 机械生成。它绑定 107 个已消费 identity，
> 防止扩数据时把旧 train/dev/confirm/reference 重新伪装成新 holdout。

## 固定分区

| Partition | Count | 来源 | 允许用途 |
| --- | ---: | --- | --- |
| `legacy_train_only` | 58 | old train-10 + expansion train-48 | 仅可与新 train 合并，不能进入 tune/evaluation |
| `forbidden_consumed` | 49 | reference-8 + old-dev-5 + fresh-16 + confirm-20 | 不得进入新训练、调参或 evaluation |

两集合无交集，union 为 107。canonical manifest 是
[`set_utility_consumed_identity_ledger_v1.json`](../../data/manifests/set_utility_consumed_identity_ledger_v1.json)：

- file SHA256：`b6f44c603b99d2f954b981e01818cf0afa028ce3a410ed935203532a097bb4ad`；
- assignment inventory SHA256：`a0d5af88fe1b007ef077d0f6e18e4fa7aac11a6bbd222cdf95778001bc97a98f`；
- union source-set SHA256：`54be8ee52248e47adcfbaea0c03a5d19793486519d6cf124474fb468f628b7b1`。

每个 assignment 只有 `{source_id, role}`，role 只允许上述两个值。细粒度 lineage、输入 path/SHA、每个历史
roster 的 ordered/set hash 和 overlap audit 保留在同一 manifest；不复制 instruction 原文。

## 可复现生成

生成器只读取六个 Git-pinned config/manifest，不联网、不读取 raw trajectories，也不载入模型：

```bash
PYTHONPATH=code .venv/bin/python \
  code/scripts/materialize_set_utility_consumed_identity_ledger.py \
  --repository-root . \
  --output /tmp/set_utility_consumed_identity_ledger_v1.json
shasum -a 256 /tmp/set_utility_consumed_identity_ledger_v1.json
```

canonical writer 使用 exclusive create，拒绝覆盖。focused tests 同时验证 58/49/107、六个细粒度 roster、
所有 preregistered hashes、跨 role overlap、malformed identity、input byte drift 与 exact-byte reproduction。

## 与 full-pool P0 的接口

ledger v1 只负责 identity firewall；历史 manifests 本身不足以在不读取 raw row 的情况下重建统一的
instruction+app group hash。P0 扫描 610 shards 时必须：

1. 用同一 canonicalizer 为全部 row 计算 group SHA256；
2. 将这 107 个 identity 从 new eligible pool 中机械排除；
3. 另存 consumed group audit（只含 identity/partition/group hash，不含 instruction 原文）；
4. 在 Freeze-B 前检查 consumed、新 train/tune/evaluation 之间的 group overlap。

因此 ledger 完成不等于 P0 已完成，更不授权 labels 或训练。
