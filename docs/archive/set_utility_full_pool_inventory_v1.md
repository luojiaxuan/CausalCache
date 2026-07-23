# Set Utility Full-Pool Inventory v1

> 状态：metadata-only P-1 已从 clean pushed `main@3a058b2` 正式完成。canonical manifest 固定
> 610/610 shards、88,186,663,372 bytes 与 610 个唯一 LFS SHA256；closed-loop、download、row decode、
> semantic census、model、GPU 和 HF mutation 均为 0。
> P0 source-only skeleton 已存在，config SHA256=
> `7e65227e710009d3626bd0063d425c831dfc59e9a6bbe871b9d3e4d15e085e8b`；它已绑定 canonical
> consumed ledger。P0 Execution-A 现已绑定本 P-1 manifest，config SHA256=
> `f01beae98432bae19d02f7811d94dc5fa569263b0d917188f2489e15edbf371f`；只有该 config commit/push
> 后才授权实际 census。

## 为什么要先做 P-1

旧 long-pool discovery 只绑定了 GUIOdyssey transport train 的 16/610 shards，共 212 rows；其中 81 rows
仅因旧 `decision_count<=12` 条件被排除。这个局部范围既不足以决定 500/1000 条 trajectory 是否可行，也不足以
冻结 train/tune/one-shot evaluation。因此新的扩数据链路先单独固定全部 source bytes，再做任何 semantic census。

P-1 只允许对
`cua-lite/GUIOdyssey@ea08072b30e523fb4492e4f4597505879ffcd63b` 调用一次
`HfApi.list_repo_tree`，读取 `mobile/use/train` 下 610 个 canonical parquet shard 的：

- repository path；
- byte size；
- LFS SHA256。

它不下载 parquet，不读取 row/image/instruction，不分配 split/query，不产生 restoration label，也不训练模型。
冻结 config 为
[`causalcache_set_utility_full_pool_inventory_v1.json`](../../code/configs/causalcache_set_utility_full_pool_inventory_v1.json)，
SHA256=`1b2b4374d1653bcf22444d8e708c71bc41ac87fa956243ddcb9bd963eeca7e96`。

## 命令

Source-only 验证不会联网：

```bash
PYTHONPATH=code .venv/bin/python \
  code/scripts/run_set_utility_full_pool_inventory_v1.py \
  --repository-root . validate-source
```

在可联网 checkout 中生成 canonical manifest：

```bash
PYTHONPATH=code .venv/bin/python \
  code/scripts/run_set_utility_full_pool_inventory_v1.py \
  --repository-root . inventory \
  --token-file ~/hf_key.txt
```

输出固定为 `data/manifests/set_utility_full_pool_inventory_v1.json`，exclusive create，已有文件时拒绝覆盖。
成功后必须先验证 610 个连续 shard index、总 byte 数、每个 LFS identity 和 manifest hash，再 commit/push；
随后才能另外冻结 P0 census 的 exact input SHA。

## 2026-07-18 正式结果

网络权限恢复后，从 clean pushed `main@3a058b2` 执行唯一 metadata 命令并成功落盘：

- manifest SHA256：`e892e7e8f226e9500d978147a9698ad206a70ad9c303ebd918350f9e10ae6c5e`；
- files-list SHA256：`e81e3ba6abe16f5da4d714c434e5e0879a54f747dab74bff531058ba47b978cd`；
- shard count：610，连续覆盖 `00000..00609`；
- total bytes：88,186,663,372；
- unique paths / unique LFS SHA256：610 / 610。

此前 DNS 失败仍是零输出的基础设施记录，不是另一次 inventory result。canonical output 使用 exclusive-create，
本次成功后不得覆盖或重跑。

## 阶段边界

P-1 完成只授权建立 byte-pinned source inventory。它不自动授权 full-pool semantic census、roster/split、
query-state selection、label generation 或训练。独立 P0 Execution-A 已绑定 consumed ledger 与 P-1 manifest
的完整 SHA256；它只授权本地 pinned shard access、row decode、policy-blind census 和一次 output write，继续禁止
role/split、query、OCR/model、labels、training、GPU、closed-loop 与 sealed test。

## P0 下游结果

Execution-A 已从 clean pushed `main@0e16bcf` 在 Hyper00 完成。610 shards 的 size/LFS SHA 均通过 runner
逐文件校验；8,146 rows 归约为 6,933 条未消费 eligible trajectory / 6,928 个 instruction-app group。
canonical P0 manifest SHA256=
`729d1e1046761177d53d0f320139331224c9f77f7add5097d04bce479566189b`。详细结果见
[`../../data/results/archive/set_utility_full_pool_census_v2/`](../../data/results/archive/set_utility_full_pool_census_v2/)。
