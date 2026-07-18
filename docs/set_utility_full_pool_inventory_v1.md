# Set Utility Full-Pool Inventory v1

> 状态：metadata-only Source-A 已冻结并通过本机验证；真实 Hugging Face inventory 尚未完成。本地受限
> 网络在第一次调用前后均未生成 partial manifest，closed-loop、row decode、model、GPU 和 HF mutation 均为 0。
> P0 source-only skeleton 已存在，config SHA256=
> `7e65227e710009d3626bd0063d425c831dfc59e9a6bbe871b9d3e4d15e085e8b`；它已绑定 canonical
> consumed ledger，但在本 P-1 manifest 产生和 commit 前不授权 P0 execution。

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
[`causalcache_set_utility_full_pool_inventory_v1.json`](../code/configs/causalcache_set_utility_full_pool_inventory_v1.json)，
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

## 2026-07-18 尝试记录

本机已确认 key file 与 `huggingface_hub` runtime 存在，但 outbound DNS 被 sandbox 拒绝，
`HfApi.list_repo_tree` 抛出 `httpx.ConnectError`。同一 sandbox 到 Hyper00 的 SSH 也返回
`Operation not permitted`。失败后 canonical output 仍不存在，因此这不是一次 inventory result，也没有可追认的
source bytes。下一步是在可联网的 Aries/Hyper checkout 从 pushed `main` 重跑上述唯一 metadata 命令。

## 阶段边界

P-1 完成只授权建立 byte-pinned source inventory。它不自动授权 full-pool semantic census、roster/split、
query-state selection、label generation 或训练。P0 必须在 P-1 manifest commit 后另立 versioned execution
contract，并绑定 consumed ledger 与 P-1 manifest 的完整 SHA256。
