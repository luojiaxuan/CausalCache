# Set Utility Full-Pool Census v2

本目录记录 policy-blind P0 的正式完成态。Execution-A 从 clean pushed
`main@0e16bcf3117c26bc8aa104412f7c743506830abe` 运行，逐个验证并扫描 P-1 固定的 610 个
GUIOdyssey parquet shard。canonical census manifest 是
[`../../manifests/set_utility_full_pool_census_v2.json`](../../../manifests/set_utility_full_pool_census_v2.json)，
SHA256=`729d1e1046761177d53d0f320139331224c9f77f7add5097d04bce479566189b`。

## 结果

- source：610 shards / 88,186,663,372 bytes / 8,146 rows；
- parser 或 consumed firewall 排除 1,213 rows；
- 新的未消费 eligible pool：6,933 trajectories / 6,928 instruction-app groups；
- strata：`6–9=1,736`、`10–17=3,601`、`18+=1,596`；decision count 范围 6–54；
- 107 个历史 identity 全部观测到：`legacy_train_only=58`、`forbidden_consumed=49`；
- role/split、query、OCR、policy、restoration、gate、training、GPU、closed-loop 与 sealed-test operation
  均为 0。

这证明 full-pool 数据规模足以进入 Freeze-B，但本结果本身不分配 train/tune/evaluation，不选择 query state，
也不授权 restoration label generation 或 predictor training。

## 复现

运行环境：Hyper00 `node-radixark-16-0000`，container
`sglang-omni-jaxan-07181624`，image digest
`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`，Python 3.12.3，
PyArrow 24.0.0。执行命令：

```bash
PYTHONPATH=code python code/scripts/materialize_set_utility_full_pool.py \
  --repository-root . \
  --execution-config code/configs/causalcache_set_utility_full_pool_census_v2_execution.json \
  --source-root /data/source/guiodyssey-full-pool-v1
```

本地 verified source cache 位于 Hyper00 host
`/data02/jaxan/source/guiodyssey-full-pool-v1`；它只是可重建缓存，canonical source 仍是
`cua-lite/GUIOdyssey@ea08072b30e523fb4492e4f4597505879ffcd63b`。完整 machine summary 见
[`summary.json`](summary.json)。
