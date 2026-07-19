# Set Utility MVP v1

## 目的

最快回答：restoration labels 扩到几十个独立 states 后，Set Transformer 是否比 recent 和 OCR/RGB 更会选 memory subset。

本轮是 exploratory MVP，不沿用旧 all-or-nothing gate。旧 D1/D1b/D2 只提供 runtime 经验，不阻塞本轮。

## 数据与标签

- Source：processor freeze v2，HF revision `c20bab8df424dc9e45ece1084f3d1dc035dd1ed8`。
- 选择：每个 processor worker 固定 12 train、2 tune、2 evaluation；总计 64 states。
- State：`stratum_anchor`、`n=4`、`maximum_labeled_cardinality=2`。
- 每 state 标签：empty、4 singles、6 pairs，共 11 个 `D(S)`。
- 选择只依赖 frozen metadata 和固定 salt，不读取 policy output、utility 或 outcome。

## State-level admissibility

每个 state 先对 full-history action 生成两次，并对同一 teacher target 重复 forward：

- canonical action 相同且 repeat KL `<=1e-4`：保留并生成全部 labels；
- 否则记录 `failure_class` 后跳过；
- 不 top-up、不让一个 state 阻塞其他 states。

## 训练与评估

训练 pairwise-additive、DeepSets、Set Transformer。没有 utility variation 的 train/tune state 不进入优化，但仍计入接受率；evaluation 不因 utility 大小过滤。

在 held-out evaluation states 报告：

- `B=1,2` true utility；
- exact-oracle recovery；
- recent、OCR/RGB、oracle-independent `J`；
- prediction error 与训练/跳过数量。

closed-loop 和 matched-NLL 暂不运行。

## 命令

```bash
python3 code/scripts/run_set_utility_mvp_labels.py all \
  --repository-root /data/CausalCache \
  --processor-root /data/artifacts/causalcache-set-utility-processor-freeze-v2-image-contract-repair-e976b99 \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --output-root /data/runs/causalcache-set-utility-mvp-v1-<git> \
  --config /data/CausalCache/code/configs/causalcache_set_utility_mvp_v1.json

python3 code/scripts/train_set_utility_mvp.py \
  --run-root /data/runs/causalcache-set-utility-mvp-v1-<git> \
  --config /data/CausalCache/code/configs/causalcache_set_utility_mvp_v1.json
```

输出目录可恢复；每个 state 独立原子写入。Raw labels/features 与 checkpoints 发布到 HF，Git 只保存 summary。

## Scale v1

MVP-64 完成后，不改 feature、model family、hidden size 或 loss，只扩大同一 deterministic selection prefix：每 worker 77 train、6 tune、6 development-evaluation，共 356 states。Teacher microbatch 从 2 提到 4；科学输出仍逐 state 保存。

除 at-most-`B` 主指标外，报告 learned fixed-`B` 诊断，用来区分 subset ranking 与负值校准失败；它不替代主指标。
