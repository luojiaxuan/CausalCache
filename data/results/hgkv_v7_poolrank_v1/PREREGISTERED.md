# v7 did_pool_rank:验收判据(**在看到结果之前写定**)

> 这份文件在两个臂训练途中写成、先于任何结果提交。理由是这个项目已经吃过
> 多次"看到数字再挑判据"的亏(中途采样误判 6–7 次、跨协议数字互相印证 1 次)。
> 判据一旦落盘就不再改;若最终要偏离,必须另起一节写明偏离理由,不许直接改这里。

## 对照设计

| 臂 | 位置 | objective | 语料 | 训练单元 |
|---|---|---|---|---|
| `poolrank` | h00 GPU4-7 | `did_pool_rank` | desktop-did-corpus-v7pool | 1,340 |
| `didbase` | h01 GPU0,1,4,5 | `did_ra_aware`(旧) | **同一份** v7pool | 1,340 |
| `v4baseline` | 已完成 | `did_ra_aware` | corpus-v4 | 1,755 |

`poolrank` vs `didbase` 是**单变量**(只换 objective,语料逐字节同源)。
`v4baseline` 只作为"历史位置"参照,**不用于判定** —— 它换了语料(v7pool 因满池
要求淘汰 25% 的组),两者之间是双变量。

## 判定量与阈值

### 主判据(必须同时满足)

1. **均匀成分下降**:训练诊断 `pool_mean_effect`(整池平均效应,新增)
   在 `poolrank` 上的末 25 组滚动值,应低于 `didbase` 的同一读数。
   *`didbase` 跑旧目标,该诊断不会产出* —— 所以退而用两臂都有的
   `adapter_on_recent` 与 `adapter_on_sparse`:判据改为
   **`poolrank` 的 `adapter_on_sparse` 不高于 `didbase` 的,而 `did_select` 不低于**。
   含义:同样的选择性,更少的绝对抬升。

2. **选择性不塌**:held-out 门控上 `did_select` 的 CI 仍排除 0,
   且点估计 **≥ `didbase` 同协议值的 80%**。

### 门控协议(两臂完全一致)

- dev 集:**corpus-v4 的 `samples-b1.jsonl`**(94 组),与 `v4baseline` 同一份。
  用它而不是 v7pool 自己的 dev,是为了三个臂共用一把尺子。
  无泄漏:episode 划分是同 seed 哈希,v7pool 的训练 episode 在 corpus-v4 里同样是训练集。
- 打分 config:**`causalcache_desktop_did_hgkv_v4.json`**(schema v2,与 dev 集匹配)。
  打分只用到适配器结构(rank/层/模块),三个 v7/v4 config 在这几项上相同;
  eps 与各项权重是损失项,不影响前向。
- 选点:各臂用自己 gate_report 的 `selection`;无可选点则回退 `did_select` 最大者
  并标 `fallback_no_selectable`,**不静默当选中**。

### 先验的失败模式(出现即判负,不许事后解释)

- `poolrank` 的 `did_select` CI 跨零 → 去掉 L_gain 把选择性一起去掉了;
- 两臂 `did_select` 差值落在 ±20% 内且 `adapter_on_sparse` 也无差别 →
  换目标没产生可测的效果,不要报告成"方向正确";
- 任一臂出现 NCCL Watchdog / ChildFailedError → 先当**代码问题**排查,
  不解释成"训练不稳定"。

## 训完的第一件事(不是看指标)

`diff` 两臂 `run_manifest.json` 的 `cli_args` 与 `dataset_manifest_sha256`,
确认**唯一差异是 `config` 与 `output_root`/`frozen_score_cache` 三项**。
2026-08-01 的语料混杂就是漏了这一步(config 严格对齐、`--dataset-root` 悄悄换了)。

## 这一步过了也**不等于**方法成立

离线门控全过而闭环无提升,是本项目已经发生过的事。过了门控之后的下一关是
135 任务快测集(`data/manifests/osworld_fast_devset_v1_meta.json`),
再往后才是同批多轮闭环。**门控只是不浪费闭环算力的过滤器,不是结论。**
