# HGKV selector v1

## 状态

Stage-1 singleton-gain scorer 已完成，并已确定 V1 对原正式 gate 为 **NO-GO**：
set-conditioned B1 与 Stage-1 完全相同，而 Stage-1 显著输给 Recent，故
B1/B2/B4 conjunctive PASS 已不可能。Stage-2 conditional label scoring 仍在 hyper01
按冻结协议执行；后续 B1 parity、完整 selected-set 重渲染/重打分和 B2/B4
set-conditioned 相对 singleton 的结果作为 V1 interaction diagnostic，不再表述为仍有
机会通过原 gate。

## 冻结输入

| 输入 | 规模 / 位置 | 状态 |
|---|---|---|
| HGKV readout features | hyper01 `/data02/jaxan/runs/hgkv-readout-v1/`；75,628 unique rows，1,280 dims，8 shards | `DONE`；`PENDING_HF_UPLOAD` |
| singleton scores | hyper01 `/data02/jaxan/artifacts/sft/hgkv-selector-scores/`；train 64,264，heldout 11,364，B0 10,680 | complete；`PENDING_HF_UPLOAD` |
| conditional renders | hyper01 `/data02/jaxan/artifacts/sft/ody-labels-cond/{train,heldout}/`；raw 255,490 / 45,227，unique score identities 219,549 / 38,836 | render `DONE`；scores in progress |
| balanced train scorer view | hyper01 `/data02/jaxan/artifacts/sft/ody-labels-cond-balanced-v1/train/` | 255,490 raw / 219,549 unique 守恒；12 shard unique 负载 18,001–18,602 |
| adapter | hyper01 `/data02/jaxan/runs/hgkv-eval/hg-s100.pt` | SHA256 `8f2cc49e1aa0b06ce231eb54937d813317f5274a799c97b09be7fdb22be46317` |

所有训练与推理代码来自
`f7cfc0006456cde96951fee4b16f4505fb18462b`；Stage-1/2 config 分别为
`code/configs/hgkv_selector_stage1_v1.json` 与
`code/configs/hgkv_selector_stage2_v1.json`。

## Stage-1 结果

Hyper01 run root：`/data02/jaxan/runs/hgkv-selector-stage1-v1/`，完成时间
`2026-07-24T08:52:05.775604+00:00`。

| 指标 | 值 |
|---|---:|
| train candidates / states | 64,264 / 9,059 |
| heldout candidates / states | 11,364 / 1,621 |
| selected fold epochs | 29, 30, 27, 21, 5 |
| final refit epochs | 27 |
| model realized `U` | 0.104780 |
| Recent realized `U` | 0.113203 |
| Random realized `U` | 0.103396 |
| Oracle realized `U` | 0.133664 |
| model − Recent | −0.008423，95% CI [−0.010526, −0.006338] |
| model − Random | +0.001384，95% CI [−0.000059, +0.002828] |
| mean regret | 0.028885 |
| mean Spearman | 0.022291 |
| top-1 hit rate | 0.173967 |
| STOP accuracy | 0.906848 |

Stage-1 显著输给 Recent；这是冻结协议下的负诊断，不据此调参或改标签。`STOP
accuracy=0.906848` 也约等于当前约 9.3% positive-STOP 分布下的 majority baseline，
不能视为正信号。Stage-1 仍按 V1 协议提供 Stage-2 encoder 初始化、第一步和 independent
baseline，但 B1 两路完全相同，因此 V1 的原正式 gate 已确定失败。完整 selected-set
评测只继续回答 set conditioning 能否在 B2/B4 修正较差首步、降低冗余或发现互补。

## Stage-1 provenance

| artifact | SHA256 |
|---|---|
| `stage1.pt` | `775bfbca851abf535b286b674f7642a14a7915a8b2aea65a640c7088b5430fb3` |
| `metrics.json` | `2125dd389b08efbd82fdaa426ad6a8972c2a1fc63511b453cbd93e4c645089d2` |
| `predictions-heldout.jsonl` | `706bc3e7775045a661e9e60aa2062989ad968866e58bf0cede73721e0b3ab08d` |

`predictions-heldout.jsonl` 为 11,364 行。run root 同时保留 5 秒 GPU preflight 和
launch-time GPU snapshot；Stage-1 使用 NVIDIA H200 GPU 0，启动时 0 MiB。

## Conditional scoring 分片

原 32 个 render file 直接按 12 个 scorer round-robin 时，每个 process 的 unique
identity 负载为 9,401–23,708；长尾会让同卡另一 scorer 提前退出。正式 train scoring
因此只重排 JSONL 行，不修改任何 row、prompt、图片或标签：

```bash
PYTHONPATH=code python -m scripts.rebalance_scoring_file_shards \
  --input-root /data/artifacts/sft/ody-labels-cond/train \
  --output-root /data/artifacts/sft/ody-labels-cond-balanced-v1/train \
  --input-glob 'samples-shard*.jsonl' \
  --output-count 12 \
  --images-target /data/artifacts/sft/ody-labels-cond/train/images \
  --source-commit f7cfc0006456cde96951fee4b16f4505fb18462b
```

分桶键与 scorer resume identity 完全相同：
`(pair_group, variant, singleton_event_step_id, restored_set_key)`；取 canonical JSON 的
SHA256 前 8 bytes 对 12 取模。同一 restored coalition 的重复 render 行始终留在同一
process 内，保留 in-process dedup。发布前验证 raw `255,490→255,490`、unique
`219,549→219,549`、跨输出 identity 重复为 0，并把输入/输出逐文件 SHA256 写入
balanced root 的 `DONE`。12 个 scorer 的 unique 负载为
`[18206,18470,18443,18128,18602,18097,18293,18257,18212,18429,18001,18411]`。

## 决策语义与正式 gate

- Stage-1/Stage-2 真正比较 candidate `rank_score` 与 `stop_logit`，不把预测
  gain/marginal 直接与 0 比较；
- B1 两路共用 Stage-1，所选集合和真实 `U(S)` 必须完全一致；
- 当前 decision query 已编码在每个 HGKV counterfactual readout 中，Stage-2 不接收独立
  query tensor；
- shortlist 只控制 conditional-label 构造成本；正式推理重排 capped inventory 的全部
  剩余候选；
- B4 第四次选择在 `|S|=3` 上属于结构外推，因为直接 conditional supervision 只覆盖
  `|S|=1/2`；
- formal gate 对 B1/B2/B4 的完整 coalition 重新渲染并由 hg-s100 打分，再 exact-join
  frozen B0；禁止用预测 marginal 或 singleton gain 求和替代真实 `U(S)`。
- selected-set render 在计分前由 `validate_selected_set_render_v1.py` 对 plan 做 exact
  unique-set coverage join，同时核对 aggregated method/budget metadata、恢复集合键、
  图片存在性和逐文件 SHA；缺一项不发射 scorer。
- `validate_selector_inference_v1.py` 在 gate plan 生成前检查两路×B1/B2/B4 全覆盖、
  candidate inventory/预算/集合合法性，并逐 state 强制 B1 selected set 完全相同。

完整冻结协议见
[`docs/selector_v1_protocol.md`](../../../docs/selector_v1_protocol.md)。

## V1 后续与论文候选 V2 裁定

- 当前 V1 不停、不改科学参数：跑完 conditional scoring、Stage-2、B1 parity 和真实
  B1/B2/B4 selected-set 评测，正式记录 B1 NO-GO，并重点报告
  `set-conditioned − singleton`、冗余、后续 STOP 与完整集合 `U(S)`；
- 当前 reusable train/heldout coalition scores 与 `cond_edge1/2` 继续有效。更换
  student 架构、loss 或训练方式不需要重打已有集合，但 heldout 不得转为训练；
- 若 B4 作为正文主结果，V2 增量增加
  `cond_edge3=U({i,j,k,l})−U({i,j,k})`。复用已有 edge2 triple denominator，仅对稀疏
  triple anchors 展开新的四元素 numerator，不重复生成 edge1/2；
- V2 优先采用 Recent-seeded set-conditioned selector：B1 与 Recent parity，B2/B4 从
  Recent-1 seed 开始学习 edge1/2/3 与 STOP；新 gate 的 B1 是 parity 审计，B2/B4 才要求
  显著超过 Recent/Similarity/Random；
- 统一从空集合学习 `edge0–edge3` 保留为更高风险备选；Stage-1 已显示当前 HGKV readout
  对 singleton 排序信号很弱，不能假设换 Stage-2 架构就能自动修复首步。

V2 是后续新实验契约，不追溯修改正在执行的 V1 config、labels、selection 或 gate。
