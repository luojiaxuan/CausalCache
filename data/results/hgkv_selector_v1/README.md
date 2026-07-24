# HGKV selector v1

## 状态

终态：**`NO_GO_HGKV_SELECTOR_V1` / `ABORTED_BY_SELECTOR_V2_SUPERSESSION`**。
set-conditioned B1 与 Stage-1 完全相同，而 Stage-1 显著输给 Recent，故
B1/B2/B4 conjunctive PASS 已不可能。2026-07-24T17:15:12Z 已停止 V1 conditional
scoring、Stage-2 launcher 和 selected-set gate 后续自动任务；V1 Stage-2 从未启动。
已完成的 exact-key coalition scores 保留为 V2 cache，未打分的旧 recent-8 /
singleton-proxy render 不再消费 GPU。V1 不进入论文主方法，Stage-1 只保留为 negative
independent baseline。

## 冻结输入

| 输入 | 规模 / 位置 | 状态 |
|---|---|---|
| HGKV readout features | hyper01 `/data02/jaxan/runs/hgkv-readout-v1/`；75,628 unique rows，1,280 dims，8 shards | `DONE`；`PENDING_HF_UPLOAD` |
| singleton scores | hyper01 `/data02/jaxan/artifacts/sft/hgkv-selector-scores/`；train 64,264，heldout 11,364，B0 10,680 | complete；`PENDING_HF_UPLOAD` |
| conditional renders | hyper01 `/data02/jaxan/artifacts/sft/ody-labels-cond/{train,heldout}/`；raw 255,490 / 45,227，unique score identities 219,549 / 38,836 | render `DONE`；V1 scoring superseded |
| conditional scores | hyper01 `/data02/jaxan/runs/hgkv-conditional-scores-v1/` | heldout `38,836/38,836 DONE`；train `160,928/219,549 ABORTED`；partial rows exact-key cache only |
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

## Supersession 与 partial-score provenance

- 停止时 12 个 train shard 合计 `160,928` 行 / `160,928` unique score identities，
  duplicate=0、invalid JSON=0；V1 expected inventory 为 `219,549`，不得继续补齐；
- `ABORTED.json` 写入 stage2 run、conditional score root/train 和 conditional artifact
  root，四份 SHA256：
  `48510c1487ec266282cce15c32ae884322fc5dcd970bd7d85ca5462228078cdf`；
- 旧 coalition score 只有在 V2 canonical key
  `(pair_group, restored_set_key, target_action_sha256, prompt_revision,
  hgkv_checkpoint_sha256, b0_policy_sha256)` exact match 时才能复用；近似相同不得复用；
- V1 的 recent-8 candidate inventory、singleton-proxy prefix distribution 和缺失 edge3
  的 label distribution 不进入 V2；
- 大规模 render、score、readout 和 checkpoint 仍为 hyper01 本地
  `PENDING_HF_UPLOAD`，后续由 V2 cache manifest 统一发布。

后继正式契约见 `docs/selector_v2_beam4_protocol.md`。V1 config、trainer、inference
源码全部保持原样作为 provenance。
