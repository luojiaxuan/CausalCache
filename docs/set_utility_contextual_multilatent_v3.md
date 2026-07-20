# Contextual multi-latent utility predictor v3

状态：`TUNE_ON_POLICY_TRUTH_COMPLETED / REPRESENTATION_ONLY_NO_GO`。

Exact B4 已达到 0.8248，而当前 Set Transformer B4 只有 0.7128；true-greedy search gap 0.0571 小于
student distillation gap 0.1121。因此 v3 先修 representation 与 deployment-search supervision，不再增加完全
同分布的 trajectories。

## 表示

每个 query/event entity 使用一个冻结 GUI-Owl 单图 prompt：task instruction、entity role、event 的
low-fidelity summary（query 无此项）、截图。缓存 final language-model layer 中：

- 全部 image-token positions 的 contextual hidden states；
- 第一枚 image token 前最多 64 个 text positions 的 contextual hidden states。

冻结 processor 下单图 image-token 长度保持原生变长集合 `{448,459,464,480}`；cache 与训练 collate 保留该
长度，不补写为固定 480。

因此 visual/text 已经过完整 VLM contextual encoder，而不是 vision merger output 与 raw input embeddings。
event context 在 arrival 时只算一次；query context 在部署时与 action policy hidden states 共享。

## Multi-latent interaction

learned resampler 产生每个 entity 的 16 个 latent tokens，不再提前 mean 成单个 vector。Set Transformer
直接接收 query latents 与全部 event latents，并对每个 event 的 latents 加 selected/unselected embedding。
DeepSets 使用相同 cache/resampler，保留 latents 到 additive pooling，确保差异只来自 set interaction。

## 执行顺序

1. 用 25% nested train trajectories + 完整 tune（3,703 states/350 trajectories/8,813 contexts）提取
   resumable contextual cache；每 8 个 contexts 原子写入一个 chunk，中断只重算当前 chunk，evaluation
   不可访问；
2. 训练共享输入的 DeepSets 与 Set Transformer，先检查 optimization/tune loss；
3. 对 predictor 搜索实际访问的 tune coalitions 生成 on-policy truth，比较 selector recovery；
4. 冻结表示、训练与 search 后，才允许触碰 untouched evaluation；
5. 只有 learned selector 真值超过 recent/OCR-RGB，才进入 policy replay，closed-loop 仍锁定。

## Tune on-policy truth

25% contextual checkpoints 已先执行完整 tune conditional-greedy path。冻结 schedule 覆盖 1,063 states、
9,814 个去重 coalitions，包含两个 learned paths、recent B1--B4 与 empty/full anchors。2026-07-20 使用
Hyper00 0--5 与 Hyper01 2--7 共 12×H200 独立生成真实 `D(S)`；state 与 microbatch 均可断点续跑。

结果 reducer 只按真实 restoration truth 选择模型：primary 为 trajectory-equal B1--B4 normalized recovery，
同时报告对 recent 的 trajectory-clustered paired bootstrap、long/very-long slice 与端到端 selector latency。
若两个模型 primary 相差不超过 0.01，选择 p95 total latency 更低者。实现见
[`evaluate_set_utility_tune_on_policy.py`](../code/scripts/evaluate_set_utility_tune_on_policy.py)。

完整 train+tune contextual requirement snapshot 已提前物化，但在 tune truth 判定前不启动 GPU extraction：
11,721 states / 1,100 trajectories / 27,867 contexts，其中 query 11,721、event 16,146；input content
SHA256=`af18388e...139c1`。科学绑定见
[`causalcache_set_utility_contextual_full_v4.json`](../code/configs/causalcache_set_utility_contextual_full_v4.json)。
该 snapshot 已发布到 private HF dataset revision `268bae32792c3b651541354f74d9a18b7b97ecd2`，tag=
`set-utility-contextual-inputs-full-v4-af18388e`，artifact path=
`artifacts/set-utility-contextual-inputs-full-v4-af18388e`。

Tune truth 完成 1,063/1,063 states、0 skip。trajectory-equal B1--B4 macro recovery 为 DeepSets
`0.4492`、Set Transformer `0.4365`、recent `0.4519`；DeepSets-minus-recent 95% CI=
`[-0.0160,0.0100]`，Set Transformer-minus-recent=`[-0.0314,-0.0004]`。因此 representation-only 改动
不能授权后续 policy 实验；按既定计划继续 full-data 双模型，随后重新测 deployment-search truth。

Full-data extraction 已使用 Git revision `e97f2d4b71f86bc526c48d1905b9462731d349fe` 完成：8 个独立
partitions，Hyper00 0--3、Hyper01 2--5，每台严格不超过 4 GPUs；8/8 workers exit 0，3,572/3,572
atomic shards 与 27,867 contexts 通过 finalizer。output root=
`/data02/jaxan/runs/causalcache-contextual-hidden-full-v4-e97f2d4`，content SHA256=
`44405c2cf96f468e6d0f087e8249bdb504c7efd150673e9bda836d2f52597604`，总 tensor bytes=
`119,159,482,488`。evaluation labels 未加载。

Full-data DeepSets 与 Set Transformer 已在 Hyper00 GPU 0/1 并行启动，output root=
`/data02/jaxan/runs/causalcache-contextual-training-full-v4-e97f2d4`。两者共享同一 cache、seed、loss 与
early-stopping contract；训练后必须用新的真实 tune on-policy `D(S)` 选择 deployment winner，不能用 tune
loss 替代 selector truth。

配置见
[`causalcache_set_utility_contextual_multilatent_v3.json`](../code/configs/causalcache_set_utility_contextual_multilatent_v3.json)。
输入清单与 GPU 提取入口分别为
[`materialize_set_utility_contextual_inputs.py`](../code/scripts/materialize_set_utility_contextual_inputs.py) 和
[`extract_set_utility_contextual_hidden.py`](../code/scripts/extract_set_utility_contextual_hidden.py)。
