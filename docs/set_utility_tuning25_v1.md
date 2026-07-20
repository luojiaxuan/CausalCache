# Set Utility Predictor 25% 超参搜索 v1

本轮只为 DeepSets 与 Set Transformer 各自选择一套配置，不做三 seed，也不读取 evaluation。

- 输入：运行中 labels 的一次冻结 snapshot；按 trajectory 抽 25% train，保留 snapshot 中全部 tune；
- candidates：每个 state 的完整 variable history，不做 recent-`n` 截断；
- 表示：完整 frozen GUI-Owl visual hidden-state sequence 与 text embedding sequence；
- budget：模型不输入 `B`，部署仍为 at-most-`B`；
- 搜索：两个 family 各 6 个单 seed 配置，覆盖 `d=256/512`、latent `8/16`、resampler `1/2`、LR `1e-4/3e-4`；
- 选择：family 内最低 `best_tune_total`；近似并列时优先 latency，再优先较小模型；
- 边界：这是 tuning-only completion snapshot，不进入正式 10/25/50/100% learning curve。全量 labels 完成后，冻结赢家配置再重建 nested trajectory fractions。

## 正式 v3 结果

最终 grid 使用固定 snapshot SHA256=`725e13ab...d526d7`、cache SHA256=`1d1eea48...aecb70`、
249 条 train trajectories / 2,535 states、完整 99 条 tune trajectories / 1,040 states。12/12 配置完成，
`evaluation_records_loaded=false`，seed 在 model initialization 前生效。

按预注册主指标，DeepSets 选择 `d512/l16/r2/lr3e-4`（best tune total `0.31137`），Set Transformer 选择
`d256/l8/r1/s2/lr1e-4`（`0.31169`）。DeepSets 赢家在 epoch 4 保存最佳 checkpoint 后出现 non-finite；该
checkpoint 仍是 finite metric 赢家，但后续 learning curve 必须保留 failure 标记，不得把它写成稳定性 PASS。

两套配置已冻结到 `code/configs/causalcache_set_utility_learning_curve_v1.json`。下一步从完成后的 labels 重建同一
trajectory ranking 下 nested 10/25/50/100% train splits，保留完整 tune，仍只跑 seed `20260720`。完整表见
[`data/results/set_utility_tuning25_v1/`](../data/results/set_utility_tuning25_v1/README.md)。当前结果不读取
evaluation，也不决定 deployment；后者仍由 held-out utility--latency Pareto 决定。

## v1 loss 诊断与 v2 修复

首个 12-config run 暴露 normalized-loss scale pathology：train/tune 中约 1% state 的
`D(empty)` 小于 `1e-4`，直接用它作分母会让少量 state 主导 objective。按旧 total 选出的配置虽然达到
`0.55` 左右，但 tune ranking accuracy 只有 21%--23%，属于近零预测退化，不能冻结为 scale 配置。

v2 保持 snapshot、seed、grid、raw/ranking loss 与 split 全部不变，只把 normalized denominator 改为：

\[
\max(D(\varnothing), 0.01).
\]

`0.01` 在查看 v2 模型结果前由 train baseline 分布固定，约为 10th percentile；它防止 near-zero state
取得不成比例权重。v1 结果只作 loss diagnostic，正式超参冻结使用 v2。

若某个 grid member 在已有有效 checkpoint 后出现 non-finite metric，trainer 立即停止且只保留此前 finite
history 与最佳 checkpoint，并记录 `termination_reason=nonfinite_metrics`；NaN 不得进入 JSON 或被当作更优结果。

第二个 diagnostic 发现旧 trainer 在 model initialization 之后才调用 `torch.manual_seed`，因此旧 summary 的
seed 字段没有控制初始参数。最终 grid 将 seed 设置移动到 cache/model construction 之前，并在 summary 写入
`seed_applied_before_model_initialization=true`。此前 v1/v2 grid 只保留为 loss 与 runtime diagnostic，不用于
冻结 architecture/LR。
