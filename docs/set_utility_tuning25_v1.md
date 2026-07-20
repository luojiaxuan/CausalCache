# Set Utility Predictor 25% 超参搜索 v1

本轮只为 DeepSets 与 Set Transformer 各自选择一套配置，不做三 seed，也不读取 evaluation。

- 输入：运行中 labels 的一次冻结 snapshot；按 trajectory 抽 25% train，保留 snapshot 中全部 tune；
- candidates：每个 state 的完整 variable history，不做 recent-`n` 截断；
- 表示：完整 frozen GUI-Owl visual hidden-state sequence 与 text embedding sequence；
- budget：模型不输入 `B`，部署仍为 at-most-`B`；
- 搜索：两个 family 各 6 个单 seed 配置，覆盖 `d=256/512`、latent `8/16`、resampler `1/2`、LR `1e-4/3e-4`；
- 选择：family 内最低 `best_tune_total`；近似并列时优先 latency，再优先较小模型；
- 边界：这是 tuning-only completion snapshot，不进入正式 10/25/50/100% learning curve。全量 labels 完成后，冻结赢家配置再重建 nested trajectory fractions。

Artifact identity、实际 state/trajectory 数、结果与 checkpoint revision 在运行完成后回填。
