# Set Utility Predictor 25% 超参搜索 v1

状态：`COMPLETED_TUNING_ONLY`。本轮只使用 train/tune，不加载 evaluation；单 seed 用于冻结两个 family 的
learning-curve 配置，不作方差或方法优劣结论。

- input：249 train trajectories / 2,535 states；99 tune trajectories / 1,040 states；完整 variable history；
- frozen input SHA256：`725e13ab...d526d7`；token cache SHA256：`1d1eea48...aecb70`；
- representation：未池化 GUI-Owl visual hidden states + text embedding sequence；
- seed：`20260720`，在 model initialization 前生效；normalization floor：`0.01`；
- runtime：Hyper00/Hyper01 H200；12 个配置并行；persistent root：
  `/data02/jaxan/runs/causalcache-set-utility-tuning25-v3-seeded-floor1e2-2d68154`；
- evaluation records loaded：`false`。

| Family | Variant | Best epoch | Tune total | Ranking acc. | Raw MAE | Termination |
|---|---|---:|---:|---:|---:|---|
| DeepSets | d256/l8/r1/lr1e-4 | 2 | 0.31609 | 0.6641 | 0.03572 | early stop |
| DeepSets | d256/l8/r1/lr3e-4 | 2 | 0.31450 | 0.6728 | 0.03639 | early stop |
| DeepSets | d256/l16/r2/lr1e-4 | 2 | 0.32430 | 0.7063 | 0.03417 | early stop |
| DeepSets | d256/l16/r2/lr3e-4 | 3 | 0.32236 | 0.6964 | 0.03512 | early stop |
| DeepSets | d512/l16/r2/lr1e-4 | 2 | 0.32537 | 0.7056 | 0.03381 | early stop |
| **DeepSets** | **d512/l16/r2/lr3e-4** | **4** | **0.31137** | **0.6963** | **0.03447** | **non-finite after best** |
| Set Transformer | d256/l8/r1/s2/lr1e-4 | 2 | **0.31169** | 0.7062 | 0.03437 | early stop |
| Set Transformer | d256/l8/r1/s2/lr3e-4 | 3 | 0.31356 | 0.7032 | 0.03582 | early stop |
| Set Transformer | d256/l16/r2/s3/lr1e-4 | 1 | 0.31583 | 0.6314 | 0.03560 | early stop |
| Set Transformer | d256/l16/r2/s3/lr3e-4 | 3 | 0.31281 | 0.6901 | 0.03535 | early stop |
| Set Transformer | d512/l16/r2/s3/lr1e-4 | 2 | 0.31989 | 0.6988 | 0.03456 | early stop |
| Set Transformer | d512/l16/r2/s3/lr3e-4 | 2 | 0.32584 | 0.6581 | 0.03690 | non-finite after best |

按预注册的 family 内最低 `best_tune_total`，冻结：

- DeepSets：`deepsets_d512_l16_r2_lr3e4`；其最佳 checkpoint 有效，但后续训练出现 non-finite，正式曲线必须保留该 failure 标记；
- Set Transformer：`set_transformer_d256_l8_r1_s2_lr1e4`；
- 两者将以同一单 seed 跑 nested 10/25/50/100% train trajectories，完整 tune 保持不变。

本结果没有选择主部署模型。正式 deployment 仍需 held-out utility 与 warm selector latency 的 Pareto evaluation。
冻结配置见 [`code/configs/causalcache_set_utility_learning_curve_v1.json`](../../../code/configs/causalcache_set_utility_learning_curve_v1.json)，机器可读结果见 [`summary.json`](summary.json)。
