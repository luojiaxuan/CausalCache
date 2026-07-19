# Set Utility Predictor v2

## 决策

v2 从论文机制出发，不再把 `q64/h64 + OCR/RGB` 小模型当作 CausalCache 主方法。旧 64 维输入只保留为
cheap-feature baseline。正式 predictor 直接学习：在当前 instruction、当前截图、候选事件高保真截图与
低保真事件文本给定时，任意 coalition 的 restoration utility：

\[
\hat U_\theta(S)=\hat D_\theta(\varnothing)-\hat D_\theta(S),
\qquad |S|\le B.
\]

模型不接收 `B`，部署时对所有 `|S|<=B` 的候选集合做 at-most-budget search。

## 完整表示

冻结 `GUI-Owl-1.5-8B-Instruct@06d5fae`，缓存而不是 mean-pool：

- 每张当前/事件截图的全部 final-main spatial-merger tokens，形状约为 `2560 x 4096`、BF16；
- instruction 与每个 deterministic low-fidelity event 的 frozen GUI-Owl input-embedding sequence；
- 11 个轻量数值字段只作为 age、step、delta 等辅助结构输入，不替代图像或文本。

训练时用 learned latent queries 对完整 token sequence 做 cross-attention resampling。该层是 student 的一部分，
因此模型可以学习保留 UI element、精确文本、局部状态与相对位置；cache 本身不做固定 mean pooling。

## 模型

每个 state 先得到一个 query entity 和四个 event entities：

```text
current visual tokens + instruction tokens
                    -> multimodal latent resampler -> query h_q

event visual tokens + low-fidelity text tokens + numeric delta
                    -> shared multimodal latent resampler -> event h_j
```

事件表示再与 `h_q` 通过 concat、product 与 absolute difference 做 query conditioning。

主模型把四个 events、selected/unselected embedding 和 query seed 输入无 positional embedding 的 Set
Transformer，预测整个 subset 的 utility。相同网络对 empty mask 再算一次并作差，因此
`U(empty)=0` 是结构约束。

DeepSets 使用完全相同的 GUI-Owl token cache、resampler、query conditioning、loss 与 split，只把最后的
set aggregator 换成 selected/unselected sum pooling。这样 `Set Transformer - DeepSets` 的差异才对应
interaction modeling，而不是输入质量。

## 部署与延迟合同

训练用的 full-token cache 不是线上 memory representation。正式部署把计算拆成两段：

1. event 到达时只编码一次截图与低保真文本，经 frozen GUI-Owl 和已训练 resampler 得到一个 `d=256/512`
   的 event embedding；线上不长期保存约 `2560 x 4096` 的 source tokens；
2. query-time 复用 action policy 对当前 observation/instruction 产生的 source tokens，编码当前 query，随后执行
   query conditioning、set aggregation 和 at-most-`B` search。

主部署 profile 是 `warm_shared_encoder`。其计时从共享的 current-query source tokens ready 开始，到 selected
subset ready 结束；event ingest latency 单独报告。为避免 encoder reuse 掩盖真实成本，还必须报告
`cold_standalone_encoder`：由 gate 独立编码当前 observation 的端到端 latency。不能把离线 full-token cache
命中时间当作 cold latency。

主部署门槛同时要求：

\[
\frac{T^{p95}_{\mathrm{warm\ selector}}}
{T^{p95}_{\mathrm{action\ policy\ forward}}}\le 0.10
\]

且 learned selector 必须在 held-out utility 上产生预注册的有效提升。只提高 utility 但不满足延迟门槛的模型只能
作为 capacity ablation，不能作为主部署方法。exact subset oracle 只提供离线上界，不参与线上 latency claim。

正式 evaluation 对每个模型和 search 组合统一报告：

- oracle utility ratio、oracle regret 与 terminal success；
- warm/cold p50、p95 latency，以及加入 selector 后的完整 policy-step latency；
- event ingest latency、persistent bytes/event、peak GPU memory；
- subset score count，以及 `N`、`B` 增长下的 latency；
- DeepSets、不同容量 Set Transformer、greedy、greedy+swap 与 beam search 的 utility--latency Pareto frontier。

因此模型选择不是“参数越多越好”，而是在通过 held-out utility 后，选择 Pareto frontier 上满足 10% 门槛的最小
配置。若没有 learned variant 同时满足两项条件，则部署 claim 为 NO-GO，即使离线 oracle signal 仍然存在。

## 训练目标与选择

每个 `n=4` state 使用全部 11 个 `|S|<=2` labels，一次 forward 共同打分：

\[
L=L_{raw}+0.5L_{normalized}+0.25L_{ranking}.
\]

- raw 与 normalized regression 使用 Smooth-L1；normalized scale 为 `D(empty)`；
- ranking 只比较同一 state 内 utility 非平局的 subsets；
- state loss 按 trajectory reweight，避免长 trajectory 因 states 更多而主导训练；
- AdamW、warmup + cosine decay、gradient clipping、tune early stopping；
- model selection 只看 tune，evaluation split 在架构与超参冻结前不加载。

首轮预注册的三个成熟配置是：共享输入的 `DeepSets-d256-L8`、`SetTransformer-d256-L8` 和容量更高的
`SetTransformer-d512-L16`。这里 `L8/L16` 指每个 entity 的 latent token 数，不是把原始视觉序列提前均值化。

## 当前 partial-label pilot

label rollout 继续在 Hyper00/Hyper01 并行。与此同时，从某一时刻已经完成的 train/tune records 建立不可变
snapshot，再执行：

1. 从 processor substrate 恢复 snapshot states 的五张图与 exact low-fidelity text；
2. 在空闲 H200 上分片缓存完整 GUI-Owl visual/text token sequences；
3. 先做小样本 overfit sanity，再并行训练三个预注册 variants；
4. 只检查 train/tune loss、ranking 和 optimization stability，不查看 evaluation，也不把 partial pilot 写成
   方法优劣结论。

该 pilot 的通过条件是至少一个 Set Transformer 配置能稳定降低 tune objective，且小样本可以被明显拟合。
正式科学比较必须等待扩大后的 frozen train/tune/evaluation labels，并用 at-most-`B` utility、oracle gap 与
selector baselines 报告。

## 后续正式 ablation

- 相同 full-token cache：Set Transformer vs DeepSets；
- learned resampler vs mean-pooled 4096-d GUI-Owl embedding；
- visual-only、text-only、visual+text；
- latent count `8/16/32`、hidden size `256/512`、set layers `2/3/4`；
- warm shared-encoder、cold standalone-encoder 与 full-event re-encode latency；
- greedy、greedy+swap、beam search 的 utility--latency Pareto；
- `B=1/2` in-distribution 与扩大数据后的 `B=3/4` cardinality transfer；
- matched next-action NLL 下 restoration mass 与 closed-loop success。

cheap-feature model 仍可作为成本/信息量下界，但不能再用它的失败否定 token-level CausalCache。

## 2026-07-19 partial pilot 结果

在 label rollout 并行期间冻结了 745 个 completed train/tune states（646/99 states，67/7 trajectories），从中
缓存 1,043 份去重 full visual token sequences 与 1,052 份 text sequences。8-state overfit objective 下降
35.5%。三个正式配置的 best tune objective 分别为 DeepSets `0.3564`、SetTransformer-d256 `0.5753`、
SetTransformer-d512 `0.3747`，相对首轮均下降。

因此 token-level optimization/data pipeline PASS，但 architecture/evaluation 仍未 PASS：tune 只有 7 条
trajectories，DeepSets 暂时最好，raw MAE 也没有随 composite objective 一致改善。正式结论必须等扩大后的
train/tune/evaluation labels，先在 tune 上冻结 loss scaling，再运行 at-most-`B` selector evaluation。完整轻量
结果见 [`data/results/set_utility_token_predictor_v2_partial/`](../data/results/set_utility_token_predictor_v2_partial/README.md)。
