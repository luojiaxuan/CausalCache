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
- `B=1/2` in-distribution 与扩大数据后的 `B=3/4` cardinality transfer；
- matched next-action NLL 下 restoration mass 与 closed-loop success。

cheap-feature model 仍可作为成本/信息量下界，但不能再用它的失败否定 token-level CausalCache。
