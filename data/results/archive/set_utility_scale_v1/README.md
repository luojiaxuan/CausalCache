# Set Utility Scale v1

状态：`DEPRECATED_ANCHOR_ONLY_PILOT`。

该实验每条 trajectory 只使用一个 `stratum_anchor` state。355 个 state 不是 trajectory 内逐 step 扩数，不能作为正式 predictor 数据规模或泛化结论；结果仅保留用于复现，不再沿此数据设计调参。

356 个预选 `n=4` states 中，355 个完成全部 11 个 `|S|<=2` restoration labels；1 个 train state 因 GUI-Owl action parse error 跳过。所有接受 state 的 reference repeat KL 为 0。

## 主要结果

| Selector | B1 exact recovery | B2 exact recovery |
|---|---:|---:|
| Set Transformer, at-most B | 0.6421 | 0.4746 |
| Set Transformer, fixed B | 0.6569 | 0.6431 |
| DeepSets, at-most B | 0.5114 | 0.6797 |
| DeepSets, fixed B | 0.5450 | **0.8594** |
| Pairwise-additive, at-most B | 0.6516 | 0.6520 |
| Recent | 0.7774 | 0.7990 |
| OCR/RGB | 0.7769 | 0.8541 |
| Oracle-independent J | **0.9939** | **0.9006** |
| Exact subset oracle | 1.0000 | 1.0000 |

DeepSets fixed-B 在 B2 的 mean utility 只比 OCR/RGB 高 `0.000271`，paired 为 12胜/1平/11负，因此结论是“边缘持平”，不是稳定胜出。Oracle-independent `J` 明显强于 learned models，下一步应优先解决 feature/distillation，而不是直接上 closed-loop。

Raw labels、完整 evaluation summary 与 checkpoints 当前保存在 Hyper00：

`/data02/jaxan/runs/causalcache-set-utility-scale-v1-ac0ef27`

HF 已发布并 fresh byte replay：

- [dataset commit `a95ce68b`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/a95ce68bd628daaec40a7575847c9db584f20dc4/artifacts/set-utility-scale-v1-ac0ef27)：362 files；
- [model commit `365f3882`](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/365f3882658b40eccb64c9565ae8639986c59e82)：3 checkpoints、config、完整 evaluation summary 和 model card；
- 两个 repo 的 tag 均为 `set-utility-scale-v1-ac0ef27`。
