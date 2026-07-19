# Token-level Set Utility Predictor v2 partial pilot

这是 train/tune-only optimization diagnostic，不是正式 held-out selector 结果。snapshot 固定 745 states / 74
trajectories，其中 646/67 用于 train，99/7 用于 tune；evaluation records 从未加载。

完整表示使用 1,043 份去重截图的 variable-length、unpooled GUI-Owl final-main merger token sequences，以及
1,052 份 frozen text embedding sequences。token cache 为 22.78GB，并在训练时常驻 H200。

| Variant | First tune objective | Best tune objective | Reduction | Best tune ranking |
| --- | ---: | ---: | ---: | ---: |
| DeepSets-d256-L8 | 2.1688 | 0.3564 | 83.6% | 0.7058 |
| SetTransformer-d256-L8 | 0.9215 | 0.5753 | 37.6% | 0.7000 |
| SetTransformer-d512-L16 | 0.5306 | 0.3747 | 29.4% | 0.6536 |

结论只限于：full-token pipeline 可优化，三种成熟配置的 tune objective 都下降，8-state overfit sanity 也下降
35.5%。当前 tune 只有 7 条 trajectories，DeepSets 暂时最好；不能据此声称 Set Transformer 优于 DeepSets，
也不能与 OCR/RGB、exact oracle 或 terminal success 比较。三个模型的 best raw MAE 并未同步稳定改善，正式训练
前还需要重新检查 raw/normalized/ranking loss 的尺度权衡。

大文件的本地 cache 位于 Hyper00：

```text
/data02/jaxan/runs/causalcache-set-utility-token-v2-pilot-ba1c480
```

canonical dataset artifacts 已上传到
[`gavinlaw/causalcache-set-utility-new-development-mobile@e1240bde`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/e1240bdeff500114097b71148ba65ae19e71e6e8/artifacts/set-utility-token-v2-partial-a73cc18)，
共 23,428,500,440 bytes / 1,878 files。四个 checkpoints 与训练 summaries 已上传到
[`gavinlaw/causalcache-set-utility-predictors-mobile@a55666c1`](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/a55666c11da6b2848a6f066b4b05980704f1bf7a/artifacts/set-utility-token-v2-partial-a73cc18)，
共 155,437,231 bytes / 11 files。两个 repo 都有 tag `set-utility-token-v2-partial-a73cc18`。

机器可读结果与 hashes 见 [`summary.json`](summary.json)。
