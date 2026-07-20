---
library_name: pytorch
tags:
  - gui-agents
  - set-transformer
  - deepsets
---

# Variable-history token predictor partial training v1

状态：`COMPLETED_DIAGNOSTIC_ONLY`。

本结果只回答 formal labels 并行生成期间，完整 GUI-Owl token sequence + variable-size set 的训练路径能否
正常优化；它使用运行中冻结的 partial snapshot，不加载 evaluation role，不执行 selector GO/NO-GO。

- code/config revision：`ca0383a`；
- immutable train/tune artifact：
  `gavinlaw/causalcache-set-utility-variable-history-mobile@771db37c9ce6d3e7057b87730c400cbae66a5398`；
- input：1,501 states（1,343 train / 158 tune）、151 trajectories（138 train / 13 tune）；
- representation：2,105 visual sequences、2,256 text sequences，cache SHA256=
  `22bbc79154924a01e7a6de5e0dcbbabac4d8a7ba369c6c4a9f57186de09b933b`；
- runtime：Hyper00 H200 GPU 4/5，一模型一卡；两个进程均 exit 0；
- persistent outputs：
  `/data02/jaxan/runs/causalcache-variable-history-training-partial-h200-ca0383a`；
- model artifact：
  `gavinlaw/causalcache-set-utility-predictors-mobile@b6bd823221e2ec4b2e68518ad40efe7e7bd5d673`，
  path=`artifacts/set-utility-variable-history-partial-ca0383a`。

| Variant | Best epoch | Best tune total | First tune total | Runtime | Checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| DeepSets d256/l8 | 5 | 0.4853 | 1.0039 | 164.7s | `b31604a6...7e4a47` |
| Set Transformer d256/l8 | 4 | 0.4714 | 1.0286 | 200.0s | `f150cf3e...c269b3` |

两个模型的 tune objective 均快速下降，证明 variable-`n_t` padding/mask、完整 token resampler 与 utility loss
可以端到端训练。Set Transformer 在本 partial snapshot 上取得更低的 best tune objective，但差值不能解释为
正式优越性；待 11,746-state labels 完成后，必须从完整 train/tune artifact 重训，再做 trajectory-disjoint
held-out selector evaluation 与 utility--latency Pareto 比较。

Taurus 的首次三容器启动因选定 A6000 在 launch 时已被其他进程占用而 OOM，未完成任何 epoch；失败容器已
删除，日志保留于 `/mnt/data2/jiaxuanluo/causalcache/logs/causalcache-variable-history-training-partial-ca0383a`。
