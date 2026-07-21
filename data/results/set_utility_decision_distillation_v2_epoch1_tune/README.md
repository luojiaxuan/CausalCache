# Decision distillation v2 epoch-best tune truth

状态：`COMPLETED / NO_GO_DECISION_DISTILLATION_V2`。

这是 train-only decision-v2 + long-oracle supervision 重训后的第一次真实 deployment-search 判定。DeepSets
与基础 Set Transformer 都已 early-stop，最佳 checkpoint 均为 epoch 1；容量版
`latent_count=64/set_layers=4` 使用训练中的 epoch-1 immutable snapshot，因此容量比较仍是 interim，但不会影响
基础 Set Transformer 的正式 fixed-tune 判定。

## 核心结果

1,063/1,063 tune states 全部完成，0 skip。聚合单位为 trajectory，预算合同为 at-most-`B`，搜索为 beam-4。

| Method | B1 | B2 | B3 | B4 | B1--B4 macro | Long+ | total p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSets | 0.2135 | 0.3774 | 0.4826 | 0.5919 | 0.4164 | 0.3152 | 19.68 ms |
| Set Transformer base | **0.2629** | **0.4293** | **0.5615** | **0.6500** | **0.4759** | **0.3885** | 71.39 ms |
| Set Transformer L64/S4 epoch 1 | 0.2455 | 0.3999 | 0.5414 | 0.6363 | 0.4558 | 0.3558 | 114.19 ms |
| Recent | 0.1911 | 0.4361 | 0.5543 | 0.6263 | 0.4519 | 0.4008 | -- |

基础 Set Transformer 首次在完整 tune deployment-search truth 上显著超过 recent：macro delta=
`+0.02401`，trajectory bootstrap 95% CI=`[+0.00430,+0.04340]`。B1/B3/B4 分别领先
`+0.07188/+0.00726/+0.02368`。

冻结 gate 仍然判定 NO-GO，原因只有两项但都是真实失败：

- B2=`0.42930 < 0.43606`，delta=`-0.00676`；
- Long+=`0.38853 < 0.40081`，delta=`-0.01228`。

因此不能访问 untouched evaluation，也不启动 policy replay、closed-loop 或 matched-NLL。这个 verdict 不应
被解释成“decision distillation 没有信号”：它已经通过 macro CI，并在四个预算中的三个超过 recent；失败集中在
B2 与 long-history 泛化。

## 容量诊断

L64/S4 epoch 1 的 macro=`0.45575`，只比 recent 高 `+0.00383`，95% CI=
`[-0.02322,+0.03007]`，并且 Long+ 更差、p95 latency 更高。因此当前不支持“增加 latent 与 set depth 会缩小
selector gap”。

训练日志中的 L64/S4 tune total=`2.7719` 也不能与 base 的 `4.0289` 直接比较：L64/S4 因显存约束使用
`evaluation_batch_size=1`，base 使用 8；当前 conditional-listwise 统计中的 complete decision groups 随 eval
batch 变化。该数值只可用于同一 L64/S4 run 内 early stopping，不能用于跨配置选模型。真实 selector truth 才是
本轮模型选择依据。

## 执行与审计

- Git revision：`650afbe921a55b603e6bd69afd941965d06cb5fb`；seed=`20260721`；evaluation records
  loaded=`false`；attention backend=`disable_cudnn_sdp`；
- base Set checkpoint SHA256=`77080c58...d873bfe`，DeepSets=`487aa9e5...9b2469`，L64/S4 epoch-1
  snapshot=`35ce0bf0...ba2be`；
- 三个 selector 均在 Hyper01 H200 GPU 2/3 完成，container=
  `sglang-omni-jaxan-07202246`，image digest=`6a8f60af...9acfa`；
- schedule=1,063 states / 15,034 unique coalitions，content SHA256=`1697ea68...1bfc9`；
- truth 使用 Hyper00 GPU `0,1,2,6` 与 Hyper01 GPU `2,3,4,5`，每卡 2 lanes；两个 container 均
  exit 0，运行区间 `2026-07-21T05:52:49Z`--`06:20:30Z`；
- runtime：PyTorch `2.11.0+cu130`、CUDA `13.0`、Transformers `5.6.0`、safetensors `0.7.0`、BF16；
- 首次 lazy selector launch 在生成任何 result 前因 CPU token 未搬到 GPU fail-fast；修复与回归测试见
  `main@650afbe`，修复后 1,063/1,063 selection 与 truth 全部完成；
- full result content SHA256=`043d3e12...d4c86`，file SHA256=`4dd81748...284c9`。

## Source of Truth

- lightweight summary：[`evaluation-summary.json`](evaluation-summary.json)；
- full result：Hyper00
  `/data02/jaxan/runs/causalcache-running-best-tune-evaluation-v2-650afbe/result.json`；
- selections：Hyper00/Hyper01
  `/data02/jaxan/runs/causalcache-running-best-tune-selectors-v2-650afbe`；
- schedule：Hyper00/Hyper01
  `/data02/jaxan/runs/causalcache-running-best-tune-schedules-v1-650afbe`；
- merged terminals：Hyper00
  `/data02/jaxan/runs/causalcache-running-best-tune-labels-v1-650afbe`；Hyper01 同路径保留 partitions
  4--7 的完整 progress；
- checkpoints：Hyper00/Hyper01 的 training roots 与
  `/data02/jaxan/runs/causalcache-running-checkpoint-snapshot-*epoch1-59cfd72`；
- intended HF dataset：`gavinlaw/causalcache-set-utility-variable-history-mobile`，planned artifact/tag=
  `set-utility-decision-v2-epoch1-tune-043d3e1`；状态=`PENDING_HF_UPLOAD`；
- intended HF model：`gavinlaw/causalcache-set-utility-predictors-mobile`，planned tag=
  `set-utility-decision-v2-epoch1-650afbe`；状态=`PENDING_HF_UPLOAD`。

