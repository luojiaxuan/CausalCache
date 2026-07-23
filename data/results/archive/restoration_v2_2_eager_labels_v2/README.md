# Restoration v2.2 exact labels

## 结论

唯一 replacement attempt `restoration-v2-2-eager-labels-v2` 已在固定双 H200、23/22 parity、microbatch 1 下
完成 45/45 states，正式结果为 `PASS_RESTORATION_V2_2_EAGER_LABELS_V2`。本次没有 retry、top-up 或
generation；canonical payload 包含：

| Payload | Count |
| --- | ---: |
| canonical $D(S)$ | 420 |
| deployment conditional-marginal labels | 435 |
| primary exact-subset oracle | 45 |
| pair interactions | 465 |
| teacher forwards | 465 |
| KL measurements | 420 |

## Policy-free 科学归约

主预算为两个 event slots。exact-subset oracle 的 mean normalized recovery 为：

| Split | States | Mean normalized recovery |
| --- | ---: | ---: |
| `v2_label_train` | 30 | 0.9028414853 |
| `v2_development` | 15 | 0.8148528628 |
| Overall | 45 | 0.8735119444 |

- 44/45 states 获得正 oracle utility；oracle 选择 cardinality 0/1/2 的 state 数为 1/3/41；
- deployment marginals 严格正/负为 360/75，22/45 states 至少包含一条负 marginal；
- pair interactions 严格正/负为 188/277；
- raw utilities 严格正/零/负为 338/45/37，其中 45 个零值来自按定义为零的 empty-coalition utility。

负 marginal、负 raw utility 与双向 interaction 表明 frozen-policy restoration set function 不是简单的独立、
单调 event-score 问题。这支持下一阶段冻结并训练 set-conditioned iterative gate；它不证明 learned gate 已接近
oracle，也不构成 matched-NLL 或 closed-loop success 证据。

## 执行与 artifact identity

- source/packaging Git commit：`5ae40d4aed4eb20b931216776b379bc6ae55629d`；
- run-contract SHA256：`b78ca1e70652c7eef68efc3472adf78cec4510e507a0c00cfcee2c2cf05285d9`；
- execution window：`2026-07-16T17:13:13.360318Z`--`2026-07-16T17:27:46.202730Z`；
- host/runtime：`hyper00` / `node-radixark-16-0000`，physical GPU 0/1 两张 NVIDIA H200，Python 3.12.3，
  PyTorch 2.11.0+cu130，CUDA 13.0，cuDNN 91900，Transformers 5.6.0，BF16 eager attention，seed 0，
  TF32 off；不声称 strict CUDA determinism；
- container：`39749a3bd0875f3c15216211f875220a4934fe62313487b537c1116e82040ff0`，image digest
  `sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- raw deterministic USTAR：101 files / 3,747,840 bytes，SHA256
  `99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e`；
- raw tree SHA256：`c5104594b0741810f3d49d007a63a74f16ee4236dd137d1dea92b9373065c45f`；
- private HF dataset：
  [`gavinlaw/causalcache-restoration-labels-mobile@8f6baae5`](https://huggingface.co/datasets/gavinlaw/causalcache-restoration-labels-mobile/tree/8f6baae5c0b23b08915fa1b0fb848dd519b4c8db)；
- tag：`v2.2-eager-train-dev-exact-v2`，已验证解析到上述 immutable revision；
- HF upload：`2026-07-16T17:30:43.268141Z`--`2026-07-16T17:30:48.243968Z`；immutable fresh download：
  `2026-07-16T17:31:07.161148Z`--`2026-07-16T17:31:09.087308Z`；tag 创建：
  `2026-07-16T17:31:33.102188Z`--`2026-07-16T17:31:33.479281Z`；
- fresh archive 是非 symlink、与 source 位于不同 inode，已重新通过完整 raw reducer，并逐 byte 相同。

机器可读 artifact binding 见 [`artifact.json`](artifact.json)，科学归约见 [`summary.json`](summary.json)。raw labels
不进入 Git。

## 边界与下一步

本结果只闭合 `v2_label_train` + `v2_development` 上的 offline exact oracle 与 conditional-marginal labels。
gate training、matched-NLL、closed-loop evaluation 和 confirm 均未运行；confirm 仍 locked。下一步先冻结独立的
set-conditioned gate-training/evaluation contract，再运行任何训练或效果评估，不直接访问 confirm/test。
