# Independent confirm-20 oracle-independent failure decomposition v1

## 目的与边界

本步骤只回答一个已经消费的 confirm-20 上的 failure-decomposition 问题：如果直接使用完整
`D(S)` 表产生的真实 independent projected target（记为 oracle-independent `J`），它能否胜过同一
confirm 上最强的廉价 heuristic；以及 `J` 与已经冻结的 learned independent selector `I` 之间还剩多少
distillation/generalization gap。

这不是新的 confirm、不是新的 GO gate，也不重新分类父结果
`NO_GO_INDEPENDENT_CONFIRM`。confirm-20 已经消费，不能再作为任何 v2 的 untouched holdout。此步骤禁止
policy forward、restoration teacher forward、gate 训练、checkpoint/model load、GPU、closed-loop、matched-NLL
与 sealed AndroidWorld test；也不能据此自动授权其中任何一项。

审计说明：用户先给出了 `J` 对 OCR/RGB 与 learned `I` 的 Case A/Case B 问题；随后在 executable Source-A
提交前发生过一次只读 exploratory 数值重放。因此本步骤不声称 code-level blind preregistration。正式
Source-A 只冻结 canonical replay、既有 estimand、用户已提出的分叉及 immutable publication；这一事实必须
保留在 config、report 与 result summary 中。

## 只读输入

唯一父输入是 private HF dataset
`gavinlaw/causalcache-independent-confirm20-mobile` 的 immutable report commit
`a0b408e58d629299be334a74ecbd0ec2fa2ed1fc`、tag `independent-confirm20-v1`。只允许 force-download：

- `independent-confirm20/v1/report/bundle-manifest-v1.json`，2,261 bytes，SHA256
  `ccc996283b89f41db77c06f320e3b1ac88f3933bc19cfeea4c8b5b508ae4b351`；
- `independent-confirm20/v1/report/raw-state-records-v1.jsonl`，1,388,707 bytes，SHA256
  `208fb36abb36b5e82eadcc8dac4ed20c5a030919902afe3e5b69cf6cad76a52a`；
- `independent-confirm20/v1/report/fixed-report-v1.json`，36,666 bytes，SHA256
  `969729deb66e610846694b0fa1139c6b47f3927db7a58b50367f8ed93365198b`。

raw records 提供 20 个 state 各自完整的 16-row `D(S)`；fixed report 提供 frozen learned `I` 与
OCR/RGB selections。所有 exact、`J`、`I`、OCR/RGB utility 必须从同一张已校验 `D(S)` 表重新计算，不能直接
拼接 summary aggregate。父 repo 的 remote mutation count 固定为 0。

## 冻结 estimand

每个 state 固定 `n=4,B=2`。对 event `j`：

\[
J_j=\frac{1}{2}\left[D(\varnothing)-D(\{j\})
+\frac{1}{3}\sum_{i\ne j}\left(D(\{i\})-D(\{i,j\})\right)\right].
\]

该定义逐字复用此前 `gate_v1_data` 与 fresh-16 failure decomposition 的 independent raw target。selector
只保留严格正分数，按 score 降序、event step id 升序打破精确同分，取 top-2，最终 coalition 按 event id
排序。exact subset oracle 在所有 `|S|<=2` 的集合中最小化 `D(S)`，依次按 exact float distance、更小集合、
lexicographically lower event ids 打破同分。

primary 口径是 20 trajectories 上 raw utility 的等权 sum/mean，与父 confirm 一致。fixed-20 zero-imputed
normalized recovery 仅作描述，不改变路线。paired bootstrap 固定为 trajectory unit、10,000 resamples、seed
`271828`、90% percentile interval、Hyndman-Fan type 7，同时报告 positive/equal/negative trajectory count。

## 冻结 A/B 分叉

最强 heuristic 仍由父 confirm 的 raw utility sum 决定；这里重点报告 exact、OCR/RGB、`J` 与 learned `I`。
Case B 只有同时满足以下四项才成立：

1. `mean_raw(J - OCR) > 0`；
2. trajectory-paired 90% bootstrap lower `> 0`；
3. positive trajectory count `>=12/20`；
4. `mean_raw(I - OCR) < 0`。

- `mean_raw(J-OCR)<=0`：情况 A，状态为 `CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB`，停止 rescue
  当前 independent objective；即使 `J>I`，student 也不是唯一 blocker。
- 四项全部成立：情况 B，状态为 `CASE_B_TEACHER_VALID_STUDENT_DISTILLATION_GAP`。它只支持另立
  representation/distillation study；必须使用新数据、新合同和新 untouched holdout。
- aggregate 为正但 Case B robustness 未全部成立：状态为
  `INCONCLUSIVE_ORACLE_INDEPENDENT_CONFIRM_DECOMPOSITION`，同样不能 rescue 当前路径。

## Source-A → Execution-B → result

Source-A 必须在 clean、已 push、`HEAD==origin/main` 的 `main` 语义基线上验证，且 runner-freeze JSON 尚不
存在。Execution-B 必须是 Source-A 的唯一 direct child，tree diff 只能新增 runner-freeze JSON；随后才允许
CPU-only selective replay。正式 child 使用独立 private HF dataset
`gavinlaw/causalcache-independent-confirm20-failure-decomposition-mobile`，tag
`independent-confirm20-failure-decomposition-v1`，只发布：

- `independent-confirm20-failure-decomposition/v1/state-decomposition-v1.jsonl`；
- `independent-confirm20-failure-decomposition/v1/failure-decomposition-report-v1.json`；
- `independent-confirm20-failure-decomposition/v1/bundle-manifest-v1.json`。

child 不复制父 `D(S)` 表，只保存 derived 20-state diagnostics 与父 immutable identity。正式结果完成后，本节
补充 Source-A、Execution-B、child commit/tag object、逐文件 SHA、完整 argv/runtime 和最终 A/B 结论。

## 正式结果：情况 A

Source-A=`d3451db74d7bdd415455b87a91718bbedfcd0884`，唯一 direct-child Execution-B=
`9f20e4b55c5e69ae1b1c26d58cd7c52bdd5e0af3`；config/runner SHA256 分别为
`e522f5cab8bfd359d8a869a5a21eac771ae3e81a204a872d619e51bdf31cf020` 与
`88a98b42f8750067d33285556c6099390328f4c92b99a88bdd60276ebd12b25d`。正式 CPU-only run 与独立
validate 均在 Hyper00 的无 GPU device request container 中完成。

raw utility sum 为 exact=`0.856431`、OCR/RGB=`0.783268`、`J=0.765311`、learned `I=0.703385`。
`J-OCR` raw mean=`-0.000898`，paired 90% interval=`[-0.006370,0.004145]`，positive/equal/negative=
`10/3/7`，所以正式状态为 `CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB`。normalized mean 下 `J-OCR=
+0.033558`，但 normalized 已冻结为 descriptive，不能推翻 raw-primary 结论。

`J-I` raw mean=`+0.003096`，paired 90% interval=`[+0.000477,+0.006533]`，positive/equal/negative=
`11/5/4`。这证明 student 的 feature/distillation/generalization gap 是真实的；但即使完全修复该 gap 并达到
`J`，仍无法超过 OCR/RGB。因此应停止 rescue 当前 independent target，而不是马上换 hidden size 或继续增加
同一 target 的训练数据。

canonical child 是 private HF dataset
`gavinlaw/causalcache-independent-confirm20-failure-decomposition-mobile@31aa5e22c08a3d56bc729fcbc88d790cce4249c0`，
tag object=`a7d6838d32e217162d99a41a7b51939bc2c5d450`。独立 validate 从 commit 与 tag fresh replay exact-three，
remote mutation=`0`、local write=`0`。Git 轻量结果见
[`../data/results/independent_confirm20_failure_decomposition_v1/`](../data/results/independent_confirm20_failure_decomposition_v1/)。
