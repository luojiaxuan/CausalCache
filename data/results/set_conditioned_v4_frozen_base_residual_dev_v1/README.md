# Set-conditioned v4 Frozen-base residual development result

状态：`NO_DEVELOPMENT_EVIDENCE_FOR_FROZEN_BASE_RESIDUAL`。

这是 non-mainline、consumed-development 结果。Fresh16 不是 holdout/test；confirm20、legacy dev-5、
matched-NLL、closed-loop、policy forward 与 GPU operation 均为 0。本结果不改变另一个 worktree 中的
restoration-guided independent gate 论文主线。

## 结论

- v4 在当前 residual parameterization/supervision 与冻结契约下检验了 v3 的一个重要歧义：5 个历史 strongest
  independent base 全程冻结，训练前后 checkpoint bytes 与 model-state SHA 均完全一致，zero residual 在
  Fresh16 feature-only 阶段 48/48 重放历史 base；
- Formal58 选择 LR=`1e-3`、epochs=`[66,0,17,40,0]`。但 safe residual OOF 没有接受任何 pair candidate，
  其 normalized/raw ratio 与 frozen base 完全相同；unguarded 略低；
- Fresh16 上 safe residual 与 frozen base 在 48/48 state 完全相同，mean normalized/raw/n=4 delta 均为 0，
  paired bootstrap 90% interval 为 `[0,0]`，五项冻结判据全部失败；
- unguarded residual 只改变 1/48 个 n=4 state，mean normalized delta=`-9.076176885394025e-07`，没有正向
  trajectory；safe guard 正确回退；
- 因此当前实验不支持“v3 失败仅由 joint training 先损坏 singleton base 导致”这一解释；这不外推到其他
  residual 设计。当前 parameterization/supervision 没有提供继续 set-conditioning 的开发证据；不调 guard、
  不再使用同一 Formal58/Fresh16 做 v5，也不打开 confirm20。

## Source of Truth

- Machine-readable summary：[`summary.json`](summary.json)
- Git Source-A：`ea3d53bd87e3d0af8d03ad8ec599a4d6ed99dd7e`
- Git Execution-B：`8e8bfd9cb65a3a943b5fdfa082d9fb39d49ccb6c`
- Private HF model：[immutable revision](https://huggingface.co/gavinlaw/causalcache-set-conditioned-v4-frozen-base-residual-exploration-mobile/tree/1641b90a4ebb05037e4710738a78e2c79f81cdc3)
- Private HF development artifacts：[immutable revision](https://huggingface.co/datasets/gavinlaw/causalcache-set-conditioned-v4-frozen-base-residual-development-mobile/tree/d38513e4c3a22e02dcef1277472c5adf8667e73f)
- Publication tag：`set-conditioned-v4-frozen-base-residual-development-v1`（本次验证时解析到上述 immutable
  revisions）

两个 HF repo 均已从 tag 解析到上述 immutable revision，在新目录使用 `force_download=True` 重取；model 的
9 个 manifest payload 与 development 的 4 个 manifest payload 均逐文件复核 SHA-256、size 和本地 sealed
bytes。report-only validation 连续两次 stdout byte-identical，SHA256 均为
`77bf39d82881ba0b25e0b28382d6a615b58f753600713d083a6a99dbe07ae750`。
