# Set-conditioned v4：Frozen-base residual 最终开发实验

## 状态与唯一问题

本实验是隔离于 paper 主线的最后一次 set-conditioning 开发 ablation。它只回答：

> 当历史最强 learned independent gate 的 checkpoint、event encoder、singleton score 和选择逻辑完全冻结时，
> 一个只训练 pair correction 的 student 能否稳定提高 set utility？

它不改写 v1 与 v3 的 `NO-GO`，也不声称 independent selector 最优。Fresh16 已被多次消费，本实验中的
Fresh16 结果只能称为 consumed development evidence，不能称为 holdout、test、confirmation 或 paper GO。
confirm20、legacy dev-5、matched-NLL、closed-loop、policy forward 和 GPU 仍保持零访问。

机器可读 Source-A 是
`code/configs/causalcache_set_conditioned_v4_frozen_base_residual_development_v1.json`。正式执行前，主任务必须在
所有 Source-A 文件完成后机械冻结 config SHA-256。本 Source-A 的固定值为
`474df3cfebad7b3a6de9a31769650c1695a0939008f542ec4fd8f31db2aa6c3f`；任何 config byte drift 都会令
Source-A/Execution-B validator fail closed。

## 为什么需要 frozen base

v3 的 additive base 在 Fresh16 上只有 `0.590359` mean normalized recovery，而历史 independent gate 是
`0.712379`。因此 v3 只能说明 joint singleton/pair model 没有稳定利用 interaction，不能排除 multi-task
interference 先损坏 singleton base 的解释。

v4 绑定 private HF model
`gavinlaw/causalcache-gate-v1-formal58-selector-mobile@23f6786075c7bff91f93fd7e8a878e070efb72a9`
中的五个 independent checkpoint。所有 base parameter 均 `requires_grad=False`，optimizer 只能包含 residual
head；训练前后的 checkpoint bytes 与 canonical model-state SHA-256 必须完全相同。

## 同量纲 residual correction

历史 independent gate 输出的是 normalized budget-conditioned projected marginal，而不是 raw singleton
utility。直接把 v3 的 raw residual 加到它上面会发生量纲错误。因此 seed `m`、chronological pair `(i,j)` 的
训练目标固定为：

\[
r^*_{m,ij}
=
\frac{U(\{i,j\})}{D(\varnothing)}
-\left(g^m_i+g^m_j\right),
\qquad D(\varnothing)>10^{-12}.
\]

这是 `set-utility correction residual`。它可以同时吸收 pair interaction 与 frozen base prediction error，
因此是对 interaction-aware student 更有利的检验；论文不得把它冒充纯二阶 interaction estimator。

每个 seed 从对应 frozen base 的第二个 GELU 读取 `z_i in R^88`，构造：

```text
[z_i, z_j, z_i * z_j, abs(z_i - z_j), q64]  # 416 dims
Linear(416,64) -> GELU(approximate=none) -> Linear(64,1)
```

共 26,753 个 trainable parameter。末层 weight 与 bias 全零初始化，因此 epoch 0 对所有 pair 严格输出零，
选择器严格退化为 frozen independent base。

## Formal58 cross-fitting

Formal58 是唯一训练与 model-selection source。不能拿 full-fit base 给 OOF held-out fold 打分，否则 residual
head 的 held-out representation 已经由包含该 fold label 的 base 产生。

对每个 seed 和 fold，先按历史 frozen independent contract 重放只使用 fold-train 的 base：LR 固定
`3e-4`，seed `0..4` 的 epoch 固定为 `60/51/6/56/54`。重放必须在 `1e-12` 绝对误差内复现历史五个
raw OOF ratio 及其 mean；这里不声称逐字段复现完整历史 OOF report。随后立即冻结 base；
residual head 只能读取该 fold base 的 embedding 和 score。final fit 才加载原始五个 full-Formal58 immutable
checkpoint。

Residual OOF 固定为：

- folds：`12/12/12/11/11` trajectories；
- LR：`3e-4 / 1e-3`；seed：`0..4`；
- epoch 0 进入 model selection；maximum 500、patience 50；
- improvement 必须严格超过 `1e-4`，否则保留更早 epoch；
- 单 seed metric 是 `n=4,B=2` 上 raw/exact 与 normalized/exact ratio 的较小值；
- 五 seed mean 选择 LR；差不超过 `1e-4` 时取 `3e-4`。

Loss 只有 pair correction SmoothL1 与系数 `0.25` 的 trainable set-ranking。ranking 只保留至少一侧是 pair
的 untied feasible-set comparison；empty/singleton-only comparison 对 residual head 没有梯度，因此不重复计权。

## 三个固定 selector

必须同时报告：

1. `frozen_base`；
2. `unguarded_frozen_base_residual`；
3. `safe_frozen_base_residual`，也是 primary selector。

Safe guard 原样继承 v3，不允许新阈值。若 ensemble unguarded candidate 与 base 不同，只有在至少 4/5 个
seed 的 argmax 等于该 candidate，且至少 4/5 个 seed 的
`total(candidate)-total(base)>0` 时才采用；否则回退 frozen base。比较严格大于零，不使用 epsilon。

## Label-blind seal 与 Fresh16 重读

`train-seal` 必须先完成五个 residual checkpoint、Formal OOF report、zero-residual/base invariant report 和
Fresh feature-only predictions，再生成 label-blind seal。此阶段不得读取 Fresh label 或 historical independent
decision artifact。zero-residual invariant 只比较同一 frozen base logits 下的 exhaustive additive selector 与
canonical `select_independent`。

Seal 后先落盘一次 versioned consumed-development claim，随后 evaluate 才允许读取 Fresh label 与 historical
decision artifact。新访问上限为：一次 claim、一次 Fresh-label semantic decode attempt、48 decoded state、一次
join 与一份 development report。实际顺序固定为先完整 parse historical artifact，再 decode Fresh label；因此
historical parser failure 会消费 claim，但不会消费 Fresh-label decode attempt。任何 repair 都必须另立版本并分别
累计 claim、historical parse 与 Fresh-label decode，不能静默重跑。post-report validate 只重读 report，不得重开
seal、prediction、history 或 label。

历史计数不能混用单位：v1 primary 封存的是 48 decoded rows；v3 封存的是一次 claim 与两次 semantic decode
attempt。v4 成功时新增一次 claim、一次 attempt 和 48 rows。只可在明确标注
`accounting_scope=v3_to_v4_lineage` 时写累计 claims=`2`、attempts=`3`，不能称为 project-global count。

## Primary route

Primary contrast 固定为：

```text
safe_frozen_base_residual - frozen_base
```

沿用 v3 的五项判据，不因 v4 结果修改：

1. mean normalized delta 至少 `0.01`；
2. 10,000 次、seed `271828`、trajectory-paired 90% percentile bootstrap lower 严格大于 0；
3. mean raw delta 严格大于 0；
4. `n=4` mean normalized delta 严格大于 0；
5. normalized delta 为正的 trajectory 至少 `8/16`。

全部满足也只能输出
`PROMISING_DEVELOPMENT_SIGNAL_FOR_A_SEPARATELY_FROZEN_FUTURE_STUDY`；任一失败输出
`NO_DEVELOPMENT_EVIDENCE_FOR_FROZEN_BASE_RESIDUAL`。无论哪条 route，都不能自动开放 confirm20，也不能再以
同一 Formal58/Fresh16 调出一个 v5。

## Source-A / Execution-B

正式 Source-A 必须位于 `luojiaxuan/set-conditioned-v4-frozen-base-residual`，并以 v3 final commit
`a50245b0edc681ec3c7f9ec06277d789586d32a8` 为 ancestor。Execution-B 必须是 Source-A 的 direct
single-parent child，唯一 diff 是新增 mode `100644` 的：

```text
code/configs/causalcache_set_conditioned_v4_frozen_base_residual_runner_v1.json
```

Runner freeze 机械绑定 Source-A commit、完整 source inventory SHA-256、contract SHA-256、CPU-only runtime、
Formal/Fresh role、frozen base revision 与 confirm20=`0`。Execution-B validation 必须发生在任何 input read、
model load 或 output mutation 之前。

## Git/Hugging Face 归档闭环

科学状态机终止于 report-only self-consistency validation；发布不是新的科学状态，也不得改变 report。之后按固定
顺序归档：

1. private model repo 保存五个 residual checkpoint、model metadata、Formal OOF report、base invariant report
   与 label-blind seal；
2. private dataset repo 保存 Fresh feature-only predictions、label-access claim、consumed-development report 与
   label-blind seal；
3. 两仓都写 README/card 和 exact-file manifest，使用 config 中冻结的同一 tag；
4. 解析 tag 到 immutable revision，在全新目录 fresh-download，并逐文件复核 SHA-256/size；
5. 最后才把 immutable revisions、manifest hashes、科学指标与 verdict 写入 Git compact result、README 和
   `docs/progress.md`。

这一步由科学 runner 之外的只读归档流程完成；未完成 fresh-download replay 时，不能把 v4 milestone 标为完成。
