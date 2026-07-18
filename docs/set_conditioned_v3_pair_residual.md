# Set-conditioned v3：pair-residual 非主线探索

## 决策与边界

这条分支只回答一个开发问题：在不重启 v1 iterative conditional gate 的前提下，显式学习
pair interaction 是否能缩小 additive selector 与 exact subset oracle 的差距。

- 分支：`luojiaxuan/set-conditioned-v3-pair-residual`；
- 基线 commit：`ed8772eaa1c22794768e99e6cbe51a47f8137ce0`；
- paper 主线仍是在另一个 worktree 执行的 restoration-guided independent gate；
- v1 的 `NO_V2_CONDITIONAL_RESCUE` 永久保留，v3 不能改写它；
- fresh-16 已经被消费，只能叫 consumed development evidence，不能再叫 holdout 或 test；
- confirm-20、legacy dev-5、matched-NLL、closed-loop、raw GUI、policy forward 和 GPU 全部保持零访问。

机器可读 Source-A 是
`code/configs/causalcache_set_conditioned_v3_pair_residual_exploration_v1.json`。本文只解释其设计；
发生歧义时，以机器可读协议和 fail-closed validator 为准。

## 为什么 v3 不是 v1 调参

v1 的 iterative conditional student 在每轮预测 marginal，再决定 continuation/stop。fresh-16
failure decomposition 已经显示，问题集中在真正受压缩的 `n=4,B=2`，且 completion/stop 占主要
distillation regret。v3 不增加 v1 MLP 的 hidden size，也不移动任何 v1 阈值，而是直接学习可枚举的
set utility：

\[
U(S)=D(\varnothing)-D(S),
\]

\[
\widehat U(\{i\})=\widehat U_i,
\qquad
\widehat U(\{i,j\})=\widehat U_i+\widehat U_j+\widehat R_{ij}.
\]

标签直接来自已有完整 restoration table：

\[
U_i=U(\{i\}),\qquad
R_{ij}=U(\{i,j\})-U(\{i\})-U(\{j\}).
\]

在 `n<=4,B=2` 时，只需枚举空集、所有 singleton 和所有 pair，最多

\[
1+4+\binom 42=11
\]

个 learned-utility 候选，不发生 frozen policy rerun，也没有 sequential stop error。

## 冻结模型

每个 event 沿用 v1 的 200 维 independent input。共享 encoder 为：

```text
Linear(200, 64) -> GELU(approximate="none")
```

singleton head 是 `Linear(64,1)`。对按 `event_step_i < event_step_j` 排序的 pair，残差输入为：

```text
[z_i, z_j, z_i * z_j, abs(z_i - z_j), q64]  # 320 dims
```

pair head 是：

```text
Linear(320, 64) -> GELU(approximate="none") -> Linear(64,1)
```

这里使用 chronological canonical order，不是声称 interaction 具有因果方向；它只是利用 GUI event
天然时间顺序给出确定、可复现的参数化，并允许模型表达“先发生 A、后发生 B”与相反顺序的不同语义。

## 训练与选择

formal-58 是唯一训练和 train-only model-selection source。每个 fold 只能用 fold-train 部分拟合
RMS scale：

\[
s=\max\left(
\sqrt{\frac{
\mathbb E_{eq}[U_i^2]+
\mathbb E_{eq}[U_{ij}^2]+
\mathbb E_{eq}[R_{ij}^2]}{3}},
10^{-12}
\right).
\]

每一类中的 `equal` 都按 trajectory、state、该类 row 依次等权。held-out fold 和 fresh-16 不得参与
scale 拟合。

冻结 loss 为：

\[
L=\frac{L_{single}+L_{pair}+L_{residual}}3+0.25L_{rank}.
\]

三个 regression term 都是对 RMS-scaled raw target 的 `SmoothL1(beta=1)`。`L_rank` 对每个 state
的空集、singleton、pair 进行全部无序比较，raw utility 差在 `1e-12` 内视为 tie 并跳过，其余使用
`softplus(-sign(y_l-y_r)(\hat y_l-\hat y_r))`。ranking 依次按 trajectory、state、untied comparison
等权。

不加入 normalized regression：fresh failure decomposition 中小 denominator 已产生约 85 倍的极端
放大，这是本轮明确拒绝的 ablation。也不额外加入 sign loss，因为 all-feasible-set ranking 已经覆盖
empty-vs-single 和 single-vs-pair 的符号次序，再加一次会重复计权。

超参数固定为：

- learning rate：`3e-4`、`1e-3`；
- seed：`0..4`；
- maximum epochs：500；
- patience：50；
- bootstrap：trajectory paired、10,000 resamples、90% percentile、seed `271828`。

单 seed 的 early stop 和 LR 选择只使用 unguarded pair-residual selector 在 OOF `n=4,B=2` 上的：

\[
\min\left(
\frac{U_{raw}}{U_{raw}^{exact}},
\frac{U_{norm}}{U_{norm}^{exact}}
\right).
\]

五个 final seed checkpoint 全部完成后，才形成以下三个 ensemble variant：

| variant | 定义 |
| --- | --- |
| `additive` | 五 seed singleton mean，pair residual 固定为 0 |
| `unguarded_pair_residual` | 五 seed singleton mean 加 pair-residual mean |
| `safe_pair_residual` | 仅在 seed argmax agreement 和 pair-vs-base margin 两个检查都通过时采用 pair candidate；否则退回 additive |

令 `S_pair` 为 five-seed mean unguarded utility 的 argmax，`S_base` 为 five-seed mean additive
utility 的 argmax。若两者相同则直接返回该 subset；否则只有同时满足以下条件才切换到 `S_pair`：

1. 至少 4/5 个 individual-seed unguarded argmax 等于 `S_pair`；
2. 至少 4/5 个 seed 上
   `F_m(S_pair) - F_m(S_base) > 0`，这里是严格大于零，不使用 epsilon。

否则返回 `S_base`。每个 state 必须封存 `pair_candidate`、`base_candidate`、五个 seed 的 argmax、
五个 pair-minus-base margin、两个通过计数和最终是否采用 pair candidate，不能只保存最终 subset。

learned subset enumeration 的 tie-break 固定为：utility 高者优先、再选 cardinality 小者、再选
event step tuple 字典序小者；不使用 epsilon。`safe_pair_residual` 是本轮 primary selector，但不参与
单 seed early stopping。

## Label-blind firewall

执行拆为 `train-seal` 和 `evaluate` 两个阶段。

`train-seal` 可以读取 formal-58 feature/label cache，也可以读取 fresh-16 feature states，但不能打开或
解析 fresh-16 label。它必须先完成：

1. 五个 seed 的 `.safetensors`；
2. formal training report；
3. fresh-16 上三个预冻结 variant 的 feature-only predictions；
4. `label-blind-seal.json`，状态必须为
   `SEALED_LABEL_BLIND_SET_CONDITIONED_V3_V1`，并绑定以上全部 bytes、Source-A commit 和 contract hash。

预下载的 label transport bytes 不等于 semantic access；但 `evaluate` 必须先验证 seal byte-identical，
再持久化唯一 label-access claim，之后才能 open/parse/decode label 或 join。`evaluate` 不得 retrain、
repredict 或修改任何 sealed selection。

状态顺序固定为：

```text
source_a_identity
-> formal_input_identity
-> formal_training_completion
-> checkpoint_completion
-> fresh_feature_only_identity
-> fresh_feature_only_prediction_completion
-> label_blind_seal
-> label_access_claim
-> fresh_label_and_historical_reference_identity
-> one_time_consumed_development_join
-> development_report_completion
-> immutable_replay_completion
```

## fresh-16 如何解释

一次 label join 要同时、完整报告：

- `additive`、`unguarded_pair_residual`、`safe_pair_residual`；
- 历史 sealed independent selector；
- exact subset oracle；
- raw 和 normalized recovery；
- `n=2/3/4` 分层；
- safe fallback rate；
- `safe_pair_residual - additive` 的 trajectory-paired bootstrap。

允许的结论只有：

- `PROMISING_DEVELOPMENT_SIGNAL_FOR_A_SEPARATELY_FROZEN_FUTURE_STUDY`；
- `NO_DEVELOPMENT_EVIDENCE_TO_CONTINUE_SET_CONDITIONING`。

`PROMISING` 不是事后主观判断。只有 `safe_pair_residual - additive` 同时满足以下五项才输出：mean
normalized delta 至少 `0.01`、normalized trajectory-paired bootstrap 90% lower 严格大于 0、mean raw
delta 严格大于 0、`n=4` mean normalized delta 严格大于 0、normalized delta 为正的 trajectory 至少
`8/16`。任一项失败都输出 `NO_DEVELOPMENT_EVIDENCE_TO_CONTINUE_SET_CONDITIONING`。

即使数字很好，也不能写 `GO`、`CONFIRMED`、`HELDOUT_PASS` 或 `TEST_PASS`，更不能因此打开
confirm-20。若要进入 confirm，必须另建版本化 study、重新冻结模型/阈值/数据契约，再由用户单独决定。

## Source of Truth 与独立 namespace

输入都绑定 immutable HF revision 和逐文件 SHA-256。v3 的本地 staging 只允许写：

```text
/data/experiments/causalcache/set-conditioned-v3-pair-residual-exploration-v1
/data/artifacts/causalcache/set-conditioned-v3-pair-residual-exploration-v1
```

计划中的 canonical HF 目标为：

- model：`gavinlaw/causalcache-set-conditioned-v3-pair-residual-exploration-mobile`；
- development dataset/report：`gavinlaw/causalcache-set-conditioned-v3-pair-residual-development-mobile`；
- tag：`set-conditioned-v3-pair-residual-exploration-v1`。

在实际 publication 完成前，上述目标状态是 pending；本地文件仍只是 staging，不得被写成 canonical。
本分支不得写 independent paper mainline 的 Git 文件、local namespace 或 HF repo。

## 验证入口

Source-A 静态验证：

```bash
cd code
PYTHONPATH=. python3 -m scripts.validate_set_conditioned_v3_contract \
  --repository-root .. \
  --contract code/configs/causalcache_set_conditioned_v3_pair_residual_exploration_v1.json
```

validator 只读取 contract、Git identity 和 source inventory，不联网、不 import torch、不授权执行。
正式 `train-seal` / `evaluate` 命令必须在 Source-A commit clean 且已 push 后由 runner 执行。
