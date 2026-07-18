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
   `SEALED_LABEL_BLIND_SET_CONDITIONED_V3_V1`，并绑定以上全部 bytes、Source-A commit、Execution-B
   commit、runner-freeze SHA-256 和 contract hash。

`train-seal` 和 `evaluate` 都必须先验证同一个 clean、pushed Execution-B，且验证发生在任何输入读取或
输出写入之前。预下载的 label transport bytes 不等于 semantic access；但 `evaluate` 还必须验证 seal
byte-identical，
再持久化唯一 label-access claim，之后才能 open/parse/decode label 或 join。`evaluate` 不得 retrain、
repredict 或修改任何 sealed selection。

## Source-A / Execution-B lineage

第一次 preliminary Source-A `c0de357096261bcc98d2acef743f197aaf290228` 暴露出一个纯执行契约错误：
runner commit 创建后，`HEAD` 不可能继续等于 Source-A。该 commit 没有被用于数据读取、训练或 HF mutation，
也没有触碰 fresh-16 label 或 confirm-20；它只作为未执行的历史 freeze 保留。

修复后的正式 lineage 固定为：

1. Source-A 包含全部科学代码、config、测试和文档；runner-freeze 文件必须不存在，worktree clean，且
   `HEAD == origin/luojiaxuan/set-conditioned-v3-pair-residual`；
2. 从 Source-A inventory 机械生成 canonical compact JSON + newline 的
   `code/configs/causalcache_set_conditioned_v3_pair_residual_runner_v1.json`；
3. Execution-B 必须是 Source-A 的 direct single-parent child，唯一 tree diff 必须是以 mode `100644`
   新增该 runner freeze；
4. B validator 要求显式 Source-A/B commit、clean worktree、local tracking 和 live remote 都等于 B，并验证
   base ancestry、runner committed blob、contract SHA 和 Source-A inventory SHA；
5. 任何 B validation failure 都必须发生在 `_read`、training、label claim 或 evaluation 之前。

runner freeze 只绑定 CPU-only execution identity、formal/fresh split role 与零 confirm/GPU 权限；它不包含
任何训练结果或 development label，因此单文件 B 仍是机械执行提交。

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

### 首次 evaluation parser failure

首次 formal train-seal 已完成，但 evaluation 在 claim 后首次 decode fresh label，随后因 historical independent
artifact 的 protocol-id contract 写错而 fail closed。实际 SHA-pinned artifact 是
`causalcache_gate_v1_fresh16_evaluation_v1`，失败 parser 错误要求
`causalcache_gate_v1_fresh16_primary_v1`。本次没有生成 development report，也没有 scientific result；证据见
`data/results/set_conditioned_v3_pair_residual_attempt_v1/`。

后续若修复，只能是另行冻结的 parser-only continuation：必须复用原 seal 与 predictions，禁止训练、预测、
selector 或阈值变化，并把首次 decode 与 repair replay 的总次数如实写入结果。该 repair 仍只是 consumed
development diagnostic，不能开放 confirm。

parser-only repair 使用独立机器可读 contract
`code/configs/causalcache_set_conditioned_v3_historical_protocol_parser_repair_v1.json`，SHA256 为
`da28f5851598703d255b5de7010002d00cb9cb7d43541d981910a67165bbbe0a`。它只接受 exact producer protocol
`causalcache_gate_v1_fresh16_evaluation_v1`，不使用 allowlist；其余 canonical bytes、top-level keys、status、
48-state order、5-seed feasibility 和 selection digest 检查全部保留。

repair 重新采用 Source-A / 单文件 Execution-B：A 绑定原 A/B、失败记录、seal、claim、prediction 和全部 source
bytes；B 只能新增 canonical runner freeze。runner 只有 `evaluate` 与 `validate`，没有 training/prediction
entrypoint。状态链为：

```text
parent seal -> unique parent claim -> parent decode attempt 1 -> protocol mismatch/no metric
-> parser-repair Source-A/B freeze -> parent byte revalidation
-> exact historical parse -> repair label replay attempt 2
-> one versioned development report -> byte replay
```

最终 report 不保留容易误解的 `fresh_label_decode_count=1`，而是显式记录 claim total=1、semantic decode
attempt total=2、historical parse attempt total=2、training total=1、prediction generation total=1、report
completion total=1；所有 confirm/GPU/policy/legacy/matched-NLL/closed-loop count 均为 0。原
`fresh16-development-report.json` 永久保持不存在，repair 只创建
`fresh16-development-report-parser-repair-v1.json`。

parser-repair v1 的 pre-label dry-run 又暴露出 positional-order contract 错误：historical 与 feature 都有相同的
48 个唯一 state-id，但前者按字典序、后者按 trajectory roster 顺序，因此不能用 `zip` 断言同序。v1 在第二次
label decode 前停止，累计 semantic decode 仍为 1。后续 v2 只允许改成 exact unique-state-id join，同时必须
继续验证两边 set equality；不能放松任一 record/seed/selection-digest 检查。失败证据见
`data/results/set_conditioned_v3_parser_repair_v1_attempt/`。

v2 overlay contract 为
`code/configs/causalcache_set_conditioned_v3_historical_protocol_parser_repair_v2.json`，SHA256
`1da7073c55437afdc8e4ca85dc59a414c6ed331619828210af54347f80b63ce4`。它不接受 arbitrary reorder：
feature 与 historical 两边都必须恰好 48 个唯一 state-id、set 完全相同，然后才按 id join；每条 record 的
schema、ensemble selection、5-seed selection feasibility 与全局 selection digest 继续逐项验证。v2 runner
依然没有 train/predict 入口，并在 exact historical join 全部通过后才调用 label loader。

v2 使用独立 Source-A/单文件 Execution-B 和独立 report
`fresh16-development-report-parser-repair-v2.json`；parent report 与 v1 repair report 必须在执行前都不存在。
access accounting 固定为：父 attempt label decode=1、v1 repair dry-run label decode=0、v2 repair replay=1、
总计=2；historical parse attempts 为 parent/v1/v2 各 1、总计=3。无论 metric 如何，fresh-16 仍是
consumed development，不能升级成 confirmatory evidence。

为保持上述 parser count，v2 不再对真实 historical artifact 做额外 parser dry-run；执行前只验证文件 bytes/SHA，
随后由 evaluate 唯一解析一次。validate 必须接收 evaluate 当场输出的 report SHA，并同时绑定 repair
Source-A/Execution-B/runner、parent seal、顶层与嵌套 authority counters；它不再次读取 label。

输入都绑定 immutable HF revision 和逐文件 SHA-256。v3 的本地 staging 只允许写：

```text
/data/experiments/causalcache/set-conditioned-v3-pair-residual-exploration-v1
/data/artifacts/causalcache/set-conditioned-v3-pair-residual-exploration-v1
```

canonical HF 目标已闭合为：

- model：`gavinlaw/causalcache-set-conditioned-v3-pair-residual-exploration-mobile`
  @`79b53aaa6017458d92e626bc4801d3b3bef9facd`；
- development dataset/report：`gavinlaw/causalcache-set-conditioned-v3-pair-residual-development-mobile`
  @`bcf7c7c8057f60b655736b5c63ec43551b3e3121`；
- tag：`set-conditioned-v3-pair-residual-exploration-v1`。

两个 private repo 的 tag 都已解析到上述 commit，manifest 中 8/7 个 payload 已 fresh-download 并逐项复验
SHA256/size。本地文件仍只是 staging，不是 canonical。本分支不得写 independent paper mainline 的 Git 文件、
local namespace 或 HF repo。

## 最终 consumed-development 结果

parser-repair v2 的 Source-A 为 `785b542a9d538bbfc80de95d5f8d956a04234e48`，唯一 Execution-B 为
`e9262338ee96f404ca37babaa2f29a24f6c6cb17`。唯一 evaluate 与 label-free validate 的 stdout byte-identical，
SHA256=`e2e1367d710b052670e0281de756cb5ecbb99c021b68afa6d10cd9b1df6219b5`。冻结结果为
`NO_DEVELOPMENT_EVIDENCE_TO_CONTINUE_SET_CONDITIONING`：

- safe residual 在 fresh16 48/48 state 都与 additive 相同，`used_pair_candidate_count=0`；
- unguarded residual 只改变 3/48 state，1 个改善、2 个恶化，normalized delta=`-0.285271`；
- safe 相对 historical v1 independent 的 normalized delta=`-0.122019`；
- Formal58 虽有显著 oracle interaction headroom，但 safe residual 在 OOF 也只接受 1/58 state，并略差于
  additive。

所以 v3 停止，不在已消费 fresh16 上调 4/5 guard，也不访问 confirm20。完整轻量结果与 failure decomposition
见 `data/results/set_conditioned_v3_pair_residual_dev_v1/`；independent paper mainline 不受该负结果影响。

## 验证入口

Source-A 静态验证：

```bash
cd code
PYTHONPATH=. python3 -m scripts.validate_set_conditioned_v3_contract \
  --repository-root .. \
  --contract code/configs/causalcache_set_conditioned_v3_pair_residual_exploration_v1.json
```

validator 只读取 contract、Git identity 和 source inventory，不联网、不 import torch、不授权执行。
随后必须机械创建、commit 并 push 唯一 runner-freeze Execution-B；正式 `train-seal` / `evaluate` 都只能在
该 clean pushed B 上执行。
