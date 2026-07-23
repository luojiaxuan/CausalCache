# CausalCache gate v1 预注册

本文件记录第一版 learned gate 在任何 48+16 扩容 policy output、正式训练或 development 评估之前冻结的
训练与判定契约。machine-readable source 是
[`code/configs/causalcache_gate_v1_preregistration.json`](../../code/configs/causalcache_gate_v1_preregistration.json)，
SHA256 为 `37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b`。

本 preregistration 本身的状态仍是 `VALID_CAUSALCACHE_GATE_V1_PREREGISTRATION`；其中 source-only 的
`training_executed=false`、`execution_authorized=false` 历史计数不追改。下游 formal-58 training、10-checkpoint
private-HF model seal 与 fresh-16 primary 已完成 immutable replay。fresh-16 按本文件的 frozen thresholds 得到
selector `NO-GO`、set-conditioning `NO-GO`；旧 dev-5、confirm-20、matched-NLL 与 closed-loop 仍未执行。

## 为什么不能只用旧 10/5

旧 exact-label artifact 只有 10 条 train 与 5 条 development trajectory。它已经足以验证 restoration signal、
interaction 和 selector geometry，但不足以支撑 learned gate 的泛化结论。因此正式分母固定为：

| 角色 | 组成 | Trajectory | State | Conditional edge |
| --- | --- | ---: | ---: | ---: |
| Formal train | 旧 10 + 新 48 | 58 | 174 | 1682 |
| Combined development | 旧 5 + 新 16 | 21 | 63 | 609 |
| Formal GO slice | 仅新 16 | 16 | 48 | 464 |
| Sealed confirm | 固定旧 20 | 20 | 未授权 | 未授权 |

在扩容 labels 变成 immutable artifact 之前，旧 10 只允许做最多两步 optimizer 的 pipeline smoke；旧 5 不得被
trainer 读取。正式模型冻结后，先一次性评估新 16 并形成 GO/NO-GO，再读取旧 5 计算 combined-21
compatibility guard。旧 5 不能选择 architecture、learning rate、epoch、loss、threshold 或 seed。

## 学习目标

Conditional teacher 保留每条部署可达边：

\[
\Delta_j(S)=D(S)-D(S\cup\{j\}),\qquad |S|\in\{0,1\}.
\]

默认 normalized target 为

\[
\Delta_j(S)/\max(D(\varnothing),10^{-12}),
\]

不做 clipping，负 gain 原样保留。若 \(D(\varnothing)\le 10^{-12}\)，该 state 只从 normalized regression
和 normalized mean 中排除；raw target、ranking、sign、raw utility 与计数仍保留。

Independent comparator 每个 state-candidate 只产生一个 projected target：

\[
\frac{1}{2}\left[
\Delta_j(\varnothing)+
\operatorname{mean}_{i\ne j}\Delta_j(\{i\})
\right]/\max(D(\varnothing),10^{-12}).
\]

它不会把同一个 projected target 重复复制到 conditional edges。两类 regression 都按 trajectory→state→
prefix cardinality→coalition→candidate 分层等权；ranking 对严格非 tie 的 candidate pairs 使用 softplus，
总 loss 为 weighted SmoothL1 + `0.25 * ranking`。

## Feature 与模型

所有输入都必须 label-blind。禁止输入 `D(S)`、marginal、oracle subset、role/split、trajectory/state identity 与
policy-vision feature。

- `q64`：instruction 与 current OCR spatial tokens 的 SHA256 signed hashing；
- `h64`：候选 event 的固定八字段 low-fidelity summary，加 candidate post-state OCR spatial tokens；
- `g8`：age、step、argument/text/token counts、screen-change 与 executor-result 的固定归一化几何；
- `phi(S)`：已选择事件 `h64` 的 sum，空集合为全零；
- arrival-time 的 64 维 OCR hash 是固定轻量 feature cache，始终保留并计入输入；policy-vision 只属于独立
  comparator，不进入 gate。

Conditional MLP 输入 330 维、hidden 64、参数量 25,409；independent MLP 输入 200 维、hidden 88、参数量
25,609。两者都是两层 `Linear + GELU` 后接 scalar linear，无 dropout、normalization 或 output activation。
参数量刻意近似匹配，使主要差异保持为 selected-set conditioning。

## 训练与选择

- CPU FP32 full-batch deterministic AdamW；weight decay `1e-4`，global grad norm `1.0`；
- learning rate 仅比较 `3e-4` 与 `1e-3`，seed 固定 `0..4`；
- formal-58 按 salted source-ID order 固定为 12/12/12/11/11 五折；
- 每个 architecture/LR/seed 最多 500 epochs，50 epochs patience；
- epoch 只由 train-only OOF 的 `n=4,B=2` raw-utility ratio 选择，改进必须大于 `1e-4`，tie 取更早 epoch；
- learning rate 只由五个 seed 的 OOF mean 选择；差异不超过 `1e-4` 时取 `3e-4`；
- final fit 在全部 58 条上分别重训五个 seed，development 不参与任何选择。

Conditional deployment 是最多两轮的 set-conditioned greedy：每选一个 event 后重算所有剩余候选；最大预测
marginal 不大于 0 即停止。Independent comparator 是一次性 static positive top-2，不允许重打分。两者都以
五个 seed 的预测均值选择，score tie 取较小 event step。

## 预先冻结的 GO 条件

正式 primary slice 是 fresh-16，按 trajectory equal、再 state equal 聚合。Selector GO 必须同时满足：

- ensemble normalized/raw exact-oracle recovery 都不低于 `0.80`；
- 对 dynamic recent、OCR/RGB v2、policy-vision v3 每个 comparator 的 mean normalized delta 都至少 `0.05`；
- 相对 strongest heuristic 至少 12/16 trajectories 为正，90% paired bootstrap lower bound 大于 0；
- 至少 4/5 individual seeds 的 exact ratio 不低于 `0.75`，seed recovery population std 不高于 `0.08`；
- fresh-16 的 hard `n=3/n=4` raw ratio 不低于 `0.70`；
- 被 selector 实际加入但 true raw marginal 非正的比例不高于 `0.10`。

Set-conditioning GO 也必须同时满足：conditional 相对 independent 的 normalized delta 至少 `0.02`、raw utility
delta 大于 0、至少 12/16 trajectory 为正、90% paired bootstrap lower bound 大于 0、至少 4/5 paired seeds
为正，并且 combined-21 mean delta 不得为负。

只有 `GO_SELECTOR && GO_SET_CONDITIONING` 才能进入下一阶段；该结果本身仍不授权读取 confirm。confirm、
matched-NLL 与 closed-loop 需要另行冻结 post-GO contract。

## 当前允许的 smoke

在 expansion labels 闭合前，只可在旧 10 条上用 seed 0 做最多两个 optimization steps，检查 schema、权重和、
200/330 维输入、finite forward/backward、负 target、conditional rescoring、tau=0 stop、temporary checkpoint
exact reload、independent one-shot 与 deterministic replay。输出不得作为 paper metric，不得保留 checkpoint，
不得读取任何 development semantic content，也不得据此修改冻结契约。

## 验证

```bash
cd code
PYTHONPATH=. python3 -m unittest tests.test_gate_v1_contract -v
PYTHONPATH=. python3 -m scripts.validate_gate_v1_contract --repository-root ..
```

validator 会重新派生 58/21/20 rosters、五折、parameter count、conditional/independent weight sums，验证所有
static section digest 与 access firewall；任一阈值、分母、hash 或 forbidden operation 漂移都会 fail closed。
