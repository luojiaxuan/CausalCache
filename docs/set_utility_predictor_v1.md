# Set Utility Predictor v1：先扩数据，再验证预算泛化

> 当前状态：仅完成 Source-A 方法与防火墙冻结；没有读取新 development 数据，没有生成 label，没有训练
> predictor，也没有执行 closed-loop、matched-NLL 或 sealed test。正式执行必须先补一份独立的 Freeze-B，绑定
> roster、state 数、split identity、训练网格、runner 与 artifact revision。

## 路线调整

当前不再把 exploratory closed-loop 作为下一步。新的顺序是：

1. 用全新的 development trajectories 扩大 restoration `D(S)` 数据；
2. 在 `B<=2` 下完整枚举集合，训练直接预测集合效用的模型；
3. 先证明 learned set utility 在新 development evaluation 上超过 `OCR/RGB`，并与 `J` 和 exact oracle
   定量比较；
4. 再用少量 `B=3/4` exact labels 检验 zero-shot 与 few-shot cardinality transfer；
5. 只有这些离线问题闭合后，才另立协议决定是否恢复 closed-loop。

这次调整不重写历史结论。conditional v1 和 independent v1 在已经消费的 fresh-16/confirm-20 上仍然是
NO-GO；那些结果只能作为 failure analysis，不能重新进入训练、调参或 model selection。

## 核心对象：预算无关的集合效用

对 query state `t` 和高保真事件集合 `S`，teacher target 固定为：

\[
U_t(S)=D_t(\varnothing)-D_t(S).
\]

模型接口是：

\[
\hat U_\theta(q_t,S),
\]

而不是 `U_theta(q,S,B)`。`B` 只进入搜索约束：

\[
\hat S_B=\arg\max_{S\subseteq C_t,\ |S|\le B}\hat U_\theta(q_t,S).
\]

因此需要严格区分：

- utility predictor 是 budget-agnostic；
- selector/optimizer 是 budget-conditioned；
- `|S|` 是输入集合自身的属性，不等于把外部 budget 当模型特征；
- `empty` 永远参与搜索，所以“不再加入 event”通过集合间全局比较实现，不需要学习第二轮 stop threshold。

集合内部送入 frozen policy 时仍按时间顺序排列，以保持合法 GUI prompt；但 predictor 的集合表示、损失与
评估必须对 event permutation 不变。

## 为什么停止把 conditional greedy 当主线

旧 conditional student 逐轮预测 marginal：第一轮选择错误会改变第二轮条件，第二轮又需要学习接近零的
敏感 stop 判据。真实 conditional marginal greedy 在 fresh-16 上能逼近 exact，并不意味着小 student 可以稳定
蒸馏这条 sequential decision chain。

新的方法直接给完整候选子集打分。对 `B=2`，搜索同时比较 `empty`、所有 singleton 和所有 pair；对更大的
`B`，接口不变，只扩大被评分集合的 cardinality。这消除了 selector 内部的 autoregressive error cascade，
也把“模型误差”和“集合搜索误差”分开。

## 新 development 数据防火墙

v1 的 tune 和 development evaluation 只允许使用一套新建的
`causalcache_set_utility_new_development_v1` 数据。训练集允许显式合并旧 formal-58 train-only rows 与新的
utility-train rows；formal-58 永远不能成为本 v1 的 tune/evaluation，已消费的 old-dev5、fresh-16 和
confirm-20 也不进入训练。正式 roster 尚未绑定，必须在任何 semantic access 或 teacher forward 之前由
Freeze-B 固定。Freeze-B 至少需要绑定：

- trajectory roster、query-state roster 与 source identity hash；
- trajectory-level train/tune/evaluation split；
- 每个 state 的 eligible candidate ids 和 `n`；
- teacher/prompt/OCR/low-fidelity schema revision；
- state 数、预计 teacher forward 数、shard layout 与 private HF revision；
- training grid、seed、selection rule、runtime 和完整 argv。

split 单位必须是 `trajectory_id`，同一 trajectory 或 source identity 不能跨 split。formal-58 的角色必须写成
`legacy_train_only`，并从原始 `D(S)` 重新机械投影 set utility；不能复用旧 event-level target。已经消费的
old-dev5、fresh-16 和 confirm-20 既不能加入训练，也不能参与调参。

### P0：先做 policy-blind long-trajectory roster discovery

旧 source selection 有 81 条 trajectory 仅因为 `decision_count>12` 被排除。这个 cap 原本服务于早期小规模
实验，不能直接推断长轨迹数据不足。因此在冻结 train/tune/evaluation 数量前，先单独做一次 P0 discovery：

- 继续要求 mobile、terminal `success` 且 terminal signal 位于最后；
- 每个 observation 仍只能对应一个 executable action；
- 继续使用同一 action canonicalizer 和同一 parser-compatible action types；
- 继续要求合法、完整、受支持的 embedded images 和安全唯一的 source id；
- discovery 长度明确取 13–64，与旧 4–12 output roles 结构不相交；
- 除长度范围外，所有 eligibility 与 parser 行为保持不变。

P0 是 roster inventory，不是实验。它可以输出 eligible source identity、decision-count histogram、app/action
coverage、deterministic selection hash 和 exclusion counts，但不能分配 train/tune/evaluation，不能选 query
state，也不能产生 `D(S)`。P0 中 policy load/forward、restoration label/distance、OCR score 和 learned gate
score 的访问都必须为零；source dataset 自带的 terminal-success metadata 不等于新 policy rollout。

因此阶段边界固定为：

1. 独立 P0 runner freeze 与 policy-blind discovery；
2. 根据 P0 的 pool size/strata 新立 label Execution-A，冻结具体 roster、split 数和 query-state sampling；
3. 再立 label Execution-B，执行 teacher forward 与 exact label production。

P0 不能自动授权后两步，也不能根据任何 policy/restoration output 决定 roster。当前文档不臆定 160 条或其他
固定规模；具体数量必须在看到 policy-blind pool inventory 后、访问任何 label 前写入新的 committed freeze。

Reusable `D(S)` tables、feature cache 和训练 checkpoints 属于 Hugging Face source of truth；Git 只保存生成
代码、config、manifest、轻量 aggregate 与 revision。预留 private dataset destination 为
`gavinlaw/causalcache-set-utility-new-development-mobile`，但 Source-A 时它仍是 uncreated/unbound，不能写成
已经发布。

## 阶段 1：`B<=2` exact labels

对每个新 development state，最多保留最近 16 个 eligible 历史事件作为候选；该 truncation 必须发生在
任何 label forward 之前，且不得根据 `D(S)` 或实验结果选候选。完整计算：

\[
\mathcal S_2=\{S\subseteq C_t:|S|\le2\}.
\]

每个 state 的 label 数为：

\[
1+n+\binom n2.
\]

`n=16` 时正好是 137。阶段 1 禁止 subset sampling，也不允许缺失某个 pair 后仍把 state 当 exact label
state。训练可以为计算吞吐分 shard，但验证时必须恢复完整 per-state table。

teacher 与已有 restoration 定义保持一致：冻结 GUI-Owl，以 full-history behavior 产生 canonical action，
对 mixed-fidelity input 完整 rerun，并在 teacher-forced action tokens 上计算 weighted KL。效用取
`D(empty)-D(S)`，因此 `U(empty)=0`。

### 模型 A：pairwise-additive

\[
\hat U_{\mathrm{pair}}(S)=
\sum_{i\in S}\hat u_i+
\sum_{\{i,j\}\subseteq S}\hat r_{ij}.
\]

它不是 `B=2` 专用模型；公式可以在任意 cardinality 上求值。它的真实限制是只显式表达一阶和二阶
interaction，因此是数据效率高、可解释的强 baseline。

### 模型 B：DeepSets

\[
\hat U_{\mathrm{set}}(S)=
\rho\!\left(q,\sum_{i\in S}\phi(q,h_i)\right)-\rho(q,0).
\]

减去 `rho(q,0)` 强制 `empty` 输出为零。DeepSets 是 v1 主候选：permutation-invariant、支持 variable
cardinality，且在当前数据规模下比 Set Transformer 更容易稳定训练。v1 不上 Set Transformer；如果两个简单
模型都失败，不允许靠无限扩大 architecture search 翻转同一次 evaluation。

两个模型必须使用相同 query/event features、相同 split 和相同 target inventory。训练目标至少包含 raw
utility regression 和 state 内 subset ranking；具体 loss 权重、hidden size、seed 和 early-stop rule 必须在
Freeze-B 中一次性绑定，不能看 evaluation labels 后补。

## 阶段 1 comparator 与判断

固定比较顺序：

1. `pairwise_additive`；
2. `deepsets`；
3. `OCR/RGB`；
4. oracle-independent `J`；
5. exact subset oracle。

`OCR/RGB` 使用已经冻结的 event heuristic，选严格正分数的 top-`B`，不得根据新 evaluation labels 重调。
`J` 不是 deployable model，而是从真实 `D(S)` table 投影出的 independent teacher diagnostic：

\[
J_j=\frac12\left[
\Delta_j(\varnothing)+
\frac1{n-1}\sum_{i\ne j}\Delta_j(\{i\})
\right],
\]

然后选择严格正的 top-`B`。exact oracle 直接在 `|S|<=B` 内最大化真实 `U(S)`；tie 依次取更小
cardinality 和字典序更小的 event tuple。

所有 selector 最终都用所选集合的真实 utility 评分。至少报告：

- trajectory-equal mean raw utility；
- trajectory-equal exact recovery ratio；
- paired trajectory win/tie/loss；
- selected-cardinality distribution；
- utility regression error 与 state 内 ranking quality；
- learned-vs-OCR、learned-vs-J、DeepSets-vs-pairwise 的 paired delta。

进入阶段 2 的最低条件是：至少一个 learned family 在 overall 和 `B=2` raw utility 上都严格胜过
`OCR/RGB`，且该 winner 对 `J` 的 overall delta 非负。该结果仍只是 development routing，不直接打开
closed-loop。

## 阶段 2：少量 `B=3/4` cardinality transfer

“架构能接收四个元素”不等于证明 budget generalization。阶段 2 使用与阶段 1 evaluation identity-disjoint 的
新 states，固定两个小规模 exact track：

| Track | 候选数 | 最大 cardinality | 每 state exact labels |
| --- | ---: | ---: | ---: |
| `B3_n6` | 6 | 3 | 42 |
| `B4_n8` | 8 | 4 | 163 |

每个 track 都做两次：

- zero-shot：byte-identical 阶段 1 checkpoint，不读取任何 B3/B4 训练 label，也不改超参；
- few-shot：在预先固定、trajectory-disjoint 的小 calibration split 上加入 higher-cardinality labels，再在另一组
  evaluation states 上测量。

few-shot 的 calibration/evaluation state 数、seed、步数与 checkpoint rule 必须在阶段 2 Freeze 中绑定；在那
之前阶段 2 保持 locked。必须同时报告 pairwise 与 DeepSets，才能回答 B2 数据是否已经足够、二阶结构是否已
足够，以及少量高阶 label 能否补上 transfer gap。

## 当前锁与下一步

以下操作全部保持 locked：

- exploratory closed-loop validation-12；
- 任何新 closed-loop episode；
- matched-NLL；
- AndroidWorld sealed test；
- paper primary table。

Source-A validator 只能读取一次 canonical config；network、HF API、file write、subprocess、torch import、
model load/forward、data access、label generation、optimizer step 和所有下游评估计数必须为零。

下一步不是启动 GPU，而是写 Freeze-B：机械生成并冻结新 development roster、明确数据规模和 split、实现 exact
label producer 与两个 predictor 的训练/evaluation runner，完成 source tests 后 commit/push。只有 Freeze-B
明确授权后，才执行 GPU preflight 和数据生产。
