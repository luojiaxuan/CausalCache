# Set Utility Predictor v1：先扩数据，再验证预算泛化

> 当前状态：全量 610-shard P-1、8,146-row P0 与 Freeze-B v2 policy-blind roster 已完成。原 v1 terminal
> query 有 one-based indexing off-by-one，已永久标记 invalid；v2 在 corrected eligible universe 上固定
> 1,200 trajectories / 2,400 queries、`1000/100/100` train/tune/evaluation、rich visual feature 与 training
> grid；没有生成本路线新 label、训练 predictor，也没有执行 closed-loop、matched-NLL 或 sealed test。
>
> Canonical SHA256：predictor=`9548159b219795b1c258c28f772f53351256e0d728b88dd409cb333bd2100fe4`；
> P-1 source=`1b2b4374d1653bcf22444d8e708c71bc41ac87fa956243ddcb9bd963eeca7e96`；
> consumed ledger=`b6f44c603b99d2f954b981e01818cf0afa028ce3a410ed935203532a097bb4ad`；
> P0 source-only=`7e65227e710009d3626bd0063d425c831dfc59e9a6bbe871b9d3e4d15e085e8b`；Freeze-B v2 manifest=
> `915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30`。focused suite
> 为 `213 passed, 15 skipped, 24 subtests passed`。

## 路线调整

当前不再把 exploratory closed-loop 作为下一步。新的顺序是：

1. 从全量 610 个 GUIOdyssey transport shards 做 census，机械排除 canonical consumed ledger 后
   建立新 train/tune/one-shot evaluation；
2. 每 trajectory 以 outcome-blind 规则取 2–4 个 query states，在 `B<=2` 下完整枚举集合；
3. 扩大 restoration `D(S)` 数据后训练 Set Transformer，Pairwise/DeepSets 作为简单对照；
4. 先证明 learned set utility 在新 offline evaluation 上超过 `OCR/RGB`，并与 `J` 和 exact oracle
   定量比较；
5. 再用少量 `B=3/4` exact labels 检验 zero-shot 与 few-shot cardinality transfer；
6. 只有这些离线问题闭合后，才另立协议决定是否恢复 closed-loop。

这次调整不重写历史结论。conditional v1 和 independent v1 在已经消费的 fresh-16/confirm-20 上仍然是
NO-GO；那些结果只能作为 failure analysis，不能重新进入训练、调参或 model selection。

## 核心对象：预算无关的集合效用

对 query state `t` 和高保真事件集合 `S`，teacher target 固定为：

\[
U_t(S)=D_t(\varnothing)-D_t(S).
\]

模型接口是：

\[
\hat U_\theta(q_t,C_t,m_S),
\]

其中 `C_t` 是冻结 candidate universe，`m_S` 是 selected/unselected bit。所有 `C_t` 中的低保真 event token
都对 predictor 可见；只有真实 padding 被 attention mask。该接口不是 `U_theta(q,S,B)`，`B` 只进入搜索约束：

\[
\hat S_B=\arg\max_{S\subseteq C_t,\ |S|\le B}\hat U_\theta(q_t,C_t,m_S).
\]

因此需要严格区分：

- utility predictor 是 budget-agnostic；
- selector/optimizer 是 budget-conditioned；
- `|S|` 是 selection mask 的结果，不等于把外部 budget 当模型特征；
- unselected candidate 仍以低保真 token 出现在模型中，selected bit 表示它会被升级为高保真；
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

v1 的 tune 和 one-shot offline evaluation 只允许使用一套新建的
`causalcache_set_utility_new_development_v1` 数据。训练集允许显式合并旧 formal-58 train-only rows 与新的
utility-train rows；formal-58 永远不能成为本 v1 的 tune/evaluation。reference8、old-dev5、
fresh-16 和 confirm-20 不得进入新训练、调参或 evaluation。正式 roster 尚未绑定，必须在任何
semantic access 或 teacher forward 之前由
Freeze-B 固定。Freeze-B 至少需要绑定：

- trajectory roster、query-state roster 与 source identity hash；
- trajectory-level train/tune/evaluation split，并以规范化 instruction+app group 作为不可跨 role 的约束；
- 每个 state 的 eligible candidate ids 和 `n`；
- teacher/prompt/OCR/low-fidelity schema revision；
- state 数、预计 teacher forward 数、shard layout 与 private HF revision；
- reference-repeat KL 的最大稳定性阈值，以及 P0 产生的 historical legacy/forbidden group
  SHA256 firewall inputs；
- training grid、seed、selection rule、runtime 和完整 argv。

split 以 `trajectory_id` 为样本单位，但相同规范化 instruction+app group 的 trajectories 必须绑定到同一
role，不能借近重复任务跨 train/tune/evaluation。group key 只用于 overlap audit，不能进入 predictor feature；
同一 trajectory 或 source identity 也不能跨 split。formal-58 的角色必须写成 `legacy_train_only`，并从原始
`D(S)` 重新机械投影 set utility；不能复用旧 event-level target。reference8、old-dev5、
fresh-16 和 confirm-20 不得进入新训练、调参或 evaluation。

### P-1/P0：先扩大到全量 transport shards，再做 policy-blind census

旧 source selection 只 byte-pin 了 `610` 个 train shards 中的前 `16` 个，共 212 rows；其中 81 条仅因为
`decision_count>12` 被排除。这个范围远不能代表[官方 GUIOdyssey 全量数据](https://github.com/OpenGVLab/GUI-Odyssey)，
也不能支撑当前扩数据目标。因此此前的 16-shard P0 在运行前取消，改成两步：

1. P-1 只访问 Hugging Face metadata，固定 `cua-lite/GUIOdyssey` 指定 revision 下全部 610 个
   `mobile/use/train/shard-?????-of-00610.parquet` 的 path、size 和 LFS SHA256；
2. P0 校验这些 bytes 后流式 census 所有 rows，同时排除 canonical consumed ledger 中的 identity。

P0 继续要求：

- 继续要求 mobile、terminal `success` 且 terminal signal 位于最后；
- 每个 observation 仍只能对应一个 executable action；
- 继续使用同一 action canonicalizer 和同一 parser-compatible action types；
- 继续要求合法、完整、受支持的 embedded images 和安全唯一的 source id；
- `decision_count>=6`，从而至少存在 4 个非 current-equivalent candidates；不再用 64 作为科学上限；
- 按 6–9、10–17、>=18 报告 prefilter candidate-capacity strata；
- 除长度范围外，所有 eligibility 与 parser 行为保持不变。

P0 是 roster inventory，不是实验。它可以输出 eligible source identity、decision-count histogram、app/action
coverage、deterministic selection hash、无 instruction 原文的 instruction+app group SHA256 和 exclusion
counts，但不能分配 train/tune/evaluation，不能选 query
state，也不能产生 `D(S)`。P0 中 policy load/forward、restoration label/distance、OCR score 和 learned gate
score 的访问都必须为零；source dataset 自带的 terminal-success metadata 不等于新 policy rollout。

consumed ledger 必须明确包含：58 条 `legacy_train_only`（旧 train10 + expansion train48）和 49 条
`forbidden_consumed`（reference8 + old-dev5 + fresh16 + confirm20）。相同 canonicalizer 还要为这些历史
identity 补齐 instruction+app group hash。当前阶段状态与边界固定为：

1. `[done]` canonical consumed 58/49 identity ledger；
2. `[pending]` P-1 full-shard metadata inventory commit；
3. 两者都 committed 后另立 P0 Execution-A，运行 policy-blind full-pool census；
4. P0 完成后根据真实 pool size/strata 冻结 Freeze-B，绑定具体 roster、split 数和每 trajectory
   2–4 个 query
   states；
5. 先做不访问 utility 的 label-throughput pilot，再立 Execution-B 运行 exact labels。

P0 不能自动授权后两步，也不能根据任何 policy/restoration output 决定 roster。当前文档不臆定 160 条或其他
固定规模；具体数量必须在看到 policy-blind pool inventory 后、访问任何 label 前写入新的 committed freeze。

Reusable `D(S)` tables、feature cache 和训练 checkpoints 属于 Hugging Face source of truth；Git 只保存生成
代码、config、manifest、轻量 aggregate 与 revision。预留 private dataset destination 为
`gavinlaw/causalcache-set-utility-new-development-mobile`，但 Source-A 时它仍是 uncreated/unbound，不能写成
已经发布。

## 阶段 1：`B<=2` exact labels

对每个新 state，先排除 post-state 与 current observation 相同的 current-equivalent history event，再取最近
16 个 eligible events。GUI-Owl 当前 32,768 context 与每图约 2,560 effective visual tokens 使 `n=16`
通常不可直接执行，因此 processor-only 预检 `input_length+256<=32768`；不满足时每次删除最老 candidate，
直到得到最终冻结的 `C_t`。该过程发生在任何 generation/teacher forward 前，不能读取 `D(S)` 或 outcome。
完整计算：

\[
\mathcal S_2=\{S\subseteq C_t:|S|\le2\}.
\]

每个 state 的 label 数为：

\[
1+n+\binom n2.
\]

`n=16` 时正好是 137。阶段 1 禁止 subset sampling，也不允许缺失某个 pair 后仍把 state 当 exact label
state。训练可以为计算吞吐分 shard，但验证时必须恢复完整 per-state table。

teacher 与已有 restoration 定义保持一致：冻结 GUI-Owl，以“全部冻结 candidates 为高保真、
noncandidate history 始终为低保真”的 reference behavior 产生 canonical action，对 mixed-fidelity
input 完整 rerun，并在 teacher-forced action tokens 上计算 weighted KL。这不表示整条 trajectory 的所有
events 都为高保真。效用取
`D(empty)-D(S)`，因此 `U(empty)=0`。

### 模型 A：pairwise-additive baseline

\[
\hat U_{\mathrm{pair}}(S)=
\sum_{i\in S}\hat u_i(q,C)+
\sum_{\{i,j\}\subseteq S}\hat r_{ij}(q,C).
\]

它不是 `B=2` 专用模型；公式可以在任意 cardinality 上求值。它的真实限制是只显式表达一阶和二阶
interaction，因此是数据效率高、可解释的强 baseline。

### 模型 B：DeepSets baseline

\[
\hat U_{\mathrm{set}}(S)=
\rho\!\left(q,\operatorname{Pool}_{i\in S}h_i,
\operatorname{Pool}_{i\in C\setminus S}h_i\right)-\hat U(q,C,m_\varnothing).
\]

### 模型 C：Set Transformer main

Set Transformer 对 `C` 中每个 event 加 selected/unselected embedding，以 query token 聚合；无 positional
embedding，只有 padding 被 mask。empty baseline 使用同一个 `C` 和全零 selection mask，差分后精确约束
`U(empty)=0`。它只在 full-pool labels 扩大后训练，不在旧几十条 trajectory 上做 architecture 结论。

三个模型必须使用相同 query/event features、相同 split 和相同 target inventory。训练目标至少包含 raw
utility regression 和 state 内 subset ranking；具体 loss 权重、hidden size、seed 和 early-stop rule 必须在
Freeze-B 中一次性绑定，不能看 evaluation labels 后补。

Source-A 只冻结 feature firewall，不替 Freeze-B 选择最终 student 表示。Freeze-B 必须在新 evaluation label
access 前，从“轻量 q64/h64 + low-fidelity + OCR/RGB/recency”和“在此基础上加入 frozen GUI-Owl final-main
normalized visual embedding”中明确选择主 schema，并把另一项的角色写清楚；不能在 evaluation 失败后再换表示。

## 阶段 1 comparator 与判断

固定比较顺序：

1. `set_transformer`；
2. `deepsets`；
3. `pairwise_additive`；
4. `recent`；
5. `OCR/RGB`；
6. oracle-independent `J`；
7. exact subset oracle。

`recent` 直接填入最近的至多 `B` 个 event。`OCR/RGB` 在 set-utility v1 中使用明确版本化的 at-most-budget
规则：选严格正分数的 top-`B`；这不同于历史实验中无条件 fill-`B` 的实现，且不得根据新 evaluation labels
重调。
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
之前阶段 2 保持 locked。必须同时报告 Set Transformer、DeepSets 与 pairwise，才能回答 B2 数据是否已经足够、
二阶结构是否已足够，以及少量高阶 label 能否补上 transfer gap。

## 当前锁与下一步

以下操作全部保持 locked：

- exploratory closed-loop validation-12；
- 任何新 closed-loop episode；
- matched-NLL；
- AndroidWorld sealed test；
- paper primary table。

Source-A validator 只能读取一次 canonical config；network、HF API、file write、subprocess、torch import、
model load/forward、data access、label generation、optimizer step 和所有下游评估计数必须为零。

下一步不是 closed-loop 或 label GPU run。consumed ledger 已完成；现在必须先在可联网 checkout
执行 P-1 metadata inventory，提交并 push 唯一 manifest，再另立绑定该 manifest SHA 的 P0
Execution-A 运行 full-pool census。看到真实 eligible pool 前不写死 500/1000 条，也不把所有
数据放进 train：必须保留 group-disjoint tune 和一次性 offline evaluation。只有这些 inventory
commit 后，Freeze-B 才能绑定 roster、
2–4 states/trajectory、训练 grid 与 HF revisions。
