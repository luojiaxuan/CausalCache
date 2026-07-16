# Interaction-aware Memory Gate Ablation

> 状态：v2.2 已生成完整 train/development $D(S)$ table；gate 仍未训练。先按
> [`../docs/restoration_v2_2_selector_geometry.md`](../docs/restoration_v2_2_selector_geometry.md) 测量真实
> search gap 与 objective-projection gap，再决定 set-conditioned student 是否进入主方法。
>
> 边界：本文档不修改 v2.1 的 `NO_GO_V2_1_FULL_45_SUBSTRATE`；后续独立的 v2.2-eager substrate 与 exact-label
> run 已各自闭合。本文仍不授权 gate training、untouched confirm 或 AndroidWorld test inference。现有 30 个
> label-train 与 15 个 development states 只能做 teacher/selector development，不能包装成 confirmatory pass。

## 合作方问题与结论

合作方指出：synthetic validation 已经证明 event 之间存在 non-additive interaction，但当前方法仍然逐个恢复、
逐个选择 event，是否自相矛盾？

结论分两层：

1. **Attribution 阶段逐个恢复 event 是正确的。** 每个 marginal 都条件于完整 coalition，因此没有假设 event
   独立。
2. **把 conditional marginals 平均成一个固定 event score，再独立做 positive-value selection，是有损近似。**
   它会丢失 redundancy 与 complementarity，形成 averaged-item objective 与真实 subset objective 之间的
   information bottleneck。

当前 independent gate 若准确拟合已经冻结的 $G_j$ teacher，并没有违反它自己的训练契约。问题在于该 teacher
target 本身是 coalition-dependent utility 的投影，不能完整监督 subset selection。只有未来把 teacher 改成原始
$(S,j,\Delta_j(S))$，再训练 set-conditioned student，才构成新的 conditional teacher--student contract。

因此候选改法不是取消逐步选择，而是让每一步都根据当前已选集合重新估计剩余 event 的 conditional marginal。

## Attribution 为什么已经包含 interaction

对当前 decision state，令 $S$ 表示已经恢复为高保真的 event coalition，$D_t(S)$ 表示相对 full-history
reference behavior 的距离。恢复候选 $j$ 的 conditional marginal 为：

$$
\Delta_{j,t}(S)=D_t(S)-D_t(S\cup\{j\}).
$$

虽然一次干预只增加一个 event，frozen policy 实际看到并重新计算的是整个 $S\cup\{j\}$ mixed-fidelity
input。因此：

- 若 $i$ 与 $j$ 冗余，$i\in S$ 后，$\Delta_j(S)$ 通常下降；
- 若 $i$ 与 $j$ 互补，$i\in S$ 后，$\Delta_j(S)$ 可能上升；
- Shapley-style averaging 只是对不同 coalition context 下的 conditional marginal 求平均，并没有把 policy
  forward 分解成独立 event effects。

定义 restoration utility：

$$
U_t(S)=D_t(\varnothing)-D_t(S).
$$

则 $\Delta_j(S)=U_t(S\cup\{j\})-U_t(S)$。二阶 interaction diagnostic 可以写成：

$$
I_{ij,t}(S)=\Delta_{i,t}(S\cup\{j\})-\Delta_{i,t}(S).
$$

在同一个确定性 set function 上，该量对 $i,j$ 对称；$I<0$ 表示局部冗余，$I>0$ 表示局部互补，接近 0
表示该 coalition context 下近似可加。它只是二阶诊断，不保证捕获高阶 interaction。

## 当前 independent student 丢失了什么

当前概念设计把 coalition-conditioned marginal 压缩成：

$$
G^{(B)}_{j,t}=\mathbb E_{S\sim\mathcal C_B(j)}[\Delta_{j,t}(S)],
$$

再让 gate 从当前 query 和单个 event 特征预测固定分数：

$$
s_{j,t}=f_\theta(g,o_t,\tilde e_j,z_j,\Delta t,B).
$$

这里的 teacher target 已经对 coalition context 做了平均，student 推理时看不到当前 $S$；即使完美拟合
$G^{(B)}_j$，也不保证最大化真实 $U_t(S)$。两个典型失败是：

- **冗余：** $U(\{A\})=10$、$U(\{B\})=10$、$U(\{A,B\})=11$。独立 gate 可能同时选择
  $A,B$，浪费一个 slot。
- **互补：** $U(\{A\})=1$、$U(\{B\})=1$、$U(\{A,B\})=10$。独立平均分数可能把两者都排在其他
  event 后面。

这不是 attribution estimator 的错误，而是把 set function 蒸馏成 context-independent item score 时产生的
approximation gap。现有 synthetic result 已给出直接信号：exact marginal-score selection 只达到 global subset
optimum utility 的 85.9%，见
[`../data/results/synthetic_phase0/README.md`](../data/results/synthetic_phase0/README.md)。

## 候选方法：set-conditioned iterative gate

候选 student 直接预测当前 coalition 下的 marginal：

$$
\widehat\Delta_{j,t}(S)=f_\theta\!\left(
q_t,h_j,\phi(S),c_j,B-c(S)
\right),
$$

其中：

- $q_t$ 编码 instruction、current observation 的轻量 query 表示；
- $h_j$ 编码 low-fidelity summary、time gap 与可选的 cached lightweight event embedding $z_j$；
- $\phi(S)$ 聚合已选 event embeddings，第一版优先使用 sum/mean pooling；
- $c_j$ 是 candidate high-fidelity block 的 visual-token cost；
- $B-c(S)$ 是剩余 visual budget。

Gate 不能为了打分而重新读取所有 candidate 的 raw screenshot；否则虽然没有把图片送入 frozen action policy，
仍会改变 selector compute/storage contract。候选 $h_j$ 只能使用始终保留的 low-fidelity channel 与在 event 到达时
一次性缓存、单独计费的轻量 embedding。只有被选中的 event 才把 archived post-state image 加入 policy-visible
context。

最小 ablation 可以先冻结 equal-cost post-state blocks，并使用 greedy：

1. 初始化 $S=\varnothing$；
2. 对所有满足 $j\notin S$ 且 $c(S)+c_j\le B$ 的候选预测 $\widehat\Delta_j(S)$；
3. 选择最大正 marginal 的 $j^*$；若最大值不大于 0，则提前停止；
4. 更新 $S\leftarrow S\cup\{j^*\}$，重新给剩余 event 打分；
5. 达到 budget 后停止。

令 $b$ 为最多选择轮数，则 $b\le\lfloor B/c_{\min}\rfloor$，推理为 $b$ 个可批处理的 rescoring rounds，
总计 $O(bN)$ 次轻量 candidate scores。这里的 $B$ 是 visual-token capacity，不是 event slot 数。若
high-fidelity block 成本不相等，纯 $\arg\max\widehat\Delta_j(S)$ 不是一般 cost-aware optimum；必须在任何
confirm output 前另行冻结 ratio、beam 或 knapsack-aware rule。

## 训练样本与损失

Restoration intervention 接口可以生成以下 conditional edge：

```text
query state t
selected coalition S
candidate event j
candidate cost c_j
remaining budget
distance D(S) and D(S union {j})
target marginal Delta_j(S)
```

v2.2 exact-label run 现在已经为 $n\in\{2,3,4\}$ 枚举完整 power set：420 条 raw $D(S)$ 可离线重建 720 条
full-hypercube edges，其中 primary $B=2$ 可达 435 条。因而当前小规模实验确实覆盖
$|S|=0,1,\ldots,b-1$ 的全部推理前缀，不需要用 sampled roll-in 猜测缺失 edge。未来扩展到更长 history 时仍须
另行冻结 random/oracle/student roll-in mixture；不能把本次 $n\le4$ 的 exact coverage 外推成普遍免费。
现有 marginal regression 可以直接使用 $(S,j,\Delta_j(S))$，而不先平均成 $G^{(B)}_j$。候选损失为：

$$
L=L_{\mathrm{marginal}}
+\lambda_1L_{\mathrm{conditional\ ranking}}
+\lambda_2L_{\mathrm{set\ utility}}.
$$

- $L_{\mathrm{marginal}}$：预测单条 evaluated edge 的 conditional gain；
- $L_{\mathrm{conditional\ ranking}}$：只比较相同 $(t,S)$ 下的多个候选；
- $L_{\mathrm{set\ utility}}$：对路径 $S_r=S_{r-1}\cup\{j_r\}$，约束：

$$
\sum_{r=1}^{k}\widehat\Delta_{j_r,t}(S_{r-1})
\approx D_t(\varnothing)-D_t(S_k)=U_t(S_k).
$$

需要特别注意：标准 permutation chain 每个 prefix 通常只评估一个“下一 event”。它天然支持 marginal
regression，却**不保证**同一个 $(t,S)$ 下有多个 candidate label 可用于 conditional ranking。若要训练
$L_{\mathrm{conditional\ ranking}}$，必须显式增加同一 coalition 的 branch-edge evaluations，或在当前
$N=4$ 场景直接复用 exact subset enumeration；不能声称 ranking pairs 完全免费。

## Iterative greedy 仍然不是任意 interaction 的解

Set conditioning 能减少大量 context-dependent redundancy，但不能保证找到任意 non-submodular set function
的 global optimum。纯互补反例是：

$$
U(\{A\})=U(\{B\})=0,\qquad U(\{A,B\})=10.
$$

若从空集出发的两个预测 marginal 都不为正，greedy 会立即停止，仍然无法发现 pair synergy。由于
$U_t(S)$ 不保证 monotone 或 submodular，这里也没有标准 greedy approximation guarantee。因此论文中只能
声称“缩小 interaction-induced oracle gap”，不能声称“解决所有 event interaction”。

为保持 AAAI MVP 简洁，主候选先使用 set-conditioned greedy；只有 development interaction analysis 显示
pure-complementarity failure 具有实际规模时，才把 two-event lookahead 或 pair-seeded greedy 作为独立 ablation。
它不能根据 untouched confirm 结果临时加入。当前四个 candidates、equal-cost capacity 为两个 events 的小规模
设置仍可穷举 exact subset oracle，并作为可靠上界。

## Interaction analysis

在 synthetic 与未来真实 restoration labels 上，只对 $i,j\notin S$ 且相关 coalition 都满足 budget 的 feasible
pairs，使用同一预先冻结的 coalition weighting 报告：

$$
M_{\mathrm{comp}}=\mathbb E\!\left[\sum_{i<j}\max(I_{ij}(S),0)\right],
$$

$$
M_{\mathrm{red}}=\mathbb E\!\left[\sum_{i<j}\max(-I_{ij}(S),0)\right],
$$

以及 normalized interaction mass：

$$
\rho_{\mathrm{int}}=
\frac{\mathbb E[\sum_{i<j}|I_{ij}(S)|]}
{\max(\mathbb E[\sum_j|\Delta_j(S)|],\epsilon)}.
$$

Interaction strata 的阈值必须只在 development 上固定，再原样用于 untouched confirm；真实估计同时报告
seed/permutation variance 或 bootstrap interval。二阶 mass 只是 diagnostic；仍需同时报告 coalition
reconstruction error 或 exact-oracle gap，以覆盖可能的高阶 interaction。

## Ablation matrix

| 方法 | 是否输入 $S$ | 选择规则 | 作用 |
| --- | --- | --- | --- |
| Independent average-value gate | 否 | 一次性 positive-value knapsack | 原方法/学生近似 baseline |
| Set-conditioned greedy, sum/mean pool | 是 | 每步重新打分，允许提前停止 | 主候选，最小实现 |
| Set-conditioned attention/Set Transformer | 是 | 同上 | 检验更强 set encoder 是否必要 |
| True conditional-marginal greedy oracle | 是 | 使用真实 $\Delta_j(S)$ 逐步选择 | 隔离 greedy optimization gap |
| Pair-seeded or two-step lookahead | 是 | 先比较单 event 与 event pair | 纯互补诊断，不默认进入主方法 |
| Exact subset restoration oracle | 完整枚举 | $\arg\max_{c(S)\le B}U(S)$ | interaction-aware global ceiling |

上述矩阵把 value estimation 与 search 放在一起概述；正式 optimizer 诊断已单独冻结在
[`subset_search.md`](subset_search.md)。特别是，既有 phase-0 的 85.9% 是 averaged attribution 经过 exact
additive knapsack 后相对真实 subset optimum 的 objective-projection gap，不是 greedy search gap。新诊断会把
exact subset、true conditional-marginal greedy、2x2 bounded exchange 与 true-utility beam-$2/4/8$ 分开计数。
使用真实 $U(S)$ 的 exchange/beam 仍需 policy rerun，只能作为 offline oracle-search ablation；learned marginal
head 若没有 direct set-utility/path-score contract，不能直接复用这两类搜索。

Independent 与 set-conditioned gate 应共享 event/query encoder、training states、visual budget 与尽可能匹配的
parameter/compute envelope，避免收益只是来自更大模型；两者还必须使用相同 label-forward budget。数据在
trajectory/app 层先划分，同一 decision state 的所有 coalitions/edges 必须留在同一 split，防止 coalition leakage。

主要指标：

- 由 frozen policy rerun 得到的 actual set utility、normalized behavior recovery 与 utility / exact-oracle ratio；
- oracle gap，按 $\rho_{\mathrm{int}}$、redundancy mass、complementarity mass 分层；
- redundant-slot rate 与 complement-pair capture rate；
- learned set-conditioned gate 对 true conditional-greedy oracle 的 distillation gap，以及 conditional-greedy
  oracle 对 global subset oracle 的 optimization gap；
- selector latency、参数量、labeling forwards、cached embedding storage 与 policy-visible visual tokens；
- 只有 offline restoration gate 通过后，才允许比较 AndroidWorld paired closed-loop success。

不能再用 selected item scores 的和替代 actual restoration utility；score sum 只可作为 student 内部诊断。

## Synthetic 与真实实验顺序

1. Synthetic 四类 set function：additive、redundant、complementary、mixed/non-monotone；验证符号、停止规则、
   greedy failure case 与 exact oracle。
2. 在已经闭合的 v2.2 label-train/development tables 上，先比较 exact subset、true conditional greedy、
   budget-conditioned independent score 与 full-path Shapley score；只有 objective-projection gap 有稳定规模才保留
   set-conditioned 主线。
3. 固定 encoder、loss weights、edge weighting、interaction strata、提前停止阈值与是否启用 pair seed。
4. 冻结新的 versioned gate contract 后，才允许训练 student；untouched confirm 的开放仍需单独 admission。
5. 只有 offline oracle 与 distilled gate 均通过预注册 gate，才进入 closed-loop。

v2.2-eager substrate 和 exact labels 已闭合，但步骤 3--5 仍未授权；selector geometry 只是 gate-contract 的前置
method-shaping analysis，不能替代该 gate。

## 论文可用表述

> CausalCache does not assume that GUI events contribute independently. Restoration attribution measures the
> marginal value of upgrading an event conditional on an already restored coalition. Because averaging these
> marginals into a context-independent event score can discard redundancy and complementarity, we evaluate an
> iterative set-conditioned gate that rescales remaining candidates after each selection.

在真实 ablation 完成前只能写成 proposal 或 planned ablation，不能把 set-conditioned gate 写成已验证贡献。

## 尚未冻结的决策

- $\phi(S)$ 第一版使用 sum 还是 mean pooling；
- conditional branch-edge 的 exact 数量与训练采样分布；
- development 上 high-interaction stratum 的固定阈值；
- pair-seeded greedy 是否只做 diagnostic，还是进入 confirm matrix；
- semantic admission 后 teacher KL 是否仍沿第一次生成的 exact tool-call token path。若是，必须称为
  “semantic admission + exact-token teacher-forced KL”，不能声称在 semantic action space 上计算 KL。
