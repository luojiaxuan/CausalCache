# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

> Target venue: AAAI
> Status: AAAI-27 paper backbone / experiment implementation

## 一句话主张

现有 GUI memory 方法主要根据 recency、similarity、attention 或 learned salience 保存历史。CausalCache 直接测量：**恢复某个历史 GUI 事件，能在多大程度上恢复完整历史条件下的后续 action behavior**，并把这种昂贵的 restoration attribution 蒸馏成在线 memory selector。

## 核心观察

两个 memory controller 即使获得近似相同的 next-action NLL，也可能产生完全不同的长期任务成功率。局部 action fidelity 无法区分：

- 当前动作暂时用不到、未来却决定任务成败的事件；
- 视觉上相似但对后续控制无关的事件；
- 需要保留精确文本、坐标、开关状态的 dependency-critical event；
- 仅仅获得较高 attention、但删除后并不改变策略行为的事件。

因此，memory quality 不应只由局部 prediction quality 衡量，还应由历史事件对未来 policy behavior 的**可恢复贡献**衡量。

## 问题定义

一条 GUI trajectory 表示为：

$$
H_t=\{e_1,\ldots,e_{t-1}\},\qquad e_j=(o_{j-1},a_j,o_j),
$$

其中每个 event 包含：

- action 前截图；
- 执行的 action；
- action 后截图；
- 可选的结构化 UI delta。

给定任务指令 $g$、当前观测 $o_t$ 和高保真事件预算 $B$，目标是选择：

$$
M_t\subset H_t,\qquad |M_t|\le B,
$$

使冻结 GUI policy 的长期 executable task success 最大。这里限制的是昂贵的**高保真多模态历史预算**；未选事件只保留固定长度的文本或结构化摘要。

## 方法

### 1. Restoration Attribution

首先用完整历史运行冻结策略 $\pi_0$，得到参考 action behavior：

$$
q_t=\pi_0(\cdot\mid g,o_t,H_t).
$$

将所有历史事件替换为低保真版本，得到 baseline memory；随后按照共享随机排列逐个恢复事件的高保真内容。对于已恢复集合 $S$，定义与完整历史行为的距离：

$$
D_t(S)=\sum_l w_l\,
KL\!\left(
q_{t,l}(\cdot\mid a^*_{<l})
\parallel
q^S_{t,l}(\cdot\mid a^*_{<l})
\right),
$$

其中 $a^*$ 是完整历史策略生成的 canonical action，KL 在 teacher-forced action token 上计算。这样无需枚举包含 coordinate 和 text argument 的完整 sequence action space。

事件 $e_j$ 的 restoration gain 为：

$$
G_{j,t}=\mathbb{E}_{\sigma}\!\left[
D_t(S^\sigma_j)-D_t(S^\sigma_j\cup\{j\})
\right].
$$

若恢复某事件显著减少与完整历史策略的差异，则该事件对当前决策具有较高 restoration value。实际使用共享的 $K=8\text{--}32$ 个 permutation，并报告估计方差和 telescoping error。

### 2. 在线 Memory Gate

离线 restoration attribution 计算昂贵，因此训练轻量 gate：

$$
s_{j,t}=f_\theta(g,o_t,\tilde e_j,\Delta t)
$$

预测 $G_{j,t}$。这里采用 **query-time scoring**：历史事件在每个决策时刻根据当前状态重新打分，解决 arrival-time label 随未来时刻变化的契约问题。

训练目标组合为：

$$
L=L_{\text{regression}}
+\lambda_1L_{\text{pairwise-ranking}}
+\lambda_2L_{\text{top-}B}.
$$

推理时保留分数最高的 $B$ 个高保真 event blocks，其余事件使用固定长度摘要。冻结 action policy，只训练 memory gate。

### 3. Mixed-Fidelity Memory

每个低保真事件固定包含：

```text
step_id
action_type
target_text_or_coordinate_bin
deterministic_ui_delta
result_status
```

截图的高保真表示仍使用原始视觉输入。第一版不做跨层 KV surgery，而是通过 mixed-fidelity input 重新运行 policy，避免位置编码与上下文依赖导致不合法的 KV 拼接。

## 实验设计

### Benchmarks

- 主要 closed-loop benchmark：AndroidWorld；
- 离线 action prediction 与 memory attribution：GUI-Odyssey、AndroidControl 或同类数据。

### Baselines

- no history、full history、summary only；
- recent top-$B$、random top-$B$；
- visual/text similarity；
- attention-based retention；
- learned salience selector；
- MementoGUI-style memory control；
- AndroTMem/anchor-style memory；
- offline restoration oracle；
- distilled CausalCache gate。

### Metrics

- closed-loop task success；
- action type、target、text argument accuracy；
- successor-action NLL；
- retained restoration mass；
- visual-token budget；
- inference latency；
- success per unit memory/compute。

## 决定论文成败的实验

构造 successor-action NLL 相近的 memory pairs，比较 terminal success：

$$
\operatorname{NLL}(M_1)\approx\operatorname{NLL}(M_2).
$$

当：

$$
\operatorname{RestorationMass}(M_1)>
\operatorname{RestorationMass}(M_2),
$$

检验 $M_1$ 是否仍显著获得更高任务成功率。这是论文最关键的证据：**restoration relevance 捕获了 one-step fidelity 无法解释的长期控制信息。**

## Ablations

- permutation 数量与 attribution variance；
- KL、JS、action log-prob recovery；
- event、screenshot、UI element 三种粒度；
- 去掉 pairwise ranking 或 top-$B$ loss；
- query-time gate 对比 arrival-time gate；
- 不同 memory budget；
- success、NLL 与 restoration mass 的独立相关性；
- 跨 app、任务长度和 policy backbone 泛化。

## 预期贡献

1. 提出 restoration-guided GUI memory attribution。
2. 提出将反事实 restoration gain 蒸馏成在线固定预算 memory gate 的方法。
3. 证明 matched next-action fidelity 下，不同 memory 仍产生不同长期成功率。
4. 在相同多模态 memory budget 下，提高 long-horizon GUI task success。

## Claim 边界

这里的 causal 含义限定为：

> 对冻结策略行为进行受控恢复干预得到的 counterfactual attribution。

本项目不声称识别真实环境结构因果，也不把该方法包装成 world model。

## 与最接近工作的差异

- [MementoGUI](https://arxiv.org/abs/2605.18652) 通过训练数据学习 memory selection、compression 和 retrieval；
- [AndroTMem](https://arxiv.org/abs/2603.18429) 使用 causally linked state anchors；
- CausalCache 的事件重要性来自：恢复该事件后，冻结策略行为的边际恢复量，而不是人工 salience、结构规则或相似度。

## Falsification Criteria

满足任一条件就应弱化主张或停止投稿：

- restoration score 不能比 recency、attention 或 similarity 更好地预测 terminal success；
- matched-NLL 后 restoration mass 与成功率不再相关；
- distilled gate 明显无法逼近 restoration oracle；
- 相同预算下 task success 没有稳定提升；
- 方法收益完全来自增加输入 token；
- attribution 成本无法通过少量 permutation 控制。

## Paper Story

> Long-horizon GUI agents do not merely need more memory; they need memory whose restoration recovers future control behavior. CausalCache learns to preserve exactly that evidence.

## 初始路线图

- [x] 固化问题定义、核心机制、实验主线与 claim 边界；
- [x] 建立并验证 AAAI-27 官方 LaTeX anonymous submission 骨架；
- [ ] 确定冻结 GUI policy、action serialization 与 mixed-fidelity prompt contract；
- [ ] 实现 trajectory/event schema 与 deterministic low-fidelity summarizer；
- [ ] 实现 teacher-forced policy distance 与 restoration attribution；
- [ ] 在小规模离线轨迹上验证 telescoping、方差和 permutation 数量；
- [ ] 构造 matched-NLL memory pairs，验证关键假设；
- [ ] 训练 query-time memory gate；
- [ ] 完成 AndroidWorld closed-loop evaluation；
- [ ] 整理论文与复现实验配置。

## Source of Truth

### Code and Documentation

- GitHub: <https://github.com/luojiaxuan/CausalCache>
- Canonical branch: `main`
- Paper source: [`paper/main.tex`](paper/main.tex)
- Build command: `make paper`
- 当前状态：研究构想与 AAAI-27 论文骨架已建立；尚无可报告的实验结果。

### Data and Models

| Artifact | Canonical location | Revision/status | Notes |
| --- | --- | --- | --- |
| Attribution/evaluation datasets | Hugging Face dataset repo（待创建） | not created | 未来采用稳定的 lowercase kebab-case repo ID，并在此记录 revision |
| Gate checkpoints/adapters | Hugging Face model repo（待创建） | not created | 记录 policy backbone、训练配置与评测 provenance |

当前没有仅存于本地、等待上传的可复用数据集、模型或评测 artifact。

## Citation

项目仍处于研究与实验阶段，正式 citation 将在论文公开后补充。
