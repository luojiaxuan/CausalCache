# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

> Target venue: AAAI
> Status: AAAI-27 paper backbone / experiment implementation

## 一句话主张

现有 GUI memory 方法主要根据 recency、similarity、attention 或 learned salience 保存历史。CausalCache 直接测量：**在实际部署预算附近，把某个 GUI 事件从固定低保真记录升级为高保真视觉证据，能在多大程度上恢复 validated frozen policy 当前决策的 executable behavior**，并把这种昂贵的 restoration attribution 蒸馏成在线 memory selector。

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
M_t\subset H_t,\qquad \sum_{j\in M_t}c_j\le B,
$$

使冻结 GUI policy 的长期 executable task success 最大。这里的 $B$ 限制 policy-visible multimodal context，不限制 persistent storage；原始事件保存在 archive，未选事件在当前 query 只暴露固定长度摘要。

## 方法

### 1. Restoration Attribution

首先只在 full-history action 通过 executor-compatible action validation 的状态上运行冻结策略 $\pi_0$，得到参考 action behavior：

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

事件 $e_j$ 在部署预算 $B$ 附近的 restoration gain 为：

$$
G^{(B)}_{j,t}=\mathbb{E}_{S\sim\mathcal C_B(j)}\!\left[
D_t(S)-D_t(S\cup\{j\})
\right].
$$

其中 $\mathcal C_B(j)$ 只包含为 $e_j$ 留出容量的 maximal near-budget coalitions。该定义是 budget-conditioned Shapley-style attribution；新意不在通用 Shapley estimator，而在 GUI mixed-fidelity intervention、validated executable behavior value 和固定预算蒸馏。实际使用共享 antithetic permutation，并报告 standard error、Spearman、top-budget Jaccard、oracle utility 与 coalition reconstruction error。

### 2. 在线 Memory Gate

离线 restoration attribution 计算昂贵，因此训练轻量 gate：

$$
s_{j,t}=f_\theta(g,o_t,\tilde e_j,z_j,\Delta t,B)
$$

预测 $G_{j,t}$。这里采用 **query-time scoring**：历史事件在每个决策时刻根据当前状态重新打分，解决 arrival-time label 随未来时刻变化的契约问题。

训练目标组合为：

$$
L=L_{\text{regression}}
+\lambda_1L_{\text{pairwise-ranking}}
+\lambda_2L_{\text{top-}B}.
$$

推理时使用 positive-value knapsack；分数不超过阈值的事件不会被强制加入，因此实际选择可以少于预算容量。冻结 action policy，只训练 memory gate。

### 3. Mixed-Fidelity Memory

每个低保真事件固定包含：

```text
step_id
action_type
target_text_or_coordinate_bin
deterministic_ui_delta
result_status
```

系统由 raw event archive、cheap summary/index（可含预计算视觉 embedding）和 policy-visible high-fidelity context 三层组成。截图高保真表示仍使用原始视觉输入。第一版不做跨层 KV surgery，而是通过 mixed-fidelity input 重新运行 policy，避免位置编码与上下文依赖导致不合法的 KV 拼接。

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

1. 提出与部署预算对齐的 restoration-guided GUI memory attribution。
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

> Long-horizon GUI memory selection lacks policy-grounded supervision. CausalCache values an event by how much upgrading it from a fixed low-fidelity record to high-fidelity visual evidence restores a validated frozen policy's executable decision behavior, then distills this budget-conditioned teacher into a query-time gate.

## 初始路线图

- [x] 固化问题定义、核心机制、实验主线与 claim 边界；
- [x] 建立并验证 AAAI-27 官方 LaTeX anonymous submission 骨架；
- [x] 冻结 action serialization、validated teacher 与 mixed-fidelity experiment contract；
- [x] 固定并评估首个 frozen policy candidate；因 full-history coverage 仅 2/9，拒绝作为主 teacher；
- [ ] 选择 GUI-tuned 主 frozen policy，并确定 transfer backbone；Qwen3-VL、UI-TARS、OpenCUA 与 ShowUI 已拒绝，GUI-Owl 单实例 AndroidWorld official-success smoke 已通过，等待完整 validation success gate；
- [x] 实现 trajectory/event schema 与 deterministic low-fidelity summarizer；
- [x] 实现并测试 budget-conditioned restoration attribution 核心；
- [x] 在 synthetic frozen behavior 上验证方差、ranking stability、负 gain 和 interaction error；
- [ ] 在已接入的真实轨迹上实现 teacher-forced policy distance；
- [ ] 构造 matched-NLL memory pairs，验证关键假设；
- [ ] 训练 query-time memory gate；
- [ ] 完成 AndroidWorld closed-loop evaluation；
- [ ] 整理论文与复现实验配置。

## Source of Truth

### Code and Documentation

- GitHub: <https://github.com/luojiaxuan/CausalCache>
- Canonical branch: `main`
- Paper source: [`paper/main.tex`](paper/main.tex)
- Experiment contract: [`docs/experiment_contract.md`](docs/experiment_contract.md)
- Frozen policy selection: [`docs/policy_selection.md`](docs/policy_selection.md)
- AndroidWorld benchmark-native stack: [`docs/androidworld_stack.md`](docs/androidworld_stack.md)
- AndroidWorld frozen task partition: [`docs/androidworld_task_partition.md`](docs/androidworld_task_partition.md)
- Progress record: [`docs/progress.md`](docs/progress.md)
- Synthetic estimator validation: [`results/synthetic_phase0/README.md`](results/synthetic_phase0/README.md)
- Qwen3-VL real-policy smoke test: [`results/qwen_policy_smoke/README.md`](results/qwen_policy_smoke/README.md)
- Qwen3-VL full-history coverage: [`results/qwen_policy_coverage/README.md`](results/qwen_policy_coverage/README.md)
- UI-TARS full-history coverage: [`results/ui_tars_policy_coverage/README.md`](results/ui_tars_policy_coverage/README.md)
- OpenCUA-7B pinned snapshot manifest: [`configs/open_cua_7b_snapshot.json`](configs/open_cua_7b_snapshot.json)
- OpenCUA-7B pinned runtime dependency: [`requirements/opencua.txt`](requirements/opencua.txt)
- OpenCUA-7B logits and mixed-fidelity smoke: [`results/open_cua_policy_smoke/README.md`](results/open_cua_policy_smoke/README.md)
- OpenCUA-7B full-history coverage: [`results/open_cua_policy_coverage/README.md`](results/open_cua_policy_coverage/README.md)
- ShowUI-2B pinned snapshot manifest: [`configs/showui_2b_snapshot.json`](configs/showui_2b_snapshot.json)
- ShowUI-2B logits and mixed-fidelity smoke: [`results/showui_policy_smoke/README.md`](results/showui_policy_smoke/README.md)
- ShowUI-2B full-history coverage: [`results/showui_policy_coverage/README.md`](results/showui_policy_coverage/README.md)
- GUI-Owl-1.5-8B pinned snapshot manifest: [`configs/gui_owl_1_5_8b_snapshot.json`](configs/gui_owl_1_5_8b_snapshot.json)
- AndroidWorld stack preregistration: [`configs/androidworld_stack.json`](configs/androidworld_stack.json)
- AndroidWorld task partition manifest: [`configs/androidworld_task_partition.json`](configs/androidworld_task_partition.json)
- AndroidWorld validation execution plan: [`configs/androidworld_validation_plan.json`](configs/androidworld_validation_plan.json)
- GUI-Owl native logits/history smoke: [`results/gui_owl_native_smoke/README.md`](results/gui_owl_native_smoke/README.md)
- GUI-Owl model-default native-resolution smoke: [`results/gui_owl_native_resolution_smoke/README.md`](results/gui_owl_native_resolution_smoke/README.md)
- AndroidWorld environment/reward smoke: [`results/androidworld_environment_smoke/README.md`](results/androidworld_environment_smoke/README.md)
- GUI-Owl AndroidWorld validation smoke: [`results/gui_owl_androidworld_validation_smoke/README.md`](results/gui_owl_androidworld_validation_smoke/README.md)
- GUI-Owl configuration-invalid validation audit: [`results/gui_owl_androidworld_validation_attempt2/README.md`](results/gui_owl_androidworld_validation_attempt2/README.md)
- Build command: `make paper`
- Test command: `make test validate-contract`
- 当前状态：论文骨架、实验契约、synthetic estimator validation、GUIOdyssey pilot、GUI-Owl 单实例 action/reward 链路与 model-default native-resolution smoke 已完成；尚无 CausalCache 方法效果结果，也尚未完成 AndroidWorld validation gate。

### Data and Models

| Artifact | Canonical location | Revision/status | Notes |
| --- | --- | --- | --- |
| GUIOdyssey pilot trajectory | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-pilot-mobile> | `1de9c34ff029d4c01665cdaca74436ae24bff276`，private | schema v0.3；10 screenshots、9 events、9 decisions |
| Rejected policy candidate | <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct> | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | full-history executable-match 2/9；不作为主 teacher |
| Rejected GUI-tuned candidate | <https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B> | `683d002dd99d8f95104d31e70391a39348857f4e` | parsed 9/9、executable-match 4/9；未通过预注册 50% gate |
| Rejected computer-use candidate | <https://huggingface.co/xlangai/OpenCUA-7B> | `a2efb7d2b104d477a4a2666a357e79550a28aafc` | parsed 7/9、executable-match 1/9；未通过预注册 gate |
| Rejected GUI navigation candidate | <https://huggingface.co/showlab/ShowUI-2B> | `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60` | parsed 9/9、executable-match 2/9；未通过预注册 gate |
| Pending AndroidWorld-native candidate | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct> | `06d5faecff74840bab2be2425e9c42667a5d04fc` | model-default 1/5-image smoke 已通过；等待重跑 validation gate |
| Full attribution/evaluation datasets | Hugging Face dataset repo（待创建） | not created | pilot 扩展为多 app、多 horizon 后创建或升级 |
| Gate checkpoints/adapters | Hugging Face model repo（待创建） | not created | 记录 policy backbone、训练配置与评测 provenance |

Pilot 的生成配置见 [`configs/guiodyssey_pilot.json`](configs/guiodyssey_pilot.json)。当前没有仅存于本地、等待上传的可复用数据集、模型或评测 artifact。

## Citation

项目仍处于研究与实验阶段，正式 citation 将在论文公开后补充。
