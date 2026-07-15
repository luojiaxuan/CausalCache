# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

> Target venue: AAAI
> Status: independent reference `NO_GO_CURRENT_REFERENCE_STACK` / oracle not opened

## 团队交接入口

当前最重要的可复核结论：冻结的 independent UI-TARS reference gate 已合法完成，但只有 27/75（36.0%）
full-history executable match，且 swipe 为 0/2，未通过 50% + required-action gate。因此当前
UI-TARS reference stack 明确 `NO_GO`，132-decision oracle split 未打开，不能生成 restoration labels 或
训练 gate。raw 结果已上传 private HF
`reference-gate-v1@b3e1245c6c6a1723fe2ca3a861148008df39df46`，轻量结论见
[`data/results/independent_reference_gate_v1/`](data/results/independent_reference_gate_v1/)。这关闭的是
当前 reference 获取路线，不是对 CausalCache 假设本身的 falsification。

方法与 synthetic estimator 接口已实现；六个早期 frozen-policy candidates 均未通过冻结 gate，当前没有
accepted validated teacher，也没有 CausalCache 方法效果结果。最新
`GUI-Owl-1.5-8B-Think@afe3707` 虽在 Hyper01 通过 1/5-image finite-logit/parser smoke，但 Aries
AndroidWorld 正式 validation 在 42/62 checkpoints 后以 success 下界 9/62、上界 29/62 判负；
512/513 actions parsed，说明失败不是主要来自 serialization coverage。raw traces 已上传 private HF
dataset `v0.2.0@0faf767e`。按预注册 change control，本轮停止，不生成 restoration labels 或训练 gate；
下一步必须先形成新的 validated-reference/primary-policy 预注册决策。final test 仍保持 sealed。

冻结的两状态 real-policy diagnostic 已完成，结果为 `INCONCLUSIVE_POSITIVE`，详见
[`data/results/go_no_go_diagnostic_v1/`](data/results/go_no_go_diagnostic_v1/)。在 step 8、1024-token cap
下，recent/similarity 恢复 48.7%，random expectation 58.1%，而 exhaustive oracle 与 restoration
selector 选择最早 event 1 + 最新 event 7，恢复 84.2%；$K=16$ 五个 seed 全部复现该选择。由于 states
来自已观察的 2/9 matched subset，且所有 memories 的离散 executable swipe 都仍正确，这只是值得进入
独立扩展集的存在性证据，不是 paper `GO`、terminal-success 结果或 gate-training labels。

当前正在执行的 go/no-go 分为两层。第一层已在任何 restoration forward 前冻结为
[`code/configs/go_no_go_diagnostic_v1.json`](code/configs/go_no_go_diagnostic_v1.json)：只用已有 Qwen
matched states 验证真实 action-path KL 与 exhaustive oracle，因 post-selection bias 禁止给出论文级
`GO`，现已按冻结判据得到 `INCONCLUSIVE_POSITIVE`。判据、失败边界与独立扩展要求见
[`docs/go_no_go.md`](docs/go_no_go.md)。正式 reference gate 与 paper-level go/no-go 仍必须使用未观察
policy/restoration 的独立多轨迹 manifest。

独立 gate 的 source pool、policy-blind eligibility/hash split、48-decision reference 最低规模、UI-TARS
exact revision、H200-to-A6000 behavioral anchor 与 fail-closed outcome 已在任何新 source row decoding 或
policy inference 前冻结到
[`code/configs/independent_reference_gate_v1.json`](code/configs/independent_reference_gate_v1.json)。
artifact 必须先上传 Hugging Face 并把 immutable revision/SHA 回写 Git；reference 失败即停止当前路线，
不会在看到输出后换样本或调阈值。

冻结 source pool 的 16 个 Parquet 文件（2.25 GB）及逐文件 SHA256 见
[`data/manifests/independent_reference_gate_v1_source_files.json`](data/manifests/independent_reference_gate_v1_source_files.json)；
这些 hash 已在 row decoding 前生成。

Hyper00 H200 的 UI-TARS behavioral anchor 已通过：旧 9-decision artifact 精确复现 A6000 的 9/9 parsed、
4/9 match 与逐 decision vector，见
[`data/results/ui_tars_hyper00_hardware_anchor/`](data/results/ui_tars_hyper00_hardware_anchor/)。因此独立 gate
可以留在 Hyper00；该 anchor 的低平均利用率仍要求在 oracle-scale attribution 前完成 batching/GPU-side
KL 优化。

multi-trajectory deterministic builder 与 formal fail-closed reference runner 已实现并通过 106 个 tests；
两次构建 byte-identical，artifact 已上传 private HF
`gavinlaw/causalcache-guiodyssey-independent-mobile@84c9f5a335e9612ccb4bd566f977574f359b2485`
并从 immutable revision 强制重下载验 hash。reference split 冻结为 8 trajectories / 75 decisions / 14 app
labels，oracle split 为 15 / 132 / 23，二者 trajectory-disjoint。formal reference 已按冻结分母完成并判负；
命令、回写与失败恢复见 [`docs/independent_gate_execution.md`](docs/independent_gate_execution.md)。

新合作者按以下顺序阅读：

1. [`docs/execution.md`](docs/execution.md)：跨芯片执行、HF/Git 回写与 Definition of Done；
2. [`docs/progress.md`](docs/progress.md)：已完成里程碑、negative results 与下一步；
3. [`docs/experiment_contract.md`](docs/experiment_contract.md)：不可静默改变的实验语义；
4. [`docs/go_no_go.md`](docs/go_no_go.md)：当前 diagnostic 与扩展 pilot 的冻结判据；
5. [`docs/independent_gate_execution.md`](docs/independent_gate_execution.md)：独立 artifact/gate 的执行与回写；
6. [`code/README.md`](code/README.md) 与 [`data/README.md`](data/README.md)：代码和数据边界；
7. [`paper/main.tex`](paper/main.tex)：AAAI 正文 source。

仓库结构：

```text
README.md          # 总索引与交接状态
AGENTS.md          # 每步 Git/HF/compute 规则
paper/             # AAAI LaTeX package
code/              # package、scripts、tests、configs、requirements
data/              # 小 fixture 与轻量 result summaries
docs/              # contract、execution、progress、decisions
```

快速验证：

```bash
make test validate-contract paper
```

计算 placement：本轮 independent policy/reference 已在通过 behavioral anchor 的 Hyper00 H200 完成；
AndroidWorld closed-loop MVP 继续使用已验证的 Aries stack。Hyper01 未参与本轮执行。

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
- [x] 冻结 action serialization、validated-reference requirements 与 mixed-fidelity experiment contract；
- [x] 固定并评估首个 frozen policy candidate；因 full-history coverage 仅 2/9，拒绝作为主 teacher；
- [ ] 选择主 frozen policy，并确定 transfer backbone；六个 candidate 均已被冻结 gate 拒绝，新的 validated-reference 方案尚未预注册；
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
- Code layout and commands: [`code/README.md`](code/README.md)
- Small-data policy: [`data/README.md`](data/README.md)
- Cross-chip execution and handoff: [`docs/execution.md`](docs/execution.md)
- Material-run metadata schema: [`code/configs/run_manifest.schema.json`](code/configs/run_manifest.schema.json)
- Experiment contract: [`docs/experiment_contract.md`](docs/experiment_contract.md)
- Frozen policy selection: [`docs/policy_selection.md`](docs/policy_selection.md)
- AndroidWorld benchmark-native stack: [`docs/androidworld_stack.md`](docs/androidworld_stack.md)
- AndroidWorld frozen task partition: [`docs/androidworld_task_partition.md`](docs/androidworld_task_partition.md)
- Progress record: [`docs/progress.md`](docs/progress.md)
- Synthetic estimator validation: [`data/results/synthetic_phase0/README.md`](data/results/synthetic_phase0/README.md)
- Qwen3-VL real-policy smoke test: [`data/results/qwen_policy_smoke/README.md`](data/results/qwen_policy_smoke/README.md)
- Qwen3-VL full-history coverage: [`data/results/qwen_policy_coverage/README.md`](data/results/qwen_policy_coverage/README.md)
- UI-TARS full-history coverage: [`data/results/ui_tars_policy_coverage/README.md`](data/results/ui_tars_policy_coverage/README.md)
- Independent gate artifact index: [`data/manifests/independent_reference_gate_v1_artifact.json`](data/manifests/independent_reference_gate_v1_artifact.json)
- Independent UI-TARS reference rejection: [`data/results/independent_reference_gate_v1/README.md`](data/results/independent_reference_gate_v1/README.md)
- OpenCUA-7B pinned snapshot manifest: [`code/configs/open_cua_7b_snapshot.json`](code/configs/open_cua_7b_snapshot.json)
- OpenCUA-7B pinned runtime dependency: [`code/requirements/opencua.txt`](code/requirements/opencua.txt)
- OpenCUA-7B logits and mixed-fidelity smoke: [`data/results/open_cua_policy_smoke/README.md`](data/results/open_cua_policy_smoke/README.md)
- OpenCUA-7B full-history coverage: [`data/results/open_cua_policy_coverage/README.md`](data/results/open_cua_policy_coverage/README.md)
- ShowUI-2B pinned snapshot manifest: [`code/configs/showui_2b_snapshot.json`](code/configs/showui_2b_snapshot.json)
- ShowUI-2B logits and mixed-fidelity smoke: [`data/results/showui_policy_smoke/README.md`](data/results/showui_policy_smoke/README.md)
- ShowUI-2B full-history coverage: [`data/results/showui_policy_coverage/README.md`](data/results/showui_policy_coverage/README.md)
- GUI-Owl-1.5-8B pinned snapshot manifest: [`code/configs/gui_owl_1_5_8b_snapshot.json`](code/configs/gui_owl_1_5_8b_snapshot.json)
- GUI-Owl-1.5-8B-Think pinned snapshot manifest: [`code/configs/gui_owl_1_5_8b_think_snapshot.json`](code/configs/gui_owl_1_5_8b_think_snapshot.json)
- Replacement teacher preregistration: [`code/configs/androidworld_replacement_teacher_v1.json`](code/configs/androidworld_replacement_teacher_v1.json)
- AndroidWorld stack preregistration: [`code/configs/androidworld_stack.json`](code/configs/androidworld_stack.json)
- AndroidWorld task partition manifest: [`code/configs/androidworld_task_partition.json`](code/configs/androidworld_task_partition.json)
- AndroidWorld validation execution plan: [`code/configs/androidworld_validation_plan.json`](code/configs/androidworld_validation_plan.json)
- GUI-Owl native logits/history smoke: [`data/results/gui_owl_native_smoke/README.md`](data/results/gui_owl_native_smoke/README.md)
- GUI-Owl model-default native-resolution smoke: [`data/results/gui_owl_native_resolution_smoke/README.md`](data/results/gui_owl_native_resolution_smoke/README.md)
- AndroidWorld environment/reward smoke: [`data/results/androidworld_environment_smoke/README.md`](data/results/androidworld_environment_smoke/README.md)
- GUI-Owl AndroidWorld validation smoke: [`data/results/gui_owl_androidworld_validation_smoke/README.md`](data/results/gui_owl_androidworld_validation_smoke/README.md)
- GUI-Owl configuration-invalid validation audit: [`data/results/gui_owl_androidworld_validation_attempt2/README.md`](data/results/gui_owl_androidworld_validation_attempt2/README.md)
- GUI-Owl native-resolution validation rejection: [`data/results/gui_owl_androidworld_validation/README.md`](data/results/gui_owl_androidworld_validation/README.md)
- GUI-Owl Think strict-parser smoke: [`data/results/gui_owl_1_5_8b_think_smoke_strict/README.md`](data/results/gui_owl_1_5_8b_think_smoke_strict/README.md)
- GUI-Owl Think passing native smoke: [`data/results/gui_owl_1_5_8b_think_smoke/README.md`](data/results/gui_owl_1_5_8b_think_smoke/README.md)
- GUI-Owl Think AndroidWorld validation rejection: [`data/results/gui_owl_1_5_8b_think_androidworld_validation/README.md`](data/results/gui_owl_1_5_8b_think_androidworld_validation/README.md)
- Build command: `make paper`
- Test command: `make test validate-contract`
- 当前状态：论文骨架、实验契约、synthetic estimator validation 与 GUIOdyssey pilot 已完成；independent UI-TARS reference 以 27/75、swipe 0/2 判负，oracle 未打开。当前仍无 accepted validated teacher 或 CausalCache 方法效果结果，主 attribution 链路按预注册停止。

### Data and Models

| Artifact | Canonical location | Revision/status | Notes |
| --- | --- | --- | --- |
| GUIOdyssey pilot trajectory | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-pilot-mobile> | `1de9c34ff029d4c01665cdaca74436ae24bff276`，private | schema v0.3；10 screenshots、9 events、9 decisions |
| Independent GUIOdyssey gate artifact | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-independent-mobile> | `v0.1.0` / `84c9f5a335e9612ccb4bd566f977574f359b2485`，private | schema v0.4；reference 8 trajectories/75 decisions；oracle 15/132；immutable re-download verified |
| Independent UI-TARS reference run | 同一 private independent dataset repo | `reference-gate-v1` / `b3e1245c6c6a1723fe2ca3a861148008df39df46` | 69/75 parsed、27/75 match、swipe 0/2；`NO_GO_CURRENT_REFERENCE_STACK`；oracle 未运行 |
| Rejected policy candidate | <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct> | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | full-history executable-match 2/9；不作为主 teacher |
| Rejected GUI-tuned candidate | <https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B> | `683d002dd99d8f95104d31e70391a39348857f4e` | parsed 9/9、executable-match 4/9；未通过预注册 50% gate |
| Rejected computer-use candidate | <https://huggingface.co/xlangai/OpenCUA-7B> | `a2efb7d2b104d477a4a2666a357e79550a28aafc` | parsed 7/9、executable-match 1/9；未通过预注册 gate |
| Rejected GUI navigation candidate | <https://huggingface.co/showlab/ShowUI-2B> | `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60` | parsed 9/9、executable-match 2/9；未通过预注册 gate |
| Rejected AndroidWorld-native candidate | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct> | `06d5faecff74840bab2be2425e9c42667a5d04fc` | 496/496 parsed；official-success 上界 30/62，未通过 50% gate |
| Rejected replacement candidate | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Think> | `afe3707fc84caebc4d7046118b34493ecf8bb060` | 512/513 parsed；official-success 上界 29/62，未通过 50% gate |
| AndroidWorld native validation traces | <https://huggingface.co/datasets/gavinlaw/causalcache-androidworld-validation-mobile> | `v0.2.0` / `0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`，private | 42 Think traces；deterministic gzip JSONL；`v0.1.0` Instruct artifact 保持不变 |
| Full attribution/evaluation datasets | 同一 private independent dataset repo | attribution records not generated | v1 reference 已失败，禁止生成 oracle records；后续只能另立 versioned preregistration |
| Gate checkpoints/adapters | Hugging Face model repo（待创建） | not created | 记录 policy backbone、训练配置与评测 provenance |

Pilot 的生成配置见 [`code/configs/guiodyssey_pilot.json`](code/configs/guiodyssey_pilot.json)，independent
artifact 见 [`code/configs/independent_reference_gate_v1.json`](code/configs/independent_reference_gate_v1.json)。
当前没有仅存于本地、等待上传的可复用数据集、模型或评测 artifact。

## Citation

项目仍处于研究与实验阶段，正式 citation 将在论文公开后补充。
