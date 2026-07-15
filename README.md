# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

> Target venue: AAAI
> Status: restoration v2 action dependency passed / remaining pre-output dependencies pending

## 团队交接入口

当前路线已经从 v1 expert-aligned admission 切换为 versioned v2 stable self-behavior estimand。科学配置
[`code/configs/causalcache_restoration_v2.json`](code/configs/causalcache_restoration_v2.json) 已在任何 v2
policy output 前冻结，SHA256 为
`9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc`。primary policy 固定为
`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faec`；reference 只要求 restricted action 可解析、logits finite、
两次 canonical action 一致，expert alignment 仅分层报告且不能过滤状态。完整定义、data exposure 和
go/no-go 阈值见 [`docs/restoration_v2.md`](docs/restoration_v2.md)。当前没有任何 v2 policy output、
restoration label、gate checkpoint 或方法效果结果。

v2 CPU interface 已独立实现并 hash-pinned，见
[`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md) 与
[`data/manifests/restoration_v2_interfaces.json`](data/manifests/restoration_v2_interfaces.json)。14 个合法
action、23 个非法 action、6,000 个完整标量坐标检查和 decision steps 4/5/6 共 28 个 coalition（含
step-6 全部 16 个）已通过；pinned AndroidWorld `JSONAction` constructor 已在 Aries 对 14/14 payload
通过，证据见
[`data/results/restoration_v2_constructor_preflight/`](data/results/restoration_v2_constructor_preflight/)。
device-side executor dispatch 已在 Aries 正式通过 14/14 cases，negative actuation control 为 HTTP 500，
独立 reducer verdict 为 `PASSED_EXECUTOR_DISPATCH`；证据见
[`data/results/restoration_v2_executor_dispatch/`](data/results/restoration_v2_executor_dispatch/)。因此 action
dependency 已闭合；derived artifact、exact IDs、exposure、OCR、baselines 与 execution config 仍阻止 policy
inference。

exact-ID/exposure materializer 已实现为 policy-blind CPU pipeline：它必须从 pinned 16 个 Parquet 重建
完整 111-trajectory eligible pool，并逐字节复现 frozen pool SHA，不能误从只含 8+15 条 trajectory 的
parent tar 继续抽样。pipeline 固定 `decision_count>=5` 后的首 20 条、step 6、8/15/20 disjoint proof，
同时为 45 个 screening states 和 20 个 confirm states 写 image/action content witnesses；当前仍需从已推送
commit 在 Hyper00 正式 materialize 后，exact IDs 与 exposure 两项才算闭合。

executor-dispatch 的 live-inspection、negative-control 与 offline-reduction 契约见
[`docs/restoration_v2_executor_dispatch.md`](docs/restoration_v2_executor_dispatch.md)。本次 run 绑定已推送
commit `b6e57c2619e88b8646657b3190bf45853a86c3d2`，未加载 policy 或使用 GPU。

v2 的干预已收窄：所有 memory 始终保留相同 strong low-fidelity summary；恢复 event 时只增加一张
post-action state image，不增加 before image 或额外 action text。confirm 固定每条 trajectory 的 decision
step 6；events 1--4 是四个 visual candidates，event 5 的 post-state 等于 current observation，因而只保留
summary、不重复计图且不能被选择。reference 使用四张历史 post-state 加 current，覆盖每个 non-current
historical event 且不重复 event 5 的 current-equivalent image；
主预算容量为四个候选中最多选两个；exact-two 作为 cardinality-matched ablation。

历史 v1 结果保持有效且不回改：independent UI-TARS reference gate 得到 69/75 parsed、27/75（36.0%）
expert executable match、swipe 0/2，输出 `NO_GO_CURRENT_REFERENCE_STACK`。这否定的是 v1 reference
stack，不是 restoration 假设。raw 结果位于 private HF
`reference-gate-v1@b3e1245c6c6a1723fe2ca3a861148008df39df46`，轻量结论见
[`data/results/independent_reference_gate_v1/`](data/results/independent_reference_gate_v1/)。旧 15-trajectory /
132-decision oracle raw trajectories 已被 builder 读取和打包，但从未产生本项目 policy/restoration output；
v2 只按预先冻结顺序把它们用于 label-train/development screening。

更早的两状态 real-policy diagnostic 为 `INCONCLUSIVE_POSITIVE`：step 8、1024-token cap 下 exhaustive oracle
恢复 84.2%，recent/similarity 为 48.7%，random expectation 为 58.1%。由于它来自已观察 matched subset，
它仍只是一条 selection-biased existence signal，不能成为 paper `GO` 或训练 label。历史 v1 配置、artifact、
H200 anchor 和执行记录全部保留，见 [`docs/go_no_go.md`](docs/go_no_go.md) 与
[`docs/independent_gate_execution.md`](docs/independent_gate_execution.md)。

新合作者按以下顺序阅读：

1. [`docs/restoration_v2.md`](docs/restoration_v2.md)：当前 scientific contract、data roles 与 gates；
2. [`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md)：action、strong LF 与
   post-state-only prompt 的冻结实现；
3. [`docs/execution.md`](docs/execution.md)：跨芯片执行、HF/Git 回写与 Definition of Done；
4. [`docs/progress.md`](docs/progress.md)：已完成里程碑、negative results 与下一步；
5. [`docs/experiment_contract.md`](docs/experiment_contract.md)：历史 v0.3 与不变的系统边界；
6. [`docs/go_no_go.md`](docs/go_no_go.md)：历史 v1 和当前 v2 判据；
7. [`code/README.md`](code/README.md) 与 [`data/README.md`](data/README.md)：代码和数据边界；
8. [`paper/main.tex`](paper/main.tex)：AAAI 正文 source。

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
make test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces \
  validate-restoration-v2-executor-dispatch paper
```

计算 placement：v2 offline substrate/attribution 默认使用 Hyper00 H200，Aries A6000 为 fallback；任何正式
GPU output 前必须闭合 [`docs/restoration_v2.md`](docs/restoration_v2.md) 所列全部八项 pre-output
dependencies，而不只是 derived HF artifact 与 execution config。AndroidWorld closed-loop MVP 继续使用
已验证的 Aries stack。Hyper01 当前不参与本轮执行。

## 一句话主张

现有 GUI memory 方法主要根据 recency、similarity、attention 或 learned salience 保存历史。CausalCache 直接测量：**在实际部署预算附近，只给某个 GUI 事件增加 archived post-state image，能在多大程度上恢复冻结策略 parseable、finite、repeat-stable 的 self-behavior**，并把这种昂贵的 restoration attribution 蒸馏成在线 memory selector。

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

使冻结 GUI policy 的长期 executable task success 最大。这里的 $B$ 限制 policy-visible multimodal context，不限制 persistent storage；原始事件保存在 archive，未选事件在当前 query 只暴露 deterministic strong summary，并单独报告实际 text tokens。

## 方法

### 1. Restoration Attribution

首先在 action-contract parse、finite-logit 与 repeat-stability 检查通过的开发状态上运行冻结策略 $\pi_0$，得到 self-behavior reference；untouched confirm 使用固定分母，失败状态不能事后删除：

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

其中 $\mathcal C_B(j)$ 只包含为 $e_j$ 留出容量的 maximal near-budget coalitions。该定义是 budget-conditioned Shapley-style attribution；新意不在通用 Shapley estimator，而在 GUI mixed-fidelity intervention、stable policy-behavior value 和固定预算蒸馏。实际使用共享 antithetic permutation，并报告 standard error、Spearman、top-budget Jaccard、oracle utility 与 coalition reconstruction error。

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

v2 每个 low-fidelity event 固定包含：

```text
step_id
action_type
action_argument
foreground_app
screen_text_added
screen_text_removed
screen_change
executor_result
```

系统由 raw event archive、cheap summary/index（可含预计算视觉 embedding）和 policy-visible high-fidelity context 三层组成。所有 memory 中 summary 序列化 byte-identical；high fidelity 只增加一张 post-action state image。第一版不做跨层 KV surgery，而是通过 mixed-fidelity input 重新运行 policy，避免位置编码与上下文依赖导致不合法的 KV 拼接。

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

> Long-horizon GUI memory selection lacks policy-grounded supervision. CausalCache values an event by how much adding only its archived post-state image to an unchanged strong-summary history restores a frozen policy's stable self-behavior, then distills this budget-conditioned teacher into a query-time gate.

## 初始路线图

- [x] 固化问题定义、核心机制、实验主线与 claim 边界；
- [x] 建立并验证 AAAI-27 官方 LaTeX anonymous submission 骨架；
- [x] 冻结 action serialization、validated-reference requirements 与 mixed-fidelity experiment contract；
- [x] 固定并评估首个 frozen policy candidate；因 full-history coverage 仅 2/9，拒绝作为主 teacher；
- [x] 冻结 restoration v2 primary policy 与 stable self-behavior reference；GUI-Owl Instruct 仅作为 v2 substrate，不回改其 v1 AndroidWorld rejection；
- [x] 冻结 v2 restricted action、strong LF、post-state-only prompt 与 interface source hashes；
- [x] 在 pinned AndroidWorld 对 14/14 payload 闭合 `JSONAction` constructor；
- [x] 在 Aries 正式闭合 14-case executor dispatch 与 negative actuation control；
- [x] 实现并测试完整 111-pool reconstruction、exact-ID selection 与 append-only exposure materializer；
- [ ] 完成 derived artifact、exposure ledger、OCR identity、baseline hashes 与 execution config；
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
- Current restoration v2 contract: [`docs/restoration_v2.md`](docs/restoration_v2.md)
- Machine-readable v2 config: [`code/configs/causalcache_restoration_v2.json`](code/configs/causalcache_restoration_v2.json)
- Frozen v2 interface semantics: [`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md)
- Frozen v2 interface hashes: [`data/manifests/restoration_v2_interfaces.json`](data/manifests/restoration_v2_interfaces.json)
- Pinned AndroidWorld constructor preflight: [`data/results/restoration_v2_constructor_preflight/`](data/results/restoration_v2_constructor_preflight/)
- Executor-dispatch contract: [`docs/restoration_v2_executor_dispatch.md`](docs/restoration_v2_executor_dispatch.md)
- Pinned AndroidWorld executor-dispatch result: [`data/results/restoration_v2_executor_dispatch/`](data/results/restoration_v2_executor_dispatch/)
- Restoration-v2 selection materializer: [`code/causalcache/data/restoration_v2_selection.py`](code/causalcache/data/restoration_v2_selection.py)
- Historical experiment contract v0.3: [`docs/experiment_contract.md`](docs/experiment_contract.md)
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
- Test command: `make test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces validate-restoration-v2-executor-dispatch`
- 当前状态：v1 UI-TARS reference 以 27/75、swipe 0/2 判负；v2 scientific/interface contracts、CPU fixtures、pinned `JSONAction` constructor、device-side executor dispatch 与 policy-blind selection materializer 已通过。derived HF artifact、正式 exact confirm IDs/exposure 产物、OCR、baselines 与 execution config 仍未闭合，因而没有 v2 policy/restoration output 或 CausalCache 方法效果结果。

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
| GUI-Owl-1.5-8B-Instruct | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct> | `06d5faecff74840bab2be2425e9c42667a5d04fc` | v1 AndroidWorld success teacher 被拒；v2 stable self-behavior substrate 已冻结，尚无 v2 output |
| Rejected replacement candidate | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Think> | `afe3707fc84caebc4d7046118b34493ecf8bb060` | 512/513 parsed；official-success 上界 29/62，未通过 50% gate |
| AndroidWorld native validation traces | <https://huggingface.co/datasets/gavinlaw/causalcache-androidworld-validation-mobile> | `v0.2.0` / `0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`，private | 42 Think traces；deterministic gzip JSONL；`v0.1.0` Instruct artifact 保持不变 |
| Restoration v2 derived dataset | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-restoration-v2-mobile> | private repo not built | 预定保存 strong summaries、OCR/UI delta、split manifests 与 attribution records；immutable revision 前禁止 policy output |
| Gate checkpoints/adapters | Hugging Face model repo（待创建） | not created | 记录 policy backbone、训练配置与评测 provenance |

Pilot 的生成配置见 [`code/configs/guiodyssey_pilot.json`](code/configs/guiodyssey_pilot.json)，independent
artifact 见 [`code/configs/independent_reference_gate_v1.json`](code/configs/independent_reference_gate_v1.json)。
v2 derived dataset 尚未构建；当前不存在已生成却只留在本地、等待上传的 v2 data/model artifact。

## Citation

项目仍处于研究与实验阶段，正式 citation 将在论文公开后补充。
