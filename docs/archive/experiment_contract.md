# CausalCache 历史实验契约 v0.3

本文档保留第一轮实验的对象、干预、预算和判定标准，不能用 v2 结果回改。当前 restoration v2 是新的
estimand，其 strong low-fidelity schema、post-state-only intervention、stable self-behavior reference 和 gates
见 [`restoration_v2.md`](restoration_v2.md) 与
[`causalcache_restoration_v2.json`](../../code/configs/causalcache_restoration_v2.json)。后续如果修改 v0.3 定义，
必须同时修改 `code/configs/phase0_contract.json`、对应测试和论文，并在 `docs/progress.md` 记录原因。

## 1. 系统边界

CausalCache 使用三层系统：

```text
raw event archive
    -> fixed-cost summary/index (+ optional cheap visual embedding)
    -> policy-visible mixed-fidelity context
```

预算 $B$ 只限制每次 policy query 可见的高保真 visual tokens，不限制 persistent storage。原始截图保存在 archive，因此历史事件可以在未来 query 被重新恢复。当前项目不研究永久 eviction。

## 2. 事件接口

低保真事件字段及顺序固定为：

```text
step_id
action_type
target_text_or_coordinate_bin
deterministic_ui_delta
result_status
```

高保真 archive event 额外记录 action 前后截图 URI、原始 action arguments、规范化 executable action、visual-token cost 和可选廉价视觉 embedding URI。低保真表示只暴露 coarse target；恢复高保真事件时必须恢复原始 action arguments，不能用 coarse canonical action 冒充原始动作。URI 必须稳定指向 Git/Hugging Face 中记录的 artifact 或测试 fixture；本地临时路径不能被写成 canonical location。

## 3. Validated Reference Policy

主实验只保留 full-history frozen policy action 与 executor-validated action 在 executable canonicalization 后一致的决策状态。必须报告：

- 总决策状态数；
- 通过验证的状态数与 coverage；
- 按 app、horizon、action type 分层后的 coverage；
- 不过滤 reference teacher 的 ablation。

成功轨迹过滤可作为辅助定义，但不能替代主实验的 executable match。

## 4. Action Behavior Distance

第一版 distance 分为 action type、target 和 text 三部分。target 映射到 UI element ID 或 coarse coordinate bin；text 做 NFKC 与空白规范化，非大小写敏感字段再做 casefold。密码或明确大小写敏感字段必须设置 `text_case_sensitive=true`。

Swipe/scroll 不比较不同 policy grammar 的原始起止坐标。它们统一为 viewport content direction：`scroll:up`、`scroll:down`、`scroll:left` 或 `scroll:right`。例如手指从屏幕底部向顶部滑动规范化为 `scroll:down`。这使等价的 swipe 与 direction-based scroll executable-match，同时仍拒绝方向错误的动作。

teacher-forced token divergence 是 canonical action path 上的 pathwise divergence，不称为完整 sequence-action KL。主文报告 executable action match；raw token KL 只作为辅助分析。

## 5. Budget-Conditioned Restoration

对事件 $j$，从能为 $c_j$ 留出容量且尽量接近预算 $B$ 的 coalition 分布 $\mathcal C_B(j)$ 中采样：

$$
G_{j,t}^{(B)} =
\mathbb E_{S\sim\mathcal C_B(j)}
\left[D_t(S)-D_t(S\cup\{j\})\right].
$$

等成本 event block 时，coalition 大小固定为 $B_{\text{event}}-1$。变长 visual-token cost 时，采样器必须满足：

1. `cost(S) + c_j <= B`；
2. 不存在尚未被选、且在剩余容量内可加入的事件；
3. 同一状态、预算和 seed 下为候选事件共享 coalition design，降低比较方差。

主配置使用 $K=16$，并扫描 $K\in\{4,8,16,32\}$ 与 5 个固定 seed。telescoping 只做实现检查；主要稳定性指标是 standard error、Spearman、top-budget Jaccard、oracle utility 与 coalition reconstruction error。

## 6. Selector Contract

gate 输入为：

$$
f_\theta(g,o_t,\tilde e_j,z_j,\Delta t,B).
$$

部署选择采用 positive-value knapsack；分数不超过阈值的事件不进入候选集。因此选择数量可以小于预算容量，负 restoration gain 不会被强制加入。

## 7. 主实验与 Negative Control

唯一主 closed-loop benchmark 是 AndroidWorld。离线 attribution/action 分析使用 GUIOdyssey 或经可复现性检查后确认的等价数据。主表必须包括：

- summary only；
- recent；
- similarity；
- 同 gate 架构、用 next-action NLL 或 salience 监督；
- 最强可复现 GUI memory baseline；
- restoration oracle；
- distilled CausalCache；
- shuffle restoration labels negative control。

所有比较共享 action policy、低保真 channel、视觉预算核算和 gate architecture（适用时）。

## 8. Matched-NLL 机制检验

在 task、budget、horizon 分层内做 NLL matching，并使用 paired bootstrap 与条件模型：

$$
\Pr(\text{success})=\sigma(
\beta_0+\beta_1\text{RestorationMass}
+\beta_2\text{NLL}
+\beta_3\text{tokens}
+\beta_4\text{horizon}).
$$

关键机制结论要求 $\beta_1>0$ 且置信区间在合理 matching tolerance 下稳定。不得用人工挑选案例代替总体分析。

## 9. Stop / Weaken Criteria

出现任一情况就弱化或停止当前论文主张：

- restoration oracle 在固定视觉预算下不优于 recency/similarity；
- matched-NLL 后 restoration mass 不再解释成功率；
- distilled gate 无法接近 oracle；
- 结果来自更多 visual tokens 或 selector 容量；
- $K\le 32$ 时 attribution 排名仍不稳定或成本不可接受。
