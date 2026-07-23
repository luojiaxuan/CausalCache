# docs/archive 索引(按实验时代)

本目录存放已被 history-gated mainline(2026-07-23 起,合同见
[`../history_gated_mainline_v1.md`](../history_gated_mainline_v1.md))取代的历史实验线文档。
**没有删除任何文件**:详细文档整目录平移到这里,对应结果目录平移到
[`../../data/results/archive/`](../../data/results/archive/README.md)。旧 progress 条目见
[`progress_2026-07-20_22.md`](progress_2026-07-20_22.md)。仍然生效的文档(mainline 合同、
sealed 矩阵、osworld_* 系列、AndroidWorld partition/stack、policy 选型、共享主机规范)留在
`docs/` 顶层。

## v1 探索与 go/no-go(2026-07-14 前后)

第一轮 attribution 实验契约与 go/no-go 路线:在固定 GUIOdyssey trajectory 上给 frozen
teacher 做 restoration 干预,判定是否存在可用的 expert-coverage teacher。UI-TARS 正式输出
`NO_GO_CURRENT_REFERENCE_STACK`,四个候选 backbone(Qwen3-VL/UI-TARS/ShowUI/OpenCUA)
full-history coverage 全部低于 50% gate,路线转入 restoration v2。文件:
[`experiment_contract.md`](experiment_contract.md)、[`go_no_go.md`](go_no_go.md);policy
选型 smoke/coverage 结果见 `data/results/archive/{qwen,ui_tars,showui,open_cua,gui_owl}_*`
与 `go_no_go_diagnostic_v1*`。backbone 候选表仍留在 [`../policy_selection.md`](../policy_selection.md)。

## Restoration v2 反事实修复时代(2026-07-15 ~ 07-17)

以 stable self-behavior estimand 重建 restoration 干预:v2 接口冻结、45-state substrate、
v2.1 official-tool interface rescue、v2.2-eager 标签协议、192-state label expansion 及其
invalid-forensic / scientific-repair / 私有 HF publication 链,外加 OCR/RGB 与 policy-vision
baseline。信号存在但从未稳定超过廉价 heuristic,后续由 gate v1 与 independent 线接手判定。
文件:`restoration_v2*.md` 共 23 篇。artifacts:HF
`gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`、
`gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile`(invalid 版取证封存)、
`gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile`(修复版)、
`gavinlaw/causalcache-restoration-labels-mobile`、OCR backend
`gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en`;结果目录
`data/results/archive/restoration_v2*`。

## Gate v1 formal cache gate 时代(2026-07-17)

第一版 learned gate 的预注册、formal-58 train-only cache/training(含 transport-identity
repair)与 fresh-16 primary evaluation(含 inventory / claim-serialization repair 与 failure
decomposition)。formal-58 gate 与 fresh-16 primary 闭合后,learned gate 未能建立对
heuristic 的稳定优势,路线并入 independent gate 对比。文件:`gate_v1_*.md` 共 9 篇。
artifacts:HF `gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile`、
`gavinlaw/causalcache-gate-v1-formal58-selector-mobile`(10-checkpoint 发布)、
`gavinlaw/causalcache-gate-v1-fresh16-{claim-serialization-repair,failure-decomposition}-mobile`;
结果目录 `data/results/archive/gate_v1_*`。

## Independent gate / confirm-20 时代(2026-07-18)

论文主线一度固定为 restoration-guided independent gate;confirm-20 在 20 个 untouched
states 上正式判 `NO_GO_INDEPENDENT_CONFIRM`(learned independent selector 输给三类冻结
heuristic),随后的 continuation 与 oracle-independent failure decomposition 定位失败层级。
该负结果保留有效,exploratory closed-loop 线因此另立协议。文件:
[`independent_confirm_closed_loop_v1.md`](independent_confirm_closed_loop_v1.md)、
[`independent_confirm_continuation_v1.md`](independent_confirm_continuation_v1.md)、
[`independent_confirm_failure_decomposition_v1.md`](independent_confirm_failure_decomposition_v1.md)、
[`independent_gate_execution.md`](independent_gate_execution.md)。artifacts:HF
`gavinlaw/causalcache-independent-confirm20-mobile`、
`gavinlaw/causalcache-guiodyssey-independent-mobile`;结果目录
`data/results/archive/independent_*`。

## Spatial reference audit(2026-07-15)

GUI-Odyssey 标注空间参照的独立审计线(v1 + 离线验证修复),为 restoration/set-utility 时代
的数据可信度提供旁证,随母线一并归档。文件:
[`spatial_reference_audit_v1.md`](spatial_reference_audit_v1.md)、
[`spatial_reference_audit_v1_validation_repair.md`](spatial_reference_audit_v1_validation_repair.md)。
artifacts:HF `gavinlaw/causalcache-spatial-reference-audit-mobile`;结果目录
`data/results/archive/spatial_reference_audit_v1`。

## Set-utility / selector v0 时代(2026-07-18 ~ 07-21,最大的一条线)

由 restoration utility 监督 set predictor 的完整攻防:MVP、Freeze-B、full-pool inventory、
processor freeze、variable-history 全量 labels(11,746 states / 461,040 coalition labels)、
25% 超参与 learning curve、held-out v1 正式 `NO_GO`、contextual v3/v4、decision
distillation v2(fixed-tune 最强正信号 +0.02401 但 B2/Long+ fail)、direct marginal v3
(Stage-A/B 后最终 `NO_GO`,learned general-`B` 路线止损)、selector-side GUI-Owl LoRA
(e1 macro +0.032 但 B2/Long+ 未动)与 budget-deferral(truth read=0 前停止)。贯穿结论:
oracle headroom 始终确认(long-oracle +0.467 CI [0.359,0.630]),但没有 learned selector
稳定胜过 recent——瓶颈被判在 policy 消化历史的能力,直接催生 policy 端与 history-gated
主线。文件:`set_utility_*.md` 共 38 篇(含 long-oracle 交接与 v2 训练执行单)。
artifacts:HF datasets
`gavinlaw/causalcache-set-utility-variable-history-mobile`、
`gavinlaw/causalcache-set-utility-new-development-mobile`,HF models
`gavinlaw/causalcache-set-utility-predictors-mobile`(各 immutable revision/tag 见
仓库根 README 历史表与各结果 README);大量中间 artifact 仍在 Hyper00/Hyper01
`/data02/jaxan/{runs,artifacts}/causalcache-*` 并标注 `PENDING_HF_UPLOAD`。结果目录
`data/results/archive/set_utility_*`。

## 闭环探索 / memory ceiling / margin-SFT 数据采集(2026-07-18 ~ 07-22)

confirm NO-GO 之后的闭环证据线:validation-12 exploratory closed-loop(H100 本地臂
36/60 interim,outcome-exposed dev probe)、AndroidWorld memory ceiling v1(180/180,
预注册裁决 `STORY_DEAD_CAPABILITY_BOUND`:B8−B0 净胜 0,frozen backbone 的记忆剂量
无闭环收益)、以及 margin-SFT 线的成功轨迹采集设计(policy 端多图 SFT 的数据入口,
下游 v3-e1 在 AndroidWorld 取得 +8.9pt CI,现为 mainline 的 fallback 路线)。文件:
[`exploratory_closed_loop_validation12_v1.md`](exploratory_closed_loop_validation12_v1.md)、
[`androidworld_memory_ceiling_v1.md`](androidworld_memory_ceiling_v1.md)、
[`androidworld_success_collection_v1.md`](androidworld_success_collection_v1.md)。
结果目录 `data/results/archive/{exploratory_closed_loop_validation12_local_arms_v1,androidworld_memory_ceiling_v1,success_action_recovery_gate_v1}`;
margin-SFT checkpoint 位置见根 README(Hyper00
`/data02/jaxan/runs/causalcache-margin-sft-v3/lora-epoch1.pt`,`PENDING_HF_UPLOAD`)。
Odyssey full-layer 零样本闭环失败线(0/90 两连败 → 修正版 s75 B0 翻倍)的记录仍在活跃目录
[`../../data/results/odyssey_zeroshot_closed_loop_v1/`](../../data/results/odyssey_zeroshot_closed_loop_v1/README.md),
因为 2026-07-24 的第三轮裁决与 osworld 交接文档仍引用它。

## 跨时代执行交接日志

[`execution.md`](execution.md):跨芯片执行与团队交接的滚动日志,头部是通用三层
source-of-truth 与容器约定,主体是 restoration v2 / gate v1 / independent confirm /
set-utility 各时代的执行边界,随这些时代一并归档。通用规则的现行版本见
[`../shared_host_docker_rules.md`](../shared_host_docker_rules.md) 与根 README。

## 旧 progress 条目

[`progress_2026-07-20_22.md`](progress_2026-07-20_22.md):`docs/progress.md` 在
2026-07-22 及更早的全部条目(含 2026-07-14 ~ 07-19 的早期时代),逐字迁移。
