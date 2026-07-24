# Selector V1 协议(冻结,2026-07-24,含用户修订裁定)

> **终态：`NO_GO_HGKV_SELECTOR_V1`，已于 2026-07-24T17:15:12Z 被
> full-history beam-4 V2 supersede。** 本文保留为不可改写的 V1 provenance；V1
> conditional scoring、Stage-2 launcher 与 selected-set gate 不再恢复。已产生的 exact-key
> coalition scores 只进入 V2 cache，不再按 V1 prefix distribution 继续消费 GPU。

主线 selector 的预注册协议。修订自初版计划,五处裁定全部为约束性条款。

## 1. 标签:hg-s100 重标 + B0 exact-key join

- U_act(j) = log p_hg-s100(a*|H={j}) − log p_frozen(a*|B0);旧 policy(frozen/full-layer)
  标签一律不得混入。
- B0 复用仅限 **exact join 命中**:键 = (episode/source_id, decision_step/pair_group,
  target action text, prompt/data revision, policy snapshot)。parity 只证明模型等价,
  不证明认证产物(n=1,220 heldout 配对组)覆盖 selector 全部训练决策点;未命中的 B0
  必须补打,禁止用均值或相似状态替代。

## 2. 标签质量门(主门 = 真实候选内部的可选择性)

真实候选 vs shuffled/donor 可分性**降级为辅助诊断**(history-gate 内容辨别的旁证),
不作为训 selector 的触发器。主门检查(全部在真实候选集合内部):

1. **Coverage**:每个 decision state 有 B0、每个候选有 singleton 分,无重复、无非有限值,100%;
2. **Within-state dispersion**:各 state 内 U_act 的 std/IQR、max−median、top1−top2 margin;
3. **Oracle headroom**:singleton-oracle-top1 相对 Recent-1 / Similarity-1 的真实增益,
   bootstrap CI 为正;
4. **STOP 信号**:max singleton gain ≤ 0 的 state 占比(不允许"天然选满");
5. Wrong-history control(辅助):true vs shuffled/donor 分布。

**关键触发条件:真实候选内部存在可预测、且能超过 Recent 的选择 headroom。**

## 3. 两个预注册候选(不做自由架构探索)

- **A. Cheap-feature baseline**:age/position、action type、summary/OCR 对齐、
  current-event 视觉相似度 → GBM/MLP;
- **B. HGKV-readout selector(主方法候选)**:与 policy adapter 机制对齐——
  第 28-35 层 frozen current-decision Q × 候选历史图 token 的 (K0,V0) 与 (K0+ΔK, V0+ΔV)
  (ΔK/ΔV 取自 hg-s100),逐层反事实 readout Δr_{l,j} = Attn(Q_l,K0+ΔK)(V0+ΔV) −
  Attn(Q_l,K0)V0,轻量 layer aggregator → gain + rank + positive/STOP 头。
- 两者共用同一 label set、trajectory split、loss family、训练预算、checkpoint 选择规则。
- 另一会话的自由形态 multimodal trainer 不进入预注册赛道。

## 4. 门禁指标:selected-set 真实效用,singleton 只做训练诊断

- singleton 指标(Spearman、top-1 hit、sign precision、singleton regret)仅用于训练诊断;
- **正式 Odyssey selector gate**:对每个门禁 state 与预算 B∈{1,2,4},将
  selector/Recent/Similarity/Random 各自选出的**完整集合真实渲染**,用 hg-s100 重算
  U_act(S) = log p_HG(a*|S) − log p_frozen(a*|B0),比较真实 set utility;
- 禁止以 ΣU({j}) 冒充集合效用(历史事件间存在 interaction,曾实测第二事件加入有相当
  比例降低效用);
- oracle:小候选集 exact,长历史 greedy/beam 并明确标注 approximate。

## 5. Split 表述纪律

- Odyssey heldout 已用于选 s100,再用于 selector 开发后其身份为
  **Odyssey development/tune set**,文中不得称 untouched test;
- 条件允许时从未参与 selector 训练的轨迹中固定 selector-test 子集;
- AndroidWorld / OSWorld sealed = 唯一最终零样本证据。

## 6. Ablation 结构(避免 policy × selector 组合爆炸)

- Policy-use ablation:{Frozen, Full-layer, HGKV} × {Recent, Full-history}(即 sealed 矩阵);
- Selection ablation:仅在 HGKV 主线上 {Random, Recent, Similarity, Cheap, HGKV-readout, Oracle};
- 不为 full-layer 配套 matched selector;
- **最终方法 = HGKV policy adapter + HGKV-readout selector**。

## 执行序(修订版)

1. singleton renderer(已完成,7235c20)→ 渲染 train/heldout;
2. hg-s100 打 singleton 分;B0 exact-join 复用 + 缺失补打;
3. 标签质量门(§2);
4. A/B 两候选同条件训练;singleton 诊断指标监控;
5. 正式 selector gate:B1/B2/B4 selected-set 重渲染重打分(§4);
6. 冻结 selector 架构、checkpoint、STOP 阈值;
7. hg-s100 + selector 进 sealed AndroidWorld/OSWorld(合同第 15 步)。

## 附:ody-labels 复用审计判定(2026-07-24,六路只读审计)

- **渲染本体可用**:ody-labels-single(75,628)+ ody-labels-b0(10,680)六项审计全过
  (坐标修复版确证、候选=历史事件除最新、cap 8 最近、schema/prompt 逐字节同 v2、
  b0 配对 100%、零重复零坏行;+22 决策态为盘点表中段孔洞,非终止合成);
- **b0 冻结分数可用**:另一会话 fl75-selector-labels 运行中 b0 半系 frozen 打分,
  与 U_act 分母定义一致,exact-join 五键全命中,直接入账;
- **singleton 既有分数必废**:系 full-layer odyv2-s75 打分(lora_modules=144、无
  adapter_type),违反"旧 policy 标签不得混入";已于 2026-07-24 用 hg-s100 重打
  (hyper00 12 分片 train 段 + hyper01 9 分片 heldout 段,输出
  runs/hgkv-selector-labels/)。fl75 那份属 full-layer 平行线,不入主线。

---

# Selector 协议修订 V2(2026-07-24,与用户+GPT 讨论后)

## 质量门 heldout 诊断结果(11,364 候选 / 1,621 决策态)

- [1] Coverage:b0 缺失 0、非有限 0,候选/态 min4/中位8/max8 —— PASS;
- [2] Within-state 离散度(均值):U_act std +0.0198、IQR +0.0286、max−median +0.0307、
  top1−top2 +0.0148 —— 真实候选内部**存在**差异;
- [3] **Oracle headroom over Recent-1:Δ +0.0205 CI[+0.0187,+0.0223](CI 下界远离 0,PASS)**
  —— 主触发条件成立:存在超过 Recent 的可预测选择空间。但绝对量薄(Recent-1 已达
  +0.1132 ≈ oracle +0.1337 的 85%);over Random-1 Δ +0.0316;
- [4] STOP 弱:73.4% 的态**全候选 U_act 为正**(天然想选满),仅 9.3% 态 max-gain≤0 ——
  与 wrong-history drift 一脉相承,预算纪律是下游最大风险。
- 关键限定:[3] 是 **B=1** 比较(singleton U_act = 单事件真实集合效用,合法);B>1
  绝不能靠 singleton 相加外推。

## interaction 结论:已被旧实验证实,不再重新 probe

旧实验(set-utility 时代)已确认:真实集合效用 ≠ singleton 可加;independent
marginal-score selection 即使拿精确分,也只达全局 subset optimum 的 **~85.9%**;根因是
redundancy/complementarity 被压成固定 item score 丢失。**结构结论可复用,数值标签不可复用**
(部署 policy 已换 hg-s100,冗余/互补关系、第二张图是否仍有正边际都可能变)。

因此:set-conditioned selector **不再需要 V1 B2/B4 失败来授权**,与 V1 并行准备,作为主
selector 候选。旧 Set Transformer 失败的根因是 teacher policy(frozen)内容盲、utility 无稳定
语义 —— hg-s100 已把 "use history" 做通(correct-B0 强正、内容对照转正、B0 parity),现在是
重试 interaction-aware selector 的合理时机。

## 两阶段 selector

**Stage 1 — Singleton HGKV scorer**(用当前 75K singleton 标签训):学
`Δ(j|∅)=U({j})−U(∅)`；由 B0 parity，`U(∅)=0`，实际 target 即
`log p_HG(a*|{j})−log p_frozen(a*|B0)`。四个头分别预测 gain、within-state rank、
positive 与 STOP。负责 B1、第一步、independent baseline、Stage 2 encoder 初始化，
以及 conditional-label 构造期的 shortlist top-K；**正式 inference 不按 shortlist
截断候选**，Stage 2 重排 capped inventory 中所有剩余候选。Stage 1 不作最终 B4
selector。

**B1 不变量**：singleton 与 set-conditioned 两路在 B1 完全共享 Stage 1 的
candidate rank 与 STOP 决策，必须选择同一集合并得到同一真实 `U(S)`；任何 B1 差异
都是实现或聚合错误。interaction advantage 只可能出现在 B2/B4。

**Stage 2 — HGKV Set-Conditioned Marginal Selector(主方法)**:学 Δ̂(j|S)。
- 实际 forward 输入:candidate features/mask、selected features/mask、remaining budget；
  当前 decision query 已编码进每个候选的 HGKV counterfactual readout，不作为独立 tensor
  再传入 Stage 2；
- 四个头分别预测 conditional marginal、within-coalition rank、positive 与 STOP；
- 实际推理:S=∅ 时由 Stage 1 决定第一张；后续在所有剩余候选中取最高
  `rank_score`，若 `stop_logit >= best rank_score` 则停止，否则加入该候选。
  marginal head 保留真实效用单位的 regression supervision 与分析用途，不直接与 0
  比较来做部署选择；
- B4 的第四个候选决策输入 `|S|=3`，而冻结 conditional labels 只覆盖
  `|S|∈{1,2}`；因此该步是同一 set-attention 的结构外推。最终 B4 只以完整集合重打分的
  真实 U(S) 报告，结果 README/Table 2 必须显式标注此 coverage limitation；
- 架构:轻量 set-attention —— 1 个 selected-set attention block + 1 个 candidate-query
  attention block + 小 MLP marginal head + STOP head,B≤4。**不用完整多层 Set Transformer;**
  DeepSets mean/sum pooling 仅作内部 control,不作正式候选。

## Conditional marginal 标签生成(近线性,非指数)

- S=∅:已有全部 singleton;
- 第一层 conditional edges:每态选 3-4 个 anchor first event(singleton-oracle top-1、
  singleton-scorer top-1、Recent-1、一个 diverse/random),对每 anchor i 在 shortlist top-K
  内打 Δ(j|{i}) = U(hg,{i,j}) − U(hg,{i});
- 第二层路径:仅沿 conditional-model greedy / Recent / beam-2,打 Δ(k|{i,j});
- renderer 对每个第二层 anchor `{i,j}` 显式物化 `cond_base({i,j})`；Stage-2 以该
  `cond_base` 分数作为 Δ(k|{i,j}) 的权威减数，并要求它与同集合 `cond_edge1`
  分数在 `atol=1e-8,rtol=0` 下 parity。singleton anchor 的 `cond_base` 同样必须与已冻结
  singleton score parity；任一漂移 fail closed，不进入训练；
- shortlist 用 singleton scorer(每态 top-6~8),数据量近线性。

## 正式主表(简化)

| Selector | 作用 |
|---|---|
| Recent | heuristic |
| HGKV singleton | independent baseline |
| **HGKV set-conditioned** | **proposed(主方法)** |
| Oracle | upper bound |

Cheap-feature selector 降为 appendix/开发分析,不占正文主表。

## correct-shuffled 小 = selection 瓶颈假说(paper joint story)

hg-s100 现测的是 **history use under recent retrieval**,非 **under causally useful
retrieval**。native recent 常选到无关/冗余/相似历史,故顺序正确 vs shuffled 差距天然被压小
(且 shuffled 只测时序敏感性,correct−irrelevant +0.0108 > correct−shuffled +0.0016 印证:
模型更依赖内容而非内部顺序;HGKV 无显式 temporal-role embedding)。这不是 adapter 的能力
天花板,而是 retrieval 天花板 —— 正好支撑 abstract 动机:memory selection 与 action policy
不能分开解决,二者是乘法关系。

**验证(selector 出来后,Odyssey development 分层表):**
Memory source {Recent-B, Similarity-B, Singleton/approx-set oracle, Learned selector-B} ×
{Correct-B0, Correct-Shuffled, Correct-Irrelevant};按 selection-regret 分层
(low/medium/high)。预期:high-regret 态中 selector 选出的 correct 与 corruption gap 明显扩大;
若不扩大,才更像 adapter 仍主要做 history-present amplification。
selected-set 的更强 corruption(优先级):**Selected top-B vs bottom-B**(最直接)、
Wrong-event replacement(同轨迹低 utility 真实事件、保数量/位置/格式)、Summary-image
mismatch、Temporal reversal。最关键量:**U(S_selected) − U(S_bottom/recent)**。

---

## V1 supersession 终态记录

- Stage-1 checkpoint、75,628 条 singleton HGKV readout、B0/singleton scores、全部
  conditional render 和 partial conditional scores均保留；
- heldout conditional scoring 完成 `38,836/38,836` unique identities；
- train conditional scoring 在 `160,928/219,549` unique identities 时停止，12 个
  JSONL shard 均可解析、跨 shard duplicate=0、invalid JSON=0；
- `ABORTED.json` 已写入 hyper01
  `/data02/jaxan/runs/hgkv-selector-stage2-v1/`、
  `/data02/jaxan/runs/hgkv-conditional-scores-v1/{,train/}` 和
  `/data02/jaxan/artifacts/sft/ody-labels-cond/`，四份 SHA256 均为
  `48510c1487ec266282cce15c32ae884322fc5dcd970bd7d85ca5462228078cdf`；
- V1 未启动 Stage-2 training，未生成 selector inference 或 selected-set gate；
- V1 config 和源码不修改；Stage-1 仅保留为 negative independent baseline。

后继契约见 `docs/selector_v2_beam4_protocol.md`。旧 V1 coalition score 的真实数值可在
V2 canonical exact key 完全一致时复用，但 V1 的 recent-8 inventory、singleton-proxy
prefix sampling 和缺失 edge3 的训练分布均不得复用。
