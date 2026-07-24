# Selector V1 协议(冻结,2026-07-24,含用户修订裁定)

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
