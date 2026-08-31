# 外审:margin 闸门 64.8% 与 Stage B 配方修订(2026-09-01)

渠道:ChatGPT 临时聊天,「极高」档,思考 1m 21s。截图逐段转录。

## 总判词

> "Do not scale Stage B exactly as written yet... The 64.8% itself would
> not stop me; failure of that argmax test would."

## 对 64.8 vs 65 的裁决

形式上未过:**预登记闸门记为未通过,不许看完数字把 64.8 说成 65**。
实践上不因 0.2 分杀线:CI ±2.3 下实验根本区分不了 64.8 与 65,为了判定
真值在 65.0 哪一侧而加数据是浪费算力。**换判据:上 2k 之前做两个检查——**
1. **行为 argmax 测试(决定性)**:margin 选出的上下文,其贪心解码正确率
   对比 recency——这正是杀死 raw likelihood 的那个测试的直接类比,比
   成对一致率重要;
2. **表面形式鲁棒性**:把 NL 动作描述改写/规范化(语义不变),若上下文
   排序大幅翻转,64.8% 是语言建模伪影而非语义判别。另做长度/模板匹配
   对照。p>0.65 的显著性检验不必再做。

## Stage B 配方修订(全部采纳)

- **负样本**:同一状态的所有上下文用**同一个负样本池**(否则 margin 差异
  部分反映"碰上哪个负样本"而非上下文质量);最好的廉价负样本 = 同状态、
  动作类型匹配、最小错误的替代(executor 的可信错误解码、邻近错坐标、
  错参数/文本、局部可信的其他 GUI 对象);随机错误动作基本浪费。
- **聚合软化**:max 换低温 logsumexp:S(ref) − τ·log Σ exp(S(neg)/τ)。
  只有 3 个负样本时 max 噪声大——哪个负样本"赢"会在上下文间突变。
- **不许把 margin 原始数值直接喂效用加权 regret**:审计只验证了
  **排序/符号**,没验证基数刻度。selector 主损失改**状态内排序**
  (pairwise/listwise),不用原始幅值。
- **NL 发现改监督不改推理输入**:selector 的职责仍是"给定任务/状态给帧
  子集打分",推理时没有也不该依赖 NL 描述;executor 的 NL 描述是构造
  标签时的 teacher-only 探针。JSON-only margin 保留为诊断通道,49.8% 的
  含义:margin 更接近"该上下文支持正确的内部动作假设的证据",而非
  "正确执行的概率"。

## 最大的 Stage B 风险(第 4 条)

**Goodhart/赢家诅咒:selector 在 37 个上下文上最大化一个中等代理。**
成对一致率 65% 看着体面,但选择是极值操作——observed margin 最大的
上下文,不成比例地可能是代理犯了大正误差的那个;完全可能"平均成对
一致率不错,而 argmax margin 的上下文行为上不如 recency"——与 raw
likelihood 的死法同型。因此 Stage B 审计改以 **top-of-list 行为**为主:
margin 选中上下文的解码正确率、行为最优上下文的 top-k 召回、margin
argmax 相对 recency 的解码 regret;成对一致率降为次要诊断。

## 我方处置

- 预登记闸门如实记未通过;go/no-go 改挂在**行为 argmax 闸门**上:
  (a) 先用现有 4 上下文数据算离线速版;(b) 发射正式版——500 状态给全部
  15 对补负样本打分(同状态共享负样本池 + logsumexp 聚合),对 margin
  argmax 对解码,与 recency 比正确率(~1.5h 双卡)。过则 2k,败则停线再议。
- 配方四处修订全部写入 Stage B 脚本;表面形式鲁棒性检查排在 argmax 闸门
  之后(若 argmax 已败则省)。
