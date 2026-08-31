# 外审:oracle 标注协议与 pair 打分头预训练方案(2026-09-01)

渠道:ChatGPT 临时聊天,推理档「极高」,思考 3m 48s。截图逐段转录。
送审方案:15k 状态量产标注(每状态 6 候选 15 组合 + recency,teacher-forcing
参考动作)→ pair 打分头 listwise 排序预训练 → RL 精调;按轨迹切分评测。

## 总判词(原文大意)

> "I would change the labeling design before spending the full 15k-state
> budget. The core idea is viable, but there are a few ways to get a very
> good selector of your offline oracle that does little—or negative—work
> for actual success."
> 并把问题拆成两层:"selector 学得会 oracle 标签"(可行)与"标签能否改善
> 在线成功率"(有多个当前评测暴露不了的辨识/分布偏移问题)。

## 潜在致命(我均采纳)

**F1. 参考动作的精确似然可能是"正确"的坏定义。** GUI 动作存在等价实现类
(同一按钮内不同坐标、快捷键 vs 菜单、等价操作顺序);log p(a_recorded)
度量的是对**单一实现**的模仿而非正确性。坐标 token 化动作尤其糟:一对好帧
可能把概率质量从记录坐标挪到同一正确按钮的另一点,反被标"更差"。

**F2. 6 候选 → 全历史部署是极值问题,不是普通分布偏移。** 训练在 15 个
组合上取 max,部署在 C(30,2)=435 乃至 C(100,2)=4950 上取 max——max 越来越
多地来自被虚高估分的组合(winner-selection bias)。要求:把"候选池规模
鲁棒性"(n=6/12/24/全历史 的 regret 曲线)做成一级评测;训练时加长历史的
hard distractors;部署强烈建议两段式:先粗检索到 M≈6–12,再枚举组合打分。

**F3. teacher-forcing 似然在量产前必须做代理有效性审计。** +0.26 nats 不
证明贪心解码的动作更正确(概率可涨而错误动作仍居 top-1、熵变化、质量来自
无关错误动作、结构化动作一部分变好一部分变坏)。**量产前取 300–500 状态,
对多个 context 比较 Δlog p(a*) 与 Δ1[解码动作正确];若关系弱,停,
更多标签修不了目标函数。**

## 严重但可修(我均采纳)

**F4. 只用成功轨迹 = 两种偏置。** 除了旧策略模仿,还有幸存者/离支撑偏置:
只在"最终成功的轨迹到过的状态"学记忆选择;部署策略一旦犯错,就进入预训练
从未覆盖的状态分布——长视野 GUI 代理的错误传播/恢复是独立问题。

**F5. 强制恰好两帧没必要且可能是错的。** 离线动作空间应为
{∅} ∪ {单帧 i} ∪ {双帧 (i,j)}:空、6 单、15 对(23 分)。单帧分尤其有价值,
可测真实协同 γ_ij = u_ij − u_i − u_j + u_∅;若 γ 多接近零,根本不需要
pair 模型。文献同向:全屏视觉历史会加重某类错误,限制到相关证据反而更好。

**F6. 按轨迹切分仍可能重泄漏。** 同任务模板的多条轨迹近乎重复。要报两个
口径:unseen trajectory/seen task 与 unseen task/template;科学主张若是
"通用历史选择",第二个口径重要得多。另查折叠摘要是否用了未来信息(标签
静默泄漏)。

**F7. 长轨迹会静默主导。** 70 状态的轨迹不该有 10 状态轨迹 7 倍的影响;
按轨迹近似均匀采样或等权;有效多样性接近"数百条轨迹"而非"15k 独立样本"。

**F8. 尽量用 executor 自己的视觉特征。** 否则学的是"编码器 A 认为编码器 B
会觉得有用的图";至少与 CLIP/DINO 做消融。

## 损失设计(反对 listwise 排序,我改判)

utilities 是**基数**不是序数,扔掉差距即扔信号。基线中心化去掉状态级
nuisance。最佳廉价损失 = **效用加权的一次性选择**:π_s(c|q)=softmax(s_c/τ),
最小化期望 regret L=Σ_c π(c|q)(u*−u_c)——平局几乎不罚、灾难性差对重罚、
与部署的一次性决策规则一致。近等价备选:软标签 CE/KL,p*(c)=softmax(u_c/τ_u),
两者对比。**平局显式处理**:ε-oracle 集 O_ε={c: u_c≥u*−ε},不对 −0.81 vs
−0.82 训 one-hot。更好的标签(若负担得起):对 hard-negative 池(recency
上下文的 top-k 解码、正确目标附近错坐标、竞争元素、其他动作类型)打
correct-action margin——"这份记忆能否把正确动作与 executor 真会犯的错
区分开";太贵则至少记录够日后构造。FOG 用聚合比(Σ增益/Σoracle增益),
不用逐状态比值均值(分母近零病态)。主指标:mean regret、聚合 FOG、
P(胜过 recency)、ε-oracle 集命中、随候选池规模的性能;top-1 一致率降为次要。

## 最小标注量(分级,不是 15k 一把梭)

1. **代理审计 ~300–500 状态**(跨轨迹广撒):验证 teacher-forcing 增益预测
   解码动作正确性;同一状态重采两套 6 候选(测候选采样方差);
2. **首轮预训练 2–4k 状态**,铺满尽可能多的轨迹、状态间隔开(300 条轨迹
   各取几个隔开的状态,远优于 50 条轨迹各取 40 个相邻状态)。
   **通过闸门:held-out selector 收回 ≥20–30% 离线 oracle 增益,按轨迹
   聚类的置信区间不含零,且数据减半→全量有可见提升**;
3. 学习曲线仍在涨再上 5–8k;"15k 是保险,不是必需"。

## 现在便宜、以后重做贵的记录(采纳入标注脚本)

空+单+对全打分(最高价值);target 段**逐 token** logprob 与 token id(标
出动作类型/坐标/参数 token——+0.26 可能全在坐标 token 里);top-k 而非全
词表;每唯一帧缓存 executor 视觉特征;时间元数据(帧步、相对当前的年龄、
组合间距、轨迹长度);任务层级元数据(轨迹 ID、任务模板、采集策略
checkpoint);摘要原文及其来源;复现元数据(checkpoint hash、prompt 序列化、
缩放设置、候选采样种子);**小型逆序实验**:几百状态上 (i,j) vs (j,i),
分数大动则时序/位置是 selector 必须显式表征的特征。

## 结论(原文)

> "The experiment I'd run next is therefore not 'label 15k.' It is:
> 500-state surrogate audit + null/single/pair labels +
> hard-negative/free-decode measurements, followed by a 2–4k-state
> trajectory-diverse learning curve. If those pass, the 15k run becomes
> much lower-risk."

## 我方处置

- 采纳:分级标注(500 审计 → 2–4k 闸门 → 视曲线加码);空+单+对 23 分
  (脚本已先行实现,与外审独立收敛);效用加权 regret 损失替代 listwise
  (与软标签 KL 对比);逐 token logprob/元数据/逆序实验入脚本;轨迹等权
  与隔开采样;双口径评测切分;候选池规模鲁棒性评测;部署两段式检索。
- 暂缓:hard-negative margin 标签(先在审计子集记录 free-decode,量产阶段
  视审计结论决定);等价类打分(若审计显示 Δlogp 预测正确性弱再上)。
- 保持:executor 视觉塔特征为第一选择(与外审 F8 同向)。
