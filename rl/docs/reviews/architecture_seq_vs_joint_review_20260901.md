# 外审:selector 架构——联合 pair 打分 vs 序贯分解(CVPR 视角,2026-09-01)

渠道:ChatGPT 临时聊天,推理档「极高」,思考 1m 53s。截图逐段转录。
背景:luojiaxuan 从 CVPR 泛化性出发主张序贯分解(一个模型任意 B 直接部署,
B=1/2/4 曲线免训练),我方原主张联合 pair 打分为主、序贯作 ablation,并
提出"贪心序贯在完美价值下占 oracle 几成"作判定实验。

## 总判词

> "I would not commit to Option B in the form you described... your proposed
> diagnostic currently confounds sequential factorization with myopic
> decoding/training."
既不选我的 A 也不选用户的 B 的字面形式,给出第三设计。

## 首选主架构(第三设计)

**预算条件化的全局集合能量模型 + 精确基数约束枚举**:
E(S|q,B) = b(q,B) + Σ_{i∈S} u_θ(q,i,B) + Σ_{i<j∈S} v_θ(q,i,j,B) + r_θ(q,S,B),
Ŝ_B = argmax_{|S|=B} E。6 候选下 C(6,1)=6、C(6,2)=15、C(6,4)=15——
**组合数小到可以全枚举:无 beam、无贪心误差、无 B 专用架构、B=2 全协同、
B=1/2/4 同一套参数**。unary+pairwise 使协同故事显式,residual 项避免假设
两两交互在 B=4 处充分。部署两段式检索把窗口压小,枚举始终精确。
"a cleaner CVPR method than either A or B: global objective, variable
budget, exact inference in the regime you actually study."

## 对双方论证的批评(各打五十大板)

**对用户的 B**:"一个模型任意 B 直接用"言过其实——能展开到 4 步不等于
泛化到 B=4:只在 1-2 步前缀上训练,3-4 步是分布偏移;要让主张成立,须跨
预算训练或给 B=4 监督。且序贯给本质无序的选择问题强加任意顺序,浪费容量、
制造多重等价标签(set-generation 文献的已知问题)。

**对我的 A**:"联合打分 = B 专用"只在字面定义 15-way pair 分类器时成立;
全局 set scorer 同样拿到协同优势而不绑死 pair。且**学出来的联合 pair 打分
头不是上界**——teacher-forcing 穷举 oracle 才是;联合网络应称"B=2 专用的
全局选择基线",不是"上界参照"。

**对我的判定实验**:c1=argmax 单帧效用是 **myopic-first greedy 的 oracle**,
不是序贯条件模型的 oracle——若模型要学 foresight,第一步的 oracle 类比是
best-completion 价值 Q(i)=max_j U({i,j}),它精确恢复 pair oracle(最优对
必含某个第一元素,链式分解无表示障碍)。应报三个量:myopic 贪心 oracle、
foresight 版(构造上=100%)、学出的序贯模型的贪心解码 vs 精确解码(区分
表示问题与搜索问题)。6 帧下序贯 pointer 的**精确解码只需枚举 30 条有序
路径**,代价可忽略;训练用 order-marginalized
P({i,j}|q)=p(i)p(j|q,i)+p(j)p(i|q,j),选最高概率的无序对——
**贪心解码由此降格为效率 ablation,不再是模型定义**。

## 判定阈值(myopic 贪心 oracle 测试)

- **≥85% oracle 增益:绿灯**,贪心序贯可作可信主架构;
- 70–85%:不足以定架构,先试 foresight 训练/精确解码;
- <70%:放弃贪心序贯作主架构。
(个人口径宁要 85–90%;0.26 nats 里 80% 意味着白扔 0.052。)
**我方预览实测(60 状态):myopic 贪心 = 91% FOG,高于绿灯线。**

## 其他要点

- R²=0.75 的重要性被高估:回归拟合 ≠ 子集决策质量,小交互项翻转两个接近
  候选就够错;**selection regret 才是中心诊断**,别报逐状态比值均值
  (oracle 改善近零的状态会爆),用聚合 η + 平均绝对 regret + 按轨迹/任务
  聚类的 bootstrap CI。
- 80 状态对架构主张太薄,同轨迹状态相关;episode 级切分 + 配对比较。
- **顺手的数据决定:把 2⁶=64 的全子集格标满**——已有 22(空+6单+15对),
  每状态只再加 42 次;直接得到 B=1/2/3/4 监督、可检验"unary+pairwise 能否
  近似 B=4 效用(是否存在高阶交互)"、budget 条件化有了精确目标。
- 论文故事:"History-frame utility is strongly non-additive; we model
  selection as cardinality-constrained set utility, explicitly capture
  interactions, and perform exact structured inference over the small
  history window." 消融梯队:additive/unary → unary+pairwise → 全集合能量
  → 序贯贪心 → 序贯精确解码 → B=2 专用联合分类器。

## 我方处置

- **采纳第三设计为推荐主架构**(待 luojiaxuan 定夺):集合能量 + 精确枚举
  同时满足他要的"一个模型全 B 曲线"与我要的"协同显式建模";他的序贯方案
  以"贪心解码=效率 ablation"的形式保留,我的联合 pair 头降格为 B=2 基线。
- **采纳全格标注**:Stage B 起每状态标满 64 子集(约 2.8× 成本,2–4k 状态
  1–2 天),无论最终选哪个架构都是监督/评测底座;需重启打分服务把
  limit-mm 从 4 提到 7。
- **采纳判定实验的修正**:91% 是 myopic 贪心的数字(仍超绿灯线);补报
  foresight 版与(训练后)贪心 vs 精确解码的分离。
- Stage A 审计照旧(其目的为代理有效性闸门,不受架构决定影响)。
