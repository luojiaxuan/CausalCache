# DCET 方案(外部设计,2026-08-12 采纳)

> 来源:用户经 ChatGPT Pro 获得的架构方案,原文收录。执行侧核验、
> 首轮实现偏差与预注册门槛见台账 §0.12;实现代码 rl/code/scripts/dcet_*。

## 核心重构

问题不再是"给每帧打分取 top-2",而是:**在候选帧完全图上,对每条帧对边
直接预测:相对 recent-2,这次替换是 0→1(赢)、1→0(输)还是无变化**。
候选 ≤30 帧、边 ≤435,全部显式枚举,无需 top-2 解码。一举解决两个已证
问题:涌现对(边分数来自两帧联合 token 交互,非 s_i+s_j)与乱动病
(recent-2 为零收益锚点,只有预测到明确正收益才替换)。

## 架构(DCET:Draft-Conditioned Counterfactual Edge Transformer)

1. **帧级多 latent 编码**:每帧 raw tokens [~2584,4096] → FrameResampler →
   K=56 latent(48 静态 = 空间锚定一半 + 全局 learned 一半;8 动态 =
   当前屏与 draft 生成的检索 query),d=512,2 层 cross-attn,8 头。
   静态 latent 帧首次出现时算一次入缓存。必须附元数据:token 二维坐标、
   帧年龄 Δt、是否 recent-2、该步动作摘要、真实 step index;候选可打乱
   (置换等变)但元数据随行。
2. **任务上下文不可省**:instruction + 全历史 action summary + 当前屏 +
   recent-2 + pass-1 draft。首选复用 pass-1 prompt hidden states(指令尾、
   动作分隔符、generation prefix 尾、动作 token 位);工程不便则回退
   GUI-Owl 冻结 token embedding + 2 层 text adapter。
3. **候选集合交互**:每帧抽 summary token 做集合级 full self-attention
   (N≤30 无需 inducing point);细节 latent 保留给 Pair Encoder;
   **绝不能从集合层输出单帧分再相加**。
4. **反事实 Pair Edge Encoder**:每对 (i,j) 4 个 pair query,cross-attn
   [Z_i, Z_j, Z_current, Z_r1, Z_r2, D, C_text, ũ_i, ũ_j];显式编码
   retained/added/removed、overlap 数、两帧时间间隔、加入-移除帧年龄差。
   学的是 Effect(把 recent-2 换成 (i,j) | draft, current, task)。
   帧对按时间序固定输入(旧、新)。
5. **边间交互**:≤435 个 edge token 上 2 层集合 Transformer(或
   edge↔endpoint 两轮消息传递)——互补/冗余判断、共享端点边交换信息、
   全体相对比较。
6. **输出**:每边 P(00),P(01),P(10),P(11)。

## 目标函数

- 收益 G = p01 − λ_harm·p10 − ε_move;G(KEEP)=0;推理仅当 max G > τ 才换。
  初始 λ=1(内层搜 {1,1.25,1.5}),ε 仅 tie-break;edge head bias 负初始化
  → 初始行为 = always-KEEP。
- 主损失 = 结构化 top-1 margin(softplus[m + smax_{a∉A*}S − smax_{a∈A*}S]),
  smax 温度退火趋近 hard max;A* = {KEEP}(recent 对时)/ 正确对集合
  (recent 错时)。
- 辅助:α·4-class CE + β·pair-correct + γ·recent-correct(+ 可选一致性);
  **pair 损失先态内平均再跨态平均**。
- 与已证伪 #3(listpos)的区别:有 draft、非加性、预测干预收益、有零收益
  KEEP 锚,不是再换一次 listwise loss。

## 推理

3 种子小 ensemble(共享静态 latent 缓存),G_LCB = mean − κ·std,
argmax 后过阈值 τ 才换。λ/τ/κ 只在外层训练集的内层 OOF 上校准。

## draft 分布编码

- 不把自回归 top-k 当完整联合分布;保存贪心动作 JSON、逐位 top-k
  (id, logit)、覆盖质量、逐 token 熵与 top1-top2 margin、parse-valid;
- 结构化 tokens:动作类型 token(type emb + margin + entropy + validity)、
  坐标 token(Fourier(x,y) + logp + rank,多候选保留多峰)、其他参数
  (button/key/text/scroll)、全局不确定性 token;
- **坐标空间偏置**:draft 的 click 坐标作为视觉注意力的高斯空间偏置
  B = −||p_t−(x,y)||²/2σ²,多 top-k 坐标多组 query。

## 数据

- 1263 态 ≈ 9.85 万边标签,但独立样本 ≈ 1263 态;够验证结构、够训
  15-35M 参数强共享模型;不够训上百 M raw-token Transformer。
  dropout/权重衰减/early stop/seed ensemble 必备;模型选择看内层 paired
  diff 不看 pair AUC。
- 扩数据优先级:①补齐 pass-1 分布(已完成);②**新增独立态**(首轮扩到
  3000-5000 winnable,AgentNet 优先,MultiApp/recent 错/长历史/稀有 UI
  加权);③再考虑容量;④不再投 B=1 逐帧 probe。
- 部分枚举策略(新态预算受限时):recent 对 + 与 recent 共端点对 +
  均匀随机对 + 当前模型 top 对 + 高不确定对 + 各 temporal-gap 桶;
  验证集必须全枚举;每态须确认至少一个正对。
- 生产化时补充 easy/hopeless 态训练 KEEP gate(不替代 winnable 主验收)。

## 流程

- **A**:pass-1 补采(固定 checkpoint hash/parser/模板/贪心配置/dtype);
  先跑 E0 诊断:P(recent 对 | draft) —— 判断 KEEP 信号是否存在。
- **B 分解式验证**:①oracle gate + learned pair(测边模型能否找对,
  尤其涌现对);②learned gate + oracle pair(测 draft 够不够判 recent
  错);③完整端到端。1 强 3 弱 → 瓶颈在 gate;1 弱 → 瓶颈在视觉/交互。
- **C 训练**:4-class+pair+recent 预热 3-5 epoch → 加 argmax 主损失
  10-20 epoch → smax 退火;batch=整态,梯度累计 16-32 态;MultiApp
  balanced sampling;候选打乱 + candidate dropout(recent-2 与当前屏
  不丢);AdamW(backbone 0.5-1e-4 / head 2-3e-4,wd 0.03-0.1,BF16,
  clip 1.0);按内层 W−L 选模型。
- **D 严格 5 折**:沿用固定划分协议;内层 OOF 做早停与 λ/τ/κ 校准;
  外层测试折只跑一次;每折 3 种子,**ensemble 本身是最终 selector**。
- **E 报告**:sel/rec acc、paired diff、CI、McNemar、W、L、W/L、move
  rate、switched hit rate、MultiApp/单应用、emergent hit@1、recent-对
  态损害率、recent-错态救回率、候选数桶、历史跨度桶、task-cluster
  bootstrap。go/no-go:OOF ≥ +3~4pp、CI 不跨零、W/L>1、超 +1.19 规则、
  MultiApp 不明显为负。

## 实验序列

| 实验 | 目的 | 部署候选 |
|---|---|---|
| E0 draft recent-correct probe | KEEP 信号是否存在 | 否(诊断) |
| E1 非加性 pair comparator | 低成本基线(PairNet(Z_i,Z_j,Z_cur,D,Z_rec) 出边收益,不得退化为 f(Z_i)+f(Z_j)) | 是 |
| E2 完整 DCET | 主候选 | **主候选** |
| E3 +动态空间检索 | 小 UI 细节瓶颈判定 | 效果优先版 |
| E4 3-seed LCB ensemble | 控移动风险 | **最终验收版** |
| E5 轻量蒸馏 | 翻正后 | 后续 |

## 算力量级(H200)

E1 1-3 GPUh/折;DCET 6-15 GPUh/折/种子;5 折 3 种子 90-225 GPUh
(5 卡并行 18-45h 墙钟)。加速:先端到端若干 epoch → 冻结 resampler →
缓存每帧 32-64 selector latent → pair/set 在小 latent 上训 → 最后解冻
微调 1-3 epoch。部署:latent 缓存 ~64KiB/帧;selector 15-35M 参数;
新帧 latent 10-50ms;pair/set 前向 10-50ms(30 帧最坏 50-200ms);
KEEP 直接复用 pass-1 贪心动作(不再跑策略),SWITCH 才追加一次策略
前向 —— 额外策略成本 ∝ move rate。新增标签:~78 对/态 × 1.75-2.5s
≈ 2.3-3.3 分钟/态;2000 态 ≈ 76-108 GPUh。

## 最终建议(原文)

第一版实现最小完整闭环:①收集 recent-2 pass-1 结构化 logits/top-k;
②每帧多个全清 raw-feature latent;③显式枚举全部帧对;④每边联合读取
两帧/当前屏/recent-2/task/history/draft;⑤预测 00/01/10/11;⑥KEEP 固定
零收益;⑦hard top-1 margin 直接优化部署 argmax;⑧ensemble LCB + 阈值
控移动;⑨oracle-gate/oracle-pair 分解定位瓶颈;⑩先在 1263 态证明结构
翻正,再优先扩独立态数。

**与全部已失败配置的本质区别:决策原子从"帧"变成"替换 recent-2 的
帧对干预",监督原子从"pair 是否正确"变成"这次干预会赢还是会输"。**
