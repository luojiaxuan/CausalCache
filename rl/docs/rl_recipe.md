# Joint Executor–Memory Policy Optimization Recipe(交接版,2026-08-25)

> 本文是 RL 线的**唯一现行方法文档**,写给接手者:动机→方法→数据→考场
> →参数→监控与验收→回退声明。方法演化的审计痕迹在
> `grpo_recipe_audit_20260824.md`(外审记录),不必读它也能开工。
> 实现与复现:`rl/cua/README.md`;运行事故账:`cua_lite_integration_20260825.md`。

## 1. 动机:为什么必须联合训练

冻结 executor 的一切选帧探针都是**双重下界**:

1. **采样下界**:random / best-of-N 每步只从 C(t,B) 子集空间抽极少数样本;
2. **分布下界**:executor 只在 recent 连续窗口的输入分布上训过,
   **非连续帧子集对它是 OOD**——random-B2 比 recent-B2 低的 -5.2pp 里,
   可能相当部分是 OOD 惩罚而非信息损失。

因此"冻结 executor 下 selector 不敌 recent"不能证伪记忆选择;联合 RL
是同时消掉两层下界的唯一办法:**selector 学"选哪几帧",executor 学
"如何用非连续历史"**。

另一头,"不选、全塞"也被实测封死(集成台账 §4.7/4.8):可行 69 题上
recent-B11 比 B2 高 +16-19pp(选帧的靶子),但 41% 任务 32k 装不下
B11;对这些长任务逐级退到最大可行 B=10 的 max-fit 臂只有 4.3%,反而
**劣于 B2 的 ~9.9%(三轮方向一致)**——塞满上下文在长任务上是负收益。
**小 B + 学会选,是唯一能同时吃两头的设计。**方法如实命名为 *joint executor–memory policy
optimization with arm-specific RLOO/control baselines*(不是标准 GRPO)。

## 2. 方法

**决策结构**:每步 selector 无放回采样 B=2 帧进 executor 的视觉上下文
(B = history_n−1,官方 CC_HISTORY_N=3 口径);executor(GUI-Owl-1.5-8B
全参)产生动作。

> ⚠ **现行 selector 实现有两个已证实的结构性缺陷(2026-08-31),接手者
> 务必先读 `summary_retrieval_design_20260831.md` 再动手**:
> 1. **表示残废**:输入是截图转灰度、缩到 **64×64**(压缩比 1898:1,
>    40-50px 高的文字只剩约 1 像素),再过一个**随机初始化、从不训练的**
>    2 层 MLP。可训练参数实为 **0.1M**(编码器 1.1M 冻结),**并非此前
>    文档所写的 41M**。这是 smoke 阶段的临时件,规模阶段从未替换。
> 2. **打分结构不成立**:逐帧独立打分再按序联乘,数学上假设两帧价值
>    可加;而帧效用探针实测**交互效应占 oracle 收益的 61-79%**,即多数
>    收益来自组合而非各帧之和。**换更好的视觉特征也救不了这个结构**,
>    选择器必须改为组合感知(见设计文档 §5 的三种可选结构)。
>
> 探针同时给出该路线的**非零天花板**:oracle 双帧比 recency 高
> **+0.142 nats**(95%CI [+0.111,+0.173]),**85% 的状态存在更优组合** ——
> 信息确实在历史帧中,只是当前 selector 提不出来。
奖励 = 终局 0/1(官方 `/task/eval`,与 `scan_finished_tasks` 同源),
**无任何 shaping**;episode 内全部选帧决策与动作 token 共享终局 credit
(per-step critic 留作全量阶段的方差缩减消融)。

**估计量与更新**:

| 通道 | 估计量 | 信任域 | 其它 |
|---|---|---|---|
| selector(侧车进程) | **G=8 全 selector 臂 + RLOO**(留一均值) | **PL 联合 slate 概率**的比率裁剪(clip 0.2;不是两个边际之积)。⚠ 注意:比率口径正确,但**打分函数本身是逐帧独立的**,见上方缺陷 2 | 熵正则按**最大可行熵归一化**(候选数随步数涨,固定系数会强度漂移);AdamW lr:mini 用 1e-4 实测过小(930 步权重漂 3.2%、行为零位移)。**全量起步 3e-4 + 行为门控升档 1e-3**(外审改判,fullrun_launch_review_20260826):连续 10 个 selector 批满足"固定态探针位移≈0 + PL clip 占比<10% + 熵健康"三条才升;clip 占比已高时加 lr 是反向修复,先查 logit 温度/优势缩放 |
| executor(slime/Megatron) | 组内基线(reward−组均值)/std | PPO 裁剪 0.2/0.28 + dual-clip 3.0 | 全参 bf16,lr 1e-6,KL 系数 0(信任域靠 clip) |

**初始化(无我方 SFT 阶段,有意设计)**:executor 起点 =
GUI-Owl-1.5-8B-Instruct 发布权重(上游已 GUI SFT,官方榜 37.6% 即此
checkpoint;但其历史格式是 recent 连续窗口——非连续 S 对它是 OOD,
教会它用非连续历史正是联合 RL 的目标之一)。selector 无 SFT:随机
初始化 PL 头 + **recency 偏置冷启动**(初始 logits 随帧龄衰减,开局
策略 ≈ recent-B2,落在 executor 分布内、继承 ~31% 的非零奖励地板)。
不做 selector SFT 的理由:MobileWorld 无枚举标签(oracle 枚举成本
不可承受);且 OSWorld 线的离线监督选帧阶梯已在部署口径全败于
recency,"先监督再 RL"的前半段被证伪过。**Contingency(现阶段不做)**:
若 RL 后 executor 仍用不动非连续历史(learned 持续低于 recent 对照且
cross-play 的"终版 exec+初始 sel"格也不涨),用已有成功轨迹按
random S 重渲染做一小段 executor SFT(只教格式不教选择),再 RL。

**任务采样**:无任何按历史成功率的硬闸门(那会恰好删掉 selector 能创造
第一次成功的任务);用**均匀采样地板(25%)+ 难度优先级**——已实现于
`sglang_omni_rl/task_priority.py` + rollout shim:convert 每批增量维护
逐任务混合组率表(`task_stats.json`,Laplace 平滑 + 访问数探索加成),
get_samples 按优先分加权选任务(批内无放回,地板保证全败任务不被永久
放弃)。无统计时严格退化为均匀;`CC_TASK_PRIORITY=0` 关闭。mini-run 以
均匀口径跑(它就是统计收集遍),其产出的混合组率表作为全量阶段的
初始权重。

**为什么 critic-free 起步而非 PPO(预答)**:PPO 的长视界优势有个
隐藏前提——critic 学得出来。VLM-GUI 场景里 V(s) 的输入是截图序列,
监督却只有同一个稀疏 0/1,早期 GAE 优势 = 噪声差分,常劣于组内基线;
且 8B 全参已顶满显存预算,第二个 critic(或共享骨干 value head 的
前向翻倍)付不起。同任务 G=8 组基线恰好吃掉环境难度混杂,50 步官方
口径下混合组率实测 ~45-50%,处在组估计量的"肥区"(30 步 cap 曾实证
"GRPO 饿死"形态,解法是环境口径而非换估计量)。让步与后手:均匀
credit 摊派是最大方差源,per-step 状态依赖 baseline(嫁接 PPO 有用的
一半,不动奖励语义)留作全量消融;slime 原生支持
`advantage_estimator=ppo`,若全量出现"混合组率健康但 loss 平台"再切换
实验,判据:critic 先证明自己学得出来,才有资格上位。

**recent 对照**:低频周期性采集(如每 5-10 step 一批),**只用于测量**
selector-vs-recent 差距,不进梯度、不做因果声明(与动作无关的 baseline
不改变梯度期望,只改方差——所以对照臂不值得花一半 rollout 预算)。

**cross-play 归因矩阵(必做,外审扩列)**:{E_0, E_t} × {S_0, S_t,
recency} 六格评测,同 heldout 实例同环境种子;另在 2-3 个 checkpoint 做
10 批 recency-训练 fork(matched-compute 对照的让步版,外审要求全程
并行臂,因预算改 fork)。没有它,"记忆选择的增益"与"executor 学会了容忍非
连续截图"不可分——而后者正是我们自己提出的 OOD 论证。

## 3. 训练环境与数据

* 环境:**MobileWorld 官方 harness**(经 CUA-Lite gym 抽象;判分与官方
  逐口径对表,复现 33.9% vs 官方 37.6%,p=0.341 无显著差异);
* **max_steps = 50(官方口径)**:官方 `mw eval --max_round 50`,对表、
  P0.5、B11 全部在 50;30 步是 smoke 期省钱口径,实测饿死奖励信号
  (批全灭率 3/4),**训练与评测一律 50**;
* 任务:117 GUI-only,**冻结切分 train 78 / heldout 39**
  (`rl/cua/fixtures/mw_split_v1.json`,seed=20260825,按主 app 分层 2:1)。
  注:"同 app 严格不跨集"与 78/39 在多 app 任务结构下不可同时成立
  (穷举:严格二分最优 63/40 且丢 14 题、heldout 被 Mastodon 统治),
  严格二分版留作备选 fixture;heldout 39 全程不碰;
* temperature 1.0(rollout)/ 0(评测)。

## 4. 闭环考场

| 考场 | 角色 | 身份与口径(2026-08-25 验明) |
|---|---|---|
| MobileWorld heldout 39 | 未见任务过拟合对照 | 官方判分,50 步,τ=0 |
| **AndroidWorld** | **跨 benchmark 主考场**(移动域,分布最近;历史基线与 frame_policy 干预面齐备) | 官方 harness |
| **MemGUI-Bench** | **记忆专项主标尺**:在线动态环境,128 任务/26 app,89.8% 任务考跨时空信息保持;MemGUI-Eval 分级 LLM-judge,SR@1/SR@3 口径 | [arXiv 2602.06075](https://arxiv.org/abs/2602.06075),[repo](https://github.com/lgy0404/MemGUI-Bench);入局前需核对其 agent 接口与我方 executor 兼容性 |
| MementoGUI-Bench | 离线记忆补充考场(200 轨迹/6,953 步,视频派生;含语义动作匹配/任务进度/记忆一致性指标)。**其方法(冻结骨干+可学记忆控制器)是最近方法竞品,必引必比** | [arXiv 2605.18652](https://arxiv.org/abs/2605.18652) |
| AndroTMem-Bench | 备选/引用(1,069 任务/34,473 步,长程 Android 锚定记忆诊断) | [arXiv 2603.18429](https://arxiv.org/abs/2603.18429) |
| OSWorld | 强 OOD 考场(桌面域;静态证据 +0.9pp null,预期保守,结果如实报) | 官方 harness,eval135 线基建齐 |

> 三个记忆 benchmark 是**三个不同的东西**(此前文档存疑已解):主选
> MemGUI-Bench(在线、移动域、专测记忆、与我方设定同构);MementoGUI
> 离线便宜且是竞品对比位;AndroTMem 作引用与备选诊断。

## 5. 阶段与参数

| 阶段 | 规模 | 参数 | 状态 |
|---|---|---|---|
| smoke | 4 题 × G8 × 3 步 | 30 步 cap(省钱口径,仅此阶段) | ✅ 四条验收全过 |
| **mini-run** | train78 × G8 × 30 步 | **50 步**、batch 4 题/步、conc 32、3×H200(1 rollout + 2 train TP2+optimizer CPU offload) | ✅ 30/30 收官,终判见台账 §4.9 |
| 全量 | train78 × G8 × 100+ 步 | hyper00 4×H200(2 train TP2 + 2 rollout);selector lr 3e-4 起步门控升档 1e-3,难度先验 warm-start;per-step critic 消融 | 待发射(外审后) |

吞吐锚点(实测):32 rollout/批,rollout-bound(train_wait 占 ~70%),
批墙钟 ~25-40 分钟;权重热换 1.7-2.0s;train-sglang logprob 失配
0.034 nats(bf16 双引擎正常带内)。

## 6. 监控与验收

**每批必看**:混合组率、nonzero_return_rate、selector 归一化熵、选帧
年龄分布(CC_TRACE)、`train_rollout_logprob_abs_diff`(>0.1 报警)、
selector `rel_drift`(参考量级;**单看权重漂移不作数**——mini 实证漂
3.2% 而行为零位移)。

**固定态探针库(外审采纳,行为位移的主判据)**:从 mini decisions 冻结
~200 个状态特征为探针库,每个 checkpoint 计算:KL(S_t‖S_0)、top-2 选集
Jaccard(vs S_0)、top-2 logit margin 分布、PL clip 占比、熵、recency
对的概率质量(替代对确定性 recency 的 KL)。写入 selector_metrics。

**判停(外审改判版,预登记)**:MDE=+3pp(绝对);批 60 是**诊断点非
kill 点**;判停需**两次连续**固定口径配对评测满足以下之一:(1) selector
固定态位移显著但 learned-vs-recency 的置信上界 < MDE 且无 executor
交互增长;(2) 3e-4→1e-3 升档后 selector 仍无法位移(梯度非零一致、无
病态 clip/熵塌缩);(3) selector 位移但与奖励无可复现关联。混合组率
**不进判停条件**(它只证 Var(R|task)>0,不证 Cov(R,∇logπ_sel)≠0)。
非连续历史能力探针(固定 random-S 批)每 checkpoint 跟踪:若"selector
有位移但零奖励关联 + executor 非连续能力零增长",启用 §2 的 random-S
重渲染 SFT contingency。

**高 reward 轨迹人工抽检(纪律,不是可选项)**:RL 的反馈延迟且间接,
指标只能推断不能证明模型学了什么——**每隔几个 checkpoint(建议每
5 个保存点)抽一批高 reward episode,逐条看截图与动作序列对不对劲**:
是否在骗评测器(提前 terminate、Q&A 靠 answer 文本碰运气、利用评测器
只查终态的漏洞)、成功是否真由选帧/操作达成。配套自动签名(OSWorld 线
验证过的口径):任务级"成功率跳升 + 成功步数塌缩"= JUMP,"全成且极快"
= FAST6,命中即人工复核。

**smoke 验收四条(已过,新环境复用)**:reward 非全 0 / 选帧分布实际
在变 / loss 全程有限 / 权重真热换(生成结果与旧权重可区分)。

## 7. TODOs(有意推迟/待测量/待接线——防遗忘清单)

**有意推迟(全量阶段做,mini 不做)**
- [ ] per-step critic / 状态依赖 baseline:episode 级共享 credit 的方差
  缩减消融(终局奖励语义不变,合法);与 selector per-step credit 同批设计;
- [x] selector lr 灵敏度:mini 已判(930 步漂移 3.2%、行为零位移);
  外审改判为 3e-4 起步 + 行为门控升档 1e-3(§2 表、§6 判据);
- [ ] 论文基线补齐(外审要求,不阻塞发射,池上并行跑):uniform-2 /
  change-point-2 / adaptive-B(装满即薄化)/ best-of-N random-S 作
  hindsight-B2 代理(oracle 枚举不可承受);random-2 已有(-5.2pp);
- [ ] 难度优先采样 warm-start:全量加载 rl/results/mini_v5/task_stats.json
  作先验(58/78 模板、混合组率 41%、18 个零混合死信号任务靠 floor 探索);
- [ ] 难度优先采样的**实跑验证**(代码已入库、单测绿,但未在真跑中生效
  过;全量首批看 `CC_PRIORITY 选中` 日志与任务分布)。

**设计解决不了、只能被 mini-run 测量**
- [ ] **选帧→成败耦合强度**(本线最根本的不确定性)。注意历史读数的
  正确标签(2026-08-27 更正):早期"+3.5pp"来自**冻结 executor** 的
  静态探针,按 §1 双重下界论证它只是**下界**(OOD+采样双重压制),
  **不构成联合训练收益的上界**——天花板多高只有联合 RL 能测。文本
  折叠削弱帧敏感度的方向性观察保留,但不得引用为定量上限。空间存在
  的更硬证据是 §4.7/4.8(信息量 +16-19pp / 长任务塞满为负)。MDE=+3pp
  的出处是外审功效论证(对照样本 CI 数 pp 宽 → 预登记最小值得效应),
  与该静态探针无关。判读口径——混合组率高(信号在)+ 行为位移在
  (在学)但 learned-vs-recent 差距不动 ⇒ 科学负结果,走 §8 回退声明,
  不是管线故障;
- [ ] executor 能否学会用非连续历史(分布下界是否可由 RL 消除):
  cross-play 矩阵的"终版 exec + 初始 sel"格是判据;不行则启用
  random-S 重渲染格式 SFT 的 contingency(§2)。

**待接线(实现存在但编排缺位)**
- [ ] **recent 对照低频测量批**:config(mobileworld_recent.yaml)就绪,
  但"每 5-10 步插一批 recent 测量"的编排未做——mini 期间用离线对照
  (P0.5 的 recent-B2 @50 步 ≈31%)代读,全量前必须接上;
- [ ] cross-play 四格评测的编排脚本;
- [ ] JUMP/FAST6 hacking 自动签名从 OSWorld 线的 rl_task_probe.py 移植;
- [ ] MemGUI-Bench agent 接口与 GUI-Owl 的兼容性核查(入局前);
- [ ] ask_user 文本格式偏差修复(interaction 44 任务入训前;GUI-only
  不受影响);
- [ ] 多节点 H20(actor-num-nodes 放开)实测;H100/H20 吞吐估算
  (handoff §5.5)用真跑校准。

## 8. 风险与回退声明(预注册)

* 117 题过拟合 → heldout 39 全程不碰,主证据放跨 benchmark;
* selector 学捷径(任务长度/app 身份而非视觉需要)→ cross-play 矩阵
  + 选帧-内容机制检查;
* 若联合训练后 selector 仍不敌 recent → 回退声明:**"联合 RL 试过且
  效果不好,再退回冻结 executor 才成立"**——负结果如实报,不改判据。
