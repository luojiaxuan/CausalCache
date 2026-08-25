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
"如何用非连续历史"**。方法如实命名为 *joint executor–memory policy
optimization with arm-specific RLOO/control baselines*(不是标准 GRPO)。

## 2. 方法

**决策结构**:每步 selector(41M 级 Plackett-Luce 打分头,输入各历史帧
特征+步位置)无放回采样 B=2 帧进 executor 的视觉上下文(B = history_n−1,
官方 CC_HISTORY_N=3 口径);executor(GUI-Owl-1.5-8B 全参)产生动作。
奖励 = 终局 0/1(官方 `/task/eval`,与 `scan_finished_tasks` 同源),
**无任何 shaping**;episode 内全部选帧决策与动作 token 共享终局 credit
(per-step critic 留作全量阶段的方差缩减消融)。

**估计量与更新**:

| 通道 | 估计量 | 信任域 | 其它 |
|---|---|---|---|
| selector(侧车进程) | **G=8 全 selector 臂 + RLOO**(留一均值) | **PL 联合 slate 概率**的比率裁剪(clip 0.2;不是两个边际之积) | 熵正则按**最大可行熵归一化**(候选数随步数涨,固定系数会强度漂移);AdamW lr 1e-4 |
| executor(slime/Megatron) | 组内基线(reward−组均值)/std | PPO 裁剪 0.2/0.28 + dual-clip 3.0 | 全参 bf16,lr 1e-6,KL 系数 0(信任域靠 clip) |

**任务采样**:无任何按历史成功率的硬闸门(那会恰好删掉 selector 能创造
第一次成功的任务);用**均匀采样地板(20-30%)+ 难度优先级**(按 selector
臂经验不确定度/混合组率加权);采到全同组时限次重采或换任务。

**recent 对照**:低频周期性采集(如每 5-10 step 一批),**只用于测量**
selector-vs-recent 差距,不进梯度、不做因果声明(与动作无关的 baseline
不改变梯度期望,只改方差——所以对照臂不值得花一半 rollout 预算)。

**cross-play 归因矩阵(必做)**:{初始, 终版} selector × {初始, 终版}
executor 四格评测。没有它,"记忆选择的增益"与"executor 学会了容忍非
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
| **mini-run** | train78 × G8 × 30 步 | **50 步**、batch 4 题/步、conc 32、3×H200(1 rollout + 2 train TP2+optimizer CPU offload) | 🔴 在跑 |
| 全量 | train78 × G8 × 100+ 步 | 32×H20,全参 FSDP/Megatron;selector lr 与 per-step critic 消融 | 交接同事 |

吞吐锚点(实测):32 rollout/批,rollout-bound(train_wait 占 ~70%),
批墙钟 ~25-40 分钟;权重热换 1.7-2.0s;train-sglang logprob 失配
0.034 nats(bf16 双引擎正常带内)。

## 6. 监控与验收

**每批必看**:混合组率、nonzero_return_rate、selector 归一化熵、选帧
年龄分布(CC_TRACE)、`train_rollout_logprob_abs_diff`(>0.1 报警)、
selector `rel_drift`(上线门槛参考 ≥0.02/迭代量级——低一个数量级的
lr 曾把 20 迭代跑成"重复测量随机初始化头")。

**smoke 验收四条(已过,新环境复用)**:reward 非全 0 / 选帧分布实际
在变 / loss 全程有限 / 权重真热换(生成结果与旧权重可区分)。

## 7. 风险与回退声明(预注册)

* 117 题过拟合 → heldout 39 全程不碰,主证据放跨 benchmark;
* selector 学捷径(任务长度/app 身份而非视觉需要)→ cross-play 矩阵
  + 选帧-内容机制检查;
* 若联合训练后 selector 仍不敌 recent → 回退声明:**"联合 RL 试过且
  效果不好,再退回冻结 executor 才成立"**——负结果如实报,不改判据。
