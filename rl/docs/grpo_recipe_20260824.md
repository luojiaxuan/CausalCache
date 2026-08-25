# 联合 GRPO 完整 recipe(送审版,2026-08-24)

## 1. 数据集

* **环境**:MobileWorld 官方 harness(唯一有信号场地;干预面 88 用例
  parity,判分用官方 scan_finished_tasks);
* **切分**:117 GUI-only 按 **app 分层**(同 app 不跨集,防 Mattermost
  登录流程等泄漏)→ **train 70 / dev 8 / heldout 39**;
  heldout 训练全程不碰,只在终局评一次;dev 用于早停与超参;
* **迁移考场**(训练后零样本):AndroidWorld(主,同为 Android)、
  OSWorld(强 OOD)、记忆专项 benchmark(MemGUI-Bench /
  AndroTMem-Bench / MementoGUI-Bench,身份与可得性入局前核实);
* **任务过滤(DAPO 式动态采样)**:只在 baseline 成功率 ∈ (0,1) 的任务上
  训练(用 multiseed recent-B2 数据估计)。全败/全成任务的组内 advantage
  恒为 0,GRPO 本来就学不到,白烧 rollout —— 76/116 双臂皆败的地板下
  这条能省约一半算力。周期性(每 ~50 step)用当前策略重估刷新任务池。
  注:这是**筛训练数据**,不是改奖励,不违反 §9。

## 2. Rollout

```
引擎        vLLM(GPU0),OpenAI 兼容端点,env 容器经 172.17.0.1 访问
采样        temperature=1.0, top_p=1.0(评测才用 greedy)
预算        B=2(history_n=3),与 P0.5 探针同口径
步数        max_round=50(与评测一致,不为省时间缩)
组结构      每题 G=8 = 4 条 selector 臂 + 4 条 recent 对照臂(见 §4)
批量        smoke: 8 题×G4;全量: 16 题×G8 = 128 rollout/step
记录        每步 (帧集合 S, messages, 响应文本, vLLM logprobs) 落盘
```

## 3. 奖励

**终局 0/1,官方判分,无任何 shaping / 步级 / gold-frame 奖励**(§9 铁律)。

## 4. ★ selector 稀疏 credit 的方案:对照臂反事实基线(非纯 RLOO)

**问题**:一个 episode 约 20-50 次选帧决策共享 1 bit 终局奖励,且该奖励
同时归功/归罪于 executor 的动作采样 —— selector 梯度信噪比极低。

**为什么纯 RLOO 不够**:RLOO(leave-one-out 组均值基线)只解决"组内
基线"的方差,**分离不出 selector 与 executor 的贡献** —— 同组各 rollout
的 selector 选择与 executor 采样同时在变,advantage 里两者纠缠。

**方案(借鉴 ATMem 的 memory-on/off 对照 rollout,改造为 on-policy)**:

* 每题 G=8 拆成 **4 条 selector 臂**(学习中的 selector 采样选帧)+
  **4 条 recent 对照臂**(固定 recent-B2,即 executor 的分布内输入);
* **selector 的 advantage** = r_i − mean(对照臂):对照臂均值估计的是
  "该任务在 executor 当前水平 + 默认记忆下"的成功率,减掉它即近似
  隔离记忆通道的反事实贡献;
* **executor 的 advantage**:8 条 rollout 全用,臂内 RLOO 基线,
  token 级 PG —— executor 从两臂都学(对照臂数据顺带让它持续适应
  自身分布,缓解漂移);
* episode 内的多次选帧决策共享该 episode 的 advantage(与 LLM RL 中
  全部 token 共享 sequence reward 同构,标准做法);
* **副产品**:对照臂就是内建的 selector-vs-recent 持续评测,训练曲线
  自带"是否已超过 recent"读数,不用额外评测。

**代价与诚实声明**:一半 rollout 不产生 selector 梯度(但产生 executor
梯度 + 基线 + 监控);闭环里无法做真正的 per-step 反事实(改一帧则后续
全变),episode 级是可行粒度的下限。

## 5. selector 架构与冷启动

* 特征:每历史帧取 executor 视觉塔 pooled embedding(不另跑 VLM,
  避免外审"假效率账"攻击)+ 步序位置编码;
* 头:41M 打分头 → **Plackett-Luce 无放回采样 B=2**;
* **冷启动(关键)**:logits 初始化带 recency 偏置,起点 ≈ recent-B2
  (executor 的分布内输入),熵正则 0.01 控制探索速度 —— 避免开局
  就把 executor 推进 OOD 深水区导致 reward 全 0 无梯度;
* selector 不加 KL(它本来就要离开初始分布),executor 加。

## 6. 优化器与同步

```
lr_executor   LoRA 1e-5(smoke)/ 全参 1e-6(32×H20)
lr_selector   1e-5
KL(executor→ref) 0.02      clip 0.2      grad_clip 1.0
on-policy:每批 1 epoch,不重放旧数据
权重同步:每 step 训后 merge/reload vLLM;验收=新旧权重生成可区分
warmup 10 step,bf16,selector weight_decay 0.01(70 题小数据防捷径)
```

## 7. 规模与墙钟

* smoke(2×H200):8 短题 × G4 × 10 step ≈ 2-3 小时,验四条
  (reward 非全 0 / 选帧分布在变 / loss·KL 有限 / vLLM 真换了权重);
* 全量(32×H20):128 rollout/step,200-400 step;
  锚点:64 rollout ≈ 40 分钟(50 步口径)→ 全量约 2-4 天,
  与 ATMem(128×H20×3 天)同量级。

## 8. 已知风险

1. 70 题过拟合/捷径(selector 偷学 app 身份、步数而非视觉需要)→
   heldout 39 + 跨 benchmark 零样本 + 机制验证(胜出案例须集中在
   缺失信息确为视觉的场合);
2. 地板效应吃掉梯度 → §1 任务过滤;
3. executor 灾难性漂移 → KL 0.02 + 对照臂持续监控 + lr 保守
   (bridge SFT -13pp 的教训);
4. 若联合训练后 selector 仍不敌 recent → 按教义回退:
   "联合 RL 试过且效果不好"成立后才允许谈冻结 executor 的方案。
