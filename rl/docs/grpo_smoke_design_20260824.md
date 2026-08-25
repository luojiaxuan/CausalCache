# 联合 GRPO 管线设计(smoke on 2×H200 → 交接 32×H20)(2026-08-24)

## 0. 定位与用户裁定

用户裁定(2026-08-24):**不再用探针结果做 GRPO 的前置闸门**,直接在
hyper00 的 2 张 H200 上把联合 GRPO 管线跑通(smoke),然后交接给同事在
32×H20 上跑全量。探针(multiseed best-of-N)降级为并行的旁证,在第 3 张
卡(GPU6 vLLM)上慢慢跑。

**关键论证(用户提出,成立)**:所有 frozen-executor 探针都是双重下界——
1. **采样下界**:random 每步只从 C(t,2) 空间抽 1 个子集,best-of-N 也只是
   max 的下界;
2. **★ 分布下界(此前没挖出来的)**:executor 只见过 recent 连续窗口的
   输入分布,**非连续随机子集对 policy 是 OOD**。random-B2 比 recent-B2
   低的 -5.2pp 里,可能相当部分是 OOD 惩罚而非信息损失。
   频次证据:frozen executor 用不好随机帧 ≠ 联合训练后的 executor 用不好
   选出来的帧。**联合 GRPO 正是同时消掉这两层下界的唯一办法**——
   selector 学"选哪几帧",executor 学"如何用非连续历史"。

## 1. 训练环境与任务

* 环境:**MobileWorld 官方 harness**(唯一有信号的场地;干预面与判分
  函数都已对表,见 mw_parity_20260824.md);
* 任务:117 GUI-only 按 app 分层切 **train 78 / heldout 39**(同 app 不
  跨集);跨 benchmark 泛化考场:AndroidWorld(主)、OSWorld(强 OOD)、
  MemGUI-Bench 族(记忆专项,身份待核实:AndroTMem-Bench /
  MementoGUI-Bench / MemGUI-Bench 可能是不同 benchmark,入局前先验明);
* 奖励:**终局 0/1,官方 scan_finished_tasks 口径,无任何 shaping**(§9)。

## 2. 架构(smoke,2×H200)

```
GPU0  vLLM rollout engine(executor 当前权重,OpenAI 兼容端点)
      ← MobileWorld env 容器池(CPU,32 台)经 172.17.0.1 访问
GPU1  训练进程:executor 8B(LoRA 起步,全参为 32×H20 目标形态)+ selector
      权重同步:训练 N step → merge/reload → vLLM 换权重 → 下一轮 rollout
```

* **selector 具体形态(smoke 版)**:对每个历史帧取 executor 视觉塔的
  pooled 特征 → 41M 打分头 → Plackett-Luce 无放回采样 B=2;
  与 executor 共享同一 episode-level advantage(GRPO 组内基线)。
* **rollout 记录**:改造 agent 补丁,把每步的 (帧集合 S, 请求 messages,
  响应文本, vLLM logprobs) 落进 traj 目录 —— 这是训练数据面。
* **credit assignment(默认,已在决策日志)**:episode 内全部选帧决策与
  全部动作 token 共享终局 advantage;不做 per-step credit(需要的
  rollout 翻倍,留给全量阶段做消融)。

## 3. smoke 参数(目标是管线通,不是效果)

```
tasks=8(train 集里最短的), G=4, temperature=1.0, max_round=20
steps=10, lr_executor=1e-6(LoRA 时 1e-5), lr_selector=1e-5
KL=0.02, clip=0.2
```

## 4. smoke 验收(四条,缺一不可,全部是本周踩坑的直接产物)

1. reward 分布非全 0(全 0 无梯度 —— 76/116 双臂皆败的地板效应是真风险);
2. selector 选帧分布在训练中**实际变化**(CC_TRACE 口径验证,不是"应该变");
3. loss/KL/grad-norm 全程有限,权重确实更新(参数哈希对比);
4. **权重同步后 vLLM 服务的是新权重**(生成结果与旧权重可区分——
   这一步最容易假成功)。

## 5. 交接 32×H20 的 runbook 要点

* 全参 8B:H20 96GB × 32 卡,FSDP/Megatron 分片,slime 或 verl 承载;
  smoke 的产物是**框架无关**的三件套:MobileWorld-gym 适配器(rollout
  采集 + reward 提取)、selector 模块、权重同步协议 —— 同事换框架时
  这三件直接复用;
* env 池:emulator 是 CPU 负载,H20 机器的 CPU 核数决定并发,
  按 2.5 核/台 估算;**就绪判据必须用容器内 adb devices,不能数容器**
  (2026-08-24 事故:64 台容器 HTTP 起了但 emulator 全没启动,4 个 run
  全废且被 .done 假标记);
* 吞吐锚点:G=8 × batch 8 题 = 64 rollout/step ≈ 40 分钟(50 步口径);
  100 step ≈ 67 小时 —— 与 ATMem 的 128×H20×3 天同量级,是真实成本。

## 6. 风险与对冲(如实)

* 117 题过拟合:heldout 39 全程不碰;主证据放跨 benchmark;
* selector 可能学到捷径(任务长度/app 身份而非视觉需要):留机制验证
  (外审第 4 条);
* 若联合训练后 selector 仍不敌 recent:回退声明按用户教义 ——
  "联合 RL 试过且效果不好,再退回冻结 executor 才成立"。
