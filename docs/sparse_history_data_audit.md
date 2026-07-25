# Sparse-History HGKV 数据与训练审计文档

供外部复核。目标:让审阅者能独立判断**数据格式、变体构造、损失定义、参考臂选择**是否正确。

## 0. 一句话背景

GUI-Owl 原生就带 `last_image=5` 的**连续最近**视觉窗口(≈ 我们的 Recent-4)。本工作要问的是:
**同样 K 张高保真历史截图的预算下,从完整历史里选出的非连续截图,是否优于连续最近 K 张。**
policy 参数全程冻结,只训练一个作用于历史图 token 的 KV adapter。

## 1. 数据来源与选材

- 原始池:GUI-Odyssey,8,334 条 annotation;成功轨迹 7,862 条。
- 长度分布(成功轨迹):<16 步 4,718;16–23 步 2,286;24–31 步 656;≥32 步 202。
- 此前 v1 只冻结了 1,200 条(≥24 步仅 155、≥32 仅 29),长历史严重不足。
- **本次新选 1,326 条未渲染的长轨迹**(≥24 步 626、≥32 步 156、≥40 步 44,最长 56 步)。
- **过滤**:`action_description` 连续重复 ≥5 次的轨迹剔除(生成式字幕退化;≥32 步组中占 35%)。
  最终留下 1,144 条。

## 2. 决策点抽样(修正了一个偏置)

旧做法每条轨迹只取**最后一个**决策点 → 目标动作 97% 是 click(最后一步总是提交按钮)。
现改为每条轨迹在 **50% / 70% / 85%** 处各取一个决策点。

结果:click 69% / swipe 13% / system_button 9% / type 8% / long_press 0.4%。

## 3. 索引对齐(全部 0 基)

三者严格对应,已用真实轨迹逐条核对:

```
annotation.steps[d]  ← 第 d 步的真实动作(含精确坐标)
action_texts[d]      ← 第 d 步的自然语言描述(来自 pool 的 action_description)
images[d]            ← 执行第 d 步动作**之前**的屏幕
```

展示用的 `Step N` = 0 基索引 `N-1`。例:决策点 d=18 → 展示 `Step19`,当前图 `obs-018.png`,
选中 `Step3` 时用 `obs-002.png`。

坐标:GUI-Odyssey 注释坐标本就归一化到 [0,1000),直接映射到 [0,999],**不除设备分辨率**
(早期版本二次归一化过,已修)。官方 `run_ma35.py` 用 `src_format="qwen-vl"`,其定义为
`x/999*width`,与此一致。

## 4. 每个决策组的六个变体

规模:**3,417 个决策组 × 6 变体 = 20,502 样本**,1,144 条轨迹,3,417 组**全部六变体齐全**。
预算分布 B1/B2/B3/B4 = 481 / 968 / 674 / 1,294 组。

| 变体 | 选哪几步 | 用哪些图 | prompt 格式 | adapter |
|---|---|---|---|---|
| `native_recent{K}` | 最近 K 步 | 对应的真实图 | **官方多轮** | 强制 bypass |
| `sameformat_recent{K}` | 最近 K 步 | 对应的真实图 | **sparse 单轮** | 生效 |
| `sparse_correct` | 非连续 K 步 | 对应的真实图 | sparse 单轮 | 生效 |
| `sparse_step_shuffled` | 同 correct | correct 的图**倒序错配** | sparse 单轮 | 生效 |
| `sparse_irrelevant` | 同 correct | **另一条轨迹**的图 | sparse 单轮 | 生效 |
| `sparse_duplicate` | 同 correct | **同一张图**重复 K 次 | sparse 单轮 | 生效 |

**六变体的目标动作完全相同**,训练只优化条件之间的相对 log-prob,CE 权重为 0。

真实样例(`pair_group=0006223132449496:19`, K=3, 决策 Step19):

```
target: Action: Open the conversation with Victor James.
        <tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[391,392]}}</tool_call>

native_recent3       steps=[16,17,18]  imgs=obs-015/016/017 + 当前 obs-018
sameformat_recent3   steps=[16,17,18]  imgs=obs-015/016/017 + 当前 obs-018
sparse_correct       steps=[3,12,15]   imgs=obs-002/011/014 + 当前 obs-018
sparse_step_shuffled steps=[3,12,15]   imgs=obs-014/011/002 + 当前 obs-018   (倒序)
sparse_irrelevant    steps=[3,12,15]   imgs=另一轨迹 2625466994960519 的 obs-000/001/002 + 当前 obs-018
sparse_duplicate     steps=[3,12,15]   imgs=obs-002 ×3 + 当前 obs-018
```

已校验:3,417/3,417 组的 irrelevant 历史图**零本轨迹污染**;donor 图逐字节 sha256 比对
40/40 与原图不同;20,502 样本的图片引用**零缺失**。

约束:sparse 选点强制含至少一张 `age>4` 的老图(100% 组满足),80% 的组完全非连续。

## 5. sparse prompt 的冻结不变量(15 项单测覆盖)

```
Instruction: ...
Previous actions before Step3:
Step1: ...      ← 被跳过的步骤仍以文本保留
Step2: ...
Historical screenshot from Step3. This screenshot is not necessarily adjacent to the next screenshot.
[IMAGE step3]
Action at Step3: <该步的真实动作描述>
Intervening actions:
Step4: ...  ...  Step11: ...
Historical screenshot from Step12. ...
[IMAGE step12]
...
Current screenshot at Step19.
[IMAGE current]
```

不变量:① 选中图按原始 step 升序;② 跳过的 step 不丢不重;③ 每张图紧跟其原始动作;
④ 显式声明 sparse/non-consecutive;⑤ 当前图永远最后;⑥ 沿用官方 `Action:` + `<tool_call>`
输出协议;⑦ K=0 退化为纯文本历史;⑧ 乱序/越界/图数不符 fail-closed。

## 6. 损失定义

记 ℓc=sparse_correct、ℓr=**reference**、ℓs/ℓi/ℓd = shuffled / irrelevant / duplicate。

```
L_gain   = [0.01 − (ℓc − ℓr)]₊
L_rank   = Σₙ scaleₙ · [0.02 − (ℓc − ℓₙ)]₊        scale: shuffled=1.0, irrelevant=1.0, duplicate=0.5
L_anchor = 2.0 · Σₙ scaleₙ · SmoothL1(ℓₙ, ℓr)
L        = L_gain + L_rank + L_anchor + 1e-4·Σ(‖A‖²+‖B‖²)      CE 权重 = 0
```

实现要点:两遍法——先 no-grad 取各 ℓ 值、按 hinge/Huber 解析算次梯度权重,再逐变体在各自
`history_adapter_scope` 内带梯度前向并立即 backward(同一时刻只活一张计算图;避免梯度检查点
重算发生在 autograd 线程时 ContextVar 读空)。ℓr 无梯度。9 项数值契约测试覆盖。

架构:LM 最后 8 层的 `k_proj/v_proj`,rank 8 / alpha 16,residual 只作用于**历史图 token**,
adapter scope 为空时完全 bypass。**不从旧 checkpoint warm start**。

## 7. 参考臂的选择(本文档最需要复核的一点)

**训练 reference 用 `sameformat_recent{K}`,不是 `native_recent{K}`。** 理由是实测数据:

冻结模型上(60 组,teacher-forced log p(a*)):

| 量 | 值 |
|---|---:|
| sparse_correct − native_recent{K}(官方多轮) | **+0.160** |
| sparse_correct − sameformat_recent{K}(同格式) | **+0.027** |
| sameformat_recent{K} − native_recent{K}(纯格式效应) | **+0.133** |

即:若用官方多轮格式当 reference,**83% 的"收益"来自 prompt 结构差异而非选点**,
adapter 只要利用格式优势就能满足 `L_gain`,学不到选点。换成同格式 reference 后,训练侧
诊断的 `lc − lr` 从 +0.15 降到 +0.02~0.05,与冻结模型实测的 +0.027 吻合。

**已知未决问题**:上述 +0.133 的格式效应可能不是官方格式的性质,而是我们实现的保真度缺陷——
官方在**保留轮**的 assistant 内容里存的是**完整响应**(`Action: ...` + `<tool_call>{...}`),
我们此前只放了裸描述,造成上下文示范与目标输出格式不一致。该缺陷已修,正在重测;
在重测出数前,**不应把"官方多轮格式不利"当作结论**。

论文里三条臂各司其职:`native_recent{K}` = 部署基线(GUI-Owl 出厂行为);
`sameformat_recent{K}` = 格式对照;`sparse_correct` = 本方法。三条都要报。

## 8. 冻结模型的基线画像(60 组)

| 量 | 值 | 读法 |
|---|---:|---|
| K0 绝对值(无历史图,文本干净) | −0.624 | 基准 |
| sparse_correct − K0 | +0.011 | 加图基本中性 |
| sparse_irrelevant − K0 | +0.028 | 无关图也中性 |
| sparse_correct − sparse_irrelevant | **−0.017** | **零内容辨别力** |

最后一行是核心病症:冻结模型分不清"对的历史"和"另一条轨迹的历史"。这与本项目另一处独立
证据一致——在 AndroidWorld 闭环上,给最老的 8 张图(oldest_B8)几乎能拿到给最近 8 张图的
全部收益("有图效应" +10.9pt CI[+6.2,+16.1] 已认证,"内容效应" +4.1pt CI[−0.5,+8.8] 未认证)。
adapter 的任务就是把这个 ≈0 的内容辨别力变正。

## 9. Checkpoint 选择规则(预注册)

四条 gate 全部 > 0:`ℓc−ℓr`、`ℓc−ℓs`、`ℓc−ℓi`、`ℓc−ℓd`;
同时 wrong-history drift 受控:`|ℓs−ℓr| < 0.02`、`|ℓi−ℓr| < 0.02`。

composite = (ℓc−ℓr) + (ℓc−ℓs) + (ℓc−ℓi) − |ℓs−ℓr| − |ℓi−ℓr|

**不得只按 `ℓc−ℓr` 选点**,否则会选出"见历史就整体放大"的 checkpoint(旧 HGKV 的失败模式:
correct−shuffled 仅 +0.0016,而 shuffled/irrelevant 相对冻结模型被整体抬高约 +0.07)。

## 10. 请重点复核的问题

1. 六变体的构造是否真的只改了该改的变量?特别是 shuffled(只错配图、文本不变)与
   duplicate(同图重复)是否构成有效的负样本?
2. 用 `sameformat_recent{K}` 当训练 reference、`native_recent{K}` 当部署基线,这个分工是否正确?
   论文的主 claim 该挂在哪一条对比上?
3. 决策点按 50/70/85% 抽样是否引入了新偏置?历史长度中位 16、P90 23、最长 45 是否够支撑
   "long-horizon" 的说法?
4. 损失里 anchor 权重 2.0 是否过强(它把 wrong-history 按在 reference 上,可能压制 rank 项)?
5. `sparse_irrelevant` 取 donor 轨迹的**前 K 张**(而非按 step 对应),是否需要改成同 step 位置?

## 11. 代码位置(GitHub: luojiaxuan/CausalCache, main)

```
code/causalcache/policy/gui_owl_sparse_history.py     sparse prompt builder
code/causalcache/policy/gui_owl_official.py           官方保真 builder / parser
code/scripts/build_sparse_history_dataset.py          数据集构建(六变体)
code/scripts/select_long_horizon_trajectories.py      长轨迹选材
code/scripts/train_success_sft_lora.py                trainer(sparse_history 分支)
code/configs/causalcache_sparse_history_v1.json       超参与 gate 定义
code/tests/test_gui_owl_sparse_history.py             15 项 prompt 契约测试
code/tests/test_sparse_history_loss.py                9 项损失数值测试
```

数据:hyper00 `/data02/jaxan/artifacts/sft/sparse-v4-final/`(20,502 样本),PENDING_HF_UPLOAD。
