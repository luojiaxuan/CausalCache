# Sparse-History HGKV 数据与训练审计文档(预审草案)

**本文是预审草案,不是"供外部复核"的成品。** 外部审计已判定本项目**外部可复核性不通过**,
主因是:(a) 数据仍 `PENDING_HF_UPLOAD`,外部拿不到 selection / samples / 图片索引;
(b) 关键数字建立在一条低保真的官方基线上;(c) 臂定义在文档、构建脚本、trainer 三处不一致。

**第二轮状态(本次修订)**:第一轮改完之后,外部对**实现本身**又做了一轮对抗审查并实跑,
给出 10 项结论:**4 PASS / 2 PARTIAL / 4 FAIL**。4 项 FAIL 是真缺陷,其中**两项是第一轮
自己引入的回归**(官方闭环 runner 被我们收紧的 builder 打断;`history_sample_context`
在缺权威 K 时静默 bypass)。同一轮还自查出**一项审计从未发现的缺陷**:`choose_sparse`
在候选池不足时会**关闭相邻性检查**,短历史组的 `S0`/`SA` 其实是连续窗口(§12)。
逐条处置见 §13,原始外部审计的 P0/P1 处置见 §14。
第一轮曾报告"测试通过",那是**假信号**:同一套用例在真解释器下实跑时 **48 个失败**(§15)。

数字状态标签统一为三种:

- `VALID` —— 已测且不受本轮缺陷影响;
- `OBSOLETE` —— 已测但前提被推翻,**必须重测**,在重测出数前不得引用、不得进论文;
- `PENDING` —— 契约已定义但尚未测量。

目标不变:让审阅者能独立判断**数据格式、臂定义、损失定义、参考臂选择、统计口径**是否正确。

## 0. 一句话背景

GUI-Owl 原生就带 `last_image=5` 的**连续最近**视觉窗口(≈ 我们的 Recent-4)。本工作要问的是:
**同样 K 张高保真历史截图的预算下,从完整历史里选出的非连续截图,是否优于连续最近 K 张。**
policy 参数全程冻结,只训练一个作用于历史图 token 的 KV adapter。

## 1. 数据来源与选材

- 原始池:GUI-Odyssey,8,334 条 annotation;成功轨迹 7,862 条。
- **轨迹长度**分布(成功轨迹):<16 步 4,718;16–23 步 2,286;24–31 步 656;≥32 步 202。
- 此前 v1 只冻结了 1,200 条(≥24 步仅 155、≥32 仅 29),长历史严重不足。
- **本次新选 1,326 条未渲染的长轨迹**(≥24 步 626、≥32 步 156、≥40 步 44,最长 56 步)。
- **过滤**:`action_description` 连续重复 ≥5 次的轨迹整条剔除,最终留下 1,144 条。
  该过滤器是一个**未验证的选择性偏置来源**,风险与待办见 §11,不要把它当作无害的清洗步骤。

注意区分两个不同的"长度":§1 是**轨迹总步数**,§2 是**决策点处的历史深度**。分层报告一律以后者为准。

## 2. 决策点抽样与历史深度画像

旧做法每条轨迹只取**最后一个**决策点 → 目标动作 97% 是 click(最后一步总是提交按钮)。
现改为每条轨迹在 **50% / 70% / 85%** 处各取一个决策点,并要求 `d >= 6`(故 `cur = d+1 >= 7`)。

结果:click 69% / swipe 13% / system_button 9% / type 8% / long_press 0.4%。

**历史深度(决策点处已完成的步数)**:中位 **16**、P90 **23**、最长约 45
(以重建后 `manifest.json` 的 `history_len_median / _p90 / _max` 为准)。

**措辞降级(审计 P1-1)**:P90 = 23 意味着**约 90% 的决策点历史 ≤ 23 步**。这不足以支撑
无限定的 "long-horizon" 说法。全文一律改称 **mid-to-long history**;
**long-horizon 只作为显式子组**出现,定义为历史深度 **≥24** 与 **≥32** 两档,
任何 "long-horizon" 结论必须在这两个子组上单独给出,不得由总体均值代表。

所有结果按四个维度分层报告,统计口径见 §10:
抽样档(50 / 70 / 85)× K(1/2/3/4)× action type × history bin(<16 / 16–23 / 24–31 / ≥32)。

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

## 4. 五臂契约(替换原六变体表;审计 P0-3 / P0-4 的修复)

本节是**臂定义的唯一事实源**。文档、`code/scripts/build_sparse_history_dataset.py`、
`code/scripts/train_success_sft_lora.py`、`code/configs/causalcache_sparse_history_v1.json`
四处必须与本表逐字一致;此前不一致(文档写 `sameformat_recent{K}` "adapter 生效",
trainer 却对它强制 bypass)正是审计 P0-4。

| `arm_id` | prompt 格式 | 选哪些历史图 | adapter | 用途 |
|---|---|---|---|---|
| `N0` | 官方多轮 `build_official_messages` | 最近 K 张 | bypass | 部署基线(GUI-Owl 出厂行为) |
| `R0` | sparse 单轮 `build_sparse_history_messages` | 最近 K 张 | bypass | **训练 reference**、格式对照 |
| `S0` | sparse 单轮 | 非连续选点 K 张 | bypass | 冻结模型的选图效应 |
| `RA` | sparse 单轮 | 最近 K 张 | active | **主 claim 的对照臂** |
| `SA` | sparse 单轮 | 非连续选点 K 张 | active | 本方法 |

**四臂共用同一预算 K**(同一决策组内 K 相同),`N0/R0/RA` 的选点相同,`S0/SA` 的选点相同,
`R0/RA` 与 `S0/SA` 各自共享同一份 prompt 渲染,**只有 `adapter_mode` 不同**。
构建端对每个 `(prompt_format, selection_mode)` 只渲染一次再分发,所以"逐字节相同"是结构性的,
不靠事后比对(用例 `test_the_main_claim_pair_differs_only_in_adapter_mode` 覆盖)。

派生量(全部为同组配对差值,单位 nats,teacher-forced `log p(a*)`):

| 派生量 | 定义 | 读法 |
|---|---|---|
| 格式效应 | `R0 − N0` | 换成 sparse 单轮格式本身带来多少 |
| 冻结选图效应 | `S0 − R0` | 不训 adapter 时,非连续选点带来多少 |
| adapter 见图放大 | `RA − R0` | adapter 是否只是"见历史就整体抬高" |
| adapter 对 sparse | `SA − S0` | adapter 在选点条件上加了多少 |
| **主 claim** | **`SA − RA`** | **同格式、同预算、同 adapter,唯一变量是"选哪几张"** |
| 部署结果 | `SA − N0` | 相对出厂行为的端到端差距(含格式 + adapter) |

以上六个是**报告量**。config 的 `gates.derived_quantities` 里还有第七个 `SA_minus_R0`,
它**只作为 gate 量**存在(§9 的 must_pass 之一,用来挡住"adapter 连同格式 reference 都赢不了"
的 checkpoint),不进主报告表,也不得当作"选图有用"的证据。

**论文主 claim 只能挂在 `SA − RA`。** `SA − R0` 混入了 adapter 效应,`SA − N0` 还额外混入
prompt 格式效应,两者都不能当作"选图有用"的证据。`SA − N0` 只作为部署差距报告,
并必须与 `R0 − N0`、`RA − R0` 同表出现,让读者看到它由哪几部分构成。

**adapter 是否生效必须来自样本的显式字段**(`"adapter_mode": "bypass" | "active"`),
**绝不允许从 variant 名字前缀推断**(审计 P0-1)。trainer 读不到该字段或取值非法时
必须 fail-closed 报错,不得默认 bypass 或 active。

三个负样本(**训练专用,不是报告臂**),均为 sparse 单轮格式 + `adapter_mode="active"`:

| 变体 | 选哪几步 | 用哪些图 | 作用 | 最小 K |
|---|---|---|---|---|
| `SA_neg_step_shuffled` | 同 `SA` | `SA` 的图**循环移位**(文本不变) | 图文对齐是否被利用 | 2 |
| `SA_neg_irrelevant` | 同 `SA` | **另一条轨迹**中 **age 向量逐位相同**的图 | 内容辨别力 | 1 |
| `SA_neg_duplicate` | 同 `SA` | **同一张图**重复 K 次 | 压制"图越多越好" | 2 |

- `step_shuffled` 用**循环移位**而不是倒序:倒序在 K=3 时中间那张仍然对位,负样本被削弱。
- **K=1 组不生成 `step_shuffled` / `duplicate`**:K=1 时它们与 `SA` 逐字节相同(倒序/移位无效、
  重复一次即原图),会退化成正样本副本。原审计条目 P1-8,处置见 §14。
- `irrelevant` 的 donor 必须**同 split、不同 episode**,并按 **age 向量**逐位取图
  (目标 ages=[16,7,4] 就在 donor 里取同 age 的位置),锚点按相对进度对齐任务阶段,
  候选中优先同分辨率。实际匹配到的 age 向量与尺寸命中数写进样本供审计。

**所有臂与负样本的目标动作完全相同**,损失只优化条件之间的相对 log-prob,CE 权重为 0。

契约示意(取自**旧语料**的一组 `pair_group=0006223132449496:19`, K=3, 决策 Step19;
旧语料已作废,重建后需重取一组真实样例,并把 donor 的真实 episode / 文件名一并列出):

```
target: Action: Open the conversation with Victor James.
        <tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[391,392]}}</tool_call>

N0                   steps=[16,17,18]  imgs=obs-015/016/017 + 当前 obs-018   官方多轮 / bypass
R0                   steps=[16,17,18]  imgs=obs-015/016/017 + 当前 obs-018   sparse  / bypass
RA                   steps=[16,17,18]  imgs=obs-015/016/017 + 当前 obs-018   sparse  / active
S0                   steps=[3,12,15]   imgs=obs-002/011/014 + 当前 obs-018   sparse  / bypass
SA                   steps=[3,12,15]   imgs=obs-002/011/014 + 当前 obs-018   sparse  / active
SA_neg_step_shuffled steps=[3,12,15]   imgs=obs-011/014/002 + 当前 obs-018   (循环移位)
SA_neg_irrelevant    steps=[3,12,15]   imgs=另一条同 split 轨迹中 age=[16,7,4] 的三张 + 当前 obs-018
SA_neg_duplicate     steps=[3,12,15]   imgs=obs-002 ×3 + 当前 obs-018
```

约束:sparse 选点**强制两两不相邻**(step 差 ≥2)且强制含至少一张 `age>4` 的老图。
这两条现在由 `choose_sparse` 的采样空间本身保证(100%,不是"尽量"),
凑不出合法选点时**降 K 而不是放宽约束**——详见 §12,这是第二轮新修的缺陷。

**现有语料的状态:作废,待全量重建。** 旧语料是"六变体 × 3,417 组 = 20,502 样本",
它同时缺 `RA` 臂(主 claim `SA − RA` 在旧语料上**根本不可计算**)、缺 `adapter_mode` 字段、
`N0` 未接通完整历史响应(§7),且短历史组的 sparse 选点其实是连续窗口(§12)。
重建后每组 6~8 条记录(5 臂 + 该组实际存在的负样本;K=1 组只有 `irrelevant`),
最终的组数 / 样本数 / 预算分布 / 齐全率一律以重建后的 `manifest.json` 为准,本文不预填数字。
旧语料的以下校验结论在重建后需**重新执行**,不得沿用:
irrelevant 历史图零本轨迹污染、donor 图 sha256 与原图不同、图片引用零缺失。

## 5. sparse prompt 的冻结不变量

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

这八条现在各有参数化用例,实跑数字与仍然缺失的部分见 §15。

## 6. 损失定义(第二轮改动:anchor → drift,并按实际存在项归一化)

记 ℓc = `SA`(带梯度)、ℓr = **reference = `R0`**(同格式、同预算、adapter bypass、冻结无梯度)、
ℓn = 各负样本在 **adapter active** 下的分数、**ℓn⁰ = 同一条负样本在 adapter bypass 下的冻结分数**。
份额 `w_n = scale_n / Σ_m scale_m`,**只对该组实际存在的负样本归一化**。

```
L_gain  = gain_w  * [m_g − (ℓc − ℓr)]₊                      gain_w=1.0,  m_g=0.01
L_rank  = rank_w  * Σₙ wₙ * [m_r − (ℓc − ℓₙ)]₊              rank_w=1.0,  m_r=0.02
L_drift = drift_w * Σₙ wₙ * SmoothL1(ℓₙ − ℓₙ⁰, 0)           drift_w=2.0
L       = L_gain + L_rank + L_drift + 1e-4·Σ(‖A‖²+‖B‖²)     CE 权重 = 0
scale(取自样本 negative_scale 字段):shuffled=1.0, irrelevant=1.0, duplicate=0.5
```

与第一轮文档相比有**两处实质改动**,都会改变训练动力学,任何第一轮的训练侧诊断数字随之作废:

1. **anchor → drift(原审计 P1-7 的一半)**。旧式 `L_anchor = 2.0·Σ scaleₙ·SmoothL1(ℓₙ, ℓr)`
   等价于断言"shuffled / irrelevant / duplicate 在**冻结模型**上本就该等于 recent reference"。
   这条断言不成立(冻结模型对这三种条件本来就各有偏好,§8 的 −0.017 就是证据),于是 anchor 会
   把 adapter 往一个错误的常数上拽,并与 rank 项直接对冲。现在每个负样本锚到**它自己**的冻结
   分数,只惩罚 **adapter 造成的位移**,不再规定负样本该落在哪。
   代价:每个负样本多一次 no-grad bypass 前向,K≥2 的组由 9 次前向变成 12 次。
   **权重 2.0 是否过强仍未消融**,这也是这一项只判 PARTIAL 的原因(§13 第 7 条)。
2. **份额归一化**。K=1 组只有 `irrelevant`(§4),若仍按固定 scale 求和,这些组的负样本项质量
   只有别组的 40%,等于按预算给梯度加权。归一化后每组负样本总质量恒为 1,与 K、与负样本
   个数无关;某条负样本的冻结锚点取不到时它整条退出损失,剩下的重新归一化到 1。

实现要点:两遍法——先 no-grad 取各 ℓ 值、按 hinge/Huber 解析算次梯度权重,再逐变体在各自
`history_adapter_scope` 内带梯度前向并立即 backward(同一时刻只活一张计算图;避免梯度检查点
重算发生在 autograd 线程时 ContextVar 读空)。ℓr 与所有 ℓn⁰ 无梯度。
次梯度权重已用有限差分逐项校验(`test_backward_weights_match_finite_differences`)。

架构:LM 最后 8 层的 `k_proj/v_proj`,rank 8 / alpha 16,residual 只作用于**历史图 token**,
`adapter_mode="bypass"` 时 scope 为空、完全 bypass(一条乘法都不执行)。**不从旧 checkpoint warm start**。

## 7. 参考臂的选择,以及三个 OBSOLETE 数字(本文档最需要复核的一点)

**训练 reference 用 `R0`,不是 `N0`。**

此前用来支撑该决策的实测数据(冻结模型,60 组,teacher-forced log p(a*)):

| 量(旧命名 → 新臂) | 值 | 状态 |
|---|---:|---|
| `sparse_correct − native_recent{K}` → `S0 − N0` | +0.160 | **OBSOLETE,待重测** |
| `sameformat_recent{K} − native_recent{K}` → `R0 − N0`(纯格式效应) | +0.133 | **OBSOLETE,待重测** |
| "**83% 的收益来自格式效应**"(0.133 / 0.160) | — | **OBSOLETE,待重测** |
| `sparse_correct − sameformat_recent{K}` → `S0 − R0` | +0.027 | **降级为 OBSOLETE**:不涉及 `N0`,但 §12 的相邻性缺陷会稀释它,且样本量不足(见 §8) |
| `RA − R0`、`SA − RA`、`SA − S0` | — | `PENDING`:旧语料缺 `RA` 臂,从未测过 |

**作废原因(审计 P0-2)**:前三行都建立在一条**未接通完整历史响应**的 `N0` 上。
官方在**保留轮**的 assistant 内容里存的是**完整响应**(`Action: ...` + `<tool_call>{...}</tool_call>`),
我们此前只放了裸描述,造成上下文示范与目标输出格式不一致,系统性压低 `N0` 的目标 logprob。

原文写"该缺陷已修"是**错的**,真实情况是:
`build_official_messages` 的**接口层**早已支持 `past_full_responses`,
但**数据构建端从未传入**——`build_sparse_history_dataset.py` 的 `render_messages`
调用官方 builder 时缺省该参数,于是退回裸描述路径,渲染出的 `N0` 仍是低保真版本。
**构建端现已接通,且接口层已收紧为 fail-closed**:`past_full_responses=None` 只允许在
`kept == 0` 时使用;传了列表就必须与 `past_action_texts` 等长,且**每一个保留轮**都要拿到
非空完整响应,否则 `ValueError`,由调用方丢弃整个 pair-group。任何涉及 `N0` 的数字都必须在
**全量重建语料后重测**。重测前不得引用上表前三行,也**不得把"官方多轮格式不利"当作结论**
——修好保真度后格式效应可能缩小、归零甚至反号。

第四行(`S0 − R0` = +0.027)第一轮标为 `VALID`(探针级),**本轮下调为 OBSOLETE**:它虽然不涉及
`N0`,但 §12 的相邻性缺陷意味着那 60 组里凡是历史深度 ≤11(即 `cur ≤ 12`)且 K≥3 的组,`S0` 量的其实是
"较早的连续窗口 vs 最近的连续窗口",不是"稀疏 vs 连续"。这个方向会**压低** `S0 − R0` 的绝对值,
所以 +0.027 既不能当上界也不能当下界用,只能重测。

**reference 选 `R0` 的理由改为契约论证,不再依赖那个待重测的 83%**:
reference 必须与 `SA` **同 prompt 格式、同预算 K**,才能让 `ℓc − ℓr` 的唯一变量是"选哪几张";
若用 `N0` 当训练 reference,格式差异会直接灌进 `L_gain`,adapter 只要利用格式优势就能满足 hinge,
学不到选点。此结论不依赖格式效应的具体数值,只依赖"它非零且方向未知"。
`N0` 的角色仅是**报告端的部署基线**,不进训练目标。

训练侧诊断此前观察到 `ℓc − ℓr` 从 +0.15 降到 +0.02~0.05(与冻结模型的 +0.027 同量级),
该观察建立在旧语料、旧 anchor 语义(§6)之上,**一并作废**,重建 + 重训后重新记录。

## 8. 冻结模型的基线画像(60 组,探针级)

| 量(新臂命名) | 值 | 读法 |
|---|---:|---|
| K0 绝对值(无历史图,文本干净) | −0.624 | 基准 |
| `S0` − K0 | +0.011 | 加图基本中性 |
| `sparse_irrelevant`(bypass)− K0 | +0.028 | 无关图也中性 |
| `S0` − `sparse_irrelevant`(bypass) | **−0.017** | **零内容辨别力** |

最后一行是核心病症:冻结模型分不清"对的历史"和"另一条轨迹的历史"。这与本项目另一处独立
证据一致——在 AndroidWorld 闭环上,给最老的 8 张图(oldest_B8)几乎能拿到给最近 8 张图的
全部收益("有图效应" +10.9pt CI[+6.2,+16.1] 已认证,"内容效应" +4.1pt CI[−0.5,+8.8] 未认证)。
adapter 的任务就是把这个 ≈0 的内容辨别力变正。

**降级说明**(探针级):本表全部在 60 组上测得,**无置信区间、未分层、未做 episode-cluster 处理**,
且 60 组的抽取方式未预注册。这些量之间不涉及 `N0`,不受 §7 缺陷影响;但涉及 `S0` 的两行
(`S0 − K0`、`S0 − sparse_irrelevant`)受 §12 的相邻性缺陷影响,涉及 `sparse_irrelevant` 的两行
还受"donor 取前 K 张"的位置混杂影响(§14 的 P1-6)。**整表只能当探针**,不得进论文结论表。
正式数字须在重建语料 + heldout 全量上按 §10 口径重测。

## 9. Checkpoint 选择规则(预注册,第二轮起**真的可执行**)

**8 条 must_pass**(量名与 config `gates.must_pass` 逐字相同;`ℓr = R0`):

```
SA_minus_RA                    > 0        ← 主 claim
SA_minus_R0                    > 0
SA_minus_SA_neg_step_shuffled  > 0
SA_minus_SA_neg_irrelevant     > 0
SA_minus_SA_neg_duplicate      > 0
step_shuffled_drift_abs        < 0.02
irrelevant_drift_abs           < 0.02
duplicate_drift_abs            < 0.02
```

`<kind>_drift` = 留出组上"该负样本 adapter active 分 − **同一样本** adapter bypass 分"的均值,
`_drift_abs` = 其绝对值。注意这是 §6 改动的下游:drift 锚点是负样本**自己的冻结分**,不再是 `R0`。

```
composite_score = SA_minus_RA + SA_minus_R0
                + mean(SA_minus_SA_neg_step_shuffled,
                       SA_minus_SA_neg_irrelevant,
                       SA_minus_SA_neg_duplicate)
                - mean(step_shuffled_drift_abs,
                       irrelevant_drift_abs,
                       duplicate_drift_abs)

selection_rule = {filter: all_must_pass, objective: maximize,
                  quantity: composite_score, tie_break: earliest_checkpoint}
```

**不得只按 `SA_minus_R0` 选点**,否则会选出"见历史就整体放大"的 checkpoint(旧 HGKV 的失败模式:
correct−shuffled 仅 +0.0016,而 shuffled/irrelevant 相对冻结模型被整体抬高约 +0.07)。
`quantity` 在解析器里被限定只能是 `composite_score`。

**第二轮之前这一整节是不可执行的文字**(审计第 9 条,FAIL):`"> 0"` / `"< 0.02"` 是从不被解析的
字符串,`composite_score` / `selection_rule` 只被断言"非空",全仓库没有任何代码在留出集上给
`N0/S0/RA` 打过分。现在:

- 比较式 `"<op> <number> [@<statistic>]"` 与算术小语言(量名 / 数字 / `+ - * /` / 一元正负 /
  括号 / `abs()` `mean()` `min()` `max()`)有真正的解析器,写不出的表达式在 config 校验期就报错;
- `code/scripts/score_sparse_history_arms.py` 按样本 `split` 取留出组,给五臂各打一次分、
  三个负样本 **active 与 bypass 各打一次**,求派生量、执行 must_pass、算 composite、跑 selection_rule;
- **解析器只有一份**,由 trainer 导出、打分脚本导入,两侧不各写一遍;
- bypass 臂(`N0/R0/S0` 与负样本锚点)的分数与 checkpoint 无关,只算一次并缓存复用——
  这既省算力,也把"bypass 必须与 checkpoint 无关"写进了流程本身;
- 统计量默认取点估计,可用 `@ci_low` / `@ci_high` 声明保守侧,或全局切到 `ci_conservative`
  (`>` 类走 `ci_low`,`<` 类走 `ci_high`);
- `--include-identity-baseline` 会把**零初始化(恒等)adapter** 也打一遍分,作为不可选中的一行。
  它是整条打分链路的自检:恒等 adapter 下 `RA ≡ R0`、`SA ≡ S0`,因此
  **`SA_minus_RA` 必须等于 `frozen_selection_effect`**。这一行对不上,说明打分链路本身有问题,
  该次 run 的所有 gate 结论都不可信。

选点在 heldout 上进行;`RA − R0`(adapter 见图放大)是这条规则的报告端对应量,
一个健康的 checkpoint 应当 `SA − RA` 显著为正、而 `RA − R0` 不显著。

## 10. 统计口径与分层报告(审计 P1-2)

- **样本不独立**:决策组来自 1,144 条轨迹,同一轨迹贡献约 3 个决策点,
  共享 instruction、app、目标与大部分前缀历史。**绝不能把决策组当独立样本**。
- **置信区间一律用 episode-cluster bootstrap**:以 **episode(轨迹)为重抽样单位**整簇放回抽样,
  B ≥ 2,000;报告有效样本量时给 **cluster 数(轨迹数)**,组数只作为附注。
  该实现已在合成语料上验证会给出比朴素按组 bootstrap **更宽**的区间
  (6 簇 / 18 组的构造下:cluster 区间宽 0.0625,朴素 0.0353),即朴素口径会把区间压窄约 1.8×。
- **配对**:所有派生量都是**同组内配对差值**(同一 `pair_group` 的两臂相减),
  不是跨组均值之差;bootstrap 也在配对差值上做。
- **分层**:主表给总体,附表给每一层——抽样档(50 / 70 / 85)、K(1/2/3/4)、
  action type(click / swipe / system_button / type / long_press)、
  history bin(<16 / 16–23 / 24–31 / ≥32)。**long-horizon 结论只在 ≥24 与 ≥32 子组上下**,
  并同时给出这两档的 cluster 数——若 cluster 数过少,如实写"不足以定论"。
- **多重比较**:六个派生量 × 四个分层维度会产生大量比较。
  **预注册主 claim 只有 `SA − RA`**;其余全部标为次要 / 探索性,并注明未做多重比较校正。
- **逐组数据公开**:每臂的逐组 logprob 随数据一起发布(§17),让外部能重算所有派生量与 CI。

## 11. 过滤器风险:连续重复 action_description(审计 P1-3,**待修**)

规则:`code/scripts/build_sparse_history_dataset.py` 中 `max_run(action_texts) >= 5` 即**整条轨迹剔除**。

**这是一个只看生成式字幕文本的启发式**,不看动作类型、坐标、文本参数,也不看截图,
因此存在明确风险:

1. **误删合法轨迹**:很多合法任务本身就含连续重复动作——长列表连续 swipe 浏览、
   连续 click 同一个"+"加数量、逐条处理同类消息。这些步骤的动作**参数不同**(坐标 / 文本不同),
   但字幕会自然写成同一句,从而被当成"字幕退化"删掉。
2. **粒度过粗**:命中即丢整条轨迹,而不是局部裁剪或只丢受影响的决策点。
3. **与主张方向相关的选择性偏置**:**≥32 步组中被过滤的比例高达 35%**,
   且过滤率随轨迹变长而上升——它恰好削掉的是支撑"长历史有收益"主张的那部分数据,
   偏置方向**未知**(既可能高估也可能低估),不能假定无害。

**必须补做,做完之前不宣称语料无偏**:

- **人工抽样审计**:从被过滤轨迹中随机抽 ≥50 条,双人独立标注"确为字幕退化 / 合法重复动作",
  报告误删率与标注一致性;抽样对 ≥32 步组单独加权,因为那里过滤率最高。
- **报告过滤前后的分布**:按 **app / action type / domain / 长度档** 各给一张过滤前 vs 过滤后
  的分布表,并给出每格的过滤率。
- **阈值敏感性**:阈值 5 → {4, 6, 8, 关闭} 各跑一遍,报告主 claim 随阈值的变化。
- 若误删率不可忽略,改判据:用**动作参数(坐标 / 文本 / 动作类型)**判重而非字幕文本,
  或改为**局部裁剪**受影响的决策点而不是丢整条轨迹。

## 12. 稀疏选点在短历史上退化成连续窗口(**第二轮自查新发现,外部审计从未提出**)

这一条不在任何一轮外部审计的清单里,是第二轮实跑 `choose_sparse` 的采样分布时发现的。
它比 §11 更严重,因为它直接污染的是**主结果的自变量**,不是选材偏置。

**旧行为**:`choose_sparse` 在候选池 `pool = 1..cur-1` 满足 `len(pool) < 3*K` 时**关闭相邻性检查**,
只保留"至少一张 `age>4` 的老图"这一条约束。于是候选池不够宽的决策点上,"非连续选点"这个
定义悄悄失效,`S0`/`SA` 的选点变成一段(位置更早的)**连续窗口**。

**穷举全部合法选点集合**(旧规则下每个集合等概率)得到的退化率,以及现实现的实测:

| `cur` (历史深度+1) | pool | K | 旧规则合法集合数 | 旧:含相邻对 | 旧:完全连续 | 现:非连续上限 | 现:含相邻对 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 6 | 3 | 16 | 75.0% | 12.5% | 3 | 0.0% |
| 7 | 6 | **4** | 14 | **100.0%** | 14.3% | 3 | K=4 不再产出(降 K) |
| 8 | 7 | 3 | 31 | 67.7% | 9.7% | 4 | 0.0% |
| 8 | 7 | 4 | 34 | 97.1% | 8.8% | 4 | 0.0% |
| 9 | 8 | 3 | 52 | 61.5% | 7.7% | 4 | 0.0% |
| 9 | 8 | 4 | 69 | 92.8% | 5.8% | 4 | 0.0% |
| 10 | 9 | 4 | 125 | 88.0% | 4.0% | 5 | 0.0% |
| 11 | 10 | 4 | 209 | 83.3% | 2.9% | 5 | 0.0% |
| 12 | 11 | 4 | 329 | 78.7% | 2.1% | 6 | 0.0% |

`cur=7, K=4` 的 100% 是鸽笼原理的必然结果(6 个位置里挑 4 个,最大两两不相邻子集只有 3 个),
不是采样噪声。现实现的一列由 4,000 次采样实测,全部为 0.0%。
测量脚本:本地脚手架 `contig.py` / `contig2.py`(未入库,见 §15 的可复现性缺口)。

**影响范围**:退化只发生在 `pool < 3K`,即

- K=2:需要 `cur < 7`,而抽样规则要求 `d >= 6`(`cur >= 7`),**从不发生**;
- K=3:`cur <= 9`;
- K=4:`cur <= 12`。

即**历史深度 ≤ 11 且 K ≥ 3** 的决策点。§2 的历史深度中位是 16,所以受影响的是分布的下半部,
主要来自较短轨迹的 50% 决策点(`int(usable*0.5)+1 <= 12` ⇔ `usable <= 23`)。
**精确占比只能由重建后的 `manifest.json` 给出,本文不预填。**

**后果**:这些组的 `S0`/`SA` 与 `R0`/`RA` 的差别只剩"窗口位置更早",而不是"是否连续"。
于是 `S0 − R0`(冻结选图效应)与 `SA − RA`(主 claim)在这些组上量的是
**老连续窗口 vs 新连续窗口**,自变量被换掉了,效应被系统性稀释。方向可判(稀释,即绝对值偏小),
但幅度取决于受影响组的占比,不能用一个系数事后校正。§7 的 `S0 − R0` = +0.027 因此下调为 OBSOLETE。

**处置(已改)**:

1. 新增 `max_sparse_budget(cur)`:候选池大小 `P = cur-1`,两两不相邻子集的最大规模是 `ceil(P/2)`;
   老事件存在当且仅当 `P >= 5`。超过这个上限的 K 在池子里**根本不存在**合法选点。
2. `choose_sparse` 只在**真正合法**的集合上采样,采不到就返回 `[]`,不再有"放宽约束"这条分支。
   采样用组合双射(从 `1..P-k+1` 取 k 个 `c_i`,令 `s_i = c_i + (i-1)`),因此在全部两两不相邻子集上
   **均匀**——不是贪心 + 拒绝那种有偏且可能死循环的写法;老事件约束用有界拒绝采样满足。
3. 抽到的 K 超过上限时**降 K**,不是放宽约束,也不是整组拒绝。理由:降 K 只动这一组的预算,
   组内 reference 仍是**同预算**的 recent-K,`S0−R0` / `SA−RA` 依旧是干净的同预算对比
   (而且 B=1..4 本来就都要校准);整组拒绝会按 K 的抽样随机丢掉短历史决策点,白白损失样本量。
   降到 0(池子小到连一个老事件都没有)才拒绝,计入 `rejected_reasons["pool_too_dense_for_sparse"]`。
4. `manifest.json` 新增 `budget_downgrades`(形如 `{"4->3": n}`)、`budget_downgraded_decisions`、
   `pool_too_dense_rejected_decisions`,降级明细必须随数据发布。
5. 给定 `d >= 6`(`cur >= 7`,`P >= 6`,上限 ≥3),**降 K 只可能在 `cur = 7` 发生且只有 4→3**;
   `cur = 7` 只出现在轨迹总步数 ∈ {8, 9, 12, 13} 的情况(50%/70%/85% 三个抽样档各自的解)。
   新选材以长轨迹为主,预期触发极少,但仍以 manifest 的实际计数为准。

**用例**:`test_max_sparse_budget_matches_the_non_adjacent_capacity`、
`test_choose_sparse_never_degrades_into_a_recent_window`(参数化多个 `cur`)、
`test_choose_sparse_returns_empty_beyond_the_capacity`、`test_sparse_equal_to_recent_is_rejected`。

## 13. 外部审计回应:第一轮 10 项结论 × 第二轮处置

第一轮改完之后,外部对**实现**做了一轮对抗审查并实跑,给出 10 项结论。
编号沿用代码注释里的"审计第 N 条"。4 项 FAIL 的描述与代码注释逐条对应,可由注释与探针复核;
PASS / PARTIAL 行是我们对结论的**复述**,原文表述以审查方的记录为准。

| # | 结论 | 判定 | 第二轮处置 |
|---|---|---|---|
| 1 | adapter 开关只读显式 `adapter_mode` 字段,不存在按名字前缀的控制流;非法值 fail-closed | **PASS** | 保持。加了守卫用例 `test_trainer_has_no_name_prefix_control_flow`(源码级扫描)与 `test_unknown_adapter_mode_is_fail_closed` |
| 2 | 五臂 + 三负样本的 pair-group 构造只改了该改的变量;`R0/RA`、`S0/SA` 共享同一份渲染 | **PASS** | 保持。构建端对每个 `(prompt_format, selection_mode)` 只渲染一次再分发,"逐字节相同"是结构性的 |
| 3 | 损失只优化相对 log-prob:CE 权重恒为 0,全局平移不变,两遍法的次梯度权重与有限差分一致 | **PASS** | 保持。`test_loss_is_invariant_to_a_global_logprob_shift`、`test_backward_weights_match_finite_differences`、`test_backward_weight_is_divided_by_the_accumulation` |
| 4 | **官方闭环 runner 被第一轮的 builder 收紧打断,且历史配对早就错位** | **FAIL** | **已修,见下方"回归 A"** |
| 5 | `SA_neg_irrelevant` 的 donor 已改为同 split、按 age 向量对齐、优先同分辨率;但"位置混杂是否真的消除"没有对照数据 | **PARTIAL** | 第一轮已改的 donor 逻辑第二轮未再动;**两版对照消融仍未做**,必须在重建语料上补(§14 的 P1-6) |
| 6 | N0 的官方保真:每个保留轮都带 `Action:` + `<tool_call>`,重建不出来就丢弃整组 | **PASS**(builder 侧) | 保持并进一步 fail-closed。注意:正是这次收紧引发了第 4 条的回归 |
| 7 | anchor 语义可疑(锚到 `R0` 等于假设三个负样本在冻结模型上本该等于 recent reference);权重 2.0 也未消融 | **PARTIAL** | **改了语义**:每个负样本锚到自己的冻结分,`L_anchor` → `L_drift`(§6)。**权重 2.0 的消融仍未做**,故仍是 PARTIAL |
| 8 | **fail-closed 有缺口:`history_sample_context` 会静默走 bypass;缺 `R0`/`SA` 时损失静默返回 `None`** | **FAIL** | **已修,见下方"回归 B"** |
| 9 | **§9 的 gate 完全不可执行**:`"> 0"` / `"< 0.02"` 是从不被解析的字符串,`composite_score` / `selection_rule` 只被断言非空,全仓库没有代码在留出集上给 `N0/S0/RA` 打过分 | **FAIL** | **已修**:比较式与算术小语言有真解析器,config 校验期即拒绝写不出的表达式;新增 `code/scripts/score_sparse_history_arms.py` 执行打分 → 派生量 → must_pass → composite → selection_rule;解析器只有一份,trainer 导出、打分脚本导入(§9) |
| 10 | **契约只能靠读代码复核**:诊断键与 gate 词表两套名字对不上;adapter 分派逻辑内联在 forward 闭包里;pair-group 构造只有跑完整条 pipeline 才看得见 | **FAIL** | **已修**:量名只在 `sparse_diagnostic_keys` 定义一次,gate 词表由它生成,对齐是结构性的;adapter 分派提为 `adapter_context_for_sample`,训练与验收共用同一份(`test_the_gate_scorer_reuses_the_trainer_dispatch`);新增纯内存入口 `build_pair_group_samples`,不落盘也能造出完整 pair-group |

### 回归 A(第 4 条):官方闭环 runner —— **我们自己打断的**

这不是原始审计里的遗留项,是**第一轮改动引入的回归**,必须记明。

- 第一轮把 `build_official_messages` 收紧为"保留轮必须带完整响应,否则 `ValueError`"。这个收紧
  **是对的**(§7)。但它在仓库里还有**另一个调用方**:`code/scripts/run_official_androidworld.py`
  ——官方基线 runner,也就是"绝对成功率 69.0"那条线的执行体。它从没被更新,于是只要
  `last_image > 1` 且已完成 ≥1 步,闭环**第一步就抛异常**,整条部署基线跑不起来。
  第一轮报告"builder 已修"时,没有检查这个调用方。
- 收紧还暴露出一个**更早就存在、一直静默**的错位:runner 用三个独立 list 维护历史,
  而 `recent_images` 只在动作**真正执行后**才 append;解析失败的 UNKNOWN 步只 append 文本、
  不执行动作、不落图。于是截图相对响应**错位一格**,而 `build_official_messages` 只检查
  `kept <= 步数`,查不出来。也就是说:官方基线在任何一次解析失败之后,模型看到的
  "第 j 步的截图 + 第 j 步的响应"根本不是同一步。
- **处置**:引入 `CompletedStep(action_text, full_response, observation)` 与**单一 append 点**;
  文本与完整响应取整个列表,截图取**同一列表的后缀**,两边截尾方式天然一致,错位在结构上
  不可能发生。UNKNOWN 步也落一条 history(观测 = 它当时看的那张图,响应只给 `Action:` 行,
  **绝不伪造 tool_call** 往上下文里塞假示范)。完整响应来源优先级:逐字节 `raw` →
  `raw` 空白但动作可解析时用 `official_response()` 反向重建(与训练语料同函数,格式逐字节一致)
  → 都不行则只给 `Action:` 行。注意 `step["raw_output"]` 是截断到 600 字符的诊断字段,
  **不能**当历史来源。
- **同一处还修了一个口径错误**:runner 原本硬性要求"起始分必须为 0",否则判 infra 故障并
  踢出分母。官方 `suite_utils.run_task` 没有这个前置检查(只在 episode 结束时评分),
  我们的检查把 `*Verify` 这类开局即满足的任务判成故障,**2026-07-25 实测 6/116 局**被误踢。
  现在只记录 `score_before` / `started_nonzero` 并照常跑完。**这意味着此前所有用该 runner 得到的
  AndroidWorld 绝对成功率,其分母都是错的,需要重跑。**
- **实跑验证**:脚手架 `official_loop.py` 在假环境上驱动 `run_official_episode`,脚本含 6 步
  (第 2 步无 tool_call、第 4 步空输出、第 6 步 terminate),对 `last_image ∈ {1,2,3,5}` 各跑一局,
  逐步比对"模型实际条件的截图 / 响应对"与 ground truth:
  `failures=[]`、`unknown_action_steps=2`、`termination=policy_terminated`、
  第 4 步 `full_response_reconstructed=true`,窗口大小、折叠步的 `StepN:` 编号全部对上。

### 回归 B(第 8 条):`history_sample_context` 静默 bypass —— **也是我们自己引入的**

- 第一轮为了让 sparse 样本复用 v2.1 的 mask 构造,写了这样一条兜底:
  样本既没有 `memory_config` 又不是 sparse v2 schema 时,`history_count` 退回
  `len(sample.get("selected_steps") or ())` → 得到 0 → `return None` → **整条样本在 adapter
  bypass 下前向**。后果不是报错,而是"以为在训练 adapter,实际全程在给冻结模型打分",
  且**日志里看不出任何异常**。这正是这类缺陷最危险的形态:它不会让 run 失败,只会让 run 无意义。
- **处置**:K 只能有显式来源——`memory_config.restored_event_step_ids` 或 sparse v2 schema 的
  `selected_steps`(且必须与 `budget` 一致);两者都没有就 `ValueError` 拒绝前向。
  同时把"读 `adapter_mode` 决定开不开"从 forward 闭包里提出来,做成唯一入口
  `adapter_context_for_sample`,`history_sample_context` 只负责几何。
  active 臂拿到空 mask(K=0 退化)也必须报错,不得静默走 bypass。
- **同一条结论的另一半**:`_sparse_history_group_loss` 在缺 `R0` 时旧代码走
  `forward("R0")` → `None` → `return None`,一组组静默跳过,主 claim 的**分母悄悄变小**,
  stdout 上只看到组数变少。上游 `validate_sparse_group` 确实也查这一条,但损失函数是被测试与
  将来其它调用方直接调用的入口,不能把 fail-closed 外包给调用方。现在缺 `R0` / 缺 `SA` 直接抛错。
- **同一族的第三处**:trainer 为 `official_multiturn` / `sparse_single_turn` 新开的
  `apply_chat_template` 编码路径,连同 v2.1 私有结构校验一起把 `prompt_aligned_input_keys` 与
  `assert_pinned_assistant_prefix` 也绕过了。于是模块开头那句"训练 prompt 与闭环推理逐 token
  一致"在**五臂上完全没有机器校验**——只是一句注释。现在:该跳过的只有 v2.1 私有结构校验和
  `tools=` 参数(官方 system prompt 已内嵌 `<tools>`,再传一次会改 prompt),
  另两项与 prompt 格式无关的保真检查照跑;`assert_pinned_assistant_prefix` 从
  `_encode_exact_batch` 里提成方法,两条编码路径**共用同一份实现**,不在 trainer 里复制。
- 相关用例:`test_a_sample_without_an_authoritative_k_is_rejected`、
  `test_active_arm_with_an_empty_history_mask_is_rejected`、
  `test_missing_required_arm_is_fail_closed`(参数化 `R0` / `SA`)、
  `test_the_geometry_function_does_not_read_adapter_mode`。

这三处的共同形态值得单独记住:**它们都不会让 run 失败,只会让 run 无意义**。
静默 bypass、静默少算分母、静默跳过保真断言——日志上都看不出来。

### 第二轮仍然没做的事(不要读成"已收敛")

- §11 的过滤器审计(人工抽样 / 分布报告 / 阈值敏感性)一项没做。
- 第 5 条的 donor 位置混杂两版对照消融没做。
- 第 7 条的 `sparse_drift_weight ∈ {0.5, 1.0, 2.0}` 消融没做。
- 语料没有重建,§7 / §8 的数字一个都没重测。
- 数据仍 `PENDING_HF_UPLOAD`(§17)。
- 测试没进 CI,也没在真 pytest + 真 torch 下跑过(§15)。

## 14. 原始外部审计(P0 / P1)逐条处置

`已修` = 已改代码 / 契约;`待修` = 已承认、尚未动工;`不同意` = 附理由,待审计方回应。
注意:标 `已修` 的契约类修复,其**数字**仍需在语料重建后重测。

| 编号 | 审计问题 | 处置 |
|---|---|---|
| P0-1 | adapter 是否生效靠 variant **名字前缀**推断(`variant.startswith("native_recent")` / `("sameformat_recent")`),语料里没有权威字段 | **已修**:样本新增显式 `adapter_mode: "bypass"\|"active"`,trainer 只读该字段,缺失或非法值 fail-closed;禁止任何按名字猜的分支。第二轮加了源码级守卫用例 |
| P0-2 | `N0` 未接通 `past_full_responses`,`+0.160` / `+0.133` / "83% 来自格式效应" 三个数字失效;原文误称"该缺陷已修" | **已修(构建端 + 接口层 fail-closed)**。**三个数字标为 OBSOLETE,待全量重建后重测**(§7)。副作用:接口层收紧打断了官方 runner,见 §13 回归 A |
| P0-3 | 主 claim 挂错对比(用 `SA − N0` / `SA − R0`,混入格式与 adapter 效应);且旧六变体语料缺 `RA` 臂,`SA − RA` **不可计算** | **已修(契约)**:主 claim 定为 `SA − RA`,六个派生量全部同表报告;语料重建后补齐 `RA` 臂才有数(§4) |
| P0-4 | 臂定义在文档 / 构建脚本 / trainer 三处不一致(文档称 `sameformat` adapter 生效,trainer 强制 bypass) | **已修**:§4 五臂表为唯一事实源,文档 / builder / trainer / config 四处对齐 |
| P1-1 | "long-horizon" 措辞过强(历史中位 16、P90 23) | **已修**:全文改称 **mid-to-long history**,≥24 / ≥32 作为显式 long-horizon 子组(§2) |
| P1-2 | 统计口径缺失:决策组被当独立样本,无 CI、无分层、无多重比较说明 | **已修(口径 + 实现)**:§10 规定 episode-cluster bootstrap、配对差值、四维分层、预注册主 claim;第二轮补上了执行它的打分脚本(§9);**数字待重测** |
| P1-3 | 重复字幕过滤器仅凭文本,≥32 组过滤 35%,可能选择性偏置 | **待修**:风险已写明(§11),人工抽样审计、分布报告、阈值敏感性**均未做** |
| P1-4 | 测试不可复现:非 pytest、硬编码主机绝对路径、stub 依赖、对源码字符串做断言 | **部分已修**:两个文件已重写为可收集的 pytest 用例、从仓库包 import、无主机绝对路径;**但仍未在真 pytest + 真 torch 下跑过,也未进 CI**,详见 §15 |
| P1-5 | 数据不可外部访问(`PENDING_HF_UPLOAD`),外部无法独立复核任何数字 | **待修**:发布清单见 §17;在 HF revision 落地前,本文档只能是预审草案 |
| P1-6 | `sparse_irrelevant` 取 donor 轨迹的**前 K 张**而非同 step 位置,混入"图的新旧 / 位置"差异,不是纯粹的内容替换 | **代码已改(第一轮)、验证未做**:donor 现在同 split、按 **age 向量逐位**取图、锚点按相对进度对齐、优先同分辨率,实际匹配到的 age 向量与尺寸命中数写进样本。第二轮未再改代码,**两版对照消融仍未做**,故判 PARTIAL(§13 第 5 条) |
| P1-7 | `L_anchor` 权重 2.0 可能过强,把 wrong-history 按在 reference 上,压制 rank 项 | **语义已改、权重未消融**:锚点从 `R0` 改为每个负样本**自己的冻结分**(§6),这解决的是"按在 reference 上"那一半;`sparse_drift_weight ∈ {0.5, 1.0, 2.0}` 的消融仍未做,判定以 `SA − RA` 与 `RA − R0` 双指标为准 |
| P1-8 | (内部复核新发现)K=1 的组里 `sparse_step_shuffled` 与 `sparse_duplicate` 的图**与 `SA` 完全相同**(倒序无效、重复 1 次即原图),两个负样本退化为正样本副本;K=1 占抽样分布 15% | **已修**:K=1 组**不生成**这两个负样本;K≥2 的 shuffled 由倒序改为**循环移位**(倒序在 K=3 时中间那张仍然对位);损失按组内**实际存在**的负样本归一化,K=1 组的负样本总质量与别组相同(§6) |
| — | 审计据 P0-2 提出:83% 既已失效,训练 reference 应回退到 `N0` | **不同意**:reference 换 `N0` 会把**方向未知且非零**的格式效应灌进 `L_gain`,adapter 只需利用格式优势即可满足 hinge,学不到选点。该论证不依赖 83% 这个具体数值。部署差距用报告端的 `SA − N0` 呈现,不塞进训练目标(§7) |
| — | 审计提出:§8 的 60 组冻结基线画像整体作废 | **改为基本同意**:第一轮答"部分不同意"(不涉及 `N0` 故不受 P0-2 影响)。但 §12 的相邻性缺陷让涉及 `S0` 的行也失效,P1-6 的位置混杂让涉及 `irrelevant` 的行也失效,四行里三行受影响。全表降级为探针,不进论文结论表,正式数字重建后重测(§8) |

## 15. 测试现状:实跑数字(审计 P1-4,**尚不能称"CI 保证"**)

### 第一轮的"测试通过"是假信号

第一轮报告"15 项 prompt 契约测试 + 9 项损失数值测试"覆盖。第二轮把同一套用例放进真解释器实跑,
**48 个用例失败**。失败原因不是逻辑分歧,而是这些"测试"从来没有真正执行过被测代码:

- 两个文件**不是 pytest 用例**:零个 `assert`,靠 `print` + `sys.exit` 汇报,pytest 无法收集与断言;
- 都通过硬编码的主机绝对路径 `exec` 被测源文件的**副本**(不是从仓库包 import),外部检出后直接跑不起来;
- 损失脚本以 `wrapped={}` 调用 `_sparse_history_group_loss`,而该函数的关键字参数叫 `adapter_parameters`
  —— **签名早已漂移**,"9 项数值契约"当时**一项也跑不起来**;
- prompt 脚本实际执行 15 个 check,汇总行却按 14 计算,通过数少报 1;
- 场景 6 对**源码字符串**做匹配(`'if variant.startswith("native_recent")' in src`)——这恰恰把审计
  P0-1 要废除的名字前缀分支**固化成了测试契约**;场景 7 写成 `... or True`,**恒为真**。

结论:**"测试通过"在第一轮不构成任何证据**。凡是"已由单测保证"的表述,都必须附实跑数字。

### 第二轮的实跑数字

重写后,两个文件共 **91 个测试函数**,参数化展开为 **137 个用例**:

| 文件 | 测试函数 | 展开用例 | PASS | FAIL | SKIP |
|---|---:|---:|---:|---:|---:|
| `code/tests/test_gui_owl_sparse_history.py` | 46 | 72 | 72 | 0 | 0 |
| `code/tests/test_sparse_history_loss.py` | 45 | 65 | 63 | 0 | 2 |
| **合计** | **91** | **137** | **135** | **0** | **2** |

2 个 SKIP 是 `@requires_real_torch` 标记的:
`test_active_mask_covers_exactly_the_history_image_tokens`(真实 mask 几何)与
`test_l2_penalty_enters_the_loss_and_the_gradient`(L2 项进损失与梯度)。
这两条恰好是**最需要真 torch 才有意义**的,目前**没有被验证过**。

覆盖到的契约:§5 的八条 prompt 不变量(参数化)、§4 的五臂契约与 `adapter_mode` fail-closed、
负样本按 `role` 而非 slot 名判定、`negative_scale` 从字段读而非查名字表、K=1 库存与份额归一化、
§6 的 drift 语义(锚到自己的冻结分、与 `R0` 位置无关、锚点前向必须是 no-grad + bypass)、
次梯度权重的有限差分校验、§12 的相邻性与降 K、留出集分桶与缺臂 fail-closed、
以及"打分脚本与 trainer 共用同一份 adapter 分派"。

### 仍然缺的(所以还不能说"契约已由单测保证")

1. **这 135 个 PASS 是在本地脚手架下跑出来的,不是真 pytest**。脚手架
   (`probe.py` / `runtests.py` / `contig.py` / `contig2.py` / `official_loop.py` / `gateprobe.py`,
   以及一个 `pytest` shim)**没有入库**,它把待测文件装配成 `causalcache/policy` + `scripts` + `tests`
   的包布局并桩掉 `torch` / `PIL`。真 pytest + 真 torch 下的结果**未知**。
2. **没有接 CI**。
3. 两条 `requires_real_torch` 用例从未执行。
4. `f2probe.py` 这个第二轮早期的探针脚本自己已经与损失的返回结构漂移(它假设返回 dict,
   现在返回 float/None),说明**脚手架本身也会腐坏**——这正是必须把用例搬进 CI 的理由。

**在 1~3 落地之前,任何"契约已由单测保证"的表述都不成立。**

## 16. 代码位置与生成命令(GitHub: luojiaxuan/CausalCache, main)

路径一律为**仓库相对路径**:

```
code/causalcache/policy/gui_owl_sparse_history.py     sparse prompt builder
code/causalcache/policy/gui_owl_official.py           官方保真 builder / parser(past_full_responses,fail-closed)
code/causalcache/policy/history_gated_lora.py         HGKV adapter(mask 门控 LoRA residual)
code/causalcache/policy/history_token_roles.py        历史图 token 掩码构造与不相交断言
code/causalcache/policy/gui_owl_v2_1_runtime.py       冻结推理 runtime;两条编码路径共用的保真断言
code/scripts/build_sparse_history_dataset.py          数据集构建(五臂 + 负样本、choose_sparse、donor)
code/scripts/select_long_horizon_trajectories.py      长轨迹选材
code/scripts/train_success_sft_lora.py                trainer(sparse_history 分支、gate 解析器、adapter 分派)
code/scripts/score_sparse_history_arms.py             留出集打分 + gate 执行 + checkpoint 选点(第二轮新增)
code/scripts/run_official_androidworld.py             官方闭环基线 runner(第二轮修复,见 §13 回归 A)
code/configs/causalcache_sparse_history_v1.json       超参、损失记号、gate 与 selection_rule
code/tests/test_gui_owl_sparse_history.py             prompt / builder 契约用例(46 函数)
code/tests/test_sparse_history_loss.py                损失 / 分派 / 留出集用例(45 函数)
```

构建(32 路并行分片,随后合并;所有路径由参数给出,不含主机绝对路径):

```bash
# 分片构建:i = 0..31
python -m scripts.build_sparse_history_dataset \
  --selection <selection.json> \
  --pool-root <guiodyssey-pool>/mobile/use/train \
  --annotations <guiodyssey-annotations> \
  --output-root <out>/shard-$i \
  --max-trajectories 1300 --decisions-per-trajectory 3 \
  --max-consecutive-repeat 5 --heldout-fraction 0.15 --seed 271828 \
  --shard-index $i --shard-count 32

# 合并:拼接 samples.jsonl,images/ 目录按 episode 归并,重算 manifest
cat <out>/shard-*/samples.jsonl > <out>/samples.jsonl
python -m scripts.merge_sparse_history_shards --shards <out>/shard-* --output-root <out>
```

留出集打分与选点(gate 真正被执行的入口):

```bash
python -m scripts.score_sparse_history_arms \
  --repository-root <repo> --model-dir <frozen-policy-snapshot> \
  --config code/configs/causalcache_sparse_history_v1.json \
  --dataset-root <out> --image-root <out> \
  --checkpoint step25=<ckpt-25> --checkpoint step50=<ckpt-50> ... \
  --include-identity-baseline \
  --bootstrap-replicates 2000 --bootstrap-confidence 0.95 --bootstrap-seed <seed> \
  --gate-statistic declared \
  --score-cache <cache.json> --heartbeat <hb.json> --progress-every 25 \
  --output <report.json> --device cuda:0
```

`--checkpoint` 可重复,**按训练顺序传入**——`tie_break=earliest_checkpoint` 读的是传入顺序,
不是文件名。`--score-cache` + `--heartbeat` + `--progress-every` 是为长跑任务准备的
断点续跑与存活信号,长跑必须开(否则共享机上被杀就得从头打分)。
`--max-groups` 只用于冒烟,它会截断留出集并在报告里打标记,**冒烟结果不得当作 gate 结论**。

训练超参**只从 config 读**:sparse 分支下 CLI 的 `--max-steps` / `--checkpoint-every-steps`
被拒绝,`config.training` 里出现任何未被消费的 key 直接报错退出(审计 P1-6),
免得"写了但没生效"的参数继续伪装成实验设定。

发布时上表每个文件的 **commit SHA** 与上面各条命令的**完整实际参数**必须一并公开(§17)。

## 17. 数据可复核性:发布清单(状态 `PENDING_HF_UPLOAD`)

产物当前 staged 于内部集群持久盘,**外部不可访问**,这是审计判定不通过的直接原因(P1-5)。
上传后本节替换为 Hugging Face repo + **immutable revision**。

**发布时必须一并公开的清单**(缺任一项都不算"可外部复核"):

1. `selection.json` **及其 sha256**(选材结果,决定了哪些轨迹进入语料);
2. `samples.jsonl` **及其 sha256**(全部臂与负样本的逐样本记录,含 `arm_id` / `arm_slot` /
   `adapter_mode` / `role` / `pair_group` / `budget` / `selected_steps` / `group_negative_slots` /
   `negative_kind` / `negative_scale` / `donor_*` / `split`);
3. `manifest.json`(轨迹数、决策组数、样本数、每组行数直方图、K 分布、action mix、
   history 深度分位、**`budget_downgrades` / `budget_downgraded_decisions` /
   `pool_too_dense_rejected_decisions`**(§12)、donor 分辨率命中数、
   `rejected_reasons` 全表、heldout 组数);
4. **每张图的索引**:相对路径 → `sha256` / 尺寸(width × height)/ 视觉 token 数;
5. **构建与评测脚本的 commit SHA**:`build_sparse_history_dataset.py`、
   `select_long_horizon_trajectories.py`、两个 prompt builder、trainer、
   `score_sparse_history_arms.py`、`run_official_androidworld.py`;
6. **完整构建 / 合并 / 打分命令**(含 `--seed`、`--shard-count`、`--max-consecutive-repeat`、
   `--heldout-fraction`、`--bootstrap-replicates` 的实际取值);
7. **冻结基线与每个 checkpoint 的逐组 logprob**(五臂 + 每个负样本的 active / bypass 各一列)
   与其 `pair_group` / `episode` / `budget`,让外部能重算 §7 / §8 / §9 的每一个数字与 CI;
8. **最终 HF revision**(不可变 commit hash,不是分支名);
9. 附:heldout 划分规则(`sparse_v3` hash salt + 15% episode 比例)与被过滤轨迹的 ID 列表
   (供 §11 的人工抽样审计复核)。

## 18. 请重点复核的问题

1. §4 五臂契约是否真的只在每一对之间改了该改的变量?特别是 `R0`/`RA` 与 `S0`/`SA`
   共享 prompt 渲染、仅 `adapter_mode` 不同,这个"唯一变量"是否成立?
2. 主 claim 挂在 `SA − RA` 是否正确?六个派生量的分解是否完整(有没有遗漏的混杂路径)?
   把 `SA_minus_R0` 只当 gate 量、不进报告表,这个分工是否合理?
3. 训练 reference 用 `R0`、部署基线用 `N0` 的分工是否正确(§14 中我们对"回退到 `N0`"的
   不同意理由是否站得住)?
4. §6 把 anchor 从"锚到 `R0`"改成"锚到每个负样本自己的冻结分"是否正确?这是对已冻结契约的
   实质改动,`drift_weight = 2.0` 在新语义下是否仍然过强?
5. §12 的处置(降 K 而不是放宽相邻性、也不是整组拒绝)是否是正确取舍?降 K 会让 K 的实际
   分布偏离预注册的 `K_DISTRIBUTION`,这本身要不要在报告里单列?受影响组要不要**单独分层报告**
   而不只是修好重建?
6. §10 的 episode-cluster bootstrap 是否足以处理"每轨迹 3 个决策点"的相关性?
   50/70/85 抽样本身是否还引入了别的偏置?
7. §11 的过滤器:整条剔除 + 仅看字幕文本,在 ≥32 组过滤 35% 的情况下,
   我们计划的补救(人工抽样 + 分布报告 + 阈值敏感性)是否够?
8. §13 回归 A 暴露的问题更一般:**收紧一个共享接口时,我们没有枚举它的全部调用方**。
   除了 CI,还有什么机制能在下次挡住同类回归?
