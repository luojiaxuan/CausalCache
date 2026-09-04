# CausalCache 记忆控制器 RL 线 实验台账(截至 2026-09-04 04:15 PT)

本文件是 RL 线的**唯一现行判断清单**。历史日志(`cua_lite_integration_20260825.md` §4.x、
`summary_retrieval_design_20260831.md` §1–28、`rl_recipe.md`、`audit_ledger_20260809.md`)只作
证据仓库,**与本文件冲突时以本文件为准**;本文件每次判断变更都要写进 §4 并注明推翻了谁。

## 0. 一句话现状

四条 RL 线中三条已判死或停线(OSWorld v1/v2、MobileWorld 联合训练 P1–P4、冻结 iter_59 的
selector 终局 RL),第四条(MemGUI 闭环基线)已出数但只作内部配对口径。当前判断:**瓶颈不是
RL 优化器,而是(a)selector 选帧精度跨模板不泛化、(b)executor 任务级地板**。执行中(2026-09-04
02:01Z 起,用户睡眠期间):executor 已换为 UI-Venus-2-9B(hyper00 GPU1,容器 `sglang-omni-jaxan-3`),
闭环探针 v2(MobileWorld 117 题 × 四臂,09:01Z 起,预计 4 小时)在跑,收官后自动接离线主闸门 G2
(oracle-2 余量,预计 1 小时);外审已落盘并采纳(闸门 G0–G2 预注册)。下一个决策点 = G2 结果。

## 1. 三十秒背景

问题:冻结的 GUI executor 之上,能否**学**出一个记忆控制器(每步从历史截图里选 B 张喂给
executor),使第三方记忆 benchmark(MemGUI-Bench,128 题闭环,LLM 判分)显著提升、常规 benchmark
(MobileWorld / AndroidWorld)不退化。成功 = MemGUI Pass@1 显著高于同一 executor 的默认历史窗,
且方法是学出来的(RL 或密集反馈的 bandit),不是手写规则。

## 2. 当前生效的判断(冲突消解后的唯一版本)

| # | 判断 | 证据(可 grep) | 推翻了什么 |
|---|---|---|---|
| J1 | **iter_59 能利用非连续历史帧**。oracle 双帧−recency 的 gold 动作 logprob:底座 +0.187,iter_59 +0.255,配对差 +0.068 显著(80 状态);行为口径 300 状态:recency 35.5%、仅当前帧 31.4%、oracle 对 53.2% | summary §10 表、§17 表 | 09-04 02:30 PT 我说的"executor 用不动非连续帧" |
| J2 | **随机远帧有害**:随机对 29.6% < recency 35.5% < oracle 53.2%;executor 只在选得准时受益 | summary §17 | — |
| J3 | **selector 精度跨模板不泛化**:密集行为标签监督预训练,宽口径(域内)+5.1pp 胜 recency,严口径(跨模板)+0.2pp 持平;训练数据减半不掉 → 数据量不是瓶颈,表征是 | summary §19–20 | "再标注扩量"的路线 |
| J4 | **终局 0/1 回报的 selector RL 在冻结 executor 上无信号**:RLOO 组 70% 零对比;22 轮策略熵 4.05≈均匀 4.19;KL(final‖init)=0.026;argmax 一致 61.5%;heldout 贪心 RL=init=recency 15.8% | summary §25–26.6 | §25"heldout 连续上行"(配对复测后判噪声) |
| J5 | **任务级地板**:MobileWorld heldout-20 有 11 题四次评测全零;MemGUI(GUI-Owl-1.5-8B,50 步口径)两臂各 7/128,118 题两臂同败,82% episode 跑满上限 | summary §26.1、§27.9 | — |
| J6 | **random-S 格式 SFT 不做**:它针对"executor 不会读非连续帧",J1 证明该前提对 iter_59 不成立 | 由 J1 推出 | 09-04 02:30 PT 我提的"先 SFT"(用户已批,本条撤回并已告知) |
| J7 | **Stage I(executor 反馈 contextual bandit)暂不启动**:同一数据上的密集监督已失败于泛化(J3),只换优化器不换数据分布/executor 不应期望不同结果 | J3 + reviews/two_stage §3 | 09-03 "两级 RL 立即开工"的排程 |
| J8 | **MemGUI 对外基线一律引用论文 11.7% / 15.6%**;我们的 5.5% 是 50 步上限口径,只作内部配对 | §27.9–27.10;用户裁定 | — |
| J9 | **判分只保留 Pass@1 所需调用**(逐步描述 + 终判),IRR/BadCase 关闭;终判模型 gemini-3.1-pro-preview 替代 2.5-pro 须脚注 | §27.10;开关已入库 | — |
| J10 | **下一步 = 换 executor 为 UI-Venus-2-9B**(MemGUI 62.6、AndroidWorld 80.2、MobileWorld 65.8;权重公开 18.8GB,Qwen3.5-9B 底座,vLLM 镜像已支持;权重许可证"待确认")。**闸门按外审改判并预注册**(reviews/executor_swap_review_20260904.md §3):G0 闭环地板 recency-2 ≥ 40%(117 题);G1 剂量/选择臂 N_IMG 0/2/8 + 启发式非连续 B=2 只作诊断;**G2 主闸门 = heldout-39 轨迹上离线 oracle-2 − recency-2 ≥ +5pp 且 random-2 ≤ recency-2 + 2pp**;MemGUI 不参与任何决策 | 外审 2026-09-04 03:40 PT;HF 仓核验 03:00 PT | 03:30 PT 版 J10 的"满历史 − 最近两帧 ≥ 3pp"闸门(外审指出不是命题的必要条件) |

## 3. 当前生效的假设与口径

| 假设/口径 | 取值 | 谁定的 | 依据 | 若错会怎样 |
|---|---|---|---|---|
| 记忆预算 B | 2 张历史帧 + 当前帧 | 外审+用户(09-01) | 可枚举 C(n,2),可变 B 排在固定 B 之后 | 结论只对 B=2 成立 |
| 研究范围 | 文本动作/推理历史之上的**视觉情景记忆选择**(Venus N_IMG=0 仍保留 assistant 文本历史) | 外审(09-04) | Venus 官方协议 | 不得表述为"记忆 vs 无记忆" |
| 探针考场 | MobileWorld GUI-only 全部 117 题(train 78 + heldout 39;executor 探针不涉及 selector 训练故可全用),heldout-39 另报;50 步上限 | 外审(09-04) | 39 题一题 = 2.56pp,分辨不了 3pp | — |
| MobileWorld 切分 | 按主 app 分层 2:1,train 78 / heldout 39,seed 20260825 | 我选的 | `cc_recipe/fixtures/mw_split_v1.json` | heldout 含训练模板则泛化数字虚高 |
| 最小值得效应 MDE | +3pp 绝对 | 外审功效论证 | rl_recipe §6 | 更小效应视为未证 |
| "可控上下文"定义 | recency 对错、∃某帧对使贪心解码对(等价类匹配) | 我选的 | summary §26.2:1,853 中 452 个(24.4%) | ∃ 带赢家诅咒,真可控率偏低 |
| MemGUI 分数口径 | 成功/128,缺失计失败;判错另列 | 我选的 | `memgui_tally.py` | 与榜单 Pass@1 定义一致 |
| MemGUI 步数预算 | **官方 golden×2.5+1**(脚本默认);09-03 两臂用的固定 50 已判偏差 | 官方 | `core/runner.py:87` | 固定上限会截断 90/128 题 |
| 判分模型 | 逐步 gemini-2.5-flash;终判 gemini-3.1-pro-preview | 用户裁定(2.5-pro 账号门控) | §26.7 | 与榜单判分不完全同源,须脚注 |
| 判分费用 | **硬停止**:未获预算不发起任何 judge 调用 | 用户(09-04 01:00 PT) | §27.9 事故 | — |
| selector 特征 | 冻结视觉塔均值池化 + 文本 embedding 均值 | 我选的 | summary §12/§19 | 已判为泛化瓶颈(J3) |
| 榜单对照 | GUI-Owl-1.5-8B 11.7/15.6;UI-Venus-2-9B 62.6 | 第三方 | `memgui/site/leaderboard.json`、Venus README | — |
| 主机/算力 | hyper00,账号 4 卡上限;executor 单卡 vLLM | 全局规则 | — | — |

## 4. 已剪枝与未做的事

| 被剪掉的 | 方式 | 为什么 | 风险(偏向哪边) |
|---|---|---|---|
| MemGUI 官方步数口径重跑 | 不做 | 用户裁定,费用与价值不匹配 | 我们的 5.5% 偏悲观,不对外用 |
| AndroTMem 离线探针线 | 全部删除 | 用户裁定:非闭环 | 失去"oracle 记忆上界"的第三方度量 |
| oracle 记忆的闭环任务级上界 | 未测 | 无 oracle 帧标注的闭环考场 | 项目余量未知,靠 J10 探针间接判 |
| IRR / BadCase | 关闭 | Pass@1 不需要 | 无榜单 IRR/FRR 列 |
| per-step 状态依赖 baseline | 未做 | 联合线停线 | 联合线方差偏大 |
| 非法动作(double_tap)训练侧惩罚 | 未做 | 联合线停线 | 联合线绝对分偏低 |
| 难度优先采样真跑验证 | 未做 | 联合线停线 | 该机制效果未知 |
| 可变 B | 推迟 | 固定 B 未过闸 | — |
| 池 p00–p31 | 已停(CPU 让 MemGUI 后端) | 探针需重启 8–16 台 | 重启约十分钟 |

## 5. 实验台账(时间倒序,只列改变判断的条目;细节指向源文档)

### 2026-09-04 03:50 PT 外审改判闸门;Venus-2 探针 v1 停、v2 重发
- 假设:换 executor 后"满历史 > 最近两帧"能作为记忆杠杆的闸门。
- 现象:外审指出该闸门非命题的必要条件(token 稀释可让满历史输而 B=2 选择赢),39 题分辨不了 3pp,
  且未诊断 selector OOD 失败就换表征是乱撞。
- 判断:采纳。主闸门改为 Venus 上离线 oracle-2 余量(G2),闭环探针改为 117 题四臂剂量/选择诊断(G1),
  地板健康 G0;全部在数据落地前预注册于 reviews/executor_swap_review_20260904.md §3。
- 改动:v1 探针(heldout-39,N_IMG 0/2/all)于 08:5xZ 停止,部分结果归档 `rl_v2/venus/partial_v1_*`;
  agent 加非连续启发臂;tally 加列联表与循环/撞限诊断;v2 探针 117 题四臂发射(见 §9 更新)。
- 结果:待。去留:待 G0–G2。

### 2026-09-04 03:00 PT 判断整合与方向改判(本文件建立)
- 假设:历史台账互相冲突导致我反复改口(SFT 一会做一会不做)。
- 现象:J1(iter_59 能读非连续帧,§10/§17)与 §4.13"executor 用不动"叙事并存;后者是联合线在
  非 oracle 选帧下的任务级现象,我把它误当作能力缺失。
- 判断:两者不冲突——能力在(J1),收益取决于选帧精度(J2),精度不泛化(J3)。
- 改动:撤回 SFT(J6),Stage I 暂停(J7),方向改为换 executor + 余量探针(J10)。
- 结果:待外审与探针。去留:待定(探针出数后更新)。

### 2026-09-04 01:00 PT 判分费用事故与止损
- 现象:用户指出 Gemini 账单约 50 美元。核算:MemGUI-Eval mode=full 每轨迹 ≈50 次 flash +
  3 次 pro(终判/IRR/BadCase),≈$0.15–0.20/轨迹,论文自报 $0.213;我们多轮重判放大到
  300–350 轨迹次。我又在未获预算时起了官方口径重跑(9 题起跑、2 题判分后停)。
- 改动:止损脚本;IRR/BadCase 开关(J9);判分列为硬停止。出处 summary §27.9–27.10。

### 2026-09-04 00:40 PT MemGUI 两臂终表(50 步上限口径)
- 臂 A 官方默认窗 7/128;臂 B recency B=2 7/128;配对 同成 4 / 仅 A 3 / 仅 B 3 / 同败 118,
  McNemar p=1.0;82% episode 跑满 50 步;B 臂死循环尾 49% vs A 22%。
- 协议偏差:固定 50 步 vs 官方 golden×2.5+1,79/75 条被截断 → 只作内部配对(J8)。§27.9。

### 2026-09-03 AndroTMem 探针(已删)
- 原版 8B 在 gold 文本历史下,远程截图无增益;结论限离线设定。用户 09-04 00:05 PT 令删除。§27.5–27.8。

### 2026-09-03 冻结 iter_59 selector RL(v2 服务)null result
- 22 轮,heldout 贪心 RL=init=recency;70% 组零对比;停。§25–26.6;外审 reviews/rl_null_result。

### 2026-09-01/02 行为标签预训练
- 宽口径 +5.1pp(3 种子稳健),严口径 +0.2pp;数据减半不掉(J3)。§19–20。

### 2026-09-01 行为天花板与侵蚀假说反转
- oracle 对 53.2% vs recency 35.5% vs 随机对 29.6%(J1/J2);iter_59 帧敏感度高于底座(§10)。

### 2026-08-25→09-01 联合训练 P1–P4(cua_lite §4.x)
- 批 60 交叉:Et×recency 45% > Et×St 40% > Et×S0 30%(§4.13);批 100 终判 learned−recency
  +8.8pp(p=0.45)与 selector-only −6.2pp(p=0.73)符号相反 → 效应≈0(§4.15/4.18);selector
  三连病理修复(§4.16);P4 因结构缺陷停线(§4.20)。

### 2026-08-04→09 OSWorld RL v1/v2
- 20 迭代 RL 81/240 vs recent 88/240,零效应;selector/LoRA 均未训动。`audit_ledger_20260809.md`。

## 6. 数字总表(口径不同不并排比较)

| 口径 | 数字 | 出处 |
|---|---|---|
| MemGUI 榜单 GUI-Owl-1.5-8B / 32B | 11.7 / 10.9 Pass@1;15.6 / 20.3 Pass@3 | leaderboard.json |
| MemGUI 榜单 UI-Venus-2-9B / 27B;Qwen-UI-Agent(235B) | 62.6 / 70.3;77.3 | memgui README 新闻栏 |
| MemGUI 自测(50 步上限,3.1-pro 终判)A / B | 5.5 / 5.5(7/128 各) | §27.9 |
| MobileWorld heldout-20 贪心(RL pv22 / init / recency) | 15.8 / 15.8 / 15.8–26.3 | §26.6 |
| 行为天花板 300 状态(iter_59) | recency 35.5,当前帧 31.4,随机对 29.6,oracle 对 53.2 | §17 |
| 可控上下文 | 452 / 1,853 = 24.4% | §26.2 |
| 监督预训练 val B=2 宽/严口径 vs recency | +5.1 / +0.2 pp | §20 |
| 联合线批 60 交叉(20 题) | 40 / 45 / 30 | cua_lite §4.13 |
| 判分成本 | ≈$0.15–0.20/轨迹(论文 $0.213) | §27.9 |

## 7. 我犯过的错与纠正

| 错误 | 如何发现 | 纠正 | 影响过的结论 |
|---|---|---|---|
| MemGUI 步数上限写死 50,违反官方口径 | 09-04 读 runner 源码 | 脚本改默认官方口径;5.5% 降级为内部口径 | "可与 11.7% 比"的表述 |
| 判分 API 花费约 $50 未先要预算;又起了会再花 $80–100 的重跑 | 用户账单 | 止损、开关、硬停止 | — |
| "executor 用不动非连续帧"标签贴错,导致提出多余的 SFT | 用户追问 iter_59 证据 | J1/J6 | 09-04 02:30 PT 的建议 |
| 配对 oracle 53.2% 被当作固定策略可达值(赢家诅咒) | 09-03 用户质疑后审计 | §27.7 更正 | "路线的肉还在"的定量表述 |
| heldout 曲线"连续上行"报为信号 | 配对 recency 复测 | 判噪声 | §25 |
| 误删并行 session 容器 sglang-omni-jaxan-1 | 对方重建 | 判僵尸看 map 与创建 session | — |
| 本地 TCC 拒读把空 blob 推上 main;map 三次被 read-filter-overwrite 抹空 | 事后核对 size / 不变量 | 每次 push 验 size;map 只走 rebuild 脚本 | cua_lite §4.20 |

## 8. 待用户裁决

| 事项 | 我的推荐 | 理由 | 若不批 |
|---|---|---|---|
| executor 换为 UI-Venus-2-9B(放弃 GUI-Owl/iter_59 主线) | 批,但以探针为闸 | J5 地板三重实证;榜单 9B 级模型已达 62.6% | 只能走"缩主张"路线 |
| MemGUI 定稿运行的判分预算 | 每臂约 $10–15(开关已开) | 只跑定稿配置一次 | 无对外可比数字 |
| 第二档磁盘清理(hyper00 `osworld/` 33G、`aw/` 13G) | 删 | 均已上 HF 或可重建 | 占盘 |

## 9. 未来 6 小时的执行计划(2026-09-04 03:50 PT 修订,用户睡眠期间)

1. [done 03:40 PT] 外审落盘 `reviews/executor_swap_review_20260904.md`,闸门 G0–G2 预注册。
2. [done 01:50–03:00 PT] rle(GUI-Owl)容器删除;GPU1 起 `sglang-omni-jaxan-3` = UI-Venus-2-9b vLLM
   (131k 上下文,64 图/请求);map 登记;Venus-2 官方 mobile 协议移植为 `ui_venus2` agent,单任务冒烟通过
   (8 步、约 5 s/步)。
3. [running 09:01Z] 探针 v2:MobileWorld 117 题 × 四臂(N_IMG 0 / 2 / 8 / 启发式非连续 B=2),池 p00–p11 并发 12,程序判分,
   零 API 费用;每臂预计 40–60 分钟;tally 报成功率、对 recency-2 的 wins/losses/ties、死循环率、撞限率、
   成功中位步数;heldout-39 另报。
4. [armed] 离线 G2:`venus_oracle.py`(Venus 官方多轮协议,23 上下文/状态,状态取自 v2 四臂成功轨迹,≤400
   状态,heldout-39 / train-78 分报)+ `venus_oracle_verdict.py`(0–999 网格 TOL=60 等价类);两状态管线
   测试通过;`venus_g2_after_probe.sh` 等 v2 收官自动执行,产物 `rl_v2/venus/oracle_v2_verdict.txt`。
5. 出数 → 更新 §2 J10 与 §5;G2 成立则起草 selector 重建方案(先诊断 OOD 失败原因,再谈表征),否则写
   负结果并停。
