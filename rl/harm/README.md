# history-harm:为什么给 GUI agent 历史截图反而有害?(机制探索线,2026-09-05 起)

分支 `history-harm` 从 CausalCache `main` 分出,专门深挖一个反常现象。方法论参照 ICLR 论文
*More Thought, Less Accuracy?*(arXiv 2509.25848)的探索流程:**现象站稳 → 错误分析 → 机制假设 → 干预检验 → 方法 → 消融**,
每一步接住上一步留下的问题。目标是一篇 ML 会议风格的机制论文,不是系统论文。

## 0. 反常现象(已有数据,来自 `rl/docs/results/history_layout_final_20260906.txt` 与台账)

在 MobileWorld GUI-only 上,对冻结的 GUI-Owl-1.5-8B,**给历史截图让它答错更多的步**(337 个自一致状态,步级动作复现率,%):

| 给图方式 | 不给图 | 官方默认最近两帧 | 15 帧对均匀 | 三裁判精选帧 | 无关任务的帧 |
|---|---|---|---|---|---|
| 正确率 | **63.2** | 55.8 | 54.8 | 51–53 | 48.1 |
| 逐状态:救回 / 伤害 | — | 2.7 / **9.5** | 5.1 / 13.8 | 3.6–5.6 / 17.2–17.5 | 5.0 / 20.5 |

三条更细的线索:
1. **伤害随"图与当前任务的相关性"单调变化**:最近帧 −6.8 < 均匀 −8.7 < 裁判精选 −12~−14 < 无关帧 −15.4。越不相关伤害越大,
   但相关的也伤害——不是"信息有误导",更像**任何额外图像都在稀释或混淆**。
2. **配对 SFT 后伤害消失、救回不涨**:三种历史布局训练后,伤害率从 9.5% 降到 4–8%,救回率仍 4–6%;训练后给真实历史、
   无关历史、不给图三者几乎相同(74.1 / 74.2 / 74.8)。训练教会的是"忽略",不是"利用"。
3. **跨模型/跨协议不同**:UI-Venus-2-9B 保留文本轨迹时最近帧中性(90.4 = 90.4),去掉文本轨迹时最近帧有益(37.1 → 55.7);
   GUI-Owl 保留文本轨迹时最近帧有害。伤害与"是否已有文本记忆"有关?与模型训练配方有关?

## 1. 研究路线(每步预注册判据,写进本目录 `docs/`)

**Step 1 现象站稳(普遍性)**:换模型(GUI-Owl-1.5-8B/32B、UI-Venus-2-9B、Qwen3-VL-8B、Qwen3.8-27B、MAI-UI 等)、
换基准(MobileWorld、AndroidWorld、GUI-Odyssey 离线、OSWorld 截图)、扫历史图数 N=0..6、两种协议(保留/去掉文本轨迹)。
输出:一张"伤害 vs N"的折线族(模型 × 协议)。判据:伤害在 ≥3 个模型 × ≥2 个基准上可复现。

**Step 2 错误分析(伤害是什么样的错)**:把"不给图对、给图错"的状态拆类——
(a) **今昔混淆**:点击坐标落在只存在于历史帧的控件上(可用历史/当前屏的 UI 树或 caption 机器判定);
(b) **过期动作复读**:重复了历史某步的动作;
(c) **指令遵循退化**:动作类型/格式错;
(d) 其它。判据:某一类占伤害的多数。

**Step 3 机制假设 + 干预检验**:候选假设——
(H1) **视觉稀释**:当前屏的图像注意力随历史图数下降(仿照该论文的 attention-to-image 曲线,逐层测);
(H2) **时序混淆**:模型分不清哪张是当前;
(H3) **上下文长度效应**:与图无关,纯粹是 token 数。
对应干预:给历史帧加显式"PAST"标记/边框/时间戳(测 H2);把历史帧降分辨率或换成空白/噪声图但保持 token 数(测 H3 vs H1);
当前图放最前 vs 最后(位置效应);测注意力分布(H1)。判据:哪个干预让伤害显著下降,机制就指向哪里。

**Step 4 方法 + 消融**:按机制设计训练目标(例如对历史帧引起的错误加权、或在训练中显式教"当前屏优先"的对比样本),
或推理时修复;主实验(伤害是否消失、救回是否上升)+ 消融(每个设计部件的作用)。

**图表**:每张图承担一个证据职责——现象图(伤害 vs N)、错误类型饼图、注意力曲线、干预对照柱、方法主表、消融表。

## 2. 可直接复用的资产(全部在 hyper00 `/data01/jaxan/`,Git 与 HF 有正本)

- 标签(23 上下文/状态,含 null/recency/6 单帧/15 帧对/无关帧):GUI-Owl 底座 `rl_v2/guiowl_selfref.jsonl`(337)、
  `rl_v2/labels_base_shard*.jsonl`(1052,Tilde);Venus `rl_v2/venus/oracle_v2.jsonl`(400,保留文本)、`oracle_v2_notext.jsonl`(116)。
- 裁判帧对与补解:`rl_v2/picks_*.jsonl`、`rl_v2/judge_decode_*.jsonl`(底座 + 三臂 + 两 seed)。
- 三臂 SFT 权重(合并版)HF:`gavinlaw/causalcache-history-layout-{recency,random,older}-s0`;数据 `gavinlaw/causalcache-history-layout-sft`;
  底座 rollout `gavinlaw/causalcache-guiowl-base-rollouts`(dataset)。
- 脚本:`guiowl_oracle.py` / `venus_oracle.py`(标注协议)、`decode_pair.py` / `decode_pair_venus.py`(任意上下文补解)、
  `harm_rescue.py`、`final_table.py`、`informed_oracle.py`(第三方选帧)、`cc_container_lib.sh`(容器纪律公共函数)。
- 文档:`rl/docs/ledger_memory_rl_20260904.md`(全部判断与错误)、`rl/docs/story_20260905.md`、`rl/docs/reviews/*`。

## 3. 纪律(继承自 CausalCache)

共享主机容器:名字现场分配(docker ps -a 与 map 两边空缺的最小号)、`--label cc.owner`、只删自己标签的容器、创建即登记
`$HOME/jiaxuanluo-map.txt`;思考型模型做结构化输出要关 thinking / 只解析 `</think>` 之后;每个结论先写判据再跑;
方向性决定前 ChatGPT 外审;台账制。
