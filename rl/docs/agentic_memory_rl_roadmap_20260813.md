# Agentic Memory RL 路线(2026-08-13 用户裁定,取代离线 selector-SFT 战役)

> 本文件是**路线的 plan of record**。离线 selector 战役(台账 §0.6-§0.14)封存为
> 负结果与初始化资产,不再调参。执行侧的实现决策与偏差记在最后一节。

## 0. 核心问题与固定原则

**能否从任务级反馈中学出一个固定预算 B 的视觉记忆策略,并超过 recent-B?**

- Primary budget **B=2**;selector 每步从全部历史截图中选 2 张高保真截图;
- **recent-2 始终是合法动作**,不设强制替换;
- 训练主奖励**只有最终任务成功 R∈{0,1}** —— 不用 milestone reward、
  step-level gold、frame-level gold、手工 process reward;
- **不以 OSWorld benchmark task 作训练数据**;先自建高吞吐程序化任务,
  再经独立真实 GUI bridge set,最后完全 held-out benchmark;
- GUI-Owl **不再永久冻结**:先 gated sparse-history SFT,再训 selector,
  最后 joint RL;视觉编码器尽量冻结以保住历史帧特征缓存;
- 不再把 pass-1 draft 作为主方案输入;
- selector 训练采样、测试 argmax —— 不存在梯度穿过图片选择或环境的问题。

## 1. Phase 0:封存离线战役

- 稳定 tag:`selector-offline-final-20260813`(主分支 + 台账 + checkpoint +
  评测结果);
- **保留为可复用组件**:高保真历史帧特征加载与缓存、candidate-set encoder、
  variable-cardinality subset encoder、policy scalar head、recent-B/candidate
  subset prompt assembler、配对评测与 wins/losses/move-rate 统计;
- **明确弃用**:rescue/harm 分类头作为主路线、离线 binary correctness
  selector SFT、v5 exact-policy objective 作为最终训练方式;
- 新建独立训练入口(`rl/code/causalcache_agentic/`),与旧离线实验不互相污染。

公共接口对象:`TaskSpec` / `Env.reset,step,render,verify` / `Trajectory` /
`HistoryFrameBank` / `PolicyInputBuilder` / `SelectorPolicy` / `GUIExecutor` /
`RolloutWorker` / `GRPOTrainer`(契约见 `causalcache_agentic/contract.py`)。

## 2. Phase 1:程序化训练环境与数据集

**形态**:轻量 mock backend + 正常渲染的视觉 GUI frontend。可 mock 应用数据库、
文件系统、文档内容、表格数据、reset、transition、terminal verifier;
**不可 mock**:模型看到的仍是 GUI 截图、selector 仍选真实历史截图、
policy 仍输出鼠标/键盘/文本 GUI 动作、历史信息仍需从屏幕内容读取。

**任务 DSL**(确定性 task graph):`READ / WRITE / COPY / COMPARE / TRANSFORM /
SEARCH / SAVE / SEND / SWITCH_APP / DISTRACT / VERIFY`。每个 primitive 至少含:
初始化逻辑、GUI 渲染状态、transition、精确终局 verifier、scripted expert action、
可组合约束。**LLM 只负责自然语言改写/实体随机化**,不参与 reward、verifier、
成功判断、task graph 合法性。

**Memory-demand 分布**(同一 workflow template 随机实例化成不同 regime,
语义与 regime 必须解耦,禁止"cross-app ⇒ 该捞旧帧"这种可猜的捷径;
每批任务随机改变各 regime 比例,不固定 50/50):
1. `recent_sufficient` —— 关键证据仍在 recent-2;
2. `one_old_frame` —— 需要一张较早截图;
3. `two_frame_complementary` —— 需两张不同历史截图的互补信息;
4. `distractor_heavy` —— 历史有相似但错误内容,乱选降低成功率;
5. `history_irrelevant` —— 当前屏已足够,历史应被忽略。

**数据划分**:先按 template/workflow family 划分,再生成实例 ——
Train / Synthetic IID val / Synthetic OOD(未见 composition)/ Real-UI bridge /
Public benchmark(完全 held out)。禁止同 template 的实例散落 train/test。

**产物**:SFT 数据(instruction、action-text history、当前截图、候选历史截图与
时间索引、提供给 policy 的 history subset、scripted expert action、terminal task
ID 与环境参数);RL 数据(selector sampled subset 与 log-prob、GUI action 与
log-prob、状态转移、terminal success、metadata)。task graph 里"哪帧含哪个变量"
可留作诊断,**不得作 selector primary RL reward**。

**Phase 1 验收**:scripted expert 成功率≈100%;reset/同 seed 可复现;
verifier 无歧义;五种 memory regime 均可程序化生成;**不同 memory subset 确实
改变任务成功概率**(需真跑冻结 policy);rollout 吞吐达大规模 GRPO 所需。

## 3. Phase 2:Gated sparse-history SFT

历史图像经独立 history adapter 注入:`h_l' = h_l + g_l · HistoryAdapter_l(H_sel, h_l)`,
gate 初始≈0(初始行为≈原 GUI-Owl),带 frame-age / history-current role / 真实
时间顺序;history subset 进 prompt 前按时间排序。**只训** history adapter、gate、
后若干层 LoRA、history position/role embedding;**冻结**主视觉编码器与多数 backbone
(保住特征缓存)。SFT 数据覆盖 memory-required / recent-sufficient /
distractor-robustness 三类,**禁止构造"输入不含答案却要求 gold"的不可解样本**。
目标 `L = −log p(a_expert | instruction, history, S, current)` + 弱 base-policy KL。
训练顺序:adapter+gate → 必要时后层 LoRA → 仍不足才更大范围,**不一开始 full
finetune 8B**。go/no-go:**给正确 sparse history 时能稳定利用历史,且 recent 已
足够时不显著退化**。产出 `policy_mem_sft`。

## 4. Phase 3:Selector-only GRPO(论文最关键阶段)

固定 `policy_mem_sft`,只训 selector。动作空间 = 全部 C(N,2) subset,
recent-2 是其中普通合法动作;接口必须是 `z_θ(S | state), |S|=B`,
不得写死 `score_pair(f1,f2)`。selector 输入:instruction、action-text history、
当前屏、全部候选历史帧缓存特征、frame age/step index、subset 联合表示;
**不用** gold / pass-1 分布 / B=1 probe / milestone。selector 大小不再受
"必须轻量"约束。

Rollout:同 task instance 同初始 state 产 G 条;selector 采样、GUI policy 冻结
greedy、terminal-only reward;GRPO 用同组相对 reward。防塌缩:entropy bonus、
温度、modest recent-B 初始化(**不是极强 KEEP bias**)、clipped ratio、
entropy 与 group reward variance 监控。**不加人工 move cost 作 primary** ——
是否保留 recent-B 应由 recent-sufficient 与 distractor 任务的 terminal reward
教会。若大量 group 全 0/全 1,优先调难度 curriculum、SFT 初始能力、group size、
rollout 数量,**不引入 milestone reward**。

Curriculum:2-stage → 3-stage cross-app → 4-5 stage composite,reward 始终 0/1。

**验收**:synthetic IID + OOD 上配对比较 `adapted policy + RL selector` vs
`adapted policy + recent-2`;报 task success、paired delta、recent-success
preservation、recent-failure rescue、regression、选 recent-2 比例、旧帧距离分布、
分 regime 结果、有 reward variance 的 group 比例。**只有 synthetic OOD 仍显著
超过 recent-2,才进 joint。**

## 5. Phase 4:Joint GRPO

初始化 = `policy_mem_sft` + Phase 3 最佳 selector(不从两个随机组件同时学)。
4A 只解冻 selector + history adapter + gate + 后层 LoRA(冻结视觉 encoder 与多数
backbone);4B 仅当 4A 明显受容量限制才扩大,**不默认 full model RL**。
双方 log-prob 进同一 trajectory-level group advantage;稳定手段:分组学习率、
对 `policy_mem_sft` 的 KL、对 Phase 3 reference 的可选 KL、交替更新、
分开监控 grad norm/KL/entropy。

**Joint 验收**(证明不是 policy 单独变强):①原始 GUI-Owl + recent-2;
②memory-SFT + recent-2;③memory-SFT + selector-only GRPO;④joint + learned
selector;⑤joint + recent-2 —— 由此拆出 SFT 收益 / selector-only 收益 /
co-adaptation 额外收益 / joint policy 是否仍依赖 learned selector。

## 6. Phase 5:真实 GUI Bridge Set

真实 Ubuntu 应用(Firefox/本地站点、LibreOffice Writer/Calc/Impress、Files、
PDF viewer、text editor、可控本地 mail/chat)上的**独立程序化任务**:程序化 task
graph、随机文件/网页/文档/数值、exact terminal verifier、terminal-only reward、
与 benchmark 不重叠的 template。可做少量 domain adaptation SFT 与 selector/joint
RL,**不得使用 benchmark task 或其 evaluator 反馈**。链条
`mock IID → mock OOD → real-UI OOD` 每层 learned selector 都超过 recent-2
才进公开 benchmark。

## 7. Phase 6:Benchmark 与论文实验

Benchmark 完全 held out(不训练、不调参),primary B=2。主表:原始 GUI-Owl+recent-2 /
memory-SFT+recent-2 / memory-SFT+random-2(sanity)/ memory-SFT+learned selector
(**selector-only 主结果**)/ joint+recent-2 / joint+learned selector /
可行时 oracle-privileged(上界诊断)。报 overall success、paired delta、
preservation、rescue、regressions、选 recent-2 频率、单 app/cross-app、
trajectory length、memory distance、synthetic 与 real 的 domain gap。

**主 claim:同一个 memory-aware GUI policy 下,RL selector 显著超过 recent-2。**
Joint RL 是系统上限,不替代 selector-only 因果对照。

## 8. 提交顺序(避免一次写完难定位)

1. 封存旧线并统一接口(history bank / subset prompt builder / selector policy API /
   trajectory schema);2. 程序化 mock 环境(DSL / renderer / transition /
   verifier / deterministic expert / template-level split);3. 专家轨迹与 SFT 数据;
4. HGKV/history-gated GUI-Owl(adapter/gate/LoRA/SFT trainer/oracle-recent-distractor
   evaluator);5. Selector policy(exact B=2 subset logits、recent-2 合法动作、
   采样与 argmax、批式候选打分);6. selector-only rollout 与 GRPO trainer;
7. joint GRPO trainer;8. real-UI bridge;9. benchmark evaluator 与论文归约。

## 9. 本路线**不得**重新引入

milestone reward;step-level action correctness;gold frame pair selector SFT;
pass-1 draft;B=1 逐帧 probe;Gumbel-softmax / straight-through 图片选择;
单 checkpoint 同时 joint train 多个 B;benchmark task 上做 RL;
full GUI-Owl 从零联合训练。

---

## 10. 执行侧决策与偏差(实现方记录,可推翻)

| # | 决策 | 理由 | 回滚方式 |
|---|---|---|---|
| E1 | Phase 1 渲染器用 **PIL 直绘**(1920×1080 真实分辨率、真实文字与控件),不用浏览器/X server | 吞吐(毫秒级/帧)、完全确定性、零外部服务;真实 UI 的 domain gap 由 Phase 5 bridge 负责,不由 Phase 1 承担 | 渲染器是纯函数 `ScreenState → Image`,换 playwright/真实应用只替换该函数 |
| E2 | **状态模型即契约**:widget 携带声明式 effect(set_value/open_app/commit/reveal/...),executor 通用解释,app 语义声明化 | 让 renderer / executor / task DSL 三者解耦,可并行实现与独立替换 | effect 词表在 contract.py,新增 effect 只加解释分支 |
| E3 | 动作口径**复用冻结路径**:policy 输出 [0,999] 归一 `computer_use` 调用,env 映射到 1920×1080 像素;历史文本经 `render_official_action_line/response` 渲染 | 与 AgentNet 语料、既有 prompt builder、既有评测逐字同源,避免量纲/格式漂移 | 契约里集中做归一↔像素换算,单点可改 |
| E4 | Phase 1 先不接 LLM 改写 instruction(用模板化多样性),LLM 改写留作 Phase 1.5 | 先把可验证的确定性骨架跑通;LLM 改写不影响 verifier 与 reward | 改写是纯文本后处理,随时可加 |
| E5 | 验收项"不同 memory subset 改变成功概率"需真跑冻结 policy(GPU),排在环境自检之后 | 它是环境**有效性**的判据,不是环境**正确性**的判据 | 见 `scripts/agentic_memory_sensitivity.py` |
