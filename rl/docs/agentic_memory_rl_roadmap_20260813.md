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
| P2A-1 | Phase 2A 用**全层 qkvo LoRA r8**,而非路线文档写的"后若干层" | 台账 §0.8/§0.9 实测:同参数量的末 8 层 k/v 适配增益 ≈0,全层 qkvo r8 拿到 +5.50pp CI[+3.00,+8.06]。硬约束仍满足:视觉编码器全冻(特征缓存有效)、非 full finetune、LoRA 零初始化 ⇒ 训练起点逐位等于原始 GUI-Owl(即"gate 初始≈0") | `--last-layers N` 一个开关切回 |
| P2A-2 | Phase 2 先做 **A 臂(历史帧走官方 prompt 主通路)**,B 臂(gate 侧通道 HGKV cross-attn)推到 Phase 3 管线打通之后 | A 臂是已验证路径,最快解锁 Phase 3(论文最关键阶段);B 臂多的是 token 成本故事与显式 g_l 旋钮,不改变"executor 会读稀疏历史"这个 Phase 2 目标 | 两臂共用 SFT 数据与评测,B 臂随时可加训并同表比较 |
| P2A-3 | SFT 训练集按 **决策步全留 + 平凡步分层欠采样**(2591 条:决策 1226 / 平凡 1365) | 记忆是否被读出只在决策步体现,是最稀缺监督;不欠采样会被"点应用图标"这类步淹没 | `mix_sft.py` 的 budget 参数 |
| P4A-1 | 联合更新的第一批数据用 **iter1+iter2 合并(48 组 / 384 rollout)**,而不是每 24 组各训一次 | 24 组太薄:GRPO 只有 58.3% 的组有奖励方差,单轮实际可用约 14 组;两轮由同一套未更新权重采样,属同一 behavior policy,合并合法 | 两批 JSONL 独立落盘,随时可只取一批重训 |
| P4A-2 | executor 采样温度取 **1.0**(模型原生分布),不为增加探索而调高 | 部署评测是贪心解码,训练分布调离贪心会引入 train/deploy 失配;先用原生温度看能否推动 | 若训练不动(ratio 恒 ≈1、grad 极小),第一个旋钮就是升温,`--executor-temperature` 一个参数 |
| P4A-3 | 4A 解冻 **executor 后 8 层 LoRA(1.70M 参数)**,而非 Phase 2A 用的全层 qkvo | 联合 RL 的方差远大于 SFT,先给最小可动面;视觉塔与前 28 层保持冻结,历史帧特征缓存因此仍然有效 | `--exec-last-layers 0` 切回全层 |
| P4A-4 | selector state 的构造**复用采样端 `resolve_state_factory`**,并在启动时对账 `selector.state_builder` origin | state 是 log π 的定义域;我第一版自建 builder 签名全错,若"侥幸"能跑就会用两个不同 state 训一个策略,而表面只表现为不收敛 | 对账不通过直接退出,不存在静默降级路径 |
| P5-1 | bridge 混合 SFT 数据 = **AgentNet ubuntu 真实腿 + Phase 2 合成记忆腿**,默认配比 **1:1**(按优化步) | 真实腿:tilde 本地 `agentnet_screening_manifest_ubuntu_v1.jsonl` 6003 决策点 + 51G 截图,每行含 history/target_text/质量位,messages 构造复用 `rl_oracle_enumerate` 的同一条 `build_desktop_official_messages` 路径(与部署逐字同源);合成腿:Phase 2 的 2591 条(hyper00,经 HF 送 tilde)。−13pp 遗忘 + writer/chrome/vlc 正迁移说明配比找得好可兼得 | 配比是 CLI 参数,扫 {1:2, 1:1, 2:1} 由验收(135 无回归复测 + 合成 hard 分层)定 |
| P5-2 | 真实腿质量过滤:只取 `target_step_last_step_correct=True` 的行;shown = recent-2(与部署 B=2 对齐) | 语料里有标注为错误/冗余的步,喂进去教坏 executor;shown 口径与部署一致避免 train/deploy 失配 | 过滤开关可关,重训对比 |

## 11. 实测数值(随执行更新)

- **冻结 GUI-Owl 在 mock GUI 上的域差(2026-08-13)**:teacher-forced 单步动作
  正确率 **52.5%**(n=160 留出记录;oracle 记忆 57.1% / recent-2 41.3%)——
  **域差不是灾难性的**,模型能解析我们渲染的界面并做对一半的动作。

### E5 验收(不同 memory subset 是否改变成功概率)—— **通过**

**先说一个被数据推翻的中间判断(记档)**:我一度写下"E5 在冻结策略上不可测"。
那只对**闭环任务成功**成立,对**决策步动作**不成立,已更正。

1. **闭环口径撞地板(n=189,4 臂)**:recent2 / oracle / random2 / none 的任务
   成功率 = **0.0 / 2.1 / 2.1 / 4.3%**;161/189 条撞 max_steps,策略从不主动
   terminate。未适配的冻结策略走不完 7-15 步的任务,四臂在地板上不可区分。
   → **闭环 E5 的正式判定挂到 Phase 2 的 `policy_mem_sft`**(执行中)。
2. **决策步口径给出决定性证据(n=189,配对:同状态同专家前缀,只换记忆子集)**:

| 分层 | oracle | recent-2 | random-2 | none | oracle−recent2 |
|---|---|---|---|---|---|
| 总体 (n=189) | 45.5% | 24.3% | 27.5% | 18.0% | **+21.2pp** CI[+15.3,+27.5] p≈0,赢/输 **40/0** |
| **需老帧合并** (n=102) | **39.2%** | **0.0%** | 9.8% | 0.0% | **+39.2pp** CI[+30.4,+49.0] p≈0 |
| one_old_frame (n=27) | 59.3% | 0.0% | 14.8% | 0.0% | +59.3pp p=3e-05 |
| distractor_heavy (n=49) | 42.9% | 0.0% | 12.2% | 0.0% | +42.9pp p≈0 |
| two_frame_compl. (n=29) | 13.8% | 0.0% | 0.0% | 0.0% | +13.8pp |
| **不需老帧合并** (n=87) | 52.9% | 52.9% | 48.3% | 39.1% | **+0.00pp,赢/输 0/0** |
| recent_sufficient (n=36) | 44.4% | 44.4% | 27.8% | 0.0% | 0.00pp(none −44.4pp p=3e-05) |
| history_irrelevant (n=51) | 58.8% | 58.8% | 62.7% | 66.7% | 0.00pp |

**读法**:①需要老帧时,给对帧 vs 给最近两帧 = 39.2% vs **0.0%** ——
记忆规则决定成败,且 40 次交锋 selector 侧全胜、零反例;②不需要老帧时
oracle 与 recent-2 **逐条完全一致**(差 0.00pp、赢输 0/0)——记忆在不该起
作用的地方确实不起作用,这是最强的对照;③random-2 在需老帧层只捞到
+9.8pp(约等于 2/5 的瞎蒙命中率),说明收益来自**选对**而非"多给帧";
④none 只在简单层显著变差(−13.8pp),因为那里的证据本来就在最近帧上。
**结论:环境的记忆需求是真的、可控的、且与任务语义解耦。Phase 1 全部
验收条款通过。**
- 口径声明:step-level 正确率在本路线**仅作诊断**(§9 禁止其作训练奖励或
  selector 监督);它回答的是"环境是否有记忆需求",不是"selector 学得好不好"。

### Phase 2A 早期读数(SFT 中途 checkpoint,800/1200 样本 × epoch 0)

同一套决策步诊断,只把 executor 换成 `policy_mem_sft`(n=57,train 划分):

| 分层 | oracle | recent-2 | random-2 | none | 对比冻结策略 |
|---|---|---|---|---|---|
| 总体 (57) | **94.7%** | 29.8% | 31.6% | 14.0% | 45.5→94.7 / 24.3→29.8 |
| **需老帧** (40) | **92.5%** | **0.0%** | 12.5% | 0.0% | 39.2→**92.5** / 0.0→**0.0** |
| 不需老帧 (17) | **100%** | **100%** | 76.5% | 47.1% | 52.9→100 / 52.9→100 |

**这正是 Phase 2 go/no-go 要求的形态,且两个条件同时满足**:
1. **"给对历史就能稳定用上"**:需老帧层 oracle 39.2% → **92.5%**,
   oracle−recent2 的差从 +39.2pp 扩大到 **+92.5pp**(赢/输 37:0);
2. **"recent 够用时不退化"**:不需老帧层 oracle 与 recent-2 双双 100%。

**为什么这不是"只学会了任务套路"**(#26 用 B=0 对照臂拦假结论的同一逻辑):
若模型只是背下流程,需老帧层的 recent-2 也会跟着涨;实测它**纹丝不动停在
0.0%** —— 因为那里的证据在最近两帧上视觉不可见,套路救不了。任务能力的
提升体现在不需老帧层(52.9→100%),记忆能力的提升体现在需老帧层
(39.2→92.5%),两者被干净地分开了。

### ★★★ Phase 2A 终判(2026-08-13):**通过,且在未见 family 上更强**

SFT 收尾:2 epoch / 2400 样本 / **零跳过**;内层留出集(按 task_id 划分)
teacher-forced 动作正确率 **53.1% → 98.1%**,五种 regime 全部 ≥96%。

**决策步诊断(配对,同状态同前缀只换记忆)**:

| 划分 | 分层 | oracle | recent-2 | random-2 | none | oracle−recent2 |
|---|---|---|---|---|---|---|
| train(IID) n=80 | 需老帧 (58) | 89.7% | 0.0% | 13.8% | 0.0% | **+89.7pp** 赢/输 52:0 |
| **syn_ood(未见 family)** n=200 | **需老帧 (147)** | **98.6%** | **0.0%** | 19.7% | 0.0% | **+98.6pp** CI[+96.6,+100] 赢/输 **145:0** |
| syn_ood | 不需老帧 (53) | 98.1% | 98.1% | 84.9% | 58.5% | **+0.00pp 赢/输 0:0** |

**闭环四臂(Phase 2 正式验收 + 闭环 E5,40 任务 × 4 臂,终局 0/1)**:

| 分层 | oracle | recent-2 | random-2 | none | oracle−recent2 |
|---|---|---|---|---|---|
| 总体 (40) | **82.5%** | 55.0% | 40.0% | 5.0% | **+27.5pp** 赢/输 11:0 p=0.001 |
| **需老帧 (15)** | **73.3%** | **0.0%** | 6.7% | 0.0% | **+73.3pp** 赢/输 11:0 p=0.001 |
| 不需老帧 (25) | 88.0% | 88.0% | 60.0% | 8.0% | **+0.00pp 赢/输 0:0** |

**结论(四条,逐条对应路线文档的验收条款)**:
1. **Phase 2 go/no-go 通过**:给对稀疏历史时稳定用得上(需老帧任务
   0% → 73.3%),recent 够用时**一点没退化**(88.0% = 88.0%,赢输 0:0);
2. **闭环 E5 通过**:不同记忆子集把任务成功率从 5.0%(none)→ 55.0%
   (recent-2)→ 82.5%(oracle)一路拉开,地板效应消失
   (主动 terminate 125/160,冻结策略时是 0/189;平均步数 18.7 → 13.0);
3. **不是背模板**:最强的一组数字出现在**未见过的 workflow family**上
   (+98.6pp,145:0),且不需老帧层 oracle 与 recent-2 逐条相同;
4. **给 Phase 3 的头寸是干净的**:需老帧任务上有 **+73.3pp** 的可捕获空间,
   而不需老帧任务上乱动要付代价(random-2 −28pp、none −80pp)——
   selector 必须学会"该动才动",这正是论文要问的问题。

**局限(如实)**:闭环 n=40(每臂),需老帧层 n=15,CI 未算;
syn_ood 只测了决策步、闭环 OOD 待补;instruction 多样性仍是模板化(E4)。

### Phase 3 首轮(2026-08-13 夜):**机器打通,尚无结论**

- 栈已验收 9/9(含 logπ 重算一致性 2.6e-07、置换等变、诊断不泄露、
  只用终局奖励);真跑链路 selector(41.2M,随机初始化 + modest recency
  先验)+ 冻结特征 + `policy_mem_sft`,22.5s/rollout;
- 首轮 rollout **519 条**(64 任务 × 8,train 划分),整体成功率 **28.5%**;
- **一次真正的 GRPO 更新已完成**:9 个优化步,entropy 1.83(未塌缩)、
  mean_ratio 0.998、clip_frac 0.118、grad_norm 1.68、π(recent-2)=0.378。

**工程事故与修复(记档)**:GRPO 更新步两次 OOM。真因不是缓存大小,而是
**整组(8 rollout × ~10 步)的计算图全部保留后才 backward** —— 每步要在
C(n,2) 个子集上过 128-latent 帧,峰值 138GB。GRPO 目标
`-mean_i mean_t[term]` 对步与 rollout 都是线性平均、优势为常数,故把
`term/(n_roll·n_step)` **就地 backward** 与整组一次 backward 的梯度逐位等价,
峰值降到 **25GB**。修完 logπ 一致性复检仍 PASS。

**当前的真正瓶颈(需要决策)**:**有效组只占 26.9%**(67 组里 37 组全失败、
12 组全成功),GRPO 只能从 18 组里学。三条可选路,代价与风险不同:
1. **难度分层采样**:优先采"oracle 成功而 recent-2 失败"的任务(Phase 2 数据
   已精确标出这类 regime)。收益最大;风险是训练分布偏向记忆关键任务 ——
   只要**评测仍在自然混合分布上**做,这个偏置是可接受的(且正是要学的
   "该动才动");
2. **加大 group size**(8 → 16/24):全 0/全 1 组的比例会下降,但 rollout
   成本线性上升(当前 22.5s/条);
3. **curriculum**:先在 2-stage 任务上学,再放长 —— 路线文档 §5 已写,
   实现成本中等。
**当前默认(可推翻)**:先做 ①+②(分层采样 + group 16),因为它们不改
目标函数、不引入新奖励,且能立刻把有效组比例抬上去。


### ★★ Phase 3 首个正结果(2026-08-14):**RL selector 在未见 family 上显著超过 recent-2**

**训练**(用户批准的方案:难度分层采样 + group 8→16,只改采样比例,
目标函数与奖励未动):3 轮迭代 × 48 任务 × 16 rollout = 2304 条 rollout。
有效组比例 26.9% → **41.7 / 43.8 / 33.3%**,可训练步 965 → 4176/轮。

| 迭代 | mean_reward | 有效组 | entropy | π(recent-2) | recent 为 argmax |
|---|---|---|---|---|---|
| 1 | 0.319 | 41.7% | 1.843 | 0.360 | 1.00 |
| 2 | 0.264 | 43.8% | 1.709 | 0.301 | 0.381 |
| 3 | 0.264 | 33.3% | 1.787 | **0.232** | **0.00** |

**验收(syn_ood 未见 family、自然混合分布、argmax 推理、80 任务配对)**:

| 分层 | learned | recent-2 | diff | 95% CI | W/L | McNemar p |
|---|---|---|---|---|---|---|
| **总体 (80)** | **37.5%** | 27.5% | **+10.00pp** | **[+3.75, +16.25]** | **8/0** | **0.0078** |
| 需老帧 (40) | 15.0% | 0.0% | **+15.00pp** | [+5.0, +27.5] | 6/0 | 0.031 |
| 不需老帧 (40) | 60.0% | 55.0% | +5.00pp | [0, +12.5] | 2/0 | 0.5 |
| one_old_frame (20) | 20.0% | 0.0% | +20.0pp | [+5.0,+40.0] | 4/0 | 0.125 |
| distractor_heavy (8) | 25.0% | 0.0% | +25.0pp | — | 2/0 | 0.5 |
| two_frame_compl. (12) | 0.0% | 0.0% | 0.00pp | — | 0/0 | 1.0 |

**preservation = 1.00,全局零回退(W/L 8:0,任何分层都没有输的任务)。**

**sanity 臂(必须一起读)**:random-2 vs recent-2 = **+3.75pp**(CI[0,+8.75],
p=0.25,**不显著**);learned vs random-2 配对 = **+6.25pp**(CI[0,+12.5],
p=0.125,n=80 下未达显著)。读法:①learned 的 +10.0pp 显著且零回退;
②"随便换帧"也能捞到 +3.75pp(需老帧层 recent-2 恒 0%,乱选也有命中率),
但不显著;③learned 约为 random 的 **2 倍**(+15.0 vs +7.5pp 在需老帧层),
方向一致但**在当前样本量下 learned>random 尚未达显著** —— 这是下一步要补的
最关键统计,不能跳过。

**如实的局限**:①n=80,分层后每格 8-22,per-regime 的 p 多数不显著;
②learned 只捕获了 oracle 头寸的约 1/5(需老帧层 15.0% vs oracle 73.3%);
③`two_frame_complementary`(需两张互补老帧)上 learned 与 recent 同为 0.0%,
完全没捕获;④move rate 66% 但**平均选中帧龄仅 1.89**,说明它主要在做
"邻近扰动"(把 (n-2,n-1) 换成 (n-3,n-1) 之类),而不是真正的远距检索 ——
这解释了 preservation=1.0,也解释了为什么远帧任务(two_frame)没起色。

### ★★★ Phase 3 判决(2026-08-14,n=300):**主 claim 成立,且关键控制臂显著**

口径与 n=80 那轮逐字相同(syn_ood 未见 family、自然混合分布、argmax 推理、
group 1、同一 `policy_mem_sft` executor,只改 memory rule),300 任务三臂配对:

| 对比 | 分层 | A | B | diff | 95% CI | W/L | p |
|---|---|---|---|---|---|---|---|
| **learned vs recent-2** | 总体 (300) | **31.3%** | 20.3% | **+11.00pp** | **[+7.33,+15.33]** | **34:1** | **≈0** |
| | 需老帧 (176) | 14.8% | 0.0% | +14.77pp | [+10.23,+19.89] | 26:0 | ≈0 |
| | 不需老帧 (124) | 54.8% | 49.2% | +5.65pp | [+1.61,+10.48] | 8:1 | **0.039** |
| **learned vs random-2** | 总体 | 31.3% | 20.0% | **+11.33pp** | **[+8.00,+15.33]** | **35:1** | **≈0** |
| | 需老帧 | 14.8% | 5.7% | +9.09pp | [+5.11,+13.64] | 17:1 | 0.00014 |
| | 不需老帧 | 54.8% | 40.3% | +14.52pp | [+8.87,+20.97] | 18:0 | 1e-05 |
| **random-2 vs recent-2** | 总体 | 20.0% | 20.3% | **−0.33pp** | [−3.33,+2.67] | 10:11 | 1.0 |
| | 需老帧 | 5.7% | 0.0% | +5.68pp | [+2.84,+9.66] | 10:0 | 0.002 |
| | 不需老帧 | 40.3% | 49.2% | **−8.87pp** | [−13.71,−4.03] | 0:11 | 0.001 |

分 regime(learned vs recent-2):one_old_frame **+27.6pp**(p=3e-05,16:0)、
distractor_heavy **+16.1pp**(p=0.002,10:0)、recent_sufficient +8.0pp、
history_irrelevant +4.1pp、two_frame_complementary 0.0pp(两臂同为 0%)。
preservation **0.984**,全局 W/L **34:1**。

**这三行一起读才是结论**:
1. **随机选帧的净收益是零**(−0.33pp):它在需老帧层捞到 +5.7pp,却在不需
   老帧层赔掉 −8.9pp(0 胜 11 负)—— "离开 recent-2"本身不值钱;
2. **学出来的 selector 两边都赚**:需老帧 +14.8pp、不需老帧 +5.7pp,净 +11.0pp;
3. 因此 **learned − random = +11.3pp(p≈0)**:收益来自**选得准**,
   不是来自"偏离 recent-2"。n=80 时这条只有 p=0.125,扩样后判定为显著。

**路线主 claim(§7)由此在 synthetic OOD 上成立**:同一个 memory-aware
executor 下只改 memory rule,RL selector 显著超过 recent-2,且显著超过随机。

**仍然如实的四条局限**:①learned 只吃到 oracle 头寸的约 1/5(需老帧层
14.8% vs Phase 2 实测 oracle 73.3%);②**平均选中帧龄 1.89** —— 学到的仍是
邻近扰动,不是远距检索;③`two_frame_complementary`(需两张互补老帧)
**零捕获**,两臂同为 0.0%;④只有**一个 selector 训练种子**,尚无重复种子
(效应 11pp 远大于历史上 ±3pp 的运行间噪声,但重复种子仍应补)。

### Phase 4A 管线打通(2026-08-16):**机器正确,但暴露一个结构性问题**

口径:hyper00 GPU2/3,任务容器 `sglang-omni-jaxan-20260813-090811-199280000`;
数据 = iter1+iter2 合并批,48 组 / 384 rollout(24 任务 × group 8 × 2 轮,
regime 分层采样),初始化 = `policy_mem_sft` + Phase 3 最佳 selector。

**先量了一件此前没量过的事:executor 的动作分布有多尖锐。**
整条动作序列(约 46 token)的概率中位数 **0.995**,**76.4% 的步 π>0.9**;
但同组两条 rollout 的逐步动作分歧率 **61.5%** —— 探索存在,却全部集中在
剩下那 ~24% 不确定的步上。

**联合更新的健康读数(修完全部 bug 之后)**

| 量 | 值 | 读法 |
|---|---|---|
| ratio_p50 / p95 / max | 1.0 / 1.39 / 7.92 | 中位数=1 说明 old 与重算同源;漂移可控 |
| **ratio_sel** p95 / max | **1.391 / 7.96** | selector 在动 |
| **ratio_pol** p95 / max | **1.012 / 2.11** | **executor 几乎不动** |
| grad_norm sel / exec | 0.908 / 0.629 | 均为 O(1),健康 |
| clip_frac | 0.256 | 正常范围 |
| nonfinite / 跳过的优化步 | 0 / 0 | 防线未触发 |
| 有效组 / 优化步 | 23 / 5 | **规模太小,不足以支撑任何结论** |

**⚠ 上面 `ratio_pol` 一栏不可直接读成"executor 学不动" —— 我先这么判过,
是错的**(口径错误,已更正,见下)。全体步上的 `ratio_pol` 把 76% 本来就
π≈1、∂logπ/∂θ≈0 的**饱和步**算进了分母;策略梯度只在策略不确定处有信号,
这是正常行为,不是病态。按不确定步(π_old<0.9,占 **11.65%**)分层重测:

| 度量(lr 2e-5 默认) | p5 | p95 | max |
|---|---|---|---|
| `ratio_pol` 全体步 | 0.994 | 1.010 | 2.11 |
| **`ratio_pol_UNCERTAIN`** | **0.821** | **1.208** | 2.11 |
| 对照 `ratio_sel` | 0.249 | 1.426 | 7.96 |

**更正后的结论:联合更新确实在同时推动两个组件。** executor 的位移集中在
它真正有决策余地的那一成多步上,量级(±20%)与 selector 可比。

**executor 学习率已用证据定档 2e-5**(不提高):

| `ratio_pol_UNCERTAIN` | p5 | p95 | max |
|---|---|---|---|
| lr 2e-5 | 0.821 | 1.208 | **2.11** |
| lr 2e-4 | **0.287** | 1.44 | **33.2** |

10 倍学习率只把 p95 从 1.21 拉到 1.44,却让 p5 塌到 0.287、尾部涨到 33.2 ——
信任域被严重破坏,属过冲而非"学得更多"。

**本轮修掉的 bug(全部是实现/口径错误,非方法问题)**

| bug | 症状 | 根因 |
|---|---|---|
| `group_id` 强转 int | 启动即崩 | schema 里是 `template::regime::seed` 字符串 |
| 自建 `SelectorStateBuilder` | 启动即崩 | 应复用采样端 `resolve_state_factory` |
| 残留 `build_extractor` | 启动即崩 | 特征源已统一由 `resolve_feature_provider` 提供 |
| log π_old 用解码路径的值 | 初始 ratio 1.52,超裁剪带 | 改为初始权重下同一前向路径重算 |
| `use_cache=True` | OOM 61.5 GB | HF 静默关掉梯度检查点(只发 warning) |
| `entropy` 的 `0·log0` | grad_norm=Inf,权重被打坏 | 概率下溢时 `p*logp` 得 nan |
| **old 缓存键冲突** | **ratio 爆到 1e+26** | `rollout_id` 只是组内序号 0..7,4341 个 (id,step) 组合被压成 128 个 |

**跨条目的教训(值得单列)**:上表后三条 bug **只在"executor 不确定的步"
上显形**,汇总指标一律正常 —— 键冲突时 ratio 中位数仍是 0.9993,因为 76%
的步两条 rollout 的 logπ 都≈0,错了也看不出来。这个系统的任何自检都必须
**按不确定步分层看**,只看均值必然漏掉真正要紧的那 24%。

### ★★ Phase 4A 首次五臂验收(2026-08-17,n=300 syn_ood):**诚实的零结果**

口径与 Phase 3 的 n=300 那轮**逐字一致**(syn_ood 未见 family、task-seed 2101、
自然混合分布、argmax、group 1、max-steps 20、同 297 配对任务),故可跨轮并排。
训练侧:96 组 / 38 有效组 / 9 个优化步,lr 2e-5,只解冻 executor 后 8 层 LoRA。

| 比较 | 含义 | 差值 | 统计 |
|---|---|---|---|
| **A** arm4 vs arm5 | selector 在**新** executor 下的增益 | **+11.67pp** | CI[+8.33,+16.00] W/L 36:1 p≈0 |
| **B** arm5 vs Phase3 recent-2 | **executor 侧净增益** | **−0.33pp** | W/L 0:1 p=1.0 |
| **C** arm4 vs Phase3 完整系统 | 完整系统变化 | **+0.33pp** | W/L 1:0 p=1.0 |

**结论:一次联合更新没有带来任何可测量的闭环变化。** B 臂精确到"executor 的
行为改变只翻转了 300 个任务中的 **1** 个";C 臂同理。selector 的 +11pp 优势
**完整保持**(Phase 3 +11.00pp → 现在 +11.67pp),说明联合更新**没有损坏**
已有能力,但也没有增加。

**这不是"方法无效"的证据,是"更新量太小"的证据**,与训练侧量级完全自洽:
executor 只在 10.4% 的步上位移 ±20%,9 个优化步不可能改变闭环行为。
分层读数也一致 —— 需老帧任务上两个 recent-2 臂都恒为 0.0%,
`two_frame_complementary` 仍是 0.0%(与 Phase 3 的零捕获一致)。

**据此启动迭代联合训练**(`joint_iterate.sh`):6 轮 × 48 任务 × group 8,
每轮 collect→update,断点续跑(已有 `selector_after.pt` 的轮次跳过),
逐轮回收截图。目标是把优化步从 9 提到约 60,再重测同一套五臂。

### ★ B=4 排除性验证(2026-08-17):**加大记忆预算无用,且这是个正面结果**

动机:§4.8 一度把 `two_frame_complementary` 的 0% 归因为"B=2 预算被两个老帧
吃满、无余量"。该措辞过头(见审计 §4.8 更正),故用**最有利于预算假说的配置**
做排除性验证:内容对齐 oracle + **B=4**,同 297 syn_ood 任务、同 executor、
同闭环口径,与 B=2 的同一 oracle 配对比较。

| 分层 | B=4 | B=2 | 差 |
|---|---|---|---|
| 总体 | 30.0% | 29.0% | +1.00pp CI[−1.00,+3.00] p=0.51 **不显著** |
| `two_frame_complementary` | **0.0%** | **0.0%** | **+0.00pp** |
| `one_old_frame` | 32.8% | 27.6% | +5.17pp p=0.25 不显著 |
| `distractor_heavy` | 16.1% | 16.1% | +0.00pp |

**两条结论**:

1. **预算假说被排除**。预算翻倍后 two_frame 仍精确为 0.0%,病因不在容量,
   而在闭环前缀质量(审计 §4.8)。
2. **这同时是论文的正面证据**:把记忆预算翻倍,总体只换来 +1.00pp 且不显著;
   而 Phase 3 的 learned selector 在 B=2 下相对 recent-2 是 **+11.00pp
   (p≈0)**。即 **选得准 ≫ 给得多** —— 正是本工作的核心主张。
   固定原则 B=2 因此不必更改(该实验仅为诊断,未用于训练)。

### 迁移到 Tilde 8×H100(2026-08-17)

hyper00 上单轮约 2h(采样 37min/4卡 + 训练 78min/**1卡**),且我把多个任务塞进
同一容器导致互抢(审计 §0.16)。按修订后的全局规则(无每主机卡上限、按优先级
取当前空闲卡、不排队)迁到 Tilde:整节点 8×H100 空闲、队列为空。

迁移前逐项验证,未"提交了再看会不会崩":

| 验项 | 结果 |
|---|---|
| PIL 字体(缺字体时 PIL **不报错**,只是画不出字) | 渲染 1920×1080,灰度唯一值 227 ✓ |
| `pylibs` 的 transformers 5.6.0 是否盖住镜像自带 | 5.6.0 ✓(顺序错会撞冻结基座校验) |
| H100 **80GB**(hyper 是 143GB H200) | 模型加载 26s,占 16.3GB ✓ |
| checkpoint 跨主机通路 | HF Hub,195MB **4 秒** ✓(对照:早期经 Mac 中转 4.4GB 耗时约 2h) |

Tilde 配置:每轮 96 任务 × group 8(hyper 上是 48),采样 8 分片、**预扫 8 分片**
(预扫是 hyper 上最大的串行瓶颈,占单轮 2/3)、更新 1 卡;8 轮;
`--requeue` + 逐轮 checkpoint 跳过,最大重放一轮。

### ★★ Phase 4A 学习曲线(2026-08-18,更新至 43 优化步):**executor 侧平坦**

迭代联合训练迁至 tilde 8×H100 后(每轮 768 rollout,约 3.1h/轮),
用 tilde round3 checkpoint(累计约 43 个联合优化步)按同一口径
(n=300 syn_ood、task-seed 2101、argmax、group 1)加测一个曲线点:

| 优化步 | selector 增益(完整 vs +recent2) | **executor 净增益**(+recent2 vs Phase3 +recent2) | 完整 vs Phase3 完整 |
|---|---|---|---|
| 0(Phase 3) | +11.00pp | — | — |
| 9(full1) | +11.67pp | **−0.33pp** | +0.33pp |
| **43(tilde r3)** | **+12.67pp** CI[+9.00,+17.00] | **−0.33pp**(逐点同 9 步) | +1.33pp p=0.29 |

**读法**:34 个额外优化步后 executor 的闭环行为纹丝不动(B 臂 −0.33pp 与
9 步时逐点相同)。机制假说(与训练读数自洽):更新只在 ~11% 的不确定步上
移动 ±20% 概率,而部署是 **argmax 解码** —— 概率挪动不翻转众数,闭环行为
就不变。selector 增益稳步上行(11.00→11.67→12.67,硬分层 +17.61pp 为
历史最高),说明 joint 里 selector 仍在受益,只是 executor 不动。

**机制验证(已完成,确证)**:teacher-forced 决策步探针,round3 adapter vs
policy_mem_sft,相同 297 任务相同输入:

| | oracle 供帧 | recent-2 |
|---|---|---|
| policy_mem_sft(0 步) | 43.0% | 23.0% |
| round3(43 步) | 43.3% | 23.7% |
| 硬分层 oracle | 34.1% | 34.1%(逐点相同) |

teacher-forced 也不动(差均 ≤0.7pp,噪声内)⇒ **43 个联合优化步挪动了概率
(训练侧 ratio 可见)但没有翻转任何 argmax 决策** —— 病因不是轨迹偏离,
是更新量对贪心部署不构成行为变化。

**三个候选方向(等用户裁决,round5–8 照跑补齐曲线末点)**:
1. **executor 侧升温采样 + 熵正则**拉开分布再训 —— 代价是训练分布偏离贪心
   部署,且需要的数据量可能仍是数量级问题;
2. **奖励塑形** —— 注意与路线 §9 冻结原则冲突(禁止 milestone/step-level
   训练信号),若走此路需先明确修约;
3. **接受"RL 动不了 executor"作为结论**,把 Phase 4 收窄为
   selector-under-joint,论文故事:**SFT 塑造 executor(53%→98%),RL 塑造
   selector(+11~13pp);在本算力尺度下 joint RL 对 executor 无增益**
   ——与 B=4 的"选得准≫给得多"互为支撑,是自洽的负结果叙事。
   (推荐:3,理由是 1 的数据量在本尺度不现实、2 违反冻结原则。)

### ★★★ Phase 4A 学习曲线 78 步末点(2026-08-19):**结论反转——joint 有显著闭环增益,通道是 selector**

8 轮 tilde 迭代全部完成(累计约 78 联合优化步),同口径 n=300 末点:

| 优化步 | selector 增益 | executor 净增益 | **完整系统 vs Phase 3** |
|---|---|---|---|
| 0 | +11.00pp | — | — |
| 9 | +11.67pp | −0.33pp | +0.33pp p=1.0 |
| 43 | +12.67pp | −0.33pp | +1.33pp p=0.29 |
| **78** | **+14.33pp** W/L 44:1 | **±0.00pp,move=0.0%** | **+3.33pp CI[+1.33,+5.67] p=0.006** |

**三个同时成立的事实**:
1. 完整系统 31.3%→34.7%,**统计显著**(硬分层 +5.11pp p=0.004),且四点曲线
   单调上行、未见顶 —— 此前"joint 无增益"的判断是**在 9/43 步下的过早结论**,
   用户坚持续训是对的;
2. 增益**全部走 selector 通道**(11.00→14.33pp 单调),executor 贪心行为
   78 步逐位不动(B 臂全格 +0.00pp、move=0.0%)——"SFT 塑造 executor、
   RL 塑造 selector"的分工在更长训练下依然成立,但系统层面 joint 是**正结果**;
3. `two_frame_complementary` 仍 0.0%(结构性,见审计 §4.8,与训练量无关)。

**决策**:已续训 rounds 9–24(至约 230 步,tilde job 186104,开放式终点:
连续 3 轮 reward 无上行 + 贪心探针无 argmax 位移才停);采样部署对比
(r8 vs Phase3,T=1.0)即将出结果,回答 executor 通道是否可能在更多步后打开。
若曲线斜率不衰减,准备 100+ H100 多节点方案(采样分片现成,数千步可达)。

**词表外漂移**(采样下 RL 改变行为的旁证 + 已修的 shard 级崩溃):round5
executor 吐过 'select',round9 吐过中文 '左点击' —— 后者进入 history 后
在下一步 prompt 构建时抛 PolicyIOError 杀死整个 shard worker(丢 72/96 条)。
已修:动作进历史前验证官方可渲染性,不可渲染 → episode 以
reason=unrenderable_action 终止、reward 0(合法终局信号,不别名化不丢样本),
worker 继续。round10 起生效。

### ★★ 采样部署对比(2026-08-19):**executor 通道判定闭合——训练收益不泛化**

r8 vs Phase 3,同 297 syn_ood 任务配对,两臂均 executor T=1.0 采样、selector
argmax,唯一差异是权重:总体 27.7% vs 27.3%(+0.33pp CI[−3.00,+4.00] p=1.0),
需老帧 +1.70pp(n.s.),two_frame 0/0。

**与 78 步贪心 B 臂(逐位 +0.00pp)合并读**:executor 在 held-out 上无论贪心
还是采样都不动 ⇒ 训练侧 reward 爬升(0.43→0.59)中属于 executor 的部分是
**train 分布上的拟合,不泛化**;能泛化的增益全部走 selector 通道
(+11.00→+14.33pp,且带动完整系统 +3.33pp p=0.006)。

**对"是否训练不够/要不要借 100 H100"的分渠道回答**:
* **selector 通道:是,继续训继续涨**(四点曲线单调、未见顶)。但它是 41M
  小模型,瓶颈在 rollout 吞吐,tilde 8 卡续训(rounds 9–24,进行中)就能画完
  这条曲线,**不需要 100 卡**;
* **executor 通道:没有证据表明同配方加算力有用**(贪心+采样双重持平)。
  要动它得改方法:任务多样性扩容(现仅少数 family,executor 可能已吃尽
  可用信号)、针对 B1 探索行为的定向 SFT、或大得多的任务空间 —— 这才是
  100 卡有意义的用法,但属于设计变更,不是"再跑几天";
* **机会成本**:论文当前真正的缺口是外部效度(Phase 5 真实 UI bridge /
  Phase 6 OSWorld 无回归),合成环境的数字已经形成完整故事
  (selector RL +14.33pp、B=4 选得准≫给得多、two_frame 结构性诊断、
  SFT/RL 分工)。建议把新算力优先投给 Phase 5/6。

### ★★★ Phase 4A 收敛判定(2026-08-20):**平台确认于 ~34.7% / +14.33pp,训练线关闭**

用户质疑"真的训到收敛了吗"之后,把定长 8 轮改为开放式续训(rounds 9–15,
累计约 145 优化步),并加测 128 步曲线点(round14 ckpt,同口径 n=300):

| 优化步 | 完整系统 | selector 增益 | vs Phase 3 完整 |
|---|---|---|---|
| 78(r8) | 34.7% | +14.33pp | +3.33pp p=0.006 |
| **128(r14)** | **34.7%** | **+14.33pp** | **+3.33pp**(与 78 步逐位同值) |

**收敛证据(三重,且排除了'评错 ckpt')**:两个 ckpt sha 不同;300 任务里
257 个轨迹逐位相同、成功数完全一致(104/300)—— rounds 9–14 的 +50 步只造成
43 个任务的轨迹抖动、净增益为零;训练侧 reward 六轮平台(0.54–0.60);
selector 熵 0.78→0.56 收敛。**病因是任务分布被吃干**(约十个模板 × 5 regime),
不是 RL 的一般上限 —— 要重开上行空间需扩模板池(Phase 1.5),不是加算力。

**执行**:按用户指令"平了就转 Phase 5/6",tilde 训练停止(scancel 191301,
8×H100 释放),最终 ckpt(round15,约 145 步)已上传
`gavinlaw/causalcache-agentic-ckpt` 的 `tilde_round15_final/`。
**合成环境最终数字定格**:系统 34.7%(vs recent-2 基线 20.3%),selector
增益 +14.33pp(W/L 44:1 p≈0),vs Phase 3 全系统 +3.33pp(p=0.006)。

**Phase 5/6 已并行启动**(hyper00):OSWorld 真实 UI smoke 12 任务 × 2 臂
全通零故障 —— policy_mem_sft 未在真实 UI 上崩坏,且在做对的任务上比冻结
基线快 2–3 倍(4–6 步 vs 15 步满额,学会了 terminate);全量 135 任务
无回归评测(2 臂 × 4 worker,8 并发 VM)运行中。

### RL 上限扩容与 Phase 5/6 选型(2026-08-20,与用户对齐)

Phase 4A 收敛判定后,用户问"怎么扩 RL 上限",并给出两条原则:eval 要选
**远端历史 image 恢复有意义**的任务;RL 训练任务要**长短类型多样**防过拟合。
结合 ChatGPT 提供的 benchmark 清单(WAA/OpenComputer/WorldGUI/WindowsWorld/
OSUniverse/CRAB/ComputerRL 等),定下三件事:

1. **训练侧:扩自己的程序化模板池,不搬真实 VM 来训。**
   吞吐差 5 个数量级(PIL 4410 steps/s vs 真实 VM ~2min/episode),我们的
   卡规模撑不起 ComputerRL 式 1000+ 并行实例;正确姿势是把外部 benchmark 的
   **任务语义**翻译成程序化模板(表格操作/邮件流/文件管理/跨应用取值-运算-
   回填,参数化 operation,4–30 步长短混合),从 ~10 family 扩到 50–100,
   乘上既有 5 regime。纯代码工作(Phase 1.5 LLM 辅助),不花 GPU。
2. **评测侧:记忆关键性要自己测,不信标签。** 没有 benchmark 标注"需要
   老帧";用审计 §4.8 的内容可见性探针审计 OSWorld-Verified(multi_apps 域
   优先)与 WindowsWorld,筛真正记忆关键的子集作 bridge set,不足则按同
   结构自造 30–60 个真实 UI 任务。**"现有 benchmark 记忆关键任务占比≈0"
   本身是可量化 finding**,论证本工作环境的必要性。
3. **切分与引用**:OSWorld-Verified 全量 held-out 永不训;按 family 切分;
   related work 引 OpenComputer/ComputerRL/WAA/WorldGUI/CRAB/OSUniverse。

**明确不做**:①shaped reward(subgoal/partial/step penalty)——违反 §9 冻结
原则(终局 0/1 是论文身份的一部分),除非用户显式解冻;②WAA/Windows 系
暂缓(需 Windows 11 VM 另一套基建,收益未证)。

### ★★★ Phase 5 首个真实 UI 判定(2026-08-20):**无回归检验不通过,bridge 训练转为必选**

管线:OSWorld docker VM(hyper00,/dev/kvm)+ 官方桌面协议 policy server
(`serve_osworld_official_policy.py`,新增 `--lora-bundle` 挂 qkvo LoRA)。
fast_devset_v1(OSWorld 135 任务,10 个域),同任务配对,B=2 recent、
max-steps 15:

| 臂 | 成功率 | 平均步数 |
|---|---|---|
| 冻结 GUI-Owl + recent-2 | **27.1%** | 14.6 |
| policy_mem_sft + recent-2 | **14.1%** | 13.7 |

**−13.04pp,W/L 6:24 —— 纯合成 SFT 造成真实 UI 灾难遗忘**(此前 12 任务
smoke 的 7→5 是真信号)。按域分层是双向的:thunderbird 100→28.6%、
gimp 71→29%、calc 67→11% 重灾;但 writer 20→**60%**、chrome 0→25%、
vlc 75→100% 反涨(学到的提前终止/记忆行为在部分域是净收益)。

**含义**:①Phase 2 的 mem_sft adapter 不能直接当真实 UI executor;
②bridge = **混合 SFT**(合成记忆数据 + 真实 UI 语料如 AgentNet 桌面集,
后者在 tilde 已有)从可选变必选,且有了量化动机(-13pp 遗忘 + 部分域正迁移
说明混合配比找得好就能兼得);③分工:数据采集在 hyper00(kvm),
训练在 tilde 8×H100。

**并行进行**:第三臂(mem_sft + r15 学习 selector 零样本,port 18803,
LoRA 与 selector 双挂载)运行中 —— 它与 memsft+recent-2 共享同一 executor,
selector 效应可分离;server 端已把 agentic selector 接入官方协议
(编号口径逐条核对:agentic 帧索引 ≡ OSWorld 事件 step_id)。

### ★★★ Phase 5 合龙(2026-08-21):bridge 混合 SFT 闭环兑现,完整系统追平冻结基线

**Bridge 混合 SFT 终版指标**(tilde job 194940,合成 2591 + AgentNet 真实
5335 决策点 1:1 混合,2 epoch,全程零样本丢失):
teacher-forced 合成 **98.75%**(纯合成训练 98.1%,记忆行为无损)/
真实 UI **75.42%**。adapter 正本
`gavinlaw/causalcache-agentic-ckpt` 的 `bridge_sft_r1.0/adapter_final.pt`。

**闭环判定(OSWorld fast_devset 135,同口径配对)**:

| 臂 | 成功率 | 读法 |
|---|---|---|
| 冻结 GUI-Owl + recent-2 | 27.1% | 基线 |
| 纯合成 SFT + recent-2 | 14.1% | −13.04pp 灾难遗忘 |
| **bridge SFT + recent-2** | **23.7%** | 遗忘坑填回 74%(−3.41pp,W/L 12:17) |
| **bridge SFT + r15 selector**(n=135 定稿) | **25.9%** | 距基线 −1.2pp(n=102 暂计曾达 27.5% 打平,最后 33 个长历史难题拉低) |

selector 增益全量定格 **+2.22pp**(W/L 7:4)——合成环境训出的选帧策略对
健康 executor 的真实 UI 增益。分域:chrome 0→66.7%、writer 20→75%;
残余赤字集中于 thunderbird(100→40)与 gimp/calc 未恢复部分。
**遗忘坑合计填回 90%**(14.1→25.9)。

**距离"主流 benchmark 跨零上涨"的最后一步**:−1.2pp → 为正。
赤字集中在少数域 ⇒ 下一轮:混合配比 1:1→2:1(真实:合成)或定向补
thunderbird 族真实轨迹;bridgedsel 剩余 33 任务补跑中,补齐后出正式统计。

### ★★★ MIX=2.0 定稿(2026-08-22):遗忘弧线闭合,r2.0 触及跨零

135 全量五臂(同口径配对):基线 27.1% / r1.0+recent2 23.7% / **r2.0+recent2
27.4%(+0.29pp,W/L 15:15)** / r2.0+selector 26.7%。

1. **灾难遗忘完整修复**:27.1 → 14.1(纯合成)→ 23.7(1:1)→ 27.4(2:1,
   回到基线之上);thunderbird 57→71%、vlc 75→100%;合成记忆 99.17% 保持。
   **正确表述:无回归 + 记忆能力零代价加入**(+0.29pp 是统计平手,不得写
   "显著提升")。
2. **selector 在 r2.0 executor 上中性**(−0.74pp,W/L 5:6)——OSWorld 缺记忆
   关键结构,selector 本无用武之地;其增益在记忆关键分布(合成 +14.33pp)
   与 executor 不完美时(r1.0 上 +2.22pp、受损 memsft 上 +4.46pp)。
   这是诊断性发现,进论文分析章。
3. **论文四支柱定型**:合成 selector RL +14.33pp;B=4 选得准≫给得多;
   混合 SFT 桥接(2:1 配比,遗忘弧线);机制诊断(two_frame/遗忘/恢复)。
   adapter 正本 `bridge_sft_r2.0/adapter_final.pt`。

**剩余最后一块**:真实 UI 记忆关键 bridge set(内容可见性探针审计
OSWorld multi_apps + 按 evidence-then-occlusion 结构自造)——selector 在
真实像素上复现两位数增益的地方;Phase 6 的 OpenComputer 纯 OOD 评测其后。

### 下一步(按优先级,2026-08-14 更新)

1. ✅ **learned > random 已做到显著**(n=300:+11.33pp,p≈0);
   剩余统计补强:**重复 selector 训练种子**(当前仅 1 个种子);
2. **攻"远距检索"**:平均选中帧龄仅 1.89、two_frame_complementary 零捕获,
   说明 selector 学到的是邻近扰动。可选手段(按代价):延长训练迭代、
   在分层采样里加大 two_frame 权重、检查 recency 先验强度是否压制了远帧;
3. 补 Phase 2 统计强度:闭环每臂 40 → 150 + bootstrap CI;
4. 闭环 OOD 四臂(补"不是背模板"的闭环证据);
5. Phase 4 joint GRPO(仅当 1 与 2 站住之后)。
