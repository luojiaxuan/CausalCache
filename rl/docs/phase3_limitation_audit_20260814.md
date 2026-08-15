# Phase 3 局限审计(2026-08-14,供外部诊断用,自包含)

> 目的:Phase 3 的主 claim 已在 synthetic OOD 上成立(learned selector 显著
> 超过 recent-2 与 random-2),但只吃到 oracle 头寸的约 1/5,且明显学成了
> "邻近扰动"而非"远距检索"。本文件给出**全部可复现数字与已排除的解释**,
> 请诊断:**瓶颈在哪一层,下一步该改什么。**

---

## 1. 系统结构(三层,逐层可换)

```
task(程序化 GUI 环境,终局 0/1 verifier)
   │
   ├── executor:GUI-Owl-1.5-8B + 全层 qkvo LoRA r8(= policy_mem_sft,Phase 2 产物)
   │      输入 = instruction + 全部历史动作文本 + B 张历史截图(高保真) + 当前屏
   │      输出 = computer_use 动作([0,999] 归一坐标)
   │      **Phase 3 期间完全冻结**
   │
   └── selector:41.2M 参数(FrameResampler → CandidateSetEncoder → 非加性 SubsetScorer)
          输入 = 冻结视觉特征(每帧 ~2584×4096,GUI-Owl 视觉塔+投影,逐帧独立)
                + instruction/动作行的冻结词嵌入 + 帧龄 + recent-B 标记
          输出 = 全部 C(n,2) 子集的 logit → softmax 采样(训练)/ argmax(推理)
          训练 = GRPO,奖励**只有终局任务成功 0/1**,GUI-Owl 不在计算图内
```

关键约束(路线文档 §9,**诊断方案不得违反**):禁止 milestone reward、
step-level 正确性作训练信号、gold 帧监督、pass-1 draft、B=1 逐帧探针、
Gumbel-softmax 软选帧、benchmark 上做 RL。

## 2. 环境与任务(已通过全部验收)

- 5 个 workflow family / 12 template,tier 2-5(2-5 阶段跨应用工作流);
- 五种 **memory regime**,语义与 regime 解耦(同一 instruction 在不同 regime
  下都成立):`recent_sufficient` / `one_old_frame` /
  `two_frame_complementary` / `distractor_heavy` / `history_irrelevant`;
- 证据"不可见"是**视觉上真不可见**(被切换/滚走/覆盖)—— 已独立验证:
  108 个老帧任务里,决策步所需证据在 recent-2 帧的顶层窗口文本中**一个都
  查不到**、且都在被标记的老帧里;72 个简单任务上证据确实可读(对照);
- 专家成功率 1.0(300 任务),渲染确定性逐字节,吞吐 4410 步/秒(无渲染)。

## 3. 已确立的事实(全部为配对实验,syn_ood = 未见 workflow family)

### 3.1 环境有记忆需求(Phase 2 决策步诊断,n=200,teacher-forced 前缀)

| 分层 | oracle | recent-2 | random-2 | none |
|---|---|---|---|---|
| 需老帧 (147) | **98.6%** | **0.0%** | 19.7% | 0.0% |
| 不需老帧 (53) | 98.1% | 98.1% | 84.9% | 58.5% |

oracle−recent2 = +98.6pp CI[+96.6,+100],赢/输 145:0;不需老帧层两者逐条相同。

### 3.2 闭环上限(Phase 2,40 任务 × 4 臂,终局 0/1)

| 分层 | oracle | recent-2 | random-2 | none |
|---|---|---|---|---|
| 总体 | 82.5% | 55.0% | 40.0% | 5.0% |
| 需老帧 (15) | **73.3%** | **0.0%** | 6.7% | 0.0% |

### 3.3 Phase 3 判决(n=300,自然混合分布,argmax 推理)

| 对比 | 总体 | 需老帧 (176) | 不需老帧 (124) |
|---|---|---|---|
| learned vs recent-2 | **+11.00pp** CI[+7.33,+15.33] W/L 34:1 p≈0 | +14.77pp | +5.65pp p=0.039 |
| learned vs random-2 | **+11.33pp** CI[+8.00,+15.33] p≈0 | +9.09pp | +14.52pp |
| random-2 vs recent-2 | **−0.33pp** p=1.0 | +5.68pp p=0.002 | **−8.87pp** p=0.001 |

绝对值:learned 31.3% / recent-2 20.3% / random-2 20.0%;preservation 0.984。
分 regime(learned−recent2):one_old_frame **+27.6pp**、distractor_heavy
**+16.1pp**、recent_sufficient +8.0、history_irrelevant +4.1、
**two_frame_complementary 0.0(两臂同为 0%)**。

## 4. ★ 关键诊断:检索失败,不是利用失败

对同一批 300 条 rollout,在**决策步**上比对 `chosen_subset` 与
`memory_probe.required_steps`(后者仅用于诊断,从不进训练):

| 臂 | 分层 | n | required 全命中 | 命中时成功 | 未命中时成功 | required 平均帧龄 | 选中平均帧龄 |
|---|---|---|---|---|---|---|---|
| **learned** | 需老帧 | 176 | **18.8%** | 24.2% | 12.6% | **6.22** | **2.00** |
| learned | one_old_frame | 58 | 24.1% | **57.1%** | 18.2% | 6.41 | 2.00 |
| learned | two_frame_compl. | 56 | **0.0%** | n/a | 0.0% | 6.45 | 2.00 |
| learned | distractor_heavy | 62 | 30.6% | **0.0%** | 23.3% | 5.63 | 2.00 |
| learned | 不需老帧 | 124 | 0.0%※ | n/a | 54.8% | 2.17 | 2.00 |
| recent-2 | 需老帧 | 175 | 0.0% | n/a | 0.0% | 6.20 | 1.50 |
| recent-2 | 不需老帧 | 124 | 49.2% | **98.4%** | 1.6% | 2.17 | 1.50 |
| random-2 | 需老帧 | 176 | 15.9% | 17.9% | 3.4% | 6.22 | 4.46 |

※不需老帧层 learned 的"全命中 0%"是标注口径问题(那里 required 常是"当前屏",
不在候选池),不影响结论。

**四条读法**:
1. **selector 的选中帧龄被钉在 2.00,而需要的帧在 6.22** —— recent-2 是
   {龄1,龄2}(均值 1.5),learned 是 {龄1,龄3} 之类(均值 2.0):它学到的是
   **把窗口往回挪一格**,不是远距检索。random-2 的 4.46 反而更"远";
2. **检索是主瓶颈**:需老帧层 required 全命中只有 **18.8%**;
3. **一旦检索成功就很值钱**:one_old_frame 命中时成功 **57.1%** vs 未命中
   18.2%(3 倍);而 executor 本身在闭环里给对帧时能到 **98.4%**
   (recent-2 在简单层的命中时成功)—— 说明 **executor 不是瓶颈**;
4. **两个异常需要解释**:
   - `two_frame_complementary` 全命中 **0.0%**(需要同时选两张远帧,
     结构上被"邻近扰动"排除);
   - `distractor_heavy` 命中 required 时成功率 **0.0%**,反而不如未命中的
     23.3% —— 怀疑是"选中了 required 帧,但第二个槽位选中了干扰帧",
     即**选对一半比选错更糟**。这条尚未单独验证。

## 5. 训练配置与轨迹(可能的病因都在这里)

- GRPO:3 轮迭代 × 48 任务 × group 16 = 2304 条 rollout;
  clip ε=0.2、entropy β=0.01、KL 关闭、lr 默认、逐步 backward;
- **难度分层采样**(训练侧):记忆关键三种 regime 各 25%,
  recent_sufficient 15%、history_irrelevant 10%(评测仍在自然分布);
- 有效组(组内 reward 有方差)比例:26.9%(v1)→ **41.7 / 43.8 / 33.3%**;
- 训练轨迹:

| 迭代 | mean_reward | 有效组 | entropy | π(recent-2) | recent 为 argmax |
|---|---|---|---|---|---|
| 1 | 0.319 | 41.7% | 1.843 | 0.360 | 1.00 |
| 2 | 0.264 | 43.8% | 1.709 | 0.301 | 0.381 |
| 3 | 0.264 | 33.3% | 1.787 | **0.232** | **0.00** |

- selector 初始化带 **modest recency 先验**:`recent_strength=4.0,
  recent_tau=2.0`(初始 π(recent-2) ≈ 0.3-0.5),按路线 §5 "不要极强 KEEP
  bias" 的要求设置;先验是**可学的**,不是硬约束;
- 采样温度 1.0;推理 argmax;
- **只有一个 selector 训练种子**(重复种子在跑)。

## 6. 已经排除的解释

- **不是 executor 不会读老帧**:给对帧时决策步 98.6%(OOD)、闭环 73.3%;
- **不是"离开 recent 就有收益"**:random-2 相对 recent-2 净收益 −0.33pp
  (需老帧 +5.68,不需老帧 −8.87,0 胜 11 负);
- **不是环境没有记忆需求**:见 §3.1 的 +98.6pp 与阴性对照;
- **不是奖励信号被污染**:只有终局 0/1;logπ 重算一致性 max|Δ|=2.6e-07;
  防作弊测试(置换等变 / 诊断不泄露 / 只用终局奖励)全过;
- **不是评测偏置**:训练侧分层采样,评测固定自然混合分布 + 未见 family。

## 7. 请诊断的问题

1. **为什么 selector 收敛到"窗口回挪一格"而不是远距检索?** 候选解释:
   (a) recency 先验强度 4.0 在仅 ~60 个优化步下无法被克服;
   (b) **探索问题**:采样分布集中在近邻,远帧子集几乎从未被试过 →
       GRPO 永远看不到它们的回报(C(13,2)≈78 个动作,远帧组合占多数却
       几乎零采样);
   (c) 信用分配:终局稀疏奖励 + 需老帧任务成功率本来就低(14.8%),
       正样本太少;
   (d) 表示问题:帧特征逐帧独立编码,selector 缺"我现在需要什么值"的
       查询信号(instruction 与动作历史只经小文本编码器进入)。
2. **该怎么修?** 具体想听的是:探索策略(温度退火 / 先验退火 / ε-greedy /
   按帧龄分层的采样)、目标函数(是否该加对远帧的探索奖励——注意
   §9 禁止 milestone/step-level reward,但"动作空间上的探索项"是否算越界?)、
   架构(是否需要显式的 query-key 检索结构)、课程(先用 required 帧在
   近处的任务?那会不会又变成邻近扰动)。
3. **`two_frame_complementary` 零捕获**:需要同时选两张远帧,组合空间里
   这类子集占比极低。是该在采样上做结构化提议(propose-then-score),
   还是该把 B=2 的联合决策拆成两次条件选择?后者会不会破坏"非加性"?
4. **`distractor_heavy` 命中 required 却 0% 成功**:如果确认是"第二槽位
   选了干扰帧",selector 该如何学会"避开干扰"——这需要它判断两帧之间
   的冲突,而不只是各自的相关性。
5. **样本效率**:当前 2304 条 rollout / 3 轮 = ~60 个优化步。要看到远距
   检索,合理的量级是多少?是该加 rollout 还是加 epoch(off-policy 重用)?

## 8. 可用资源与成本

- 4×H200(与他人共享,实际可用 2-4 张);rollout 22.5s/条(闭环含
  executor 生成);GRPO 更新 ~20 分钟/轮(逐步 backward,峰值 25GB);
- 环境吞吐:纯逻辑 4410 步/秒、带渲染 81 步/秒 —— **任务生成不是瓶颈,
  executor 的自回归解码才是**;
- 可无限生成新任务实例与新 family(程序化);
- 冻结视觉特征可缓存复用(每帧 ~21MB fp16)。

---

## 9. 代码位置(如需具体实现细节)

`rl/code/causalcache_agentic/`:`contract.py`(状态模型)、`render.py`、
`executor.py`、`apps.py`、`tasks.py`(DSL 与 regime)、`env.py`、`expert.py`、
`policy_io.py`(prompt 装配与 selector 接口)、`features.py`、
`selector_model.py`(选择器);
`rl/code/scripts/`:`rollout_selector.py`、`train_selector_grpo.py`、
`agentic_selector_eval_reduce.py`、`agentic_selector_diagnose.py`(本文件 §4 的
数字由它产出)。路线文档:`rl/docs/agentic_memory_rl_roadmap_20260813.md`。
