# 外审:两级 RL(executor-in-the-loop 密集反馈 → 闭环终局)设计(2026-09-03 17:45 PT)

工具:网页 ChatGPT 临时会话,推理档"极高",思考 1m54s(含联网检索)。审前判断见 §1,原文
§2(逐字),采纳/不采纳 §3。

## 1. 审前判断(先写后审)

用户确认项目必须有 agentic RL,并支持两级设计。我的方案:第一级在 AndroTMem-Bench
(1069 题/34k 步,按构造含决定性中间状态)的轨迹上做 executor-in-the-loop 密集反馈 RL——
每步策略采样帧子集,冻结 executor 用它出动作,reward = 与记录动作一致;第二级只在可控
任务分布(自造记忆关键训练任务,永不用测试集)上做闭环终局 RL;评测 MemGUI-Bench(在线)
+ MementoGUI(离线)。可变 B 作为扩展(序贯选帧 + 停止动作,λ 扫描),放在固定 B 站住之后。
风险自评:MemGUI 上基座 11.7% 可能重演"两头堵";动作一致 reward 可被 hack;离线轨迹到
在线 rollout 的分布漂移。

## 2. 外审原文(英文,逐字)

Yes, Level 1 is formally RL, but the safest paper framing is contextual-bandit policy optimization with executor feedback, not "long-horizon RL." The overall project makes sense; the main danger is not terminology but learning a selector that scores well on teacher-forced action matching and does almost nothing for live task success.

### 1) Level 1: RL or supervised learning?

Let the controller choose memory M∼π_θ(M|x), the frozen executor produce E(x,M), and r(x,M)=1[E(x,M)≃a*]. Optimizing E_{M∼π_θ}[r] with REINFORCE/RLOO is legitimately a one-step contextual-bandit RL problem. The controller's discrete action is the frame subset; its reward is only known after actually querying the frozen executor. There is no provided "correct memory subset" label.

But a reviewer has a very easy attack: "The states are fixed offline, the target GUI action is already labeled, there are no policy-dependent environment transitions or delayed credit assignment, and the reward can be converted into labels by querying candidate subsets. This is supervised/ranking learning with an expensive black-box labeler, with policy gradient used as the optimizer." That criticism is basically fair. Especially with fixed B=2, your Level-1 MDP has horizon 1. Calling the whole thing "two-level RL" invites an argument you don't need.

I would write it as: Stage I: executor-feedback contextual-bandit pretraining. Stage II: closed-loop task-level RL. Then explicitly say Stage I is an RL formulation of a combinatorial retrieval problem, while Stage II is actual environment RL.

The experiment that settles the issue scientifically is a query-budget-matched comparison against supervised alternatives. Take the exact same executor calls and compare RLOO against: train-on-discovered-positive subsets, pairwise ranking/DPO-style training, and direct anchor supervision where available. If PG wins because it explores the combinatorial subset space efficiently, then the RL choice has substance. If all methods tie, don't make "RL" the contribution.

AndroTMem is particularly suitable for this because it has 1,069 long-horizon tasks / 34k+ steps and explicit causal dependencies/anchors, not merely long trajectories.

### 2) Risks, ranked

#1 is not offline-to-online shift. It is whether memory selection has enough causal headroom with your frozen 8B executor. GUI-Owl-1.5-8B is competent on shorter/general mobile tasks (model card ~69% AndroidWorld, 41.8% MobileWorld) but its reported MemGUI-Bench Pass@1 is only 11.7%. Your "24% of contexts are controllable" result is your most important preliminary result, but context controllability is not task-level headroom. You need an oracle curve before spending heavily on RL: recency/no-memory executor; random distant memory; controller memory; oracle memory (search/annotate the best available historical frame set for the frozen executor). Do this on a MemGUI-like held-out development suite, not by tuning on MemGUI-Bench. The crucial number is frozen-executor Pass@1 with oracle memory. If baseline is 11.7 and oracle-memory performance is, say, 14%, this project cannot produce a large benchmark win regardless of RL quality. If oracle memory reaches 25–35%, you have a very strong story. MemGUI-Agent reports an 8B system at 23.4% Pass@1, but that involves SFT of the 8B executor rather than keeping GUI-Owl frozen.

#2: all-step action agreement will probably train the wrong thing. AndroTMem has 34k steps, but "long trajectory with decisive states" does not imply every action requires distant memory. If you reward every step equally, the controller can obtain excellent action agreement while effectively ignoring memory. Train hardest where memory intervention actually changes the frozen executor's correctness: filter or strongly upweight contexts with E(x,M_recent)≠a* and ∃M: E(x,M)=a*. Even better, replace raw reward with a counterfactual memory advantage r_mem(M)=Match(E(x,M),a*)−Match(E(x,M_0),a*), where M_0 is recency/no-memory. If you can cheaply obtain GUI-Owl token likelihoods, an even denser version is the change in gold-action log likelihood; still retain actual sampled/action-match reward for evaluation. Use AndroTMem's tolerant AMS-style action equivalence, not exact string/coordinate identity.

#3: "gold action agreement" has a route-imitation problem. A recorded action is not necessarily the only valid action; the controller may learn which screenshots manipulate GUI-Owl into mimicking the dataset policy. Report strict recorded-action agreement and semantic/action-equivalence agreement; for taps, element-level equivalence over coordinates.

#4: offline-to-live distribution shift is serious, but Level 2 is the correct answer. Make the self-built tasks contain: off-nominal/recovery states; distractor historical screens; facts relevant after long gaps; cross-app transfer; obsolete old memory; cases where no historical frame is useful; multiple valid paths; randomized values/layout/content. Reproduce the observed lag distribution (88% decisive frame ≥5 steps back). Define controllability with an independent oracle intervention, not by whether the current controller succeeds. Mix in some non-memory-critical tasks or retain a KL/reference loss to the Level-1 policy.

#5: selector reward hacking needs explicit causal audits. Canonicalize the selected screenshots chronologically before feeding them to GUI-Owl (otherwise permutation becomes a control channel). Include causal tests: replace selected screenshots with matched irrelevant ones; shuffle temporal metadata; occlude the purported decisive region; compare selected-frame overlap with human/AndroTMem anchors.

### What I would add/drop

Add four diagnostics: oracle-memory upper bound; recovery on controllable contexts, Recovery=P(policy correct | recency wrong, oracle correct); harm on already-correct contexts; oracle retrieval gap. Make MemGUI-Bench's memory-oriented metrics (IRR/FRR/MTPR) first-class. Drop from the initial version: the phrase "two-level RL" as the main hook; equal weighting of all AndroTMem steps; variable-B from the core contribution until fixed-B works.

### 3) Variable B: worthwhile, but absolutely not first

Order: (1) fixed B=2 Stage I only — strong recovery on memory-controllable contexts; (2) fixed B=2 Stage II — survives closed-loop; (3) cheap fixed-budget curve B∈{0,1,2,4}; (4) a simple retrieve-or-not gate (adaptive B∈{0,2}); (5) only then sequential select→…→STOP pretrained with Stage-I feedback; (6) Stage-II fine-tune and sweep λ. Report a Pareto curve (Pass@1 vs mean historical frames / image tokens / latency), and prefer a constrained/Lagrangian interpretation over one universal λ.

### The paper I think you actually have

"Terminal RL fails for memory controllers because most GUI outcomes are invariant to memory intervention. We identify controllability as the key condition for learning memory policies, use executor-feedback contextual-bandit training at memory-sensitive states to acquire retrieval competence, and then perform closed-loop RL only where memory has causal task-level leverage." If your oracle-headroom experiment is strong, pursue this. If oracle memory barely moves the frozen GUI-Owl on matched live tasks, stop optimizing RLOO and change the executor/interface — the bottleneck is elsewhere.

## 3. 采纳 / 不采纳

- **采纳(改判)**:叙事上不以"两级 RL"为主钩子,改为"Stage I:executor 反馈的 contextual-bandit
  预训练;Stage II:闭环任务级 RL";Stage I 的 reward 改为**反事实记忆优势**
  Match(E(x,M))−Match(E(x,M_recent)) 并只在记忆敏感上下文上训;动作一致用 AMS 式容差;
  选中帧按时间序规范化后再喂 executor;加四项诊断(oracle 上界、可控上下文恢复率、
  已正确上下文的伤害率、oracle 检索差);MemGUI 的 IRR/FRR/MTPR 作一级指标;
  **先做 oracle 记忆上界实验再烧 RL**——这是最高优先级的下一步。
- **采纳(与我原判一致)**:可变 B 放最后,且先做 B∈{0,1,2,4} 预算曲线与"取/不取"门控;
  Stage II 保留,训练任务加入干扰帧/过时记忆/无用记忆/多解等负样本并复现帧龄分布。
- **坚持原判(附依据)**:仍保留 RLOO 作为 Stage I 的优化器,但按其建议加"同预算下 vs
  发现正样本监督 / pairwise ranking"的对照,让 RL 的价值由实验决定而非定义。
- **不采纳/待定**:换 executor(32B)只在 oracle 上界实验证明 8B 无余量时才考虑。
