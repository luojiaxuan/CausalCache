# 全量发射前外审(ChatGPT Pro,临时会话,2026-08-26,思考 10m36s)

## 发审前我方预判(先行锁定,防背书)

(1) selector lr 直接 1e-4→1e-3 一步到位,不加中间档;(2) 4 卡分 2 train
TP2 + 2 rollout;(3) 批 8 题 × G8 = 64 rollout;(4) 难度先验 warm-start +
25% floor;(5) recent 对照每 8 批一批只测不训;(6) 判停线:60 批时混合组
率 ≥35% 但 learned-vs-recent 差距≈0 且 selector 行为位移≈0 → 停判负结果;
(7) executor lr 1e-6。

## 外审原文(全文,机器翻译未做,保留英文)

> 思考了 10m 36s

**Bottom line** — The systems plan is sensible: the 2-train/2-rollout
split, larger rollout batch, uniform sampling floor, and unchanged
executor learning rate all look defensible. The experimental design is
not yet decisive. As written, both a positive result and a null result
will be easy for a reviewer to explain away. The main danger is spending
several days and ending with an uninterpretable outcome.

### (a) The three weakest points

**1. Your frozen-executor probes motivate adaptive history length, not a
learned fixed-B=2 selector.** The claim that "sparse learned selection
wins both ends" is not supported yet. On short tasks, your own result
says 11 frames beat 2 frames by 17 percentage points. That is currently
evidence against hard-coding B=2. A learned selector might find two
sufficient keyframes, but you have not shown that such a two-frame
subset exists. On long tasks, B=2 beating B=10 establishes that extra
history can hurt under context pressure or attention dilution. It does
not establish that learned selection is better than recency, uniform
thinning, change-point keyframes, or simply using all frames until the
token budget becomes tight. A reviewer will ask for: an oracle or
hindsight-best B=2 result on short tasks; "all history while it fits,
then thin" as a baseline; adaptive B rather than fixed B=2; recency-2,
uniformly spaced-2, random-2, and simple visual-change/keyframe-2
baselines; equal-token-budget comparisons, not merely equal frame
counts. The strongest hypothesis supported by your probes is: history
should be budget-adaptive. The learned-selector hypothesis remains
unproven.

**2. The frozen-recency measurement batch is not a valid control for
joint training.** Let E_t be the executor after training with learned
selector S_t. Evaluating E_t with recency is an asymmetric swap test:
E_t(S_t) vs E_t(recency). A positive gap can arise because the executor
co-adapted to S_t, not because S_t provides better information. A null
can arise because the selector changed faster than the executor could
adapt. Either interpretation is available to a skeptical reviewer. A
measure-only recency batch every eight batches does not solve this. The
real causal control is an equal-compute arm: E_t^learned(S_t) vs
E_t^recency(recency), both from the same checkpoint with matched task
schedules. A full parallel run is expensive, but at minimum use
checkpoint forks for shorter matched horizons. For diagnosis, run a
crossed evaluation at fixed checkpoints: E_0 with {S_0, S_t, recency};
E_t with {S_0, S_t, recency}; same held-out task instances and
environment seeds. This separates selector improvement, executor
adaptation, and selector–executor interaction.

**3. Mixed outcomes are not selector learning signal, and the proposed
stop comparison is probably underpowered.** A 41% mixed-outcome-group
rate establishes only Var(R|task)>0. Selector learning requires
Cov(R, ∇logπ_selector)≠0. The mixed outcomes may be almost entirely
caused by executor action sampling or environment stochasticity. If the
G=8 rollouts choose similar frames, the selector can receive essentially
no consistent direction despite substantial reward variance. With binary
terminal reward, every selector decision across as many as 50 steps
receives the same trajectory-level advantage. RLOO removes a task-level
baseline; it does not solve temporal credit assignment or separate
executor noise from selector effects. The 18 of 58 zero-signal templates
make this worse. You need to measure: within-group diversity of selected
frame sets; reward difference conditioned on selector differences;
gradient agreement across independent halves of a batch; selector
gradient signal-to-noise, not merely reward variance; per-template
effects with hierarchical uncertainty. The stopping evaluation is also
thin: by batch 60, a control every eight batches gives about seven
control batches ≈ 56 task groups. Even under optimistic independence
assumptions the uncertainty on a binary learned-vs-control difference is
several percentage points; with task clustering, wider. "Gap ~0" is not
a statistical criterion. Predeclare a minimum worthwhile effect (+2 or
+3 pp absolute) and stop based on a confidence bound around that
threshold. Evaluation should remain fixed and uniform even while
training uses difficulty-priority sampling.

### (b) 10x jump to 1e-3 or 3x to 3e-4?

Use **3e-4 first, with a precommitted escalation to 1e-3**. I would not
blind-jump directly to 1e-3. The 3.2% weight drift says the optimizer
was not doing nothing. Zero movement in mean frame age can instead mean:
the metric is too coarse; parameters moved in directions that barely
affect PL logits; logit rank margins are large; updates cancel across
noisy groups; the selector changed conditionally while aggregate age
remained constant; the rollout task distribution moved at the same time
as the policy. A 10x jump risks pushing most PL ratios into the clipped
region, collapsing entropy, or changing the selector distribution faster
than the executor can adapt — making a negative result even less
interpretable. Use a fixed bank of observation histories and monitor
actual policy movement: KL to selector initialization and to the
previous checkpoint; change in per-frame inclusion probabilities; top-2
set overlap (Jaccard); distribution of top-2 logit margins; PL ratio
clip fraction; entropy and logit scale; gradient agreement across batch
splits. Do not rely on mean frame age alone. KL to a deterministic
recency policy is awkward or infinite unless smoothed — probability mass
assigned to the recency pair is cleaner; KL to S_0 is the more useful
learning diagnostic. Escalate from 3e-4 to 1e-3 when all three are true:
fixed-state policy movement remains negligible; PL clip fraction is low
and entropy healthy; selector gradients are nonzero and reasonably
consistent. If 3e-4 produces clear selector movement but no reward
effect, the learning rate is not the bottleneck. If clipping is already
high, increasing lr is actively the wrong fix; inspect logit
temperature, advantage scaling, update epochs, entropy pressure instead.

### (c) Is the batch-60 stop rule premature?

**Yes, as an end-to-end negative-result criterion. Batch 60 should be a
diagnosis checkpoint, not an automatic kill point.** There is a real
bootstrap problem: the pretrained executor may not know how to use
temporally non-contiguous screenshots → selector choices initially have
little causal effect on success → the selector receives little useful
reward signal → it does not discover better histories. Simply increasing
selector lr does not solve that loop. Run an explicit
executor-adaptation phase first: keep the selector frozen or strongly
anchored near recency while training the executor on a mixture of
recency and randomly/heuristically sampled non-contiguous histories;
include original frame indices or ages if the executor does not already
receive them; end this phase based on a fixed non-contiguous-history
probe, not merely elapsed batches. Then begin counting selector-active
batches. At checkpoints the useful comparisons are: E_t(S_0)−E_0(S_0)
(executor adaptation); E_t(S_t)−E_t(S_0) (selector learning beyond
adaptation); E_0(S_t)−E_0(S_0) (selector value to unchanged executor);
E_t(S_t)−E_t(recency) (deployed swap advantage, co-adaptation
confounded). For the final scientific claim you still need the matched
training-arm comparison.

**A better stop rule** — after the explicit adaptation phase, stop only
after two consecutive fixed, uniform, paired evaluations show one of:
(1) selector moved materially on fixed states but the upper confidence
bound on learned-vs-matched-recency is below the predeclared minimum
worthwhile gain, with no growing executor–selector interaction; (2) the
selector cannot be made to move even after the 3e-4→1e-3 escalation,
despite nonzero coherent gradients and without pathological clipping or
entropy collapse; (3) the selector moves but its changes have no
reproducible association with reward across tasks or checkpoint forks.
Remove the mixed-group-rate threshold from the stopping rule — it is a
weak necessary condition, not evidence the selector is learnable. One
nuance: if selector behavior is genuinely unchanged on a fixed-state
probe after the lr escalation, waiting for executor adaptation does not
rescue the claim that the selector is learning; the executor may still
benefit from adapting to the initial non-contiguous distribution —
report that separately as history-augmentation or co-adaptation, not
selector learning.

### My launch decision (reviewer's)

Keep the 2+2 GPU allocation, 8 tasks by G=8, the 25% uniform training
floor, and executor lr 1e-6. Change four things before committing the
full run: (1) add adaptive-B and simple keyframe baselines — your
current probe most strongly supports those; (2) start the selector at
3e-4 with a behavior-gated escalation to 1e-3; (3) replace the sparse
swap-only control with crossed evaluations and at least
checkpoint-forked matched recency training; (4) treat batch 60 as an
adaptation/optimization diagnosis and start the actual selector-learning
stop clock only after the executor has demonstrated competence on
non-contiguous histories. Without those changes, a null will not
establish that learned selection fails, and a positive learned-vs-recency
swap gap will not establish that the selector learned better frame
selection.

## 我方裁决(分歧处理,依实测数据)

**它同意的**:2+2 卡分工、64 rollout/批、难度采样+25% floor、executor
lr 1e-6。不展开。

**它反对、我改判的(4 条,全部采纳)**:

1. **selector lr:3e-4 起步 + 行为门控升档 1e-3**(弃我方一步到位)。
   它指出的"clip 占比高时加 lr 是反向修复"是硬逻辑;门控升档的成本
   接近零(我全程盯批,门槛三条:固定态位移≈0 + clip 低 + 熵健康,
   连续 10 个 selector 批满足即升)。
2. **固定态探针库替代 mean_frame_age 单指标**。mini 已实证该指标太粗
   (权重漂 3.2% 它纹丝不动)。实现:从 mini decisions 缓存 ~200 个
   状态特征作冻结探针库,每 checkpoint 算 KL(S_t‖S_0)、top-2 Jaccard、
   margin 分布、clip 占比、熵;写进 selector_metrics。
3. **判停:预登记 MDE=+3pp,批 60 改为诊断点(非自动 kill),去掉
   混合组率条件,加两连续配对评测判据**。它的功效计算(7 个对照批
   ≈56 组,CI 数 pp 级)正确,我方"差距≈0"确实不是统计判据。
4. **checkpoint 交叉评测矩阵扩为 {E_0,E_t}×{S_0,S_t,recency}**(在
   我方原 cross-play 四格上加 recency 列),同 heldout 实例同环境种子。

**它反对、我坚持的(3 条,附依据)**:

1. **"发射前先补 adaptive-B / keyframe-2 / uniform-2 基线"→不阻塞
   发射**。这些全是冻结 executor 的评测臂,跑在池上(CPU 容器+单卡
   vLLM),与 4 卡训练零冲突,可并行补;random-2 已有实测(-5.2pp
   vs recent-B2,即 OOD 惩罚证据)。作为论文基线必补,但不是发射
   前置。oracle-B2 枚举在 MW 不可承受(C(t,2)^T 组合),用 best-of-N
   random-S 重采样做 hindsight 代理,预算封顶后如实报"代理下界"。
2. **"全程 matched-compute recency 训练臂"→採其让步版**:选 2-3 个
   checkpoint 做 10 批的 recency-训练 fork(单 fork ≈5-6h,可在主跑
   间隙/收尾后补),不做全程并行(4 卡没有双倍预算)。
3. **"硬两阶段:先 executor 适应期再放开 selector"→软化执行**:我方
   recency 偏置冷启动本就是连续版适应期(开局分布≈recent-B2,漂移
   渐进);硬冻结阶段费墙钟且其"结束判据"同样要探针。折中:非连续
   历史能力探针(固定 random-S 批)加入每 checkpoint 监控,若诊断点
   出现"selector 有位移但零奖励关联 + executor 非连续能力零增长",
   再启用预注册的 random-S 重渲染 SFT contingency(recipe §2 已有)。

**采纳落地清单**:selector lr 3e-4 + 门控升档纪律;探针库指标进
trainer;MDE=+3pp 与新判停三条进 recipe §6;交叉评测矩阵加列;
launch 配置按此更新。
