# 外审:selector-RL 零结果的诊断与三条路线(2026-09-03 14:40 PT)

工具:网页 ChatGPT 临时会话,推理档"极高",思考 1m26s。审前我的判断见 §1,原文见 §2,
采纳/不采纳见 §3(结果落地后补)。

## 1. 审前判断(先写后审)

heldout 配对 r18:learned 21.1% vs recency 20.0%;r21 learned 15.8%;训练域配对持平。
诊断:(A)部署策略熵 3.7 nats(均匀 4.19),从初始到 pv22 不变——行为策略≈top-12
里随机抽对,已知随机对比 recency 差;(B)RLOO 组仅 29% 有混合结果;(C)服务只有
采样、预训练 +5.1pp 是 argmax 下测的。计划:暂停 RL,先做 argmax 侦察评测(RL 权重
与预训练初始各在 heldout-20 / train-20 上贪心 vs recency),再在 (a) 修探索温度继续
RL / (b) 升级特征 / (c) 放弃 RL 只报预训练 selector 之间选。

## 2. 外审原文(英文,逐字)

Yes—for the deployed sampled policy, the entropy finding largely explains the null. It does not explain why RL failed to move the policy. That distinction is important.

### 1) What I think is happening

You have three separate failures layered together:

A. Serving mismatch. Your only demonstrated selector gain was under argmax, but deployment/RL evaluation samples from a 66-way distribution with H=3.7. That distribution has perplexity e^3.7 ≈ 40: you are effectively spreading probability over dozens of pairs. If random-ish pairs are worse than recency, sampling can absolutely wash out a useful ranking.

B. Almost no RL signal. With only 29% mixed RLOO groups, 71% of groups contribute zero within-group reward contrast. For binary terminal reward, all-success or all-failure groups give RLOO essentially nothing useful. Out of 1,300 groups, only about 377 groups are gradient-informative at all. And even those are contaminated by executor/environment stochasticity and long-horizon credit assignment.

C. Representation/generalization problem. The strongest evidence here is not the RL curve; it is that offline pretraining gave +5 pp in-distribution and zero gain on new templates. That says your selector's features were already learning something template-specific rather than a robust notion of "which past frame is useful now." Fixing exploration alone doesn't address that.

So I would describe the current result as: RL failed to measurably alter the selector policy; high-entropy sampling additionally prevents the serving policy from exploiting whatever ranking signal the selector has; separately, the pretrained representation already showed weak template generalization. Those are much more defensible claims than "RL failed because entropy was too high."

Before touching another long RL run, I would check these:

- Did parameters actually move? Measure per-state KL(π_final ∥ π_init), argmax agreement, top-1/top-5 probability mass, logit-margin changes, gradient norms, PPO clip fraction, and parameter delta. Entropy alone can hide reshuffling.
- Audit PPO/RLOO bookkeeping. Old logprobs must correspond exactly to the rollout policy; check stale data across rounds, masking, accidental detach/frozen params, optimizer parameter inclusion, reward/advantage scaling, and any entropy bonus. "Entropy unchanged for 22 rounds" is enough that I would actively suspect an implementation/scaling problem until disproved.
- Look at mixed groups in more detail. Histogram 0…8 successes per RLOO group. If most informative groups are 1/8 or 7/8, signal is still extremely noisy.
- Measure selector causal sensitivity. On identical task states/seeds where possible, vary only the chosen frame pair. Does executor action accuracy or eventual success materially change? If pair choice barely affects behavior compared with executor noise, no policy-gradient trick fixes this.
- Temperature sweep both checkpoints. Evaluate pretrained init and RL final at argmax and T=.1,.25,.5,1, ideally paired. Success-versus-entropy is probably your most diagnostic plot right now.

One result would be especially revealing: if RL-final argmax beats pretrained-init argmax, then RL may actually have improved the ranking while your sampling path concealed it. If final argmax ≈ init argmax, RL really did essentially nothing.

### 2) Which option I would take

Given the evidence you have now, I would choose (b), upgrade the features, after finishing the argmax diagnostics you're already running.

I would not make (a) "fix temperature and continue RL" the main bet. It is a worthwhile controlled sanity experiment, but your offline result already says the representation doesn't transfer to unseen templates. Better exploration can produce a stronger optimizer of the wrong features.

The feature upgrade should specifically encode the relation between the current state and candidate history, rather than just making the frame encoder bigger: current↔past visual difference/change maps, patch-level matching, cross-attention or similarity, perhaps action/time metadata. The selector's actual question is relational: which previous observation contains information relevant to what changed / what I need now?

Then train/evaluate offline first. I would require the new selector to beat recency on unseen templates under argmax before paying for more RL. If it cannot generalize with dense behavioral labels, terminal-success RL is unlikely to magically teach it that representation.

I'd rank your choices: (b) > (a) > (c) for the research program.

But there is a conditional branch: if your current argmax evaluation shows RL-final clearly > pretrained-init, then serving mismatch becomes interesting enough that a carefully controlled version of (a)—temperature/entropy targeting, not just "more PPO"—moves ahead of a large representation rebuild.

Option (c) is only attractive if the pretrained argmax selector now gives a reproducible heldout improvement. Based on what you wrote, it currently does not: its +5 pp was in-distribution and unseen-template performance was flat. Reporting it as a generally better selector would therefore be vulnerable.

### How a reviewer will attack this

The strongest reviewer attack isn't actually the RL hyperparameters. It's evaluation methodology.

Your heldout-20 has been evaluated at seven RL rounds and is now guiding the decision between (a)/(b)/(c). It is no longer a pristine test set; operationally it is a validation set. The 32% peak is particularly vulnerable to "best checkpoint over multiple looks" / winner's curse. For a final paper claim, I would freeze all decisions and evaluate once on a new untouched collection of unseen templates.

Other obvious attacks: 20 templates are too few for a template-generalization claim; repeated rollouts may be pseudoreplication; recency is a weak sole baseline; RL shows no paired training-domain gain; the executor's stochasticity may dominate selector effects; and sampled learned-policy versus deterministic recency may be an unfair inference comparison.

I'd include random-pair, recency, pretrained-init, final-RL, and at least one simple nonlearned relevance heuristic such as image/patch similarity or frame-difference selection. An oracle analysis—"best pair among the 66 in hindsight"—would also tell reviewers whether there is actually enough headroom for selection to matter.

### 3) Sample size

For a claim about unseen task templates, your independent unit is primarily the template/task, not each of the 19 repeated rollouts. Repeats improve your estimate of each task's success probability; they don't give you 19× as much evidence about generalization to new templates. Twenty independent unseen tasks is very weak for distinguishing effects of the size you're discussing.

As a rough power reference, ignoring clustering, for baseline success around 20% at two-sided α=.05 and 80% power: 5 pp (20→25%): ~1,090 unpaired observations per arm; 8 pp: ~445; 10 pp: ~292; 12 pp: ~207. Pairing helps substantially: for a 5 pp effect roughly 300–900 independent paired units depending on disagreement rate; 10 pp roughly 70–230; 12 pp roughly 50–160.

Because yours are clustered repetitions within templates, estimate the intra-template correlation from your current 19 repeats and power a paired cluster-level design. With 19 repetitions per task, even an ICC of 0.2 produces a design effect of 1+18(0.2)=4.6, so "380 rollout results" may behave more like ~83 independent observations. At ICC 0.5, closer to ~38.

For a paper where the intended claim is "generalizes to unseen templates," my practical threshold would be at least ~80–100 genuinely new templates if you expect a large ~10 pp effect. For a credible 5 pp claim, think hundreds of new templates, not 20 tasks with more seeds. And make the final test set completely fresh: the existing heldout-20 is dev-unseen, not test-unseen.

The key decision rule from your upcoming argmax experiment is simple: final argmax > init argmax ⇒ investigate train/serve temperature mismatch; final argmax ≈ init ⇒ stop spending RL compute, fix signal/representation first.

## 3. 采纳/不采纳(待贪心侦察结果落地后补)
