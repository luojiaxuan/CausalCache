# 外审:这构不构成顶会论文;缺什么(2026-09-06 22:20 PT,GPT-6 Pro,用户点名顶配档)

- 审阅方:网页 ChatGPT 临时会话,GPT-6 **Pro**(用户要求);"思考了 16m 7s";读了钉住 commit `9a9a562` 的台账、故事、设计稿,并检索了 MementoGUI / ATMem / HiMPO / MemGUI-Bench / AndroTMem / STAMP 与 NeurIPS/ICLR/ACL 的 CFP。
- 提示词(单行英文,约 1,490 字符)见文末。

## 审前判断

同意用户:现有的是一组精细的因果诊断——(1) 视觉回看只在文本历史缺证据时有价值(Mail +46.7、零件图 +80 的交互项);(2) 该保留的是请求轮 + 证据轮而非最近两轮;(3) 冲突时形状听图、数字听文本。但没有方法/系统、没有第三方 benchmark 收益、结论有"意料之中"的味道、触发条件未操作化。我倾向回到 RL 记忆控制器方向(用本线因果信号做奖励),代价是周级工作。

## 外审全文(逐字落盘)

### 1. Verdict and missing contribution class

Agree on the current manuscript; disagree that a learned controller is the missing checkbox. The missing contribution is a reusable methodology with demonstrated external validity: either an intervention-valid evaluation framework that changes conclusions about memory systems, or a deployable method that improves their success–cost frontier.

Your current estimand is the effect of supplied context at a checkpoint, not the value of an autonomous memory policy or the cause of a training-induced capability. The +46.7/+80 pp figures are text×image interactions, not unconditional task-success gains.

Neither "add RL" nor "scale the benchmark" establishes novelty against current work:

| Proposed contribution | Already occupied territory | Remaining differentiation to establish |
|---|---|---|
| Learned multimodal controller around a frozen GUI executor | MementoGUI learns selection, compression, and retrieval without finetuning the executor. | Text-conditioned residual utility, complementary evidence-set selection, and externally validated total-cost improvements. |
| Counterfactual memory rewards plus cost penalties | ATMem/STR-GRPO contrasts memory-on/off rollouts with memory-cost-aware rewards; HiMPO estimates local counterfactual memory-write utility. | A materially different estimator or optimization problem; "our reward is counterfactual" is insufficient. |
| Constructed memory-intensive GUI benchmark | MemGUI-Bench, AndroTMem, and STAMP already provide memory-focused tasks, diagnostics, or controllable memory-variable generation. | Separate evidence contribution from text redundancy, prompt-format effects, and oracle-selection privilege; demonstrate that this separation changes a practical evaluation conclusion. |

Analysis is eligible: NeurIPS explicitly welcomes in-depth analysis yielding new insights, and ICLR includes datasets and benchmarks. The blocker is significance and generalization, not absence of learning.

### 2. Smallest credible additions, ranked by ex-ante acceptance odds

Ranking is my judgment from the current assets; sample sizes and GPU budgets below are proposed experimental packages, not venue requirements.

| Rank | Route | Minimum credible addition | Samples, experiments, benchmarks | Estimated H100-80GB GPU-days |
|---|---|---|---|---|
| 1 | Benchmark + intervention-valid diagnostic methodology | Reusable protocol separating evidence, textual coverage, and serialization; demonstrate one diagnostic-driven policy choice transferring beyond constructed checkpoints. | 500 fresh parent scenarios: 100 development + 400 untouched test, each with two twins; ≥20 held-out workflow templates, ≥6 apps; 4 frozen actors across ≥2 backbone families. External: 200 AndroTMem traces for diagnostics; all 128 MemGUI tasks for closed-loop validation. | 12–30 |
| 2 | Learned residual-utility budgeted controller | Beat strong online text-memory/retrieval baselines, not merely recency; transfer without retraining across executors and budgets. | 5k–10k training prefixes, 8–16 counterfactual context sets/prefix; 3 training seeds; 400 fresh test parents, 2 executors. External: full MemGUI plus the unmodified MobileWorld GUI-only suite; keep previously inspected MobileWorld tasks a regression test, not an unseen-generalization claim. | 30–70; optional online RL +15–30 |
| 3 | Analysis-only | A predictive principle or evaluation correction that survives matched controls and predicts held-out behavior; repeating "deleted information helps when restored" does not qualify. | 400–800 parent tasks, ≥2 corpora, 4 actors; crossed text/image/format interventions; held-out predictions. Use AndroTMem and MemGUI traces; add executable continuations where supported. | 6–12 |

Venue-fit judgment: route 1 favors ACL main / ICLR main; route 2 has the strongest conditional case for NeurIPS main / ICLR main once external gains exist; route 3 has the lowest ex-ante odds. ACL explicitly includes analysis, evaluation/resources, and multimodal grounding; NeurIPS's separate Evaluations & Datasets track does not satisfy your main-track constraint.

Controller specification, should route 2 be pursued: train a query-conditioned 1–3B selector/value model for u(S|T,h) = E[Y(S,T) − Y(∅,T) | h] − λ[C(S,T) − C(∅,T)], where T is existing textual memory, S is an evidence set, Y is environment-verified decision/short-continuation success, and C includes retrieval and execution costs. Include empty-set wins, harmful retrievals, and complementary multi-source sets; gold source indices are training supervision, never inference inputs.

Train SFT/value ranking first; use a contextual bandit only after supervised generalization succeeds. Required ablations: text conditioning removed; independent top-k versus set selection; evidence-location supervision versus counterfactual utility; cost penalty removed; SFT versus bandit. A practical target is ≥10 pp same-cost success improvement or ≥25% measured total-cost reduction, with an appropriate confidence interval and predeclared noninferiority margin for the latter—not a recency-only win.

Compute accounting: assume effective batched throughput of 0.3–0.5 executor decisions/s/GPU, then replace that assumption with a representative 1,000-call timing run. Budget approximately 0.2–0.5M / 0.6–1.0M / 0.1–0.2M executor-equivalent decisions for routes 1/2/3, respectively: D ≈ 1.5 Σ_m N_m / (86,400 r_m) + D_training. The factor 1.5 provisions reruns; route 2 includes approximately 4–8 training GPU-days. These are summed accelerator-days, excluding emulator CPUs, annotation, and API judges; price larger actors separately.

### 3. Choose benchmark + diagnostics; make the contribution measurement, not task count

Choose route 1. The documented strong text controls already eliminate substantial image-retrieval headroom; a controller currently requires an additional positive result against those controls before its methodological case begins.

**3.1 Run the cheapest discriminating experiment first.** Use 80 fresh development parents, two actors, known-at-start versus genuinely late-bound queries, and competent online structured notes plus caption/OCR retrieval; sweep 2k/4k/8k/16k historical-context tokens. Allocate 0.5–2 GPU-days within the route-1 budget. If text memory dominates throughout realistic budgets, drop the image-controller thesis, not the text baseline. A text-dominant frontier remains a valid diagnostic result; manufacturing image necessity by deleting competent notes does not establish deployable value.

**3.2 Define and separate the estimands.** For verified local outcome Y, measure Δ_image(t) = E[Y(source,t) − Y(matched control,t)], I = Δ_image(fact absent) − Δ_image(fact retained).

| Contrast | Required control | Permitted interpretation |
|---|---|---|
| Historical-image content | Identical native serialization, roles, positions, replies, image preprocessing, and matched token counts; change image content only. | Conditional effect of supplied visual evidence. |
| Operational retained-turn policy | Preserve each selected turn exactly as deployment would; additionally run a request-preserved comparison. | Effect of selecting whole turns, including their text. |
| Textual evidence coverage | Replace target facts with matched irrelevant content; audit actual retained facts rather than summary length. | Text–image substitution/complementarity. |
| Consistent counterfactual twins | Change the underlying task fact and regenerate dependent observations/text consistently. | Whether both versions produce their respective correct outcomes. |
| Image–text conflict | Deliberately contradict the channels; vary reliability and task operation independently. | Arbitration preference—not an optimal modality law. |

The current pilot explicitly bundles screenshots with original replies in whole-turn treatments; that is a valid operational treatment, but not a screenshot-only estimand.

**3.3 Make memory pressure natural and splits genuinely fresh.** Generate the late query after browsing and memory commitment; a query file present at initialization but requested later is only procedurally hidden. The proposed delayed-reveal implementation currently initializes such a file early. Allow notes and backtracking; charge both. Vary record count, possible queried attributes, evidence age, intervening information, and version changes independently. Delay alone does not defeat a competent summary. Use the existing 48 cases only for development. Split fresh data by workflow template and parent scenario, not screenshots, substitutions, or decoding seeds. Cover delayed lookup, multi-source joins, visual binding, and version/state updates; randomize values and layouts without treating these variants as new workflow families. Release generators, snapshots, native adapters, complete valid evidence sets, verifiers, exclusions, and failure traces. Audit at least 100 parent pairs manually for solvability, twin consistency, and unintended answer exposure.

**3.4 Compare against the baselines capable of falsifying the thesis.** Run text-only/zero historical images, matched-cost recency and random selection, online structured notes with text/OCR retrieval, summary-plus-ROI/image retrieval, and a strong learned memory baseline where runnable. Include generator-defined evidence oracles only as diagnostic upper bounds, not deployed policies. Keep the original request available in a dedicated controlled comparison; otherwise "request+evidence beats recent-2" partly measures whether the task specification was forgotten. Distinguish a read-context budget from a persistent-memory budget. An archive may remain available under the former, with retrieval charged; evicted information cannot remain secretly accessible under the latter. Count selector archive scans, summaries/captions, image encoding, executor input/output, cache behavior, rereading actions, and p50/p95 latency. Report native token counts and measured total cost separately.

**3.5 Require external consequences, not necessarily leaderboard superiority.** On 100 held-out parent pairs, run natural closed-loop episodes with 2 actors × 3 deployable policies × 3 runs. Report checkpoint reach rate, conditional checkpoint accuracy, and complete-task success separately; teacher-forced continuations never count as autonomous completion. For MemGUI, run 128 tasks × 2 actors × 3 policies × 3 runs = 2,304 episodes, using the official step-budget and judging protocol. Its 128 tasks contain 64 mirror pairs; repetitions do not create 384 independent task families. Use 200 prespecified AndroTMem task traces to test diagnostic transfer; label replay/action evaluations as offline evidence, not live benchmark gains. AndroTMem provides 1,069 tasks and dependency-oriented annotations suitable for this check. Lock one diagnostic-derived policy choice on development data and test whether its predicted success–cost advantage transfers. A larger suite with no external consequence remains vulnerable to "another synthetic benchmark."

**3.6 Power the claims, not the number of decode calls.** Cluster inference by workflow and parent; keep twins and repeated runs within their clusters. Define twin success as both answers correct, not merely an action change or correct swapped-answer marginal. For a paired binary difference δ, approximate 80%-power planning at two-sided 5% significance gives n ≈ 7.84 q / δ², where q is the policy-discordance probability. At q = 0.2–0.4, detecting 10 pp requires approximately 160–315 independent parents; 5 pp requires 630–1,260, before template-cluster inflation. Power the factorial interaction separately using development-set cluster variance. Treat 400 test parents as a starting budget, not assurance of power for every family. Predeclare a noninferiority margin, such as −5 pp, before claiming unchanged accuracy; a nonsignificant difference is not equivalence. Use gold-information controls to identify perception/action bottlenecks, but report all attempted tasks and a prespecified solver-capable stratum separately; do not erase inconvenient model failures through post-hoc validity filtering.

### 4. Cuts

| Cut from the main claim or work plan | Retain instead |
|---|---|
| Joint executor–memory RL now. | Frozen executors; supervised selector only as a bounded baseline unless residual headroom survives strong text controls. |
| The story headline that training-history layout determines retrieval benefit. Native-format reevaluation did not distinguish the SFT arms; earlier conclusions changed under serialization repair. | Protocol sensitivity as motivation for the diagnostic methodology. |
| Universal "recency is harmful," "images help only without text," and "numbers=text/shapes=image" laws. | Conditional effects; cross content type with the same operation and source reliability before claiming modality-specific behavior. |
| Oracle maxima, best-of-selector scores, and self-action reproduction as deployment achievements. | Generator-defined oracle bounds, verified local outcomes, and separately reported closed-loop results. |
| "46% efficiency gain at equal accuracy." | Observed prompt-token reduction; add full-pipeline cost and noninferiority evidence before making an efficiency claim. |
| More seeds on inspected templates; additional platforms before protocol validation; the failed-experiment chronology in the main narrative. | Fresh workflow families, a locked external evaluation, and an audit appendix. |
| Treating recent-2 as the universal official baseline. The ledger corrects that assumption for GUI-Owl. | Each executor's native default plus separately named project baselines. |

Do not cut: zero-old-image text memory, competent online notes, matched irrelevant evidence, native-format controls, complete failure accounting, or external task verification.

### 5. One-sentence defensible claim

For the tested frozen GUI agents, controlled MobileWorld checkpoints show that historical-image utility depends on which task facts survive in text, and oracle request-plus-evidence retention can outperform comparable-token recency.

## 我的采纳与分歧(2026-09-06 22:25 PT)

| 外审意见 | 判定 | 说明 |
|---|---|---|
| 现稿不是顶会论文;缺的是"有外部效度的可复用方法论",不是"学个控制器" | **采纳** | 与审前判断一致;它把我倾向的"回到 RL 控制器"否了:MementoGUI / ATMem / HiMPO 已占住那块地,"奖励是反事实的"不够新。 |
| 选路线 1(benchmark + 干预有效的诊断方法论),且**先跑 §3.1 的判别实验**:80 个新场景、两个 actor、已知/真正延后的请求、称职的在线笔记 + OCR 检索基线、历史预算 2k/4k/8k/16k 扫描;文本记忆若在现实预算内全程占优,就放弃"图像控制器"论点 | **采纳,作为下一步的唯一闸门** | 这是对本线的生死判:我们所有正向数字都是"把称职的笔记删掉之后"得到的,外审明说这不构成部署价值。先做这一步再谈路线。 |
| 延后揭示的 todo.txt 方案"只是程序上隐藏"(文件在初始化时就存在) | **采纳,设计稿 §7 作废** | 真正的延后 = 浏览与记忆写入之后才生成请求(需要 ask_user 或运行时注入)。 |
| 砍掉:联合 RL、"训练布局决定收益"的故事标题、各种"普适定律"措辞、oracle 上界当成果、"46% 效率且精度不变"(需全流程成本 + 非劣性)、recent-2 当官方基线 | **采纳** | 故事与 artifact 的措辞下次改稿时统一改为条件效应。 |
| 样本量:检出 10pp 需 160–315 个独立场景;500 个新场景、≥20 个留出模板、≥6 个 app、4 个 actor;MemGUI 2,304 个闭环 episode;AndroTMem 200 条轨迹 | **采纳规模估计,资源待用户批** | 12–30 GPU-天 + MemGUI 判分预算;这已是多周项目,不是本 session 能自行决定的。 |
| 一句话主张 | **采纳** | "在所测冻结 GUI agent 上,受控 checkpoint 显示历史图的价值取决于哪些任务事实留在文本里;oracle 的请求+证据保留可胜过同 token 的 recency。" |

分歧:无实质分歧。唯一保留:路线 2(残差效用控制器)在 §3.1 判别实验通过后仍是 NeurIPS 主赛道的更强路径,外审也这么说;顺序上先 §3.1。

## 提示词原文

```
Repo pinned at https://github.com/luojiaxuan/CausalCache/blob/9a9a562/ ; read: rl/docs/ledger_memory_rl_20260904.md, rl/docs/story_20260905.md, rl/docs/b_pilot_design_20260906.md. STATE: frozen GUI agents (UI-Venus-2-9B, GUI-Owl-1.5-8B) on 48 constructed MobileWorld memory tasks with twins; interventions show (a) old screenshots help only when task evidence is absent from text history (+46.7pp Mail, +80pp parts), (b) the right kept turns are request+evidence, not recent-2 (43% vs 0% at equal tokens, GUI-Owl), (c) numbers follow text, shapes follow image when they conflict; dropping recent-2 under verbose text saves 46% tokens at equal acc. No learned controller, no third-party benchmark gains, n=15-30 checkpoints. PI: not a top-venue paper. ASK: (1) agree? name the missing contribution class; (2) smallest additions for a NeurIPS/ICLR/ACL main-track paper, ranked by acceptance odds, with experiments, benchmarks, sample sizes, GPU-days; (3) choose: learned budgeted memory controller (RL/SFT on our counterfactual signal) vs benchmark+diagnostic suite vs analysis-only; (4) what to cut; (5) the one-sentence claim you would defend. FORMAT: (agent 格式段,同前)
```
