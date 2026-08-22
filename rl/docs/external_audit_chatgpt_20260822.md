# 外部审计记录:ChatGPT 深度审阅(2026-08-22)

背景:用户对 32dff2c 定下的新路线(AndroidWorld 训练 + AndroTMem/MementoGUI
评测)自提三问:①mobile-only 是否满足 CVPR;②闭环教义 vs 两个第三方记忆
benchmark 都是离线的;③我们论证的是"老帧有时比新帧有用"而非"更好的记忆方法",
正面对比已发表记忆方法可能打不过,要不要比。经 chatgpt-review skill 提交
外审(临时聊天,gist + GitHub connector 读私有仓库,模型思考 20m46s、
检索 60+ 来源)。本文档为其回答全文存档;结论采纳情况见 roadmap 同日
★★★ 条目。**外审观点不自动成为我们的结论**——采纳/驳回均在 roadmap 里
逐条说明理由。

## 审计请求

见 gist(公开,用户批准):
https://gist.github.com/luojiaxuan/9c3f9bf8ac81369ae91a2d41c224f533
(内容与 scratchpad/cvpr_audit_request.md 相同:系统定义、全部实测数字、
训推 mismatch 问题陈述、三问、我方立场、"be blunt" 指令。)

注:ChatGPT 的 GitHub reader 抓不了 gist,但它经用户 ChatGPT 里挂的
GitHub connector 直接读到了 CausalCache 私有仓库的 roadmap 与 08-22
benchmark 转向记录(含同一批 bridge/OSWorld 数字),并声明其回答基于
仓库正本 + 独立文献核查,小节标题为重构而非照抄 gist。

## ChatGPT 回答全文(原文照录)

> I could not retrieve the Gist itself: both the connected GitHub reader and
> the public fetch path fail on Gist URLs, so I cannot claim verbatim
> Section-6 fidelity. I did recover the active CausalCache plan-of-record and
> the August 22 benchmark pivot from the connected repository—including the
> same bridge/OSWorld measurements—and independently re-audited the 2026
> GUI-memory literature. I'm therefore answering the four substantive
> decisions; the section labels below are reconstructed rather than copied
> from the inaccessible Gist.

### 1. What do the measurements actually establish?

At the moment, they do not establish the intended headline claim.

The most damaging pair of numbers is not the synthetic result; it is the
external-validity contrast:

- on the self-built 60-task bridge, learned selector = 8.3% versus recent-2
  = 1.7%;
- on the OSWorld-135 development evaluation, the selector is −0.74 pp,
  essentially neutral;
- the bridge oracle-frame condition reaches 35%.

That means the selector is responding to the memory structure you
deliberately manufactured, but you have not shown that the learned behavior
matters on a natural third-party distribution. Calling 8.3% versus 1.7%
"5×" is mathematically true and rhetorically bad: it hides that absolute
success remains 8.3% and that roughly three quarters of the oracle headroom
is still unrealized.

The bridge is still useful. It demonstrates two things cleanly:
recent-window memory can catastrophically fail when evidence becomes
non-revisitable, and the executor can exploit the right old frame. That
makes it a mechanism/stress-test appendix, not a convincing primary
benchmark.

There is also a serious flaw in the latest pivot. The roadmap proposes
training with live AndroidWorld terminal reward, then treating AndroTMem
and MementoGUI as the main third-party memory evaluations. That changes the
estimand halfway through the paper. You train for closed-loop task success,
then headline offline action prediction / VLM judgment. A reviewer can
reasonably ask why the claimed closed-loop benefit disappears precisely
when the agent is allowed to act.

So my verdict on Item 1 is: the mechanism is credible; the general method
claim is not yet supported.

### 2. Which genuinely CLOSED-LOOP memory-oriented GUI benchmarks/environments did you miss?

This is where the literature search was materially incomplete.

| Candidate | Interactive/live? | Memory-oriented? | Programmatic end-state? | My classification |
|---|---|---|---|---|
| LearnGUI-Online / LearnAct | Yes | Yes — demonstration/experience memory | Yes, inherits AndroidWorld evaluation | QUALIFIES |
| MemGym computer-use → WebArena-Infinity | Yes | Yes — memory-isolated evaluation | Yes, WAI has programmatic verifiers | QUALIFIES |
| DataScope / ATMem | Yes | Very much yes | Apparently state/progress based, but I could not independently verify released verifier code | PROVISIONAL / must investigate |
| MobileWorld | Yes | No, generic environment; heavily used by memory work | Yes: DB, local-storage, textual and callback verification | QUALIFYING ENVIRONMENT, not memory benchmark |
| AndroidWorld | Yes | No, generic environment | Yes | Already known |
| MemGUI-Bench | Yes in the current runtime | Yes | No under your definition: MemGUI-Eval uses progressive semantic/VLM judgment | DO NOT COUNT |
| AndroTMem-Bench | No | Yes | No | DO NOT COUNT |
| MementoGUI-Bench | No | Yes | No | DO NOT COUNT |

**The biggest omission: LearnGUI-Online.** LearnGUI explicitly contains 101
online tasks for demonstration-based learning. More importantly, its online
evaluation is built directly on AndroidWorld with the same emulator,
metrics, and evaluation protocol; the only added capability is retrieval of
demonstration trajectories. AndroidWorld itself has dedicated
initialization, success-checking, and teardown logic that inspects device
system state. That makes it closed-loop by your definition. The caveat is
conceptual, not technical: LearnGUI studies memory of prior
demonstrations/experience, whereas CausalCache is principally about
within-episode visual working memory. So it is not an exact task match. But
saying "there is no closed-loop memory-oriented GUI benchmark" would still
be false.

**The second real omission: MemGym + WebArena-Infinity.** MemGym has a
dedicated computer-use memory track on WebArena-Infinity: 140 tasks, live
web interaction, and a same-reasoner no-memory comparison. Their reported
structured-memory score is 38.6% versus 34.3% without memory. The
underlying WebArena-Infinity environments are live, self-contained web
applications with reliable programmatic verifiers, and they support
coordinate-based computer-use agents as well as browser agents. Again,
caveat: MemGym's published computer-use setup observes the GUI mainly
through a text accessibility tree, so it is not a direct raw-pixel-memory
competitor. But as a closed-loop agentic-memory benchmark for computer use,
it qualifies considerably better than AndroTMem.

**The most dangerous overlooked work: DataScope / ATMem.** The June 2026
paper "What Memory Do GUI Agents Really Need?" is a much bigger conceptual
threat to your paper than AndroTMem. It proposes Active Task-Driving Memory
plus STR-GRPO, an online RL procedure that contrasts memory-on and
memory-off rollouts based on task completion. It also introduces DataScope
specifically to stress long-horizon data tracking, distractors, and scope
preservation. Secondary material describes DataScope as 32 task families /
96 instances with success rate, app-level progress, and scope-aware F1. I
would not yet write "programmatically scored" in the paper because I did
not find released verifier code sufficient to audit how those metrics are
computed. But this should be your first follow-up. Methodologically, it is
perilously close: GUI memory + online RL + task-completion utility + a
purpose-built memory-stress benchmark.

**MobileWorld is a better generic testbed than you are giving it credit
for.** MobileWorld is not a memory benchmark, but it is a very sensible
third-party environment for this project: 201 tasks, long-horizon and
cross-app workflows, live Android execution, with task verification through
backend database queries, local-storage inspection, textual matching, and
application callbacks. It is also already becoming a standard evaluation
venue for memory systems: MemGUI-Agent reports live MobileWorld results,
and ATMem reports transfer there. I would prefer AndroidWorld + MobileWorld
for the closed-loop part of the paper over AndroidWorld + two offline
memory datasets.

**AndroTMem is much weaker evidence than your current roadmap implies.**
AndroTMem is explicitly an inference framework that reads recorded
multi-step JSONL data, supplies a stored screenshot at each step, predicts
an action, and computes AMS/TCR from annotations. It does not execute the
proposed action and observe the resulting GUI state. Worse, its released
evaluator automatically credits a missing screenshot as correct
(AUTO_PASS). That doesn't invalidate the benchmark as a diagnostic dataset.
It absolutely disqualifies it from being evidence for your closed-loop
claim.

**MementoGUI** is even easier to classify: its own project page calls
MementoGUI-Bench an offline benchmark and uses VLM-judged VAM/TPS/MCS
metrics.

**MemGUI-Bench** is subtler. Its 2026 code has been refactored into a
genuine live MobileWorld-style Android runtime—the agent acts, screenshots
change, trajectories are real. But its final evaluation uses Progressive
Scrutiny involving trajectory description, semantic judgment, and targeted
visual verification, rather than a pure programmatic end-state predicate.
Thus it fails your stricter scorer criterion, even though calling it
"offline" is now outdated.

**Strict bottom line**: I found two clean memory-oriented hits you were
missing—LearnGUI-Online and MemGym/WebArena-Infinity—plus one extremely
important provisional hit, DataScope. If you narrow "memory" to
within-episode, raw-pixel visual working memory, I still do not see a prior
benchmark that simultaneously has your exact combination of fixed visual
budget + live execution + purely programmatic terminal reward. That
narrower statement is defensible. The broad statement is not.

### 3. What experimental route should replace the current pivot?

I would not make "AndroidWorld training → AndroTMem/MementoGUI evaluation"
the main experiment. It takes your strongest methodological
principle—learn memory from actual consequences—and evaluates it somewhere
consequences no longer exist.

The clean experiment is a matched-executor causal comparison. Hold the GUI
policy fixed and run recent-2, random-2, learned-2, and privileged oracle-2
on exactly the same live tasks. Learned-2 versus recent-2 is your main
causal result; oracle-2 tells you whether the environment actually contains
usable memory headroom. Only after that works should you show a jointly
trained executor+selector as a system-level upper bound.

This matters because the current idea of joint GRPO weakens attribution. If
both selector and executor change, "joint+learned beats frozen+recent" does
not tell me that better memory selection caused anything. Worse, testing a
jointly adapted executor with recent-2 can itself be an out-of-distribution
intervention: the executor learned under the selector's induced history
distribution.

I would structure the evidence as follows: train only on task families that
are disjoint from evaluation; establish selector-only improvement first;
then joint-train as an optional second stage; evaluate the same checkpoint
on AndroidWorld and MobileWorld; additionally run LearnGUI-Online if its
demonstration-memory protocol can be adapted fairly. Keep
AndroTMem/MementoGUI as offline diagnostics showing whether the selector
retrieves causally useful states, not as the evidence that the agent
accomplishes more tasks.

AndroidWorld also has an overfitting trap. "Millions of unique task
variations" sounds much stronger than it is: they come from 116
hand-authored task templates with randomized parameters. New entity values
are not new reasoning structures. If you train RL on AndroidWorld,
instance-level randomization is not enough. You need held-out task
templates, workflow families and preferably held-out apps—or a second
environment such as MobileWorld.

One more evaluator issue: terminal verification should penalize undesired
side effects, not merely detect the requested positive state. DataScope's
scope-aware evaluation is a useful lesson here. A task such as "delete all
qualifying X but preserve Y" is much better for memory evaluation than a
verifier that merely checks whether X disappeared, because an amnesic agent
can otherwise succeed by destructive over-action.

And keep the bridge. Demote it; don't delete it. Its role is an
intervention study answering "when old evidence physically cannot be
revisited, does memory selection matter?" That is a good analysis figure.
It just cannot be the only place your method works.

### 4. Is there still a CVPR paper here, and what is actually novel?

The broad novelty position is much worse than the current roadmap suggests.

By August 2026 there is already a dense cluster of GUI-memory work: UI-Mem
does self-evolving experience memory with online RL; ATMem does task-state
memory with STR-GRPO; MAGNET dynamically evolves stationary/procedural
memory and evaluates online on AndroidWorld; EchoTrail-GUI builds
actionable memories through autonomous exploration and appeared at CVPR
2026; EAM uses executable state-machine/KG memory in live AndroidWorld;
AGMem specifically studies visual memory representation on OSWorld; and
MemGUI-Agent makes context management itself an action in long-horizon live
mobile execution.

So I would strike any claim resembling "first GUI memory," "first online-RL
GUI memory," "first closed-loop GUI-memory system," or "first long-horizon
GUI memory." Those claims are no longer credible.

There is still a narrower and substantially better CVPR story:

**Budgeted visual working-memory allocation**: from a long within-episode
screenshot history, learn which small fixed set of raw visual states to
retain, using only terminal GUI task outcomes, while keeping the executor
fixed for the primary causal comparison.

That distinguishes you from ATMem's structured symbolic task state,
UI-Mem's cross-task experience templates, EAM/ActionEngine's reusable state
machines, MementoGUI's multimodal summary/ROI controller, and AGMem's
action-grounded cross-trajectory crops. But every qualifier above is doing
work. Remove "fixed budget," "raw visual," "within episode," "terminal
outcome," or "matched executor," and the novelty rapidly collapses into
existing 2026 work.

For CVPR specifically, I would lean into visual memory allocation, not
generic agent memory. The interesting question is not "can memory help
agents?"—2026 already answered that several times. It is closer to: under
a hard visual-token budget, which past visual evidence should an embodied
GUI policy preserve, and can terminal consequences teach that allocation
without frame supervision?

My go/no-go criterion would be severe: if the learned selector remains
neutral on independent live evaluation and only wins on your bridge or
offline action-matching metrics, I would not submit this as a CVPR methods
paper. That result says the mechanism works under constructed memory
pressure, not that the method improves GUI agents. At that point the more
defensible paper would be about the evaluation gap itself—standard GUI
benchmarks under-measure memory—but a self-built bridge alone is not enough
evidence for that either.

If, however, you get a statistically credible same-executor learned-2 >
recent-2 result on one genuinely memory-oriented live benchmark and confirm
it on a second independent live environment such as MobileWorld, while also
beating or cleanly differentiating from ATMem/UI-Mem/MemGUI-Agent-style
baselines, then there is a real CVPR paper.

The immediate correction I would make to the August 22 roadmap is
therefore: do not replace the bridge with AndroTMem/MementoGUI. Replace the
headline evaluation with third-party live execution, and retain all three
offline/constructed suites as diagnostics.

## 引用与待核实项

其回答引用了(未逐一独立核实,标记为**待核实**):LearnGUI/LearnAct
(ACL Anthology)、MemGym(Wujiang Xu 主页)、WebArena-Infinity(GitHub)、
"What Memory Do GUI Agents Really Need?"(arXiv,2026-06)、DataScope
规模(Moonlight 二手材料)、MobileWorld(GitHub,201 任务)、AndroTMem
evaluator AUTO_PASS 行为(GitHub 源码)、MemGUI-Bench 2026 重构(GitHub)、
UI-Mem / MAGNET / EchoTrail-GUI(CVPR 2026)/ EAM / AGMem / MemGUI-Agent。
**采纳任何一条进论文前必须自己读原文核实**,尤其:DataScope/ATMem 的
STR-GRPO 与我们的重叠度、AndroTMem AUTO_PASS、LearnGUI-Online 的判分链、
MemGym computer-use 轨道的观测形态(a11y tree vs 像素)。
