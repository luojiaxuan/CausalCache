# 外审:换 executor 为 UI-Venus-2-9B 与余量探针闸门(2026-09-04 03:40 PT)

## 1. 审前判断(先写后审)

- 瓶颈在 executor 任务级地板(MemGUI 7/128、MobileWorld heldout 11/20 全零)与 selector 跨模板精度,
  不在 RL 优化器;冻结 GUI-Owl/iter_59 主线终止,换 UI-Venus-2-9B(MemGUI 62.6、MobileWorld 65.8)。
- 闸门(我原方案):MobileWorld heldout-39 程序判分三臂 N_IMG = 0 / 2 / 全部;满历史 − 最近两帧 ≥ +3pp
  才继续;继续 = 在 Venus 上重做 selector 表征(patch 级 cross-attention)。
- 已否决:random-S 格式 SFT(J1 证明能力已在)、Stage I bandit(同数据不解泛化)。

提示词(单行英文,<1500 字符)与设置见 ledger §2 J10;ChatGPT 临时会话,极高推理档,思考 3m10s。

## 2. 外审原文(英文,按段整理转录;取回时 `=`/`&`/`?`/`%`/`;` 被过滤,已按上下文还原;
## 约 250 字符的一段(MobileWorld 为何合适)只剩摘要)

I would swap to UI-Venus-2-9B. The GUI-Owl setup has become a capability-floor experiment, not a viable
substrate for learning a memory policy: the oracle gap proves frame choice can matter locally, but 7/128
closed-loop success plus pervasive step-cap/loop failures means terminal reward cannot tell you whether the
selector is good. The RLOO behavior is exactly what that floor predicts.

The swap itself is well motivated. UI-Venus-2-9B is in the same rough parameter class, its public interface
explicitly supports varying the number of retained historical screenshots while keeping the current screenshot
and assistant text history, and its reported scores are 65.8 on MobileWorld GUI-only at 50 steps and 62.6
Pass@1 on MemGUI. That gives you enough successful trajectories for memory selection to actually move end-task
success.

### (a) Top 3 reviewer attacks

1. Your gate does not test your hypothesis. "All history beats recency-2" tests whether more visual context
   helps UI-Venus. Your claim is that choosing the right two non-contiguous frames helps. Those are not
   equivalent. All-history can lose because of visual-token dilution/interference while a B=2 selector wins;
   conversely, all-history can win even if no learnable B=2 policy can recover the useful frames. Therefore,
   failing your proposed gate is not evidence that selective memory is pointless, and passing it is not evidence
   that your selector is viable.
2. You have not diagnosed why the supervised selector failed OOD, so "add patch-level cross-attention" looks
   like architecture thrashing. +5.1pp in-domain versus +0.2pp on unseen templates, with no sensitivity to
   halving the data, screams shortcut/generalization problem, not insufficient capacity. Patch-level features may
   help if the failure is fine-grained visual correspondence, but they can just as easily increase template
   memorization. Before rebuilding, you need to establish whether the label itself transfers: run the new
   executor with oracle-selected pairs on unseen templates. If oracle headroom also collapses OOD, there is
   nothing for a better selector to learn. If oracle headroom remains large but your learned selector collapses,
   then redesigning the selector is justified.
3. The go/no-go evaluation is too small and too post-hoc to support a 3pp threshold. On 39 binary tasks, one
   task is 2.56pp. Your ≥3pp criterion therefore jumps from +2.56 to +5.13; in practice, two net task flips pass
   the gate. That is noise-level evidence for a major project decision. There is also an obvious reviewer
   narrative: "The authors failed on one executor, then selected a new executor after seeing that it scores 62.6
   on their target benchmark." You can neutralize that only by freezing the swap criterion now and making the
   gate entirely on non-MemGUI data. Also be precise about scope: in UI-Venus, N_IMG=0 still retains assistant
   text history, so you are studying visual episodic memory selection on top of textual action history, not
   "memory versus no memory."

### (b) The probe is directionally right, but I would change the gate

39 MobileWorld held-out tasks are fine for a smoke test. They are not fine for a 3pp gate. MobileWorld is a good
choice because its GUI-only tasks have deterministic/backend verification rather than an LLM judge, and the
official suite is long-horizon and heavily cross-app.

I would use this minimal design:

| Arm | Purpose |
|---|---|
| N_IMG=0 | Measures contribution of historical images beyond retained text |
| recency-2 | Actual baseline you intend to beat |
| recency-8 or all-history | Tests whether the executor benefits from a larger visual window |
| non-contiguous heuristic B=2 | Tests the actual selective-memory hypothesis without training |

For the fourth arm, use something intentionally dumb but non-contiguous: most recent frame + last
app-transition frame, or most recent + highest visual-change frame. It need not be good. Its job is to
distinguish "old frames can be useful" from "more frames can be useful."

I would prefer recency-8 over all-history as the main large-context arm, with all-history only as an auxiliary
diagnostic. "All" changes input length dramatically with episode length and can create a context-load
confound. A finite 8-frame arm gives a cleaner dose-response: 0 → 2 → 8. UI-Venus itself exposes N_IMG exactly
this way.

The metric should remain paired programmatic task success, but report the contingency table against recency-2:
wins / losses / ties, not just percentage points. Add loop rate, step-cap rate, and median steps-to-success as
diagnostics. Given what killed GUI-Owl, a memory policy that changes 0 completed tasks but halves looping is
still scientifically informative, even if it should not trigger the full project.

For the actual gate, choose one of two regimes. If you insist on only 39 tasks, make this an effect-size
screen: I would want roughly ≥4 net task wins over recency-2 (~10.3pp), or a smaller task-success gain
accompanied by a very clear reduction in memory-related loops/cap hits. If your scientifically meaningful target
really is 3–5pp, run the full 117 GUI-only MobileWorld tasks instead; 39 simply cannot resolve that scale
credibly. The official MobileWorld GUI-only suite has 117 tasks under a 50-step protocol.

More importantly, I would add one cheap offline oracle test before any selector training. On fixed held-out
trajectories, run UI-Venus with current-only / recency-2 / random-2 / oracle-2, exactly mirroring your GUI-Owl
diagnostic. This directly answers the crucial question: Does UI-Venus-2 itself have an OOD action-level
oracle-vs-recency gap? If the answer is no, stop. If it is large, but all-history fails, do not stop—that is
actually evidence in favor of selective B=2 memory.

So my gate would be: continue when UI-Venus shows substantial oracle-B=2 headroom on unseen templates and has
a healthy closed-loop floor on MobileWorld. I would not make all-history > recency-2 a necessary condition.

### (c) Missing executor / benchmark options

The most important missing comparison is actually MemGUI-Agent / ConAct, not another backbone. It is now
directly overlapping related work: proactive context management for long-horizon mobile GUI agents, with
released code and an 8B SFT checkpoint. The authors report 23.4 MemGUI Pass@1 and 17.9 MobileWorld for the 8B
model. Even though UI-Venus-2-9B is much stronger overall, reviewers will ask how "external learned screenshot
retrieval" differs from or complements "model-generated context folding/UI memory." You need that
comparison/discussion.

For a second executor, I would use MAI-UI-8B as a robustness control rather than as the main backbone. Its
weights are public under Apache-2.0, and it is a genuine mobile GUI model, but its reported MobileWorld/MemGUI
performance is much lower than UI-Venus-2. If your selector gives the same qualitative benefit on Venus and
MAI-UI, that substantially strengthens the claim that this is a controller contribution rather than a
Venus-specific trick.

Qwen-UI-Agent-27B is the obvious high-ceiling future option—82.1 on MobileWorld is substantially stronger—but I
would not build the project around it right now. The current public MAI-UI repository points to released MAI-UI
2B/8B weights, while Qwen-UI-Agent is presented as the newer follow-up; I did not find a corresponding public
27B checkpoint in the current official Hugging Face releases.

For benchmarks, I would not add AndroidWorld as another primary gate. It is reproducible, but MobileWorld was
explicitly built because AndroidWorld is easier/closer to saturation and has far fewer cross-app workflows. A
more useful addition is a small controlled memory diagnostic: tasks where an early screen exposes one needed
fact, intervening screens remove it, and the final action is programmatically checkable. That gives you causal
evidence that gains come from retained visual information rather than generic navigation improvements.

Also use MemGUI's memory-specific metrics in the final paper, not only Pass@1. The current benchmark exposes
IRR, MTPR, FRR, step ratio, etc. If your controller raises Pass@1 without raising information
retrieval/retention, the "memory" story is weak. Conversely, a clean IRR gain makes the argument much stronger.
I would manually inspect every episode on which recency-2 and your selector disagree in the final MemGUI
evaluation, because a few-point result should not rest entirely on LLM-judge boundary cases.

### (d) I agree with the swap—with one change in framing

Keep GUI-Owl in the paper as a negative-control executor, not as a failed preliminary experiment you hide. The
story becomes much cleaner: Necessary condition 1: executor utilization. GUI-Owl has a huge oracle-vs-recency
step gap, proving frame identity matters, but its task-level floor is so low that terminal learning is
uninformative. Necessary condition 2: learnability/generalization. Dense-label selector SFT overfits task
templates; terminal RLOO cannot repair that under sparse reward. Intervention: move to a capable frozen executor
where task reward has dynamic range, then test whether selective non-contiguous visual memory adds value.

That is defensible. What I would not do is immediately spend weeks on the patch-level selector after UI-Venus
clears only an all-history probe. First establish UI-Venus oracle-B=2 OOD headroom. That experiment tells you
whether the project still has a causal target.

My decision tree would therefore be: UI-Venus recency-2 has healthy MobileWorld success → oracle-B=2 beats
recency on unseen templates → build selector. If oracle does not beat recency OOD, stop/downscope. If oracle
beats recency but all-history does not, continue—that is exactly the regime where selective memory is
potentially most useful.

One practical footnote: the 9B weights are publicly available, but the current Hugging Face material has
inconsistent/pending license language, so verify the final model-weight license before making
reproducibility/distribution claims.

Net: swap yes; proposed gate no. The critical gate is oracle selective B=2 > recency-2 on OOD, not all-history
> recency-2.

## 3. 采纳 / 不采纳(2026-09-04 03:50 PT,探针数据落地前预注册)

| 外审意见 | 处置 | 理由 |
|---|---|---|
| 换 executor 正确;GUI-Owl 作阴性对照 executor 写进论文 | 采纳 | 与 J5 一致;叙事上比"藏起失败"更强 |
| "满历史 > 最近两帧"不是必要条件 | **采纳,改判** | 反例成立:token 稀释可使满历史输而 B=2 选择赢;原闸门会误杀 |
| 剂量臂 0 / 2 / 8,全历史降为辅助 | 采纳 | 8 帧上限消掉上下文长度混杂 |
| 加"非连续启发式 B=2"臂(最近帧 + 视觉变化最大帧) | 采纳 | 零训练即可区分"旧帧有用"与"多帧有用" |
| 39 题太小;3pp 闸门分辨不出;跑全部 117 GUI-only 题 | 采纳 | 117 题 = train 78 + heldout 39 全集,executor 探针不涉及 selector 训练,可全用;heldout-39 另报 |
| 报 wins/losses/ties 列联表 + 死循环率 + 撞上限率 + 成功中位步数 | 采纳 | 进 tally 脚本 |
| 关键闸门 = 该 executor 上 oracle-B=2 在未见模板上的离线动作级余量(current-only / recency-2 / random-2 / oracle-2) | **采纳为主闸门** | 与 §17 协议同构;闭环探针只作"地板健康"与"剂量响应"辅证 |
| 先做 oracle 余量再动 selector 表征;"patch 级 cross-attention"未经诊断即上是架构乱撞 | 采纳 | selector 改造推迟到 oracle 余量成立之后 |
| 现在冻结换 executor 的判据、闸门只用非 MemGUI 数据 | 采纳 | 本节即预注册;MemGUI 不参与任何决策 |
| 范围表述:N_IMG=0 仍保留文本动作历史,研究的是"文本历史之上的视觉情景记忆选择" | 采纳 | 写进 ledger 口径表 |
| 必须对比/讨论 MemGUI-Agent / ConAct(8B SFT,MemGUI 23.4 / MobileWorld 17.9) | 采纳(论文期) | 直接重叠的相关工作 |
| MAI-UI-8B 作第二 executor 稳健性对照 | 待定 | 主线成立后再加 |
| 最终论文报 MemGUI 的 IRR/MTPR/FRR | 待定 | 与判分费用决策(只保 Pass@1)冲突,由用户裁定 |
| 小型受控记忆诊断任务集 | 待定 | 有价值但不阻塞 |

**预注册闸门(数据落地前写定)**:
- G0 地板健康:UI-Venus-2 recency-2 在 117 题 MobileWorld 上成功率 ≥ 40%(论文报 65.8%;低于 40% 视为
  harness/移植问题先排查)。
- G1 剂量与选择(闭环,117 题,配对):报 N_IMG 0/2/8 与启发式 B=2 对 recency-2 的 wins/losses/ties;
  **不作为继续/停止条件**,只作叙事与诊断。
- G2 主闸门(离线,heldout-39 轨迹上的动作级贪心正确率,协议同 §17):oracle-2 − recency-2 ≥ +5pp 且
  random-2 ≤ recency-2 + 2pp(即"选得准才有用");满足则进入 selector 重建,不满足则停线/缩主张。
- 违反 G0 时不解读 G1/G2。
