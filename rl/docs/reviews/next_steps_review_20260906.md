# 外审:主线下一步该怎么走(2026-09-06 11:30 PT)

- 审阅方:网页 ChatGPT 临时会话,模型选择器实际列出的最新旗舰 GPT-6,档位"极高"(次高档);"思考了 3m 23s"。
- 给它的材料:公开仓库钉住 commit `b2c5fdf`,要求先读 `rl/docs/ledger_memory_rl_20260904.md`、`rl/docs/story_20260905.md`、`rl/docs/b_pilot_design_20260906.md`、`rl/cua/scripts/pilot_checkpoint_eval.py`,引用时注明文件与节。它实际用 GitHub 搜索读了钉住 commit 的这四个文件(回复内引用了台账 J20/J22、设计稿 §0–§5、评测器 `run()`/`state_of()`/`irr2`/`gold_text` 与 416–435 行),属于"读了仓库后说的",非凭空猜。
- 提示词(单行英文,1498 字符):见文末。

## 审前判断(发审前写下)

1. 部署布局 recency 臂(337 态)说明加图的大头是格式效应(无关图 +11.4 vs 最近图 +16.5),内容贡献 ≈ +5pp;裁判老帧换进最近图位与 recency 打平,附加在 recency 之上更差。自然态分布里记忆依赖态 1/337 → 在 GUI-Owl 分布内"选哪张老图"没有余量。
2. 论点"视觉历史不该默认最近图"只能在两处成立:Venus 无文本(裁判帧 +20~31pp)和构造的记忆关键任务(B-pilot)。下一步顺序:deploy2 收尾 → B-pilot checkpoint 干预矩阵(部署布局,Venus 为主、Owl 为异质性)→ go/no-go。
3. go 判据(我拟):决策步上 src_keep/src_pickimg 的救回数显著高于 irr2 且高于 rec2,且在遮蔽文本与保留文本两种协议下方向一致;no-go 则 pivot 到动态 B 的效率叙事。
4. 最大风险:Owl 的文本摘要把关键值写进历史 → text_only 已解 → 视觉记忆无余量;次风险:延后揭示做不到(SMS 第 1 步可见)。
5. 现在不上联合 RL:先看 pilot 读数。

## 外审全文(逐字落盘,未删减)

### 1. Single next experiment

Run the Venus-primary B-pilot as a same-checkpoint, same-layout counterfactual frame-content intervention; deploy2 becomes supporting context, not the gate. Natural MobileWorld contributes only 1/337 clearly memory-dependent states, while current deployment-layout gains contain a large generic-image component (irrelevant2 +11.4pp), so deploy2 cannot decide whether which old frame matters. [rl/docs/b_pilot_design_20260906.md §0; rl/docs/ledger_memory_rl_20260904.md §5, 2026-09-06 11:10 PT]

| Factor | Conditions | Requirement |
|---|---|---|
| Model | UI-Venus-2-9B primary | Use Venus-native history protocol; do not route Venus through the current GUI-Owl decode_ctx.py constructor. |
| Text | native_text, no_history_text | Native = deployment relevance; no-text = visual-channel identifiability, matching the prior Venus diagnostic where selected old frames beat recency by ~17–27pp. [ledger §5 J20] |
| Frames, B=2 | source, twin_wrong_source, recency2, irrelevant2 | All four must use the same image-slot/layout constructor. source and twin_wrong_source should both be pickimg, not hybrid. |
| Unit | 24 base scenarios × 2 twins = 48 checkpoints | Cluster inference by the 24 base twin pairs; stratify family, do not treat 48 twins as independent. |
| State | Same current screenshot, task goal, recent-turn text/layout | Only historical image content changes. No conditioning on text_only failure; rescue/harm is secondary. |
| Primary causal check | source ↔ opposite-twin source swap | For the same state, replacing A's source frames by B's must make the action follow B's answer. |

The existing evaluator is close but not sufficient: src_pickimg holds recent-turn structure fixed, whereas swap_hybrid appends PAST images and therefore changes the intervention itself; add swap_pickimg = messages_deploy(st,2,irr=twin_source_frames). [rl/cua/scripts/pilot_checkpoint_eval.py checkpoint-intervention note / run()]

src_keep is diagnostic only because it jointly changes preserved assistant turns; src_hybrid is diagnostic only because appending old frames already shows substantial format/harm effects. The ledger explicitly requires decoupling "which frames" from "which turns retain original text." [ledger §5, 2026-09-06 10:20 PT]

### 2. Go/no-go thresholds

The current preregistration is too permissive: "one backbone," twin-follow >50%, and source beating recency in "at least one family" permits backbone/family selection with only eight pairs per family. [rl/docs/b_pilot_design_20260906.md §5]

| Gate | GO threshold for joint RL | Failure disposition |
|---|---|---|
| Source specificity, native text | V(source)-V(irrelevant) ≥ +10pp, paired 95% CI lower bound >0 | Fail ⇒ no joint frame-selection RL. |
| Anti-recency, native text | V(source)-V(recency2) ≥ +10pp, paired 95% CI lower bound >0 | Fail ⇒ deployment thesis "old source > recency" unsupported; dynamic-B/efficiency branch. |
| Source specificity, no-text | Same: ≥+10pp, lower 95% CI >0 | Fail ⇒ visual channel itself is not robust on constructed tasks. |
| Anti-recency, no-text | Same: ≥+10pp, lower 95% CI >0 | This should be easy if the prior Venus +17–27pp old-frame-over-recency effect transfers; failure is substantive. [ledger §5 J20] |
| Counterfactual twin flip | P(source→own answer AND twin-source→twin answer) ≥70%; lower 95% bound >50% | Fail ⇒ apparent source advantage can be task/layout correlation rather than evidence use. |
| Family robustness | pooled gate above + no family has source-recency < -10pp | Do not demand per-family significance at 8 pairs/family. |
| Positive-control validity | gold_text ≥85%; otherwise checkpoint invalid, not a model failure | Current evaluator directly injects the expected fact for this control. [pilot_checkpoint_eval.py gold_text] |

Decision rule: require every native-text gate and every no-text gate; a no-text-only pass is NO-GO for joint RL but a clean protocol-dependence result, because deployment retains text. This matches the existing evidence that Venus visual dependence explodes when text history is removed, while text can substitute for screenshots when retained. [rl/docs/story_20260905.md §2, first chapter; ledger §5 J20]

Use full-sample paired ΔV as the gate, not "rescue among B=0 failures"; the evaluator correctly states that checkpoints should not be selected on B=0 failure. [pilot_checkpoint_eval.py checkpoint-intervention note]

### 3. Pilot flaws

| Flaw | Fatality | Required fix |
|---|---|---|
| Venus protocol mismatch | Fatal implementation blocker. Current script imports /data01/jaxan/decode_ctx.py and constructs GUI-Owl deployment messages, despite Venus being the intended primary model. [pilot_checkpoint_eval.py lines 416–435] | Implement Venus-native messages_*; rerun only after protocol parity is asserted. |
| Text-summary "leakage" | Causal self-written text containing the fact is not leakage; it is a competing memory channel, exactly as the design states. Future-conditioned teacher summaries, answer-bearing metadata, filenames, or tool outputs are fatal leakage. [b_pilot_design §3] | Before every checkpoint, audit goal, extracted concls, filenames/tool output for expected + aliases. Keep causal text untouched in native_text; separately run no-text. The current evaluator extracts every prior conclusion, so this audit is mandatory. [pilot_checkpoint_eval.py state_of()] |
| Late reveal is not late | Fatal for the selector/RL claim; nonfatal for the narrower frozen-executor "can this old frame causally affect the decision?" test. The design requires future demand to be invisible to memory-writing components, but current SMS exposes it at step 1. [b_pilot_design §3; ledger §5 11:10 PT] | Before joint RL, make the request appear only after evidence browsing via ask_user or a carrier requiring later navigation. Otherwise the learned policy is told in advance what to remember. |
| Twins | Twins are an asset only as within-state image swaps; A-trajectory vs B-trajectory comparisons are confounded by differing histories. | Add swap_pickimg; preserve A's current state/text and swap only B's evidence pixels. Use the twin wrong-source condition as the strongest negative control. Current swap_hybrid does not satisfy this. [pilot_checkpoint_eval.py checkpoint matrix / run()] |
| Irrelevant control | Potentially fatal to "semantic source content" if visually/domain mismatched. Current code simply takes the first two screenshots from irrelevant_dir. [pilot_checkpoint_eval.py irr2] | Match app, rendering, screenshot age, image count, and workflow stage; prioritize opposite-twin wrong evidence over arbitrary irrelevant screenshots. |

The early-reveal flaw directly contradicts the pilot's original intended construction: family A was supposed to browse records before the user specifies which record/property is needed. [rl/docs/b_pilot_design_20260906.md §1–§3]

### 4. Top 3 reviewer attacks

1. "This is another layout/prompt artifact, not a memory result." This attack is unusually strong because the project already had a major conclusion reverse after replacing the flattened evaluator with deployment-faithful interleaved history; GUI-Owl recency moved from apparently harmful to strongly useful, and old-frame keep interventions destroyed recent-turn structure. [ledger §5, 2026-09-06 10:20/J22] Defense required: Venus-native protocol + identical pickimg layout for source/twin-wrong/irrelevant/recency + source must beat both controls.
2. "You manufactured tasks where the answer is old, exposed the query from step 1, and then declared old frames useful." Natural MobileWorld has only 1/337 identified memory-dependent states, and the current implementation loses the preregistered delayed binding. [b_pilot_design §0, §2–§3; ledger §5 11:10 PT] Defense required: scope the pilot as a causal capacity test; restore genuine late reveal before claiming learned storage/selection; retain deploy2 as external-validity evidence rather than the causal gate.
3. "Your pilot is underpowered and researcher-flexible: 24 independent twin clusters, 8/family, many context arms, 'at least one family/backbone' success, and a next-action substring endpoint." The preregistered pilot gate explicitly allows one backbone and one family, while the main-study design itself anticipates 300–400 base instances and notes that ~5pp effects require hundreds of pairs. [b_pilot_design §5] Defense required: Venus-primary pooled endpoint fixed before decoding; twin-pair clustered inference; ≥10pp practical-effect floor; family results descriptive; counterfactual flip as causal endpoint; short checkpoint continuation/task-state verification for every claimed rescue.

## 采纳表(2026-09-06 11:50 PT)

| 外审意见 | 判定 | 理由 / 落地 |
|---|---|---|
| 下一实验 = Venus 为主、同 checkpoint 同布局的反事实"帧内容"干预;deploy2 只作外部效度背景,不作闸门 | **采纳** | 与审前判断一致;deploy2 的 +11.4(无关图)已说明它分不清"哪张老图"。 |
| 评测器把 Venus 也走 GUI-Owl 的 `decode_ctx` 构造器 = 致命阻断 | **采纳(它读仓库后指出的真问题)** | 评测器加 `--backend venus`:复用 `venus_oracle.messages`(Venus 原生多轮协议,`no_text` 为无文本口径);Venus 下老帧放回原 turn 位置(`src_at_turn`)是原生干预,不改任何文本轮。 |
| source / twin_wrong_source / recency2 / irrelevant2 必须同一构造器;加 `swap_pickimg`,不用 `swap_hybrid` 作主反事实 | **采纳** | GUI-Owl 后端加 `swap_pickimg = messages_deploy(st,2,irr=twin_source)`;`swap_hybrid` 降为诊断。 |
| 闸门:V(source)−V(irrelevant) 与 V(source)−V(recency2) 均 ≥ +10pp 且配对 95% CI 下界 > 0;原生文本与无文本两协议都要过;孪生翻转 ≥70%(下界 >50%);gold_text ≥85% 为有效性前提;按 24 对聚类;不要求逐族显著 | **采纳,加一档** | 48 个 checkpoint 上 +10pp ≈ 5 个净不一致对,CI 下界 >0 需要方向近乎一边倒;点估计 ≥+10pp 但下界 ≤0 不判 NO-GO,判 **EXPAND**(扩到 ≥100 对再判)。三档预注册见设计稿 §5。 |
| 仅无文本口径过线 = 对联合 RL 是 NO-GO,但是干净的"协议依赖"结果 | **采纳** | 部署保留文本;这条与 Venus 保留文本 +5.8pp / 去文本 +30pp 的既有读数一致。 |
| 自写文本含答案不是泄漏而是竞争通道;要审计 goal / 抽取的 conclusion / 文件名 / 工具输出里是否含 expected 及别名 | **采纳** | 评测器每个 checkpoint 输出 `leak={goal,hist,filename}` 三个标志;原生文本臂不动文本,另跑无文本臂。 |
| 延后揭示未延后:对 selector/RL 主张致命,对冻结 executor 的因果测试不致命 | **采纳(范围界定)** | pilot 定位为"因果容量测试";联合 RL 之前必须恢复真正的延后揭示(ask_user 通道或需导航的载体)。 |
| 无关对照必须匹配 app / 渲染 / 步龄 / 张数 / 工作流阶段;优先用对侧孪生的错误证据 | **采纳** | 主负对照 = 孪生另一版的源帧(构造上完全匹配);`irr2` 取同族另一对同步位截图,只作次对照。 |
| 三个攻击点(布局伪影 / 人造任务且步 1 揭示 / 功效与研究者自由度) | **采纳为写作约束** | 主终点在解码前固定为 Venus 合并 ΔV;逐族只描述;每个救回做短续跑核实。 |

它没读到 / 我补充的实测:外审发出时前缀采集尚无读数;审后 15 分钟发现底座 GUI-Owl 在 18/48 个 Mail 任务上全部 40 步循环、从不进入决策步(见台账 11:50 PT 条目),因此"自然前缀 + 决策步 checkpoint"在 GUI-Owl 上不成立——这与外审"Venus 为主"的要求同向:前缀改由 Venus 自然生成;若 Venus 同样到不了决策态,退回脚本化(teacher-forced)前缀。

## 提示词原文

```
Public repo pinned at https://github.com/luojiaxuan/CausalCache/blob/b2c5fdf/ ; read first, cite file+section: rl/docs/ledger_memory_rl_20260904.md, rl/docs/story_20260905.md, rl/docs/b_pilot_design_20260906.md, rl/cua/scripts/pilot_checkpoint_eval.py. SETUP: GUI-Owl-7B on MobileWorld; thesis: visual history should not default to recency (fixed B=2 old frames). Deployment-layout recency arm, n=337 states: text-only 76.1; +recent2 +16.5pp; +irrelevant2 +11.4pp; judge frames in recent slots +9..+16; appended as PAST +6..+8. Venus no-text: judge frames +20..31pp over recency. Memory-dependent natural states 1/337. Running: deploy2 random/older/base arms; B-pilot: 48 twin Mail/Files tasks -> checkpoint intervention matrix, Venus primary. PLAN: finish deploy2 -> pilot go/no-go (source-frame rescue > irrelevant and > recency at decision step, with and without text) -> go: joint RL with frame selection on memory-critical tasks; no-go: dynamic-B efficiency framing. ASK: (1) single most decision-relevant next experiment; (2) go/no-go thresholds; (3) fatal flaws in pilot (text-summary leakage, late reveal, twins); (4) top 3 reviewer attacks. FORMAT: Your reader is an AI agent, not a human. Be terse and information-dense: no preamble, no restating my setup, no hedging boilerplate, no motivational or summary paragraphs. Every sentence must carry a claim, a number, a citation, or an instruction. Prefer numbered claims and compact tables. Keep all technical content; drop everything else.
```
