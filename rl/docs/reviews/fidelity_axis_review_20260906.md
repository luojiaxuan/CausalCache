# 外审:文本保真度轴主张(2026-09-06 18:05 PT)

- 审阅方:网页 ChatGPT 临时会话,最新旗舰 GPT-6,档位"极高";"思考了 2m 35s"。读了钉住 commit `9081ec7` 的台账 §5(16:00–17:50 PT 条目)与故事 §2.12–2.13,回复按文件+节引用。
- 提示词(单行英文,1506 字符)见文末。

## 审前判断

主张:GUI agent 回看老截图的余量由文本历史保真度决定。证据:Venus 同批 30 checkpoint 上 full → oneline → action-only,源帧收益 0 → +7 → +53pp;GUI-Owl 教师强制(一行 conclusion)同批 0 → 43%(请求轮+证据轮 vs 最近两轮,同 token);GUI-Owl PartMatch 自然前缀 12.5 → 37.5。效率:扔掉最近两帧 −46% 精度不变;缩放检索帧(0.35,≈350 tok/帧)50% vs 压缩文本 87%——对文本可写事实"以图换文"不成立。预期外审会攻:保真度轴的操作化(oneline 仍含值)是否算"一行摘要";教师强制与自然轨迹的差别;n=30/16 对;效率负结果是否让控制器失去意义;第三方 benchmark 缺位。

## 外审全文(逐字落盘)

Fidelity-axis: defensible as a task-relevant text-sufficiency axis; not yet as a smooth generic "fidelity" law. Use: "The marginal value of retrospective screenshots is gated by whether task-relevant evidence survives in textual history." Venus gives 30/30 target retention with full think and text-only 100%, 29/30 retention with first-sentence and text-only 86.7%, versus 3/30 retention with action-only and text-only/recency/age-control all 0%; under the cleaned request-controlled condition, request+evidence reaches 53.3% [39.3,67.9] with gold-text 93.3%. rl/docs/story_20260905.md §2.13; rl/docs/ledger_memory_rl_20260904.md §5, 17:15 PT.

| Issue | Judgment | Required fix |
|---|---|---|
| "Fidelity" | Full and first-sentence are effectively the same task-relevant fidelity: 100% vs 96.7% target retention; current data mainly identify high-sufficiency vs low-sufficiency, not a graded axis. story §2.13. | Plot measured target-evidence retention, not text length or nominal compression. |
| Main missing control | Compression, style, OOD prompt form, and fact deletion move together between full/first/action histories. | Create same-length/same-style one-line histories with the target fact retained vs omitted, crossed with age-control vs true evidence frame. |
| Strong estimand | The causal claim is an interaction, not three marginal accuracies. | Test [(source−control)_fact omitted − (source−control)_fact retained] > 0 with pair-clustered CI. |
| "Axis" language | Three nominal modes provide only ≈2 effective retention levels. | If retaining "axis," add controlled 25/50/75% target-retention doses; otherwise say sufficiency gate. |

A stronger nontriviality control is delayed relevance: expose evidence before the future query is known, then ask for a detail omitted by the summary; §2.12 already identifies "relevance revealed later + lossy summary" or non-textual referents as the regimes where visual memory should genuinely matter. rl/docs/story_20260905.md §2.12.

Teacher forcing is valid for the conditional mechanism; it is not a stand-in for natural end-to-end agent behavior. GUI-Owl writes its own one-line conclusions on the exact Venus screens, then the actions are forced down the Venus path; this removes trajectory differences and cleanly tests p(a_t | same UI path, Owl summaries, frame policy), with request+evidence 43.3% [26.7,59.4] versus recent-2/request+age-control/text-only 0% at ≈9.1k equal-token two-frame budgets and gold 100%. rl/docs/ledger_memory_rl_20260904.md §5, 17:30/16:55 PT.

Teacher forcing does not identify the natural state distribution d^{π_Owl}(s): GUI-Owl naturally failed to reach the Mail decision state on the attempted prefixes, so "natural one-line-summary agents obtain +43pp" would overclaim. rl/docs/story_20260905.md §2.12–2.13.

GUI-Owl PartMatch is the correct ecological complement: natural prefixes give text-only/age-control 12.5%, source 37.5%, source−control +25 [6.2,43.8], with zero shape leakage in conclusions; its weakness is n=16/8 twin pairs and gold-text 75%, below the stated 85% validity gate. rl/docs/ledger_memory_rl_20260904.md §5, 16:10 PT.

Use terminology: "teacher-forced controlled replay replication", not "natural-agent replication."

Main figure: make the interaction the figure, not raw memory-policy bars.

| Panel | x | y | Required points |
|---|---|---|---|
| A: mechanism | Measured task-relevant evidence retained in text, % | targeted evidence − age-matched control, pp | Venus full: 100% retention / +3.3; first-sentence: 96.7% / +6.7; action-only: 10% / +53.3 using the cleaned request-controlled condition; add matched fact-retained/omitted controls. story §2.13; ledger §5. |
| B: cross-model deployment form | Protocol | Accuracy | Venus action: control 0 → request+evidence 53%; Owl TF: 0 → 43%; Owl natural PartMatch: 12.5 → 37.5%; plot gold 93/100/75 as validity markers. ledger §5, 17:15/17:30/16:10. |

Do not put first-sentence at an arbitrary "medium fidelity" x-coordinate; its target-value retention is 29/30, so its measured fidelity is near full text. rl/docs/story_20260905.md §2.13.

Minimum new experimental work is two additions: (i) the matched one-line fact-retained vs fact-omitted × source vs age-control interaction on the same 30 Venus checkpoints; (ii) expand/repair one natural-prefix low-summary condition to ≥30 valid checkpoints with gold-text ≥85%, preferably PartMatch or delayed-relevance rather than another known-request Mail task.

Top three reviewer attacks.

| Rank | Attack | Why it lands | Best defense |
|---|---|---|---|
| 1 | "This is answer deletion, not a fidelity result." | Full/first retain the target in 30/30 and 29/30 histories; action retains it only 3/30, so the result can be summarized as "images help when you delete the answer." story §2.13. | Matched-format fact-retained/omitted interaction + delayed-relevance task. |
| 2 | "GUI-Owl replication is off-policy." | The 43% result uses Owl summaries but Venus-forced actions; Owl could not naturally traverse the Mail paths. ledger §5, 16:55/17:30; story §2.12. | Call it controlled replay; use natural PartMatch as external validity; expand natural n/gold. |
| 3 | "The controller contribution is tautological/too synthetic." | Request+evidence is hand-identified, Mail facts are directly textualizable, and the natural visual replication is only 16 checkpoints with gold 75%. ledger §5, 16:10–17:15. | Demonstrate one regime where a competent summary cannot know what to preserve—delayed query—or where text is intrinsically lossy—PartMatch—then learn/retrieve the evidence rather than supplying oracle turns. |

Keep the efficiency results; reframe them as modality allocation, not a separate efficiency regime. Full text without old screenshots dominates the default full-text+recent-2 context: 6,048 versus 11,160 tokens, both 100%, giving the clean −46% result. rl/docs/ledger_memory_rl_20260904.md §5, 17:25 PT.

At matched ≈4.5–4.6k budgets, compressed first-sentence text gives 86.7% at 4,542 tokens while 0.35-scale request+evidence frames give 50.0% at 4,615 tokens; therefore images do not efficiently substitute for textualizable history, even though the two scaled historical frames themselves cost only ≈700 tokens. rl/docs/ledger_memory_rl_20260904.md §5, 17:50 PT.

Use the efficiency claim: "Allocate memory by information type: preserve textualizable facts in compressed text; set historical-image budget B=0 when text is sufficient; retrieve targeted old frames only for information omitted by text or poorly representable in text." This is exactly the current §5 interpretation and avoids the contradicted "image-for-text" efficiency claim. rl/docs/ledger_memory_rl_20260904.md §5, 17:50 PT.

Put the 100%@6.0k, 86.7%@4.54k, and 50%@4.62k points in a supplementary Pareto panel; keep the fidelity/sufficiency interaction as the main figure.

## 采纳表

| 外审意见 | 判定 | 落地 |
|---|---|---|
| 不叫"保真度轴",叫"任务相关证据是否留在文本里"的充分性闸;横轴画实测的证据留存率,不画文本长度 | **采纳** | 主张改写:"回看截图的边际价值由任务相关证据是否留在文本历史里决定";图 A 横轴 = 证据留存率(full 100% / 首句 96.7% / action 10%)。 |
| 缺的对照:同长度同风格的一行历史,只差"目标事实留/删",交叉 源帧/对照帧,估计交互项 | **采纳,已发射** | 评测器加 `--text-mode action_fact`(只留 action,证据轮加一句"Noted: <事实>"),与 `action` 配对;条件 text_only / ctrl_at_turn / src_at_turn;交互 = (src−ctrl)_删 − (src−ctrl)_留。 |
| 教师强制 = "受控回放复现",不是自然 agent;自然分布上 GUI-Owl 到不了 Mail 决策态,不许说"自然一行摘要 agent 获得 +43" | **采纳** | 措辞改;PartMatch 自然前缀作外部效度,v4 扩到 ≥30 有效 checkpoint、gold ≥85%。 |
| 最强的非平凡对照 = 延后揭示(先看证据、后知道要什么、问摘要没写的细节) | **采纳为下一任务设计** | 需要 ask_user 通道(用户模拟器)或需导航才可见的请求载体;设计后先外审。 |
| 效率结果保留但改为"按信息类型分配模态",不另立效率制式;Pareto 图放附录 | **采纳** | 与 17:50 PT 条目一致。 |
| 主图 = 交互项(证据留存 × 源帧/对照),不是三根记忆策略柱 | **采纳** | 待交互实验出数后画。 |

## 提示词原文

```
Public repo pinned at https://github.com/luojiaxuan/CausalCache/blob/9081ec7/ ; read first, cite file+section: rl/docs/ledger_memory_rl_20260904.md (section 5 top), rl/docs/story_20260905.md (2.12-2.13). CLAIM: a GUI agent's need to look back at old screenshots is set by the fidelity of its text history. Same 30 checkpoints, same UI-Venus-2: full think text -> text_only 100%; first-sentence text 87%; action-only text -> text_only 0, recent-2 0, age-matched control turn 0, request+evidence turns 53% [39,68], gold-text 93%. GUI-Owl teacher-forced on the same paths (own one-line conclusions): recent-2 0%, request+evidence 43% [27,59] at equal tokens, gold 100%. GUI-Owl PartMatch natural prefixes 12.5->37.5%. Efficiency: dropping recent-2 under full text saves 46% at 100%; scaled retrieval frames (0.35, ~350 tok/frame) 50% @4.6k vs compressed text 87% @4.5k -> image-for-text fails. ASK: (1) fidelity-axis claim defensible? missing control? (2) is teacher-forcing a valid stand-in for natural one-line-summary agents? (3) main figure + minimal experiment set; (4) top 3 reviewer attacks; (5) drop efficiency regime or reframe as modality allocation? FORMAT: Your reader is an AI agent, not a human. Be terse and information-dense: no preamble, no restating my setup, no hedging boilerplate, no motivational or summary paragraphs. Every sentence must carry a claim, a number, a citation, or an instruction. Prefer numbered claims and compact tables. Keep all technical content; drop everything else.
```
