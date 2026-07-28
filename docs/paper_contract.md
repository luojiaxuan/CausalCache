# CausalCache 论文工作合同

状态：`WORKING`

> 本文件记录持续更新的论文状态，不冻结 Git revision。只有用户明确要求
> submission snapshot 时，才切换到冻结流程。

## 本次观测状态

- `observed_at`: `2026-07-27`
- `branch`: `main`
- `Git remote`: `https://github.com/luojiaxuan/CausalCache`
- `revision policy`: 精确 artifact revision 以本文件及对应源码的 Git history
  为准；working paper 不额外冻结 snapshot
- `AAAI template revision`: AuthorKit27，hash 见 `paper/README.md`
- `main PDF page count`: 9；正文不超过 7 页，references 在第 7 页自然开始，
  第 8--9 页仅 references
- `supplement status`: 独立 2 页 PDF，working，未冻结
- `paper sources`: `paper/main.tex`、`paper/supplement.tex`
- `build command`: `make paper-all`
- `outputs`: `output/pdf/causalcache_aaai27.pdf`、
  `output/pdf/causalcache_aaai27_supplement.pdf`
- `link policy`: 两份 PDF 均无 embedded links/bookmarks；AAAI-27 禁止
  `hyperref`/`navigator`

## 一句话 identity

CausalCache 在完整 event-summary trace 上、固定 active history-image budget
`B` 下，学习哪些已总结事件值得恢复为 summary-plus-archived-image，并用 HGKV
让 policy 选择性消费这些重新挂接的真实像素。

## 主张边界

- 主张：对象是 event fidelity，不是 event inclusion；主比较是 matched-`B`
  fidelity reallocation。
- 不主张：旧截图通常优于近期截图，不主张加图带来收益，也不把一个 qualitative
  episode 当作总体效果。
- `causal` 的精确定义：固定 policy 与剩余 prompt 时，对 archived screenshot
  restoration 做受控 intervention；不声称环境结构因果。
- offline utility 与 closed-loop success 的关系：teacher-forced likelihood 只作
  训练/离线 surrogate，闭环 task success 是最终裁决。
- `k=|S\setminus Recent-B|` 是 realized statistic，不是训练或绘图预设。
- CausalCache-P 是 proposal-conditioned 默认路径；CausalCache-LA 是效率分支。

## Headline claims

| ID | Claim | Status | Main evidence | Allowed strength |
|---|---|---|---|---|
| C1 | CausalCache 在 complete summary trace 上重分配相同的 `B` 张 active history images。 | confirmed | Figure 1；protocol identity | establishes |
| C2 | HGKV 在 desktop DiD gate 上产生选择性且保持 recent/wrong drift cap。 | confirmed | Table 1；`desktop_did_policy_v1` | supports |
| C3 | Desktop-only CausalCache-P 在 MobileWorld 全 roster 上取得最高均值，但当前整体 paired CI 跨零。 | inconclusive | closed-loop table；三轮聚合 | descriptive |
| C4 | 改善集中在 construction-defined memory-critical split。 | confirmed | `memory_critical_split.json` | supports |
| C5 | Cart 案例展示 event-level pixel restoration 的实际机制。 | confirmed | Figure 2；`qualitative_cart_audit.json` | descriptive |

完整原子 claim、统计边界和 revision 见 `docs/paper_claim_evidence.tsv`。

## 当前待补实验或资料

| Experiment / item | Question | Planned location | Blocking? | Status |
|---|---|---|---|---|
| MobileWorld canonical citation 与最终 roster hash | benchmark 的正式出处和 denominator provenance 是否完整？ | Related Work / Setup | freeze 前 blocking | pending |
| 正文 draft marker 清理 | `Pending` 是否都有最终证据？ | 全文 | freeze 前 blocking | pending |
| AAAI reproducibility checklist | 独立 checklist 是否完整？ | 独立上传件 | freeze 前 blocking | pending |

## 七页预算

| Block | Target pages | Current placement | Action |
|---|---:|---|---|
| Title/Abstract/Introduction | 1.35 | page 1 | 保持 |
| Figure 1/Related Work/Problem | 1.65 | pages 2--3 | Figure 1 已固定 page 2 top |
| Method | 1.55 | pages 3--4 | 压缩时优先删重复实现说明 |
| Experimental Setup/Results | 2.20 | pages 4--6 | 保留 matched-budget 与主统计 |
| Figure 2/Ablations/Conclusion | 1.25 | pages 6--7 | Figure 2 保持机制例证，不扩写 gallery |

## 主图主表

| Artifact | Question answered | Claim IDs | Status |
|---|---|---|---|
| Figure 1 | 方法究竟控制 event inclusion 还是 fixed-`B` event fidelity？ | C1 | complete；page 2 top |
| Figure 2 | 一次真实闭环状态里，restoration 怎样改变可见证据和动作？ | C5 | complete；page 6 |
| Table 1 | HGKV 是否在 drift envelope 内产生 selectivity？ | C2 | confirmed |
| Table 2 | fixed-budget offline/closed-loop 主结果是什么？ | C1/C3/C4 | working |

## Supplement inventory

| Section | Content | Main-paper pointer | Status |
|---|---|---|---|
| A | fixed-budget protocol、完整 per-budget gate 与 selector ablation | Method / Results | complete for current working results |
| B | MobileWorld per-round/split、OSWorld paired statistics 与 cost | Results | complete for current working results |
| C | 额外 qualitative/failure gallery | Figure 2 | pending |
| D | prompts、schemas、artifact manifests | Reproducibility | pending |

## Open decisions

- submission freeze 前决定 Figure 2 留在 main 还是移入 supplement；当前保留在 main，
  因为它直接澄清 “summary 一直存在、像素只在 promotion 时恢复”。
- 新科学结果进入正文后继续保持 `WORKING` 并更新 observed revision。
- 只有用户明确授权 freeze 后，才能改为 `FROZEN_SUBMISSION_CANDIDATE`。
