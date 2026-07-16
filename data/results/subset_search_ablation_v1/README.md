# Subset-search ablation v1

> 这是 CPU-only optimizer diagnostic 与旧 v1 development table 的 post-hoc replay；不是新的 policy evidence，
> 不评估 learned gate/closed-loop success，也不改变 v2.1 的 frozen NO-GO。

## 核心结果

- Phase-0 average-marginal knapsack / exact subset：`0.858594`；这是 objective-projection gap，不是 greedy search gap。
- Phase-0 true conditional greedy / exact subset：`1.000000`。
- Complementary trap：raw greedy 选择 `[0, 1]`，utility ratio `0.500`；2x2 exchange 选择 `[2, 3]`。
- Beam-2 / Beam-4 ratio：`0.500` / `1.000`。
- Mixed/non-monotone：raw greedy ratio `0.667`，2x2 exchange 与 beam-4 均为 `1.000`。
- Variable-cost：raw greedy ratio `0.833`，density greedy 为 `1.000`，说明 cost rule 必须预先冻结。

## Deterministic scale sweep

| Events / slots | Exact queries | Raw greedy ratio / queries | Exchange ratio / queries | Beam-2 ratio / queries | Beam-4 ratio / queries | Beam-8 ratio / queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 / 3 | 93 | 1.000 / 22 | 1.000 / 73 | 1.000 / 33 | 1.000 / 51 | 1.000 / 69 |
| 12 / 3 | 299 | 0.951 / 34 | 1.000 / 215 | 0.951 / 53 | 1.000 / 88 | 1.000 / 139 |
| 16 / 3 | 697 | 0.951 / 46 | 1.000 / 411 | 0.951 / 73 | 1.000 / 125 | 1.000 / 209 |
| 24 / 3 | 2,325 | 0.971 / 70 | 1.000 / 1,185 | 0.971 / 113 | 1.000 / 197 | 1.000 / 347 |

在这一个冻结 generator 上，beam-4 用 197/2,325 的 unique utility queries 找到 $n=24$ exact optimum；这只是
query-cost diagnostic，不是通用 guarantee。2x2 exchange 也恢复 exact，但需要 1,185 queries，成本明显更高。

## 旧真实 policy coalition-table replay

| State | Budget | Exact subset | True greedy | Greedy / exact | Exact queries | Greedy queries |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 4 | 512 | `[3]` | `[3]` | 1.000 | 4 | 4 |
| 4 | 1024 | `[1, 3]` | `[1, 3]` | 1.000 | 7 | 6 |
| 8 | 512 | `[1]` | `[1]` | 1.000 | 8 | 8 |
| 8 | 1024 | `[1, 7]` | `[1, 7]` | 1.000 | 29 | 14 |

四个缓存 state/budget 上 true greedy 都等于 exact。样本只有两个 selection-biased states、成本相等且最多选两个 event，
所以这只能验证 replay/search implementation 一致性，不能证明 stronger search 在真实任务上优于 greedy。

step 8 / budget 1024 在空集处有 2 个正、19 个负二阶 interactions，但仍未形成 greedy trap。若把旧 state
sensitivity threshold `1e-4` 错当 search stopping threshold，step 4 / budget 1024 会提前停在 `[3]`，ratio
降到 `0.944896`；这是 abstention-threshold sensitivity，不是 interaction search regret。

## 成本与声明边界

- `unique_set_utility_evaluations` 计入所有被评估但未采用的 coalition；exchange 还包含 seed greedy 成本。
- true-U greedy/exchange/beam 的一次 query 在真实部署中意味着一次 policy rerun；本地 CSV lookup wall time 不是线上 latency。
- equal-cost scenario 省略 density greedy，因为它与 raw greedy 完全相同。
- learned conditional-marginal head 不能直接无定义地用于 beam/removal；需要独立的 direct set-utility 或 path-score contract。
- Exact 只保证冻结 single-step restoration utility 最优，不保证 terminal task success 最优。

## Provenance

- Source Git commit：`7569ce2ac1e63f565be4e0d4dcc9626285aa355c`
- Scientific payload SHA256：`26846d509d421dcb49f2d1554598893a65829c6a25610ef399ce69b383274edf`
- Device：CPU；GPU/policy operations：`0`
