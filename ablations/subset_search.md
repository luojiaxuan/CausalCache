# Subset Search Ablation

> 状态：首次 CPU attempt 在 pre-commit JSON round-trip replay fail closed；serialization fix 待 push 后 canonical rerun。
>
> 范围：synthetic implementation validation，加上旧 v1 selection-biased development coalition table 的
> post-hoc replay。零新 policy/GPU operation；不训练 gate、不读取 confirm、不修改
> `NO_GO_V2_1_FULL_45_SUBSTRATE`。

## 要回答的问题

Interaction-aware attribution 与 subset optimization 是两件事。即使 teacher 能给出真实 conditional marginal，
greedy 也未必得到最优 subset；反过来，即使 additive-score knapsack 被精确求解，它优化的也可能只是 averaged
event score surrogate，而不是真实 restoration utility。

本 ablation 固定三个正交轴：

1. **Value source**：真实 set utility、真实 conditional marginal、averaged event attribution 或未来 learned
   surrogate；
2. **Search algorithm**：exact、greedy、local exchange、beam；
3. **Evidence source**：controlled synthetic、既有 phase-0 synthetic、旧 v1 cached policy table。

不能把某个 value source 上的优化误差写成另一个 value source 上的 search gap。

## 统一目标与 exact oracle

令：

$$
U(S)=D(\varnothing)-D(S),\qquad
\mathcal F_B=\left\{S\subseteq E:\sum_{j\in S}c_j\le B\right\}.
$$

Exact subset oracle 定义为：

$$
S^*=\arg\max_{S\in\mathcal F_B}U(S).
$$

空集始终可选，因为 restoration utility 不保证 monotone。若 utility 在数值容差内相同，统一按更低 cost、更小
cardinality、lexicographically smaller sorted event IDs 决胜。

本项目的 $B$ 是 visual-token budget，不是 event 个数。一般 exact 查询数是 $|\mathcal F_B|$；只有所有 event
成本相等时，令 $b=\lfloor B/c\rfloor$，才能写为：

$$
Q_{\mathrm{exact}}=\sum_{k=0}^{b}\binom nk.
$$

因此 exact 只在短历史、小 visual budget 的离线 analysis states 上使用。对允许任意 interaction 的 black-box
$U(S)$，没有额外可加性、submodularity 或 admissible bound 时，不声称存在通用 polynomial-time exact method。
lazy greedy 需要 submodularity；branch-and-bound、A* 或 MILP 的效率也依赖额外结构。搜索器是可替换模块，
不是 CausalCache 的核心 novelty。

## 冻结的近似搜索

### True conditional-marginal greedy

每轮对所有 feasible candidate 计算：

$$
\Delta_j(S)=U(S\cup\{j\})-U(S).
$$

raw greedy 最大化 $\Delta_j(S)$；variable-cost density greedy 最大化 $\Delta_j(S)/c_j$，再以 raw gain、较低
cost、较小 event ID 决胜。最终被选 candidate 的 raw gain 不大于冻结 stopping threshold 时停止。equal-cost
scenario 中两者严格相同，因此结果矩阵省略 density，不能把同一个算法算成两条 ablation。

若 raw greedy 最终选入 $r$ 个 event，查询数会包含空集和每轮全部 feasible candidates，而不是只数最终路径。
equal-cost、选满 $b$ 时上界为：

$$
Q_{\mathrm{greedy}}
=1+bn-\frac{b(b-1)}2.
$$

### Greedy + bounded local exchange

从 true raw-greedy seed 出发，每轮完整评估：

$$
\mathcal N_{p,q}(S)=left\{
(S\setminus A)\cup C:
|A|\le p, |C|\le q,
(S\setminus A)\cup C\in\mathcal F_B
\right\}.
$$

v1 固定 $p=q=2$、最多 8 passes，采取 best total-order improving move；utility 在容差内相同时只允许向更低
cost/cardinality/lexicographic key 移动。查询成本包含 seed greedy、所有 evaluated-but-not-
selected neighbors 和最后一次无改进 pass。对 equal-cost、满预算的互补 pair，1-out/2-in 通常不可行，必须允许
2-out/2-in；但在 $N=4,b=2$ 时，这个 neighborhood 已接近 exact，只能作为 optimizer diagnostic，不能包装成
可扩展的新算法。

### True-utility beam search

从空集开始逐层扩展所有 one-event children，按 coalition 去重并用真实 $U(S)$ 排序，只保留 width
$w\in\{2,4,8\}$。为了能越过 pure-complementarity 的零/负 singleton prefix，v1 固定允许 non-positive
prefix；最终 incumbent 仍与空集比较。查询数统计所有 unique evaluated children，另报 sequential rounds。

Beam/local 若用真实 $U$，一次 score 就意味着一次 frozen-policy rerun，所以它们只是 offline oracle-search
diagnostic。learned conditional-marginal head 也不能直接无定义地用于 beam 或 removal：同一个 set 的累计预测可能
依赖加入顺序，removal 没有直接 set score。未来若做 learned beam/local，必须先冻结 direct set-utility head 或
明确的 path-score/dedup contract。

## 两个 gap，不能混称

既有 phase-0 先对 exact averaged marginal $G_j$ 做 positive-value knapsack。这个 knapsack 已精确优化 additive
surrogate；其 utility 只达到 global subset optimum 的 85.9%，是：

> interaction / objective-projection gap

而不是 greedy search gap。只有使用真实 conditional marginal 的 greedy 与 exact subset 的差才定义 search
regret：

$$
R_{\mathrm{search}}=U(S^*)-U(S^U_A).
$$

未来若同一 search algorithm $A$ 使用 learned score，完整分解是：

$$
U(S^*)-U(S^{\hat U}_A)
=
\underbrace{U(S^*)-U(S^U_A)}_{\text{search regret}}
+
\underbrace{U(S^U_A)-U(S^{\hat U}_A)}_{\text{signed student residual}}.
$$

第二项可能为负，因为 learned error 可能偶然避开 greedy trap，所以不能无条件称为非负 “distillation gap”。

## 指标与 query accounting

当 $U(S^*)>\epsilon$ 时报告：

$$
\rho(S)=\frac{U(S)}{U(S^*)},
$$

以及 actual utility、absolute regret、unique set-utility evaluations、candidate evaluations、sequential rounds。
若 oracle utility 不为正，ratio 记为 `null` 并单独标记，不能人为填 1 混入均值。selected score sum 不能替代
actual $U(S)$。

正式结果还单列：

- exact averaged-attribution knapsack 的 teacher distance evaluations；
- interaction at empty 的 positive/negative pair counts 与 mass；
- deterministic $n=8/12/16/24,b=3$ scale sweep 的 query count；
- 旧 real table 的 cached lookup 数，但明确不把本地 CSV wall time当作 VLM latency。

## 输入与运行边界

machine-readable config：
[`../code/configs/subset_search_ablation_v1.json`](../code/configs/subset_search_ablation_v1.json)，当前 SHA256
`161518c3e951829528b63b9146878236d918d4d620b4a6acbecc53f7c4db921f`。它绑定：

- additive、redundant、complementary trap、mixed/non-monotone、variable-cost 五个 controlled fixtures；
- 既有 `synthetic_phase0.json`；
- 旧 v1 `coalition_distances.csv`，SHA256
  `3679d990166c6907e20f1eb141152fee221fcb2bf72f5392f6c8210a505dbed2`；
- 对应 compact summary，SHA256
  `8180c749a4de11208f780c0bc0d2f1739e6d375280de2b44c595b71800261489`。

旧 table 只含两个事后挑选的 Qwen policy development states：step 4 的 3 个 events 与 step 8 的 7 个
events；每 event 成本 476，完整覆盖 budget 512 的 singleton 与 budget 1024 的 pair。它可以无新 GPU forward
重放 exact/greedy/exchange/beam，但仍是 selection-biased、最多选两个 event、没有 learned gate 或 terminal
success 的 implementation smoke。

搜索 stopping threshold 固定为纯数值 0；旧 state sensitivity threshold `1e-4` 只作单列 abstention sensitivity。
二者不能静默混用，否则 step 4 budget 1024 会把小而正的第二步 marginal 错记成 search failure。

正式 run 必须从 clean pushed `main` 使用：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.run_subset_search_ablation \
  --repository-root /absolute/path/to/CausalCache \
  --config /absolute/path/to/CausalCache/code/configs/subset_search_ablation_v1.json \
  --source-git-commit <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --output-dir /absolute/path/to/CausalCache/data/results/subset_search_ablation_v1
```

本次结果是轻量 deterministic summaries，进入 Git 即足够；没有新的 reusable dataset/model，不创建多余 HF
repo。正式结果回写并 push 后，再由独立 CLI 对 committed summary、source commit/blob、input hashes 与完整
scientific payload 复算。

## 首次 CPU attempt：artifact validation fail closed（2026-07-15）

首次 attempt 从 clean pushed `main@d5cd389f6a7b2720cc1daaa3373cefa52480cf7b` 运行 14 个 scenarios，CPU
wall time 51.668 秒，policy/GPU operation 均为 0。runner 写出后，pre-commit 独立复算发现 scientific values
一致，但 Python in-memory trace 中的 `coalition` 是 tuple，JSON round-trip 后变成 list；validator 使用严格
payload equality，因此按设计 fail closed。该 attempt 的 result 目录已删除，没有 commit、push 或升级为正式证据。

已在 `_trace` serialization boundary 显式转换 list，并加入 `json.loads(json.dumps(result)) == result` regression。
必须先把该 source fix commit/push，再从新的 clean source 重新生成；不能沿用旧 output 或只放宽 validator。

以下只记录 failed attempt 的 provisional diagnostics，canonical rerun 前不能称为 formal result：

- 既有 phase-0 中，exact averaged-marginal knapsack / exact subset ratio 为 `0.858594`，而 true conditional
  greedy 为 `1.0`。因此这里的已知缺口确实来自 value projection，不是 greedy optimization；
- controlled complementary trap 中 raw greedy/beam-2 ratio 为 `0.5`，2x2 exchange 与 beam-4/8 为 `1.0`；
  controlled mixed/non-monotone 中 raw greedy 为 `0.666667`，exchange/beam-4/8 为 `1.0`；
- variable-cost case 中 raw greedy 为 `0.833333`，density greedy 为 `1.0`，证明 cost-aware rule 必须先冻结；
- scale sweep 的 exact query 数随 $n=8/12/16/24,b=3$ 为 `93/299/697/2325`。raw greedy 只用
  `22/34/46/70` queries，ratio 为 `1.0/0.95098/0.95098/0.970588`；beam-4 用
  `51/88/125/197` queries 在该 generator 上都找到 exact。2x2 exchange 也都 exact，但到 $n=24$ 使用
  `1185` queries，明显高于 beam-4；这些只描述该冻结 generator，不构成通用 guarantee；
- 旧 real cached table 的 step 4/8 × budget 512/1024 共四个 cases 中，true greedy、exchange、beam 与 exact
  全部选同一 coalition，greedy/exact ratio 均为 `1.0`。step 8 / 1024 虽有 2 个正、19 个负 pair
  interactions，仍未观察到 real greedy trap；这两个事后挑选 states 只能作 consistency smoke；
- 若把旧 state sensitivity threshold `1e-4` 混作 search stopping threshold，step 4 / 1024 会从 exact
  `[1,3]` 提前停在 `[3]`，ratio `0.944896`。这属于 abstention-threshold sensitivity，不是 search regret。

若 canonical rerun 逐项复现，最稳妥的方法决策仍是：主线上使用 set-conditioned greedy；exact 只作小规模 ceiling；beam/local 是
可替换的 offline diagnostics。synthetic 已证明 stronger search 可能修复 greedy trap，但旧 real table 没有证明
它值得增加线上复杂度。只有未来 development data 出现稳定真实 search regret，且 direct set-utility contract
闭合后，才考虑 learned beam/local。

## 对论文的安全表述

> Exact enumeration defines a small-scale offline ceiling for the frozen restoration utility. Deployment uses a
> conditional greedy selector; local exchange and beam search are reported only as replaceable offline search
> diagnostics. We separately report the objective-projection gap from averaging coalition-dependent values and the
> search regret of conditional greedy under the true set utility.

不能声称 greedy 有近似保证、stronger search 已在线部署、exact subset 对 terminal success 最优，或旧两个 real
states 构成 paper-level generalization evidence。
