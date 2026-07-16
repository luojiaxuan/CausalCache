# Restoration v2.2 selector geometry

> 状态：v2 reporting-only repair 已从 clean pushed source 完成；result push 后已通过 clean-descendant
> committed-byte replay。v1 output 未提交。
> OCR/RGB 与 policy-vision feature baselines 仍待补齐。
> 本阶段只消费已经闭合的 train/development raw $D(S)$，不训练 gate、不读取 confirm/test，也不运行
> action policy、matched-NLL 或 closed-loop episode。

## 为什么先做这一步

45-state exact-label run 已证明 restoration utility 存在，但 `B=2` overall recovery 把三个不同问题混在一起：

- decision step 4 只有两个 candidate，`n=2,B=2` 等于恢复全部 candidate，是 ceiling sanity；
- decision step 5 是 `n=3,B=2`；
- decision step 6 是真正的半容量压缩设置 `n=4,B=2`，也是本阶段 primary slice。

此外，出现负 marginal 或非零 pair interaction 并不自动证明 set-conditioned selector 必要。真正需要测的是：

1. 使用真实 conditional marginal 的 greedy 距 exact subset oracle 有多远，即 search gap；
2. 把 coalition-dependent marginal 压成一次性 event score 后又损失多少，即 objective-projection gap。

只有第二个 gap 在 development trajectories 上具有稳定规模，才有证据把 set conditioning 作为主方法；否则应把
方法简化成 independent restoration gate，而不是凭 interaction count 选择更复杂的 student。

## 固定输入与边界

canonical labels 是 private HF dataset
`gavinlaw/causalcache-restoration-labels-mobile@8f6baae5c0b23b08915fa1b0fb848dd519b4c8db` 的
`raw/v2.2-eager-train-dev-exact-v2.tar`：101 files、3,747,840 bytes、SHA256
`99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e`。runner 必须先走完整
`read_label_evidence_archive` validator，再从 420 条 raw distance rows 重算全部 selector；不能从 Git compact
summary 或 artifact 自报的 derived rows反推。

本阶段固定 15 条 trajectory：10 条 `v2_label_train`、5 条 `v2_development`，每条各含 decision step
4/5/6。trajectory 是唯一统计与 bootstrap 单位；420 个 coalition 和 435 条 edge 不是 420/435 个独立样本。
confirm 的 prompt、image、action、policy output 与 AndroidWorld sealed test split 均保持 locked。

OCR/RGB 与 policy-vision similarity 只在 primary `n=4,B=2` slice 上比较。它们需要 immutable derived dataset
`gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@89f136abaff797e14fe758a198996e51032a10a6`：
OCR/RGB 是 CPU feature stage；policy-vision 是单独的 vision-only feature stage，只允许调用冻结 GUI-Owl vision
encoder 的 final main merger，不允许 language model、LM head、`generate`、新 teacher KL 或新 $D(S)$。

## Selector 定义

所有 event cost 在本分析中固定为一个 slot。令：

$$
U_t(S)=D_t(\varnothing)-D_t(S),\qquad |S|\le B.
$$

对每个 `n` 单独报告 `B=0,\ldots,n`；`B=0` 与 `B=n` 是 sanity endpoints。primary 是 `n=4,B=2`，
secondary 是 `n=3,B=2`，另报每条 trajectory 对 n=3/4 等权的 hard-history slice。禁止把 `n=2,B=2`
混入 primary compression claim。

### Exact subset oracle

$$
S^*=\arg\min_{|S|\le B}D_t(S).
$$

允许空集和提前停止。tie 使用 exact float comparison，再按更小 cardinality、lexicographically lower event IDs。
另报 `|S|=\min(B,n)` 的 forced-cardinality oracle，量化非单调性造成的“选满预算”代价。

### True conditional-marginal greedy

从空集开始，每轮用 raw table 计算所有可行候选的真实
$\Delta_j(S)=D(S)-D(S\cup\{j\})$，选择最大者；只有严格正 gain 才加入，tie 取更小 event ID。
它是 offline diagnostic oracle，不是可部署 selector。其与 exact 的差是 search gap。另报不允许提前停止的
forced-fill sensitivity。

### Independent average-score projection

主 comparator 是 exact budget-conditioned average marginal。单位成本下：

$$
a_j^{(B)}=\mathbb E_{|S|=B-1,\,j\notin S}
[D(S)-D(S\cup\{j\})].
$$

每个 state 只算一次固定分数，取严格正的 top-$B$，选择后不重新打分。secondary comparator 使用完整
permutation path 上的 Shapley-style average marginal，再做相同 positive top-$B$。两者都以 raw $U(S)$ 评价，
不能用 selected score sum 替代 actual utility；同时报告 forced-fill sensitivity。

### Heuristic baselines

- `recent`：固定选时间上最后 `min(B,n)` 个 event；
- `random`：解析地平均所有 exact-`min(B,n)` subsets，不做 Monte Carlo；
- OCR/RGB：冻结的 `0.5 * OCR Jaccard + 0.5 * 16^3 joint-RGB-histogram cosine`，primary slice 选满 2；
- policy vision：冻结的 post-state/current final-main-merger mean/L2 embedding cosine，primary slice选满 2。

heuristic 没有 utility abstention signal，因此 primary 定义选满容量；negative recovery 保留，不 clip。视觉基线若
尚未生成，table-only result 必须显式标记 `pending_feature_stage`，不能把它们从 baseline 清单中静默删除。

## 统计与 gap 分解

state recovery 固定为：

$$
r_t(S)=\frac{D_t(\varnothing)-D_t(S)}{D_t(\varnothing)}.
$$

若 denominator 不大于 `1e-12`，该 state 的 normalized metric 记为 `null` 并单列；不能写成 1。所有均值先在
trajectory 内对目标 slice 的 states 等权，再让 trajectory 等权。ratio 使用 trajectory-weighted means 的比：

$$
\rho_A=\frac{\operatorname{mean}_{traj}r_A}
{\operatorname{mean}_{traj}r_{exact}},
$$

不平均逐 state ratio。三个主 gap 是：

$$
g_{search}=r_{exact}-r_{true\ greedy},
$$

$$
g_{projection}=r_{true\ greedy}-r_{independent},
$$

$$
g_{total}=r_{exact}-r_{independent}=g_{search}+g_{projection}.
$$

`g_projection` 是 signed quantity，independent selector 偶然胜过 greedy 时可以为负。每个方法同时报告 selected
cardinality、exact-subset match、Jaccard、trajectory paired win/tie/loss 与 actual recovery。

paired bootstrap 固定 10,000 resamples、seed `271828`、percentile 90% interval。train 和 development 分别按
trajectory 整块有放回采样；overall 只能按 split 分层抽 10+5 后合并。development 只有 5 个 clusters，因此还
必须列出 5 条 trajectory paired delta 与 sign count，interval 只作 method-shaping screen，不能写成统计显著性。

## Interaction 描述

不同 `n` 的 pair-interaction row 数不同，禁止直接按 state total mass 分层。每 state 使用：

$$
m_t=\frac{\operatorname{mean}_{i<j,S}|I_{ij}(S)|}
{\max(D_t(\varnothing),10^{-12})}.
$$

同时报告 negative-marginal flag/rate 与 redundancy share
$\sum\max(-I,0)/\sum|I|$。low/mid/high interaction cutpoints 只从 train、按 `n` 分别取 tertiles，再原样应用到
development；它们只作描述性 strata。判定 set conditioning 是否值得的直接证据仍是 `g_projection`，而不是
interaction 符号或 mass 本身。

## 冻结的内部 method-shaping 规则

这些规则只决定下一步 student/search 设计，不是 paper success gate，也不打开 confirm：

- primary development `g_projection >= 0.05`、至少 4/5 trajectory 同方向且 90% bootstrap lower bound 大于 0：
  保留 set-conditioned gate 为主方法；
- `0.01 < g_projection < 0.05` 或 interval 包含 0：yellow，后续用同架构/同参数量训练 independent 与
  set-conditioned 两版，不预先声称 conditioning 必要；
- `g_projection <= 0.01`：不以 interaction-aware gate 为核心贡献，优先简化为 independent restoration gate；
- true-greedy/exact recovery ratio 至少 0.90 且 `g_search <= 0.05`：greedy 作为充分的默认搜索器；
- ratio 小于 0.85 或 `g_search > 0.10`：在 training/development 上增加已经实现的 swap/beam diagnostic；
  两组阈值之间是 yellow，只保留 stronger-search diagnostic，不自动升级线上算法。

无论哪种结论，gate checkpoint、matched-NLL 与 closed-loop 仍需新的独立 contract；当前 geometry 结果不能
替代 untouched confirm 或 AndroidWorld task success。

## 执行顺序

1. commit/push 本 contract、validator、table-only runner 与 tests；
2. 从 immutable labels fresh projection 运行 CPU table geometry，写入轻量 Git result；
3. commit/push table result 后，冻结并运行 OCR/RGB feature join；
4. 若仍需要 policy-vision baseline，再从 clean source 运行单独的 vision-only feature stage；
5. baseline matrix 完整后才冻结 gate training contract。

任何一步都不得根据新 aggregate 修改 selector、tie、bootstrap、primary slice 或 internal decision rule；若实现契约
有 bug，只能以新的 versioned repair 记录原失败并重新冻结。

table-only formal command 固定为：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.run_restoration_v2_2_selector_geometry run \
  --repository-root /absolute/path/to/CausalCache \
  --contract /absolute/path/to/CausalCache/code/configs/causalcache_restoration_v2_2_selector_geometry.json \
  --labels-archive /fresh/immutable/raw/v2.2-eager-train-dev-exact-v2.tar \
  --source-git-commit <CLEAN_PUSHED_MAIN_SHA> \
  --output-dir /absolute/path/to/CausalCache/data/results/restoration_v2_2_selector_geometry_v1
```

runner 要求当前 `main` clean、`HEAD == origin/main == source_git_commit`，并重新校验 raw archive size/SHA 与完整
101-file label evidence。输出只允许 `README.md`、`summary.json` 和单个 canonical JSONL state-budget shard。
result commit/push 后用同一 argv 把 `run` 改为 `validate`；validator 从 immutable raw table 重算完整 payload，逐
byte 比较三份文件，并要求 formal reducer source 相对 execution commit 没有变化。

## V1 preliminary replay 与 versioned reporting repair

第一次 table replay 从 clean pushed `main@7a1a8cc28a9a8f6a745f372406fe751b9bf4ff53` 读取 immutable label
revision `8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`，101-file label validator 与 180 条 state-budget
records 的 selector 数值均通过。随后 independent contract-completeness audit 发现两个 output-schema 缺口：

1. 冻结 contract 要求联合报告
   `role × n × B × interaction-strength × has-negative-marginal`，旧 reducer 只在 primary `n=4,B=2`
   分别给出 strength 与 negative-marginal 两张边际 gap 表；
2. analytic exact-cardinality random 已知精确 cardinality，并可解析计算 exact-match probability 与 expected
   Jaccard；旧 compact reducer 却把这些字段省略。

这两个缺口不改变任何 selector coalition、utility、recovery、bootstrap 或 method-shaping 数值，但属于预冻结
reporting contract 没有完整实现。按预先写下的 bug 边界，旧 v1 output 不进入 Git source of truth，也不能静默
覆盖。repair 使用独立 protocol/config/module/output identity，必须显式物化 train/development 共 144 个联合 cells
（包括统一 empty schema），并为 random 报告解析集合几何；正式 v2 result 仍须从 clean pushed source 重新生成、
逐 byte replay，并再次接受 implementation-independent coverage audit。

repair machine-readable config 为
[`../code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json`](../code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json)，
SHA256 `2d312f54559f67aafe7efec2656d23171000e8b0f41d2d3c923f6a3c8b43be4c`。source-only validation：

```bash
cd code
python3 -m scripts.validate_restoration_v2_2_selector_geometry_v2_contract \
  --contract configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json \
  --repository-root ..
```

primary `n=4,B=2` 的 trajectory-weighted normalized recovery 为：

| Selector | Train | Development | Overall |
| --- | ---: | ---: | ---: |
| Exact subset | 0.877376 | 0.925342 | 0.893364 |
| True conditional greedy | 0.877376 | 0.925342 | 0.893364 |
| Budget-conditioned independent | 0.835977 | 0.766303 | 0.812753 |
| Full-path Shapley independent | 0.815367 | 0.884128 | 0.838287 |
| Recent | 0.554136 | 0.019666 | 0.375979 |
| Random exact expectation | 0.430032 | 0.139471 | 0.333178 |

true greedy 在 primary 15/15 states 与 exact subset 的 coalition 完全一致，因此 train/development/overall search
gap 都是 0。budget-conditioned independent 只在 7/15 states 命中 exact coalition，development 是 0/5；其
development objective-projection gap 为 `0.159039`，5/5 trajectories 同方向，冻结的 90% trajectory-bootstrap
interval 为 `[0.007836, 0.355867]`。按预先冻结的规则，method-shaping 输出
`set_conditioned_main_candidate + online_greedy_sufficient`。

这个 green 结论必须保留两个 caveat：

1. development 只有 5 条 trajectories；normalized gap 的 74.1% 来自一个
   `D(empty)=0.000616` 的 state。删除它后其余 4 条仍为 4/4 同方向、mean gap `0.05152`，但这只能作
   sensitivity，不能包装成显著性或普遍稳定效应；
2. 更强的 static comparator 会缩小差距。full-path Shapley 与 forced-fill budget-conditioned independent 在
   development 都达到 `0.884128`，相对 greedy 的 normalized gap 是 `0.041214`；预算条件 independent 的
   primary raw-utility gap 也只有 `0.0021944`。因此本结果支持“set-conditioned student 值得进入主候选”，
   尚不能证明任意 independent gate 必然明显更差。

secondary `n=3,B=2` 的 independent/recent/random normalized mean 会被一个极小 `D(empty)` state 放大为负值。
独立 raw-table audit 证明这不是实现错误：该 state 的所有 singleton/pair 都比 empty 差，exact/greedy 正确停在
空集；但在 singleton bases 上平均的 conditional event scores 都为正，static independent 因而选入负 utility
pair。这是 objective projection 的真实 pathology，同时也是报告 raw utility 与逐 trajectory delta 的理由，不能
概括为所有 n=3 states 普遍崩溃。

旧 v1 核心表由两个不 import 新 selector/result reducer 的独立实现重算：primary 225 个 state-level fields、
15 个 aggregate fields 与全部 selected coalitions 的 mismatch 都为 0；第三个审计独立复算了 180 rows、
budget grid、paired bootstrap 与 method-shaping，数值也完全一致。上述 primary 数字已用于 repair 前后
invariance check；v2 repaired artifact commit/push 后又通过 clean-descendant validation。新 policy forward、
generation、teacher/KL、gate、matched-NLL、closed-loop、confirm/test 和 policy-vision forward 仍全为 0。

## V2 repaired formal result

canonical result 位于
[`../data/results/restoration_v2_2_selector_geometry_v2_repair/`](../data/results/restoration_v2_2_selector_geometry_v2_repair/)，
execution source 为 `9a4eca5a53c2a9a3340c6274b9fa5ff9012a5a64`。complete payload SHA256 是
`cc505443a7efdc68c8eeca754f24c9143cabf72a090f024fde05011e783cbb21`；180-row JSONL SHA256 是
`b3f67714bb5667ceda945a3cb953b8987108aef607d5819a617272d48450cb03`。144 个联合 cells 已全部物化，
83 nonempty / 61 empty；random 在 primary overall 的 selected cardinality、exact-match probability 与 expected
Jaccard 分别是 `2.0 / 0.155556 / 0.379630`（development 为 `2.0 / 0.166667 / 0.388889`）。

primary selector recovery、search/objective-projection gap 与旧 v1 independent audits 完全一致；冻结
method-shaping 仍为 `set_conditioned_main_candidate + online_greedy_sufficient`。该决定只授权下一步把
set-conditioned student 放入主候选，不授权 gate training、confirm 或 paper claim。development 小样本、
small-`D(empty)` normalized amplification 和 stronger static comparator caveat 均继续适用。

result commit `d0f25d812869d5fc7b58284a25abe3aa8049b0aa` push 后，clean descendant validator 从 immutable raw
labels 重建并逐 byte 验证三份 files，返回 `VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR`。
