# Long-horizon development v1

> 归档终态：`SUPERSEDED_BEFORE_SCIENTIFIC_SCORING`。本协议已停止，禁止创建 runner-C 或继续运行旧
> independent / conditional / v4 residual 的 restoration、GO、matched-NLL 与 closed-loop。修复后的
> Source-A/Selection-B、24/18 split、24 条 development substrate、OCR 与读取/验证代码保留供后续独立研究复用。
> 终止前完成的 576-record label-blind selector seal 只作 forensic archive，不构成科学结果。

## 归档终态

- repaired Source-A：`90287134630eb2407582e547c246c93b9fccd55e`；repaired Selection-B：
  `bc611552d65e1348c6d68475da0e4a4b97c34850`；contract SHA256：
  `82965790bf541dea348997ecd1849473d5cfb89f53da9f5a615568f57ca67a96`；
- 24 条 development / 18 条 reserve 的 selection 已冻结。development 侧 408 events、48 states、576
  candidate occurrences、432 image/OCR records 已构建并 replay；reserve semantic access 为 0；
- repaired substrate 已由 private HF revision
  `d237271e3266a72cce7aa730d0708936f1552365` immutable fresh-readback，并完成第二次 432-record OCR/raw replay；
- 用户终止消息到达前，576-record label-blind seal 已本地构建且 validator 返回
  `PASSED_LONG_HORIZON_LABEL_BLIND_SELECTOR_SEAL_VALIDATION`。这表示 preparation bytes 自洽，不表示旧 selector
  假设得到支持；
- archive payload 已上传 private HF revision
  `5800f150f34d454ca72ae3eaeee3f30f564d834e`，tag 为
  `long-horizon-development-v1-superseded-archive`。从该 immutable revision fresh-download 的 4 个文件与 staging
  bytes 逐 byte 一致；
- restoration distance、restoration label、policy generation、teacher forward、GPU KL、matched-NLL、
  closed-loop 与科学 GO/NO-GO verdict 均为 0。没有生成 long-horizon 科学结果，也没有打开 sealed test；
- 本文余下内容保留为历史协议说明，不再代表待执行计划。任何新方法必须在独立 task/branch/worktree 中另立
  数据合同、untouched split 与评测权限，不能沿用本文 GO 入口。

机器可读终止清单、HF revision 与 byte roster 见
[`data/results/long_horizon_development_v1_archive/`](../data/results/long_horizon_development_v1_archive/)。

## 目标与边界

这是一条独立的 development-only study，用来回答现有 `n<=4,B=2` exact analysis 无法回答的问题：

1. 当候选历史扩展到 `n=8/16` 时，restoration-guided independent gate 是否比 recent 与
   OCR/RGB similarity 更有优势；
2. v1 iterative conditional 与 v4 frozen-base residual 是否能在长历史中利用 interaction headroom；
3. `B=2/4` 下的 utility、oracle gap、选择延迟与压缩率如何变化。

它不重解释已完成的 `NO_GO_INDEPENDENT_CONFIRM`，也不打开原先锁定的 confirm-20、matched-NLL、
AndroidWorld paired closed-loop 或 sealed test。development 通过只能授权另立新的 closed-loop Source-A；
不能直接运行旧协议。

## 为什么需要 `n=8/16`

项目定义中，最近 event 的 post-state 与 current observation 重复，因此 zero-based decision index `i` 的
可恢复候选数为 `n=max(0,i-1)`。对既有 AndroidWorld early-stopped traces 的只读审计显示：

| stack | `n>=8` decision incidence | `n>=16` decision incidence |
| --- | ---: | ---: |
| GUI-Owl-1.5-8B-Instruct | 179/496 = 36.09% | 64/496 = 12.90% |
| GUI-Owl-1.5-8B-Think | 257/513 = 50.10% | 160/513 = 31.19% |

这些是当前 policy stack 的 early-stopped observed incidence，不是 AndroidWorld benchmark prevalence。完整
provenance 与复现入口见 [`data/results/long_horizon_incidence_v1/`](../data/results/long_horizon_incidence_v1/)。

## Source-A 数据合同

- source pool：沿用 `independent_reference_gate_v1` 固定的 GUIOdyssey 前 16 个 Parquet shards；
- structural eligibility：完整继承原 validator，只把 decision count 改为 `[18,60]`；
- 期望 exact eligible pool：42 trajectories；多一条或少一条都判为 invalid；
- 顺序：`sha256(salt + NUL + source_id)`，再按 source ID、shard path、row index 稳定 tie-break；
- 前 24 条为 development，后 18 条为 unopened reserve；不 padding、不拼接、不输出后 top-up；Parquet
  reader 可能读取同时包含两类 locator 的物理 row group，但 reserve row 不会被转换为 Python object、返回给
  builder 或做 semantic inspection，这一 transport overread 会在 manifest 中显式记录；
- historical role manifests 中出现过的 trajectory 必须与本次 42 条零交集；
- reserve 只允许保留 selection locator/hash metadata，不生成 state、不跑 OCR、policy、selector 或 restoration。

每条 development trajectory 固定两个 prefix state：

| decision step | candidates | current-equivalent event | 用途 |
| ---: | --- | ---: | --- |
| 10 | events `1..8` | 9 | exact long-history analysis |
| 18 | events `1..16` | 17 | scalable pairwise deployment analysis |

历史是原 trajectory 的严格未拼接 prefix。24 条 trajectory 形成 48 states、432 个 observation image members、
408 events 与 576 candidate occurrences。

## Policy context 可行性

冻结 policy 为 GUI-Owl-1.5-8B-Instruct v2.2 eager BF16 official-tools stack，文本 context 上限为 32,768。
在同一 processor profile 下已观察到：

- 8 个 full-fidelity history images + current image：23,631 tokens，可行；
- 16 个 full-fidelity history images + current image：44,320 tokens，超过上限。

因此不能把 `n=16` 包装成 full-history exact oracle。正式执行还必须对每个真实 request 做 processor-only
length check；任何超限 state/pair 原位记为 invalid，不能替换 trajectory。该 check 要求 prompt token 加冻结的
256-token generation/teacher suffix reserve 后不超过 32,768，而不是只检查 prompt 本身。

## Label-blind selectors

所有 selector sets 必须在任何新 restoration generation、teacher forward 或 `D(S)` 访问之前序列化成
canonical JSON seal，并绑定 byte SHA256。

正式 seal 同时保存两个预算，唯一键为 `(budget, selector, state)`，不能让同一 selector/state 的 `B=2`
与 `B=4` 记录互相覆盖。48 个 states 的固定分母为
`48 x (7 B2 arms + 5 B4 arms) = 576` 条 selection records。

`B=2`：

- restoration independent gate；
- dynamic recent；
- OCR/RGB v2 similarity；
- v1 iterative conditional gate；
- v4 safe frozen-base residual；
- deterministic random；
- summary-only。

`B=4`：

- restoration independent gate；
- dynamic recent；
- OCR/RGB v2 similarity；
- deterministic random；
- summary-only。

v1 conditional 与 v4 residual 的训练合同都只定义 `B=2`，因此禁止把它们无定义地外推到 `B=4`。
formal-58 independent/conditional checkpoints与 v4 residual-only checkpoints均按 immutable HF revision、
file size 与 SHA256 读取；本 study 不训练或修改任何 selector 参数。

v4 输入必须经过两层验证：先以 Source-A 的 `manifest_path/manifest_sha256` 验证 canonical outer
`manifest.json`，再由其 file roster 绑定 `label-blind-seal.json`，并以 Source-A 的独立 seal path/SHA
复核。outer manifest、inner seal 与实际 metadata/checkpoint bytes 必须三方一致；seal 还必须保持 label、
confirm、policy、GPU access counters 全为 0。旧 lineage 错把 outer SHA 直接用于 inner seal，首次加载即
fail closed，因此旧 substrate revision 只保留 forensic evidence，不能进入本次 formal runner。

## `n=8`：exact at-most-budget evaluation

完整 8-event history定义 reference behavior。reference canonical action 必须独立生成两次、两次都 strict
parse 且 canonical action 完全一致。对 valid state 计算：

\[
D(S)=\sum_l w_l\,\mathrm{KL}(q_l\Vert q_l^S),
\]

其中 `q` 来自 full 8-event reference，候选请求按相同 teacher action 完整 rerun。`B=4` 表需要全部：

\[
\sum_{k=0}^{4}{8\choose k}=163
\]

个 feasible coalitions；`B=2` 的 37 个 coalitions 是同一表的严格子集。由此可给出 at-most-`B` exact
subset oracle，而不是 greedy proxy。utility 与 normalization 固定为：

\[
U(S)=D(\varnothing)-D(S),\qquad
\bar U(S)=
\begin{cases}
U(S)/D(\varnothing), & D(\varnothing)>10^{-12},\\
0, & \text{otherwise}.
\end{cases}
\]

小 denominator state 不删除。negative utility、non-monotonicity 与 empty-set oracle 都原样保留。

## `n=16`：selector-pair union reference

对每个预注册 comparator pair `(A,B)`，先从 label-blind seal 取固定 sets `M_A,M_B`，再定义：

\[
R=M_A\cup M_B.
\]

policy 只在 `R` 上生成该 pair 自己的 reference behavior，然后测 `D_R(empty)`、`D_R(M_A)` 与
`D_R(M_B)`；`D_R(R)=0` 是同一 reference 的 self row。每个 pair 都是独立实验单元，不能跨 pair 合并成
shortlist、不能枚举额外 set、不能声称 exact oracle。`B=2` 时最多 5 images，`B=4` 时最多 9 images，
均含 current image。

预注册 pairs：

- `B=2`：independent–recent、independent–OCR/RGB、conditional–independent、v4-safe–independent；
- `B=4`：independent–recent、independent–OCR/RGB。

pair 内 raw/normalized delta 使用同一个 `q_R` 与 `D_R(empty)`，因此可以公平比较；不同 pair 的绝对
utility 不直接互相比大小。

## Validity 与 routing

- `n=8` 至少 18/24 trajectories 通过双生成 parse/agreement；
- 每个 `n=16` comparator pair 至少 18/24 通过；
- 低于任何 coverage threshold 时输出
  `NO_GO_INSUFFICIENT_LONG_HORIZON_POLICY_COVERAGE`，不补样本；
- coverage 通过但仍有 invalid state 时，所有 GO metric、paired bootstrap、positive support 与 exact-utility
  ratio 继续使用冻结的 24 条 denominator：invalid contribution 预注册为 0，而不是删除失败样本；若 valid
  state 缺失或重复 selector join，则整次执行判为 invalid；
- independent 主假设与 set-aware exploration 使用 Source-A 中输出前冻结的 trajectory-equal、paired
  bootstrap、exact-ratio 与 positive-support thresholds；
- 任一 development GO 只允许起草一个新的 AndroidWorld closed-loop Source-A。旧 confirm、matched-NLL、
  closed-loop 与 sealed test access count 仍必须为 0。

## 执行与芯片分工

冻结采用三段 Git lineage：Source-A commit 先冻结全部科学代码与阈值；它的 direct child 只新增 deterministic
selection manifest；从这个 clean pushed selection commit 构建并上传 substrate 与 label-blind seal；最后一个
direct child 只新增 Execution-B runner config，并绑定 Source-A inventory、selection commit/SHA、两个 HF
immutable revisions。这样 selection 必须先进入 Git，runner 又无法在 substrate/selector identity 未知时提前
占位。

- raw scan、OCR、feature construction、selector inference 与 report reduction 是 CPU/policy-blind 阶段；
- GUI-Owl reference/restoration 在 Hyper00 的 4 张空闲 H200 上按 6 trajectories/worker 静态分片；
- 每个 worker 单 GPU、单 runtime，不做跨卡 tensor parallel；失败可按已完成 trajectory receipt resume，
  但不允许科学 retry、改 selection 或 top-up；
- Taurus/Aries A6000 可用于 builder/tests 与 processor smoke；48 GB 不作为 9-image eager exact run 的正式
  capacity claim；
- 不同芯片不要求 bitwise logits 相同。正式 result 必须记录 host/GPU UUID、container digest、torch/
  transformers/CUDA、dtype、完整 argv、Git/HF revisions、起止时间与 failure classification。

GPU job 前按仓库规则做 10 秒 idle preflight；启动后仅在 warmup 与代表性 steady-state 窗口检查每张分配卡
利用率至少 80%。

## Source of truth 与产物

- Git：合同、runner、tests、轻量 selection/result summaries、本文与进度；
- private HF dataset：`gavinlaw/causalcache-long-horizon-development-mobile`；repaired substrate revision 为
  `d237271e3266a72cce7aa730d0708936f1552365`，superseded archive revision 为
  `5800f150f34d454ca72ae3eaeee3f30f564d834e`；
- private HF models：formal-58 gate 与 v4 residual checkpoints 的既有 immutable repositories；
- raw GUIOdyssey 与 AndroidWorld traces 不复制进 Git。

原计划的三段提交在 policy-blind substrate 后停止；label-blind selector preparation 仅进入显式 superseded
archive，restoration/report 从未生成。archive 的 fresh-download SHA/size/byte replay 已完成。
