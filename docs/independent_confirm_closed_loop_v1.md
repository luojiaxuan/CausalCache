# Independent 主线 confirm / closed-loop v1

本合同把论文主线固定为 **restoration-guided independent gate**。它不是对失败的
set-conditioned v1 做阈值救援，也不重新训练 gate。formal-58 的 5 个 independent checkpoints
原样复用；fresh-16 只保留为已经消费的 development evidence。此前结论
`NO_V2_CONDITIONAL_RESCUE` 永久保留，conditional gate 只进入 negative ablation。

机器可执行合同是
[`code/configs/causalcache_independent_confirm_closed_loop_v1.json`](../code/configs/causalcache_independent_confirm_closed_loop_v1.json)。

## 一次性 confirm-20

confirm 集不是 raw-unseen：构造 derived artifact 时程序已经读取过 raw rows、expert actions、截图和 OCR。
它仍然是 **policy-output / restoration-output untouched**：此前没有在这 20 条 trajectory 上生成 reference
policy action、距离表、gate evaluation 或结果驱动筛选。论文和文档只能使用后一种表述。

固定 denominator 是 20 条 trajectory、每条一个 decision step 6 state；候选 event 恒为
`(1,2,3,4)`，视觉预算 `B=2`。不允许 retry、top-up、过滤、小分母删除或 confirm 后改阈值。

执行严格分两段：

1. 验证 immutable derived bytes，构建 q64/h64/g8 feature；运行 5-seed independent ensemble、
   dynamic recent、OCR/RGB 和 policy-vision；把所有 score 与 selection 封存并发布 HF payload commit。
2. 只有远端 payload commit receipt 已存在，才允许生成 full-history canonical action、计算 16 个 coalition
   的 `D(S)`、生成 exact-subset oracle 和固定 denominator report。report commit 必须是 payload commit 的
   direct child，最后创建 annotated tag。

4 个 policy-vision worker 与 4 个 restoration worker 都固定每卡 5 states。phase wall-clock timeout 分别为
7,200 秒和 43,200 秒；超时或任一 worker 异常会 terminate 全部存活 worker、保留 failure evidence，并且不
允许用同一 one-shot identity 重试。

正式 run 前必须先在同一个 container、同一组 4 个显式 CUDA devices/physical UUID、同一 model directory
和 snapshot manifest 上运行 data-blind topology smoke。smoke 先用 `spawn` 并行加载四个 policy-vision
runtime、全部 join 后才用 POSIX `fork` 并行执行四个 eager teacher-forced `wait`；两段都只处理程序生成的
5 张固定 RGB 图，不读取 confirm data、不生成 policy action、不访问 HF。stdout 的 canonical JSON receipt
保存到 Git 外，正式 CLI 必须显式传 receipt 绝对路径及 SHA256，并在任何 confirm semantic decode、model/HF
访问前验证 allocation、model/runtime/snapshot、phase order 和 operation counts。smoke 每段 timeout 固定
7,200 秒。

每个 state 在 full-history 输入上独立生成两次 action，只有 canonical action 完全一致才定义 reference。
随后进行 17 次 teacher forward：1 次 reference、1 次同输入 repeat，以及 15 个非 full coalition；
`D(full)=0`。任一 reference 失败都留在 20 条固定 denominator 中，并直接路由
`NO_GO_REFERENCE_COVERAGE`，不能在剩余子集上宣称 confirm 通过。

### Confirm GO

主聚合使用 20 条完整 denominator 上的 raw utility 总和，避免通过逐 state 小分母剔除改变样本：

- valid reference `20/20`；
- `D(empty)>1e-12` 的 memory-sensitive state 至少 `8/20`；
- independent / exact raw utility ratio 至少 `0.85`；
- independent raw utility / baseline-distance mass 至少 `0.50`；
- independent 相对每个 heuristic 的 mean raw delta 都严格大于 0；
- 相对最强 heuristic 至少 `12/20` trajectory 为正，paired 90% bootstrap lower 严格大于 0；
- 5 个 seed 中至少 4 个达到 exact raw ratio `0.75`，seed ratio population std 不高于 `0.10`。

固定 report 同时公开两套 **不参与 GO** 的 normalized descriptive：一套把
`D(empty)<=1e-12` state 记为 0 后对固定 20 条求均值，另一套复刻 fresh-16 的 eligible-only 口径并显式报告
eligible count。两者都报告 independent、exact、三个 heuristic 与五个 seed；fixed-20 mean 不能与 fresh-16
eligible-only mean 直接比较，也不允许从两种口径中事后选择更好看的一个作为门槛。

全部满足才输出 `GO_TO_PAIRED_CLOSED_LOOP`。这只解锁 closed-loop 的实现、跨机 smoke 和冻结的
60-template train development，不解锁 sealed test。

## Closed-loop 的真实 estimand

AndroidWorld 环境栈已经在 Aries 验证，但现有 GUI-Owl Instruct 和 Think 都没有通过旧的 50% teacher
success gate。因此本版本研究的是：

> 在同一个固定弱 backbone 下，independent memory controller 相对等预算 recent controller 是否提高
> executable terminal success。

不能把 confirm GO 写成 reference teacher 已被重新验证，也不能把已经消费的 62-instance validation
重新当作新方法主证据。

主对照是 `independent-B2` 与 `recent-B2`。当前截图始终可见且不计 memory budget；与当前截图等价的
最近 post-state 不作为历史候选；所有更早 event 都保留同一份固定摘要。Independent selector 使用 5 个
checkpoint raw score 的算术平均，只选严格正分 event，按 `(-score, step_id)` 排序，最多选择 2 个。
还记录 `summary-B0`、固定随机 `random-B2` 和 `OCR/RGB-B2` 作为 secondary arms。不同预算的 full 或
`B4` 只能是 descriptive reference。

正式拓扑固定为 Hyper H200 policy 与 Aries AndroidWorld emulator 的 versioned 跨机 HTTP/SSH tunnel。
在第一次 episode 前必须验证 tunnel、reset、screenshot、execute、score、tear-down 和 paired arm
identity。不得静默切换到 A6000 policy runtime；若跨机方案不可用，需要新合同，而不是修改本版本。

Development 机械取 AndroidWorld train partition 的 60 个 template、每个 `task_index=0`，seed
`314159`，即 60 个 paired instances / 120 个 primary-arm episodes。前 12 个可作为 infrastructure smoke
查看，但仍属于同一固定 60 分母，不能据此改协议或删除。
两臂顺序由 SHA256 counterbalance：key 精确串联
`protocol_id\0partition\0suite_seed\0task_type\0task_index`；digest 首 byte 的最低 bit 为 0 时
`independent→recent`，为 1 时反序。每臂独立 initialize/reset/tear-down；任何 infrastructure failure 按 ITT
计 0 并单列原因，不允许 hidden retry。secondary `random-B2` 不调用语言 RNG，而对上述 key、decision index
和 candidate step 做 SHA256 排序后取前 2 个。

线上 independent feature 精确复用 formal gate 的 200d `[q64,h64,q64*h64,g8]` 与 pinned RapidOCR
PP-OCRv5 backend；requested budget 和 policy-vision feature 都不是模型输入。OCR/feature 失败不能 fallback，
必须记入 ITT failure classification，并使 100% feature-audit gate 失败。具体 adapter 已在 Source-A 冻结：
每个 raw transition 使用 canonical `GUIOwlV2Action`、before/after encoded screenshot bytes 和同一 pinned
backend 的 canonical OCR records，重建 `low_fidelity_v2`；screen-text delta 与 256×256 RGB MAD 精确复用
formal 规则，`foreground_app`、`executor_result` 固定为训练时的 `unknown`。AndroidWorld 的 RGB PNG 直接走
generic image preparation，不错误套用 GUIOdyssey-only RGBA input contract。history 强制 oldest→newest，age
只能是 `N-1-index`；最新 event 的 post OCR 必须等于 current OCR，并且只排除这一项 current-equivalent event，
其余任意长度历史全部重新 query-time scoring。函数绑定见 config 的 `online_feature_binding`。

只有以下条件全过，才可在新的 direct-child authorization 中打开 sealed 75：

- 两臂 parse coverage 都至少 0.95；
- budget 和 feature audit coverage 为 1.0；
- hidden retry 为 0；
- independent-minus-recent 的 template-cluster paired 90% bootstrap lower 严格大于 0。

每个 `(partition, task_type, task_index)` 是一个 paired unit；两臂在同一个 frozen instance record 上分别
initialize/reset/tear-down。单元效应为 `success_independent-success_recent`，失败或缺失 episode 的 success 按
ITT 固定为 0。template 效应先对该 template 的全部预注册 instance indices 求算术平均，partition 效应再对
template 等权平均。train 使用 `[0]`；test 使用 `[0,1,2]`，因此三条 test instance 不能被当作三个独立
bootstrap clusters。

按 task-partition 的固定 template order 做 10,000 次 template-cluster bootstrap：每次有放回抽取 `T` 个
template index，并携带所抽 template 的全部 instance pairs；`T=60/25` 分别对应 train/test。RNG 固定为
`random.Random(271828)`，每个 replicate 精确调用 `T` 次 `randrange(T)`；区间为 90% percentile、
Hyndman--Fan type 7、lower quantile `0.05`。该统计配置同时用于 train-60 与未来 sealed-test-75，不得根据
observed success 更换。

sealed test 是 25 templates × 3 instances，共 75 个 paired instances / 150 个 primary-arm episodes，seed
`161803`。不得结果后 top-up。主指标是 template 等权、template 内三 instance 等权的 official terminal
success paired difference，同时报告冻结的 template-cluster bootstrap、win/tie/loss、
parse/executor/infrastructure failure、真实 visual tokens、latency 和 horizon。

## Matched-NLL 预注册

虽然 matched-NLL 只在对应 partition 的 immutable closed-loop primary report 后执行，其统计合同现在已经
冻结。train-60 只能产生 development diagnostic，不能支持论文 primary mechanism claim，也不能修改 caliper、
eligibility 或 sealed-test contract；primary mechanism evidence 只来自 development GO 后才允许打开的
sealed-test-75。长期轨迹可能超过 policy
的 5-image envelope，因此 reference 不是不可执行的 full history，而是同一 state 上
`M_independent ∪ M_recent` 的 pair-union：最多 4 张历史图加 current。pair-union 独立生成两次且 canonical
action 必须完全一致；在同一 action 上计算两种 memory 的 mean action-token NLL、full-vocabulary KL，以及
raw restoration mass `R_m=D(empty)-D(M_m)`（允许为负）。

每个 paired instance 使用两条实际 primary episode 的 policy-decision state multiset，不在两个 origin 之间
去重；每个 state 都反事实重建并评估两种 memory。聚合依次为：origin episode 内等 state、两个 origin 各
`1/2`、template 内固定 instance indices 等权、最后 template 等权。train 只含 index 0；test 必须包含
indices 0/1/2。success 对每个 arm 同样在 template 内平均全部预注册 instance 的 official terminal ITT 0/1。
一个 paired instance 只有在两条 episode 都存在且各至少一个 decision state，并且每个 state 的 reference、
finite NLL/KL、selection 与 budget audit 全过时才 eligible；一个 template 只有在其全部固定 instance pairs
都 eligible 时才进入机制分析。ineligible 原因完整保留，不 retry/top-up，它仍保留在 closed-loop ITT 主分母。

caliper 只在上述 template aggregation 后应用。主 caliper 固定为 `0.05` nats/token，`0.02` 和 `0.10`
只作 sensitivity。对 NLL-matched template 定义
`z=sign_eps(R_ind-R_recent)*(success_ind-success_recent)`，`eps=1e-12` 且 restoration tie 贡献 0。主机制
统计是 template-equal mean `z`，使用 10,000 次、seed `271828`、90% percentile、Hyndman--Fan type 7
bootstrap；每个 replicate 从冻结的 `m` 条 matched-template records 有放回抽 `m` 条。只有 sealed-test-75
主 caliper 至少 15 个 matched template clusters 且 bootstrap lower 严格大于 0 才允许声称：在
one-step NLL 匹配后，保留更高 restoration mass 的 memory 更可能成功。sensitivity 不能替代 primary，
也不得根据 success 或实际 NLL 分布选择 caliper。这是 controller-conditioned state distribution 上的
post-treatment mechanism association，不是 mediated causal effect。

## 当前执行顺序

1. Source-A 全部代码、测试、阈值、closed-loop 与 matched-NLL 统计合同冻结并 push `main`；
2. 唯一 direct-child Execution-B 只增加 runner-freeze JSON，再次 push `main`；
3. Hyper GPU preflight 后运行 label-blind payload phase，HF commit；
4. 执行 confirm restoration、固定 denominator evaluation、HF report/tag 与 fresh replay；
5. 只有 confirm GO 才实现并运行跨机 paired closed-loop train-60；
6. 只有 development GO 才另行生成 test authorization，打开 sealed 75。

任何一步失败都保留原始 evidence 和固定 denominator，不通过改数据、阈值、模型、seed 或 comparator
把结果调成 GO。

## v1 formal attempt 结果：execution INVALID

唯一 v1 formal attempt 已在 Hyper00 四张 H200 上运行。data-blind topology smoke 通过；20-state
label-blind payload 已本地 seal，并发布为 private HF commit
`6d0cd95997186293e01c65276f3c082c11a9f52d`。8 个 payload targets 经 fresh download 全部逐字节一致，
report targets 与 annotated tag 均不存在。

执行随后在第一个 restoration worker 创建 runtime 时失败。formal 父进程在 POSIX `fork` 前已经初始化 CUDA，
子进程触发 PyTorch 的 `Cannot re-initialize CUDA in forked subprocess`。此时 reference generation、teacher
forward、`D(S)`、exact-subset oracle 与 report 都是 0，因此本次状态永久为 execution `INVALID`，不是 confirm
`NO-GO`。机器可读证据见
[`data/results/independent_confirm20_v1_attempt/`](../data/results/independent_confirm20_v1_attempt/)。

同一 v1 identity 不得重试。允许的下一步仅是独立 Source-A/B 的 restoration-only continuation：它必须绑定旧
failure、A/B、topology receipt、HF base/payload history、8 个 exact payload bytes、report/tag absence，并在任何
restoration semantic access 前完成 fresh replay；不得重新运行 policy-vision/independent selector，不得修改
threshold、dataset、model、seed、selection 或 GO contract。旧 attempt 不因 continuation 成功而重分类。
