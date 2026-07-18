# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

> Target venue: AAAI
> Status: restoration-guided independent gate 的一次性 confirm-20 continuation 已完成并得到有效科学
> `NO_GO_INDEPENDENT_CONFIRM`。20/20 reference coverage 与 memory-sensitive checks 通过，但 independent / exact
> raw utility ratio=`0.821298 < 0.85`；independent 相对 recent、OCR/RGB、policy-vision 的 mean raw delta
> 全部为负。按预注册权限链，paired AndroidWorld closed-loop、matched-NLL 与 sealed test 均未执行并继续
> locked；不允许事后改 threshold、data、model、seed 或 comparator。轻量结果见
> [`data/results/independent_confirm20_continuation_v1/`](data/results/independent_confirm20_continuation_v1/)，
> canonical report 为 private HF commit `a0b408e58d629299be334a74ecbd0ec2fa2ed1fc`。协议见
> [`docs/independent_confirm_continuation_v1.md`](docs/independent_confirm_continuation_v1.md)。

> Completed diagnostic: 已完成 confirm-20 的只读 oracle-independent `J` failure decomposition。raw utility
> sum 为 exact=`0.856431`、OCR/RGB=`0.783268`、`J=0.765311`、learned `I=0.703385`；`J-OCR` raw
> mean=`-0.000898`，因此正式为 `CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB`。`J-I` 的 paired 90%
> interval=`[+0.000477,+0.006533]`，证明 student gap 存在，但 projection ceiling 已经低于 strongest
> heuristic，student 不是唯一 blocker。结果见
> [`data/results/independent_confirm20_failure_decomposition_v1/`](data/results/independent_confirm20_failure_decomposition_v1/)，协议与边界见
> [`docs/independent_confirm_failure_decomposition_v1.md`](docs/independent_confirm_failure_decomposition_v1.md)。

> Frozen P0: 已另立并验证 development-only AndroidWorld validation-12 repeated-selection 协议、roster、纯
> selector/prompt/evaluator 与 source-only validator；尚未接入 live environment，也没有运行任何 episode。根据
> 2026-07-18 的路线调整，该 closed-loop P0 保留为可恢复协议但暂停执行。见
> [`docs/exploratory_closed_loop_validation12_v1.md`](docs/exploratory_closed_loop_validation12_v1.md)。

> Current route: 历史 conditional/independent NO-GO 与 Case-A failure decomposition 全部保留；closed-loop P0
> 继续暂停。主线改为先清点全部 610 个 GUIOdyssey transport train shards，扩充 exact
> restoration tables，再训练 budget-agnostic set utility predictor。`U_theta(q,C,m_S)` 永远看到
> 全部有效候选与 selected/unselected bit，不接收 `B`；预算只进入包含 empty 的
> joint at-most-`B` search。Set Transformer 是主模型，DeepSets 和 pairwise-additive 是对照。
>
> 当前已实现 variable-`n` features、`|S|<=K` exact label schedule/producer、Set Transformer/
> DeepSets/pairwise 模型、trajectory-uniform trainer、joint search 与 true-`U` evaluator。predictor config
> SHA256=`9548159b219795b1c258c28f772f53351256e0d728b88dd409cb333bd2100fe4`；focused suite 为
> `193 passed, 15 skipped, 24 subtests passed`，skip 仅因本机无 PyTorch。
>
> 数据防火墙已机械物化 107 个历史 identity：58 条 `legacy_train_only` + 49 条
> `forbidden_consumed`，ledger SHA256=
> `b6f44c603b99d2f954b981e01818cf0afa028ce3a410ed935203532a097bb4ad`。P-1 metadata-only
> contract SHA256=`1b2b4374d1653bcf22444d8e708c71bc41ac87fa956243ddcb9bd963eeca7e96`；P0
> source-only contract SHA256=`7e65227e710009d3626bd0063d425c831dfc59e9a6bbe871b9d3e4d15e085e8b`，已绑定
> ledger，但故意未绑定尚未产生的 P-1 manifest。
>
> 本地唯一真实 P-1 尝试在调用 `HfApi.list_repo_tree` 时因 sandbox DNS 失败，未收到
> metadata，也未生成 partial
> manifest；因此真实 610-shard inventory、semantic census、新 restoration labels、checkpoint 和
> offline delta 都仍不存在。下一步是在可联网 checkout 运行并 push 唯一 P-1 manifest，
> 再单独冻结 P0 Execution-A；不使用旧 13–64 小池子替代全量数据。formal-58 仅可
> train-only；reference8、old-dev5、fresh-16、confirm-20 永不进入新训练/调参/评估。
> matched-NLL 与 sealed test 继续 locked。见
> [方法契约](docs/set_utility_predictor_v1.md) 与
> [实现交接](docs/set_utility_implementation_v1.md)。

> Gate data status: 旧 train-10 只允许与新 train-48 合并为 formal-58，不能单独产出 metric；原 preregistration
> 只允许在 fresh-16 outcome 冻结后另立旧 dev-5 combined-21 compatibility 阶段，当前因 primary NO-GO 选择不执行。
> 新的 48 train +16 fresh-dev
> policy-blind expansion split manifest、pre-output exposure ledger 与 gate v1 preregistration 均已冻结，
> policy-blind derived artifact 已绑定 private HF immutable revision。Planned final denominator 为 58/21/20；
> 192-state substrate 已从 runner source A=`1a3833d`、execution B=`642feb2` 完成唯一双 H200 formal run：
> 192/192 states、0 failed，384/576/384 operations 精确命中，185 states memory-sensitive，正式 `PASS`；
> raw archive 已绑定 private HF immutable revision `25ac19cf6ef98adc243d421cd0039ac104ddb539` 并 fresh-download
> 逐 byte 复验。expansion exact-label config、coalition input、双 H200 runner 与 raw reducer 已由 source
> A=`2c00c9118dc00cc1bda361325d24d79d8c14f8b6` 冻结；config SHA256 为
> `65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5`。runner freeze SHA256
> `c0447acda3f09bccc65461ed08092fa6b166370721767cf5f35eb19cc59583d6` 已作为 execution
> B=`bd5cc78838c09a50214b1108fb18f62139c7419e` 单独 commit/push，committed validator 重建 76-file
> inventory 并返回 GPU authorization=true。唯一 Hyper00 v1 attempt 已完成 192/192 states 与全部固定 operation
> counts，内部 raw reducer 得到 `PASS_V2_2_EXPANSION_EXACT_LABELS_V1`；但 source-locked monitor 的 3,849 个
> intervals 中 5 个超过冻结 3 秒上限（max 3.883721 s），post-worker validator 因而将全局 attempt 永久封为
> `INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`。这些 bytes 当前不能作为 formal expansion labels。下一步是独立
> CPU-only child validation repair 与 separate invalid-evidence immutable archive；不重跑 GPU、不改阈值、不追认
> v1 formal PASS。invalid-forensic source-only P0 已冻结：402 root files 与 3 external ledgers 使用双 namespace、
> 406-member deterministic USTAR、append-only failure chain 和 permanent formal-ineligible manifest。真实
> local archive 已在 Hyper00 package/readback：SHA256 `8e205d73196604c0d8d342f9415755b090b96e83555ca55c386b12b8427ca489`，
> 3,880,960 bytes。独立 P1 private-HF publication contract 已 source-freeze：config SHA256
> `affb54cf44a656394c983cebe5d004d3e1ff216537c43da942dc031fa6baaccb`，固定 claim-first、single-pair
> commit provenance、no-overwrite tag、crash recovery 与 immutable fresh readback。唯一真实 invocation 已创建
> 正确的 private pair commit `5efe1ae8...b416` 与 tag，但 Hub refs API 返回 annotated-tag object SHA、而
> tag resolution 返回 pair commit SHA；v1 parser 将两者强制相等，因而在 completion 前 fail closed。remote
> 原样保留。read-only versioned tag-resolution child 已从 clean `main@d61dff5` 完成正式 replay 与第二次幂等
> replay，config SHA256 为 `f257dcdebb218533a080739e8a5067cc44fe35fd533ad57808db5ce789cfd956`；它分别
> 绑定 annotated-tag object `ca652858...5f44` 与 resolved commit `5efe1ae8...b416`，immutable force-download、
> 406-member P0 strict readback、pre/post identity/provenance 均通过，remote mutation API 调用为 0。该完成只
> 闭合永久 INVALID 证据的 transport，formal label eligibility 与 gate unlock 仍为 false。
> ledger-neutral no-GPU child 随后从 clean `main@b14f489` 完成正式 run 和完整只读 revalidation；192 raw /
> 192 derived states、1,792 distances、1,856 deployment edges、3,072 full edges、1,984 interactions、576
> attributions 与 192 exact oracles 全部重算一致。4-member local USTAR SHA256 为 `1a9fdcc0...e01`，原
> producer 仍永久 INVALID。新的 repaired private-HF publication 已由 clean `main@e47c506` 闭合：private
> dataset tag `v2.2-expansion-exact-labels-scientific-repair-v1` 的 annotated-tag object 为
> `6a907ba2...a4a57`，tag-resolved immutable pair commit 为 `7a6c254b...5cef3`；archive/sidecar、exact-pair、
> 幂等 replay 与独立只读 postflight 均通过。由此只解除 formal-58 的 label-data prerequisite；Gate v1
> trainer/evaluator source 与 synthetic-only CPU smoke 已闭合；formal-58 OOF/model selection/final fit 现已完成并
> immutable replay。fresh-16 learned-gate primary metric 也已完成并 immutable replay，但预注册 selector 与
> set-conditioning 两个 GO gate 都未通过；在该 fresh-16 milestone 当时，matched-NLL、closed-loop 与 confirm
> 均未执行。formal-58 train-only cache 的 v1 Source-A
> `990f015` 与单文件 Execution-B `079c095` 已冻结并 push；首次 Hyper00 CPU-only run 在 global claim 后的
> feature transport byte check fail-closed：config 把 expansion trajectories SHA 错写为 `00fe93...353a6d`，
> immutable bytes 与 Git-pinned producer witness 均为 `fe93e9...353a6d`。失败早于任何 semantic decode，未生成
> cache/HF repo/tag/completion，也未创建 label-access claim。旧 claim 永久保留。独立 namespace 的 transport-repair
> 已由 Source-A `4f8c01b` 与唯一 Execution-B `f96c197` 完成；只修正该一个 SHA leaf，并完成 private-HF immutable
> replay（tag-resolved commit `a61b31b...cc386`）。formal cache 已作为正式训练输入消费，训练封存结果见
> [`data/results/gate_v1_formal58_train_v1/`](data/results/gate_v1_formal58_train_v1/)；上游数据协议详见
> [`docs/restoration_v2_2_label_expansion.md`](docs/restoration_v2_2_label_expansion.md) 与
> [`docs/restoration_v2_2_expansion_exact_labels.md`](docs/restoration_v2_2_expansion_exact_labels.md)、
> [`docs/gate_v1_preregistration.md`](docs/gate_v1_preregistration.md)、
> [`docs/gate_v1_formal_cache.md`](docs/gate_v1_formal_cache.md)、
> [`docs/gate_v1_formal_cache_transport_repair.md`](docs/gate_v1_formal_cache_transport_repair.md)、
> [`docs/gate_v1_execution.md`](docs/gate_v1_execution.md)。
> formal-train Source-A=`e20f004…b9` 与唯一单文件 Execution-B=`bad28b7…a2f2` 已 push。Hyper00 CPU-only
> formal run 和只读 replay 均闭合；private HF model tag `gate-v1-formal58-train-v1` 指向 manifest commit
> `23f6786…72a9`，包含 5 conditional + 5 independent checkpoints、2 full OOF reports 与 4 manifests。
> `gate_trained=true` 只表示 train-only model seal 完成。fresh-16 primary 已在独立 claim-serialization repair
> namespace 上完成；在该 historical milestone 当时，旧 dev-5、confirm、matched-NLL 与 closed-loop 尚未打开。
> fresh-16 evaluation Source-A 冻结了 exact 16 trajectories / 48 states / 144
> candidate features / 448 distances、variable-`n` heuristics、双 H200 49 processor batches / 97 vision
> forwards、all-selections-before-label firewall 与 9-target payload + 4-target report 的两段 publication；config
> SHA256 为 `c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e`。其唯一 v1 formal attempt 在
> derived exact-tree preflight 永久 pre-semantic fail closed：fixed revision 的 15 paths 中有 9 个历史 paths 未被
> v1 allowlist 绑定，download/semantic decode/GPU/HF mutation 均为 0。full-inventory repair Source-A=
> `6fb3e868e293bce191ce30a5c6a15ecc544c4591`、Execution-B=
> `c22734ffc85935882f57ddb081c9194d6dae92d0` 已冻结并执行；它完成 16 trajectory / 80 OCR decode、48 feature
> states、144 candidates、10 checkpoint loads、双 H200 2 workers / 97 forwards 后，在写入 label-access claim 前因
> `mappingproxy` 序列化失败永久 `INVALID`。独立 claim-serialization repair 随后由 Source-A=
> `f0dd53b9a0249f259a833b4d8ad3ff26096a0ed1` 与唯一 Execution-B=
> `ce523ff54ebfdc19a9c2bd49ad21548f0934a634` 完成。它只把 `LabelAccessClaim.claim` 改成 canonical-JSON deep
> snapshot，并在全新 namespace 上闭合 16 个 durable receipts、13-target private-HF publication 与独立 immutable
> replay。有效结果为 `go_selector=false`、`go_set_conditioning_primary=false`：conditional ensemble 相对最强
> heuristic 的 normalized delta 为 `+0.3261`、90% bootstrap lower 为 `+0.0490`，但 normalized recovery / exact
> 只有 `0.6942`，5 个 seed 均未达到 `0.75`；conditional 相对 independent 的 normalized delta 为 `-0.1224`。
> 这是否定 frozen v1 gate 与 set-conditioning claim 的科学 `NO-GO`，不是实现失败。
> 完整边界见 [`docs/gate_v1_fresh16_evaluation.md`](docs/gate_v1_fresh16_evaluation.md) 与
> [`docs/gate_v1_fresh16_inventory_repair.md`](docs/gate_v1_fresh16_inventory_repair.md)、
> [`docs/gate_v1_fresh16_claim_serialization_repair.md`](docs/gate_v1_fresh16_claim_serialization_repair.md)。只读
> failure-decomposition 的 input、estimand、routing 与 split-consumption 边界见
> [`docs/gate_v1_fresh16_failure_decomposition.md`](docs/gate_v1_fresh16_failure_decomposition.md)。

## 团队交接入口

本节保留此前 v2 substrate→label→gate 的历史执行链；当前科学终态与下一步以本文顶部 `Status` 为准。

当前路线已经从 v1 expert-aligned admission 切换为 versioned v2 stable self-behavior estimand。科学配置
[`code/configs/causalcache_restoration_v2.json`](code/configs/causalcache_restoration_v2.json) 已在任何 v2
policy output 前冻结，SHA256 为
`9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc`。primary policy 固定为
`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faec`；reference 只要求 restricted action 可解析、logits finite、
两次 canonical action 一致，expert alignment 仅分层报告且不能过滤状态。完整定义、data exposure 和
go/no-go 阈值见 [`docs/restoration_v2.md`](docs/restoration_v2.md)。第一次固定 development screening 已产生
45 个 v2 native policy outputs，但 strict parse 为 0/45，因此没有 teacher forward、KL、restoration label、
gate checkpoint 或方法效果结果；在该 historical milestone 当时 confirm 仍 locked。

旧 15-trajectory v2.2 exact labels 已闭合：30 个 label-train + 15 个 development states 已得到完整
420-row $D(S)$ table、435 条部署可达 conditional edges 与 exact-subset oracle，raw artifact 已绑定 private HF
immutable revision。训练 gate 前先执行
[`docs/restoration_v2_2_selector_geometry.md`](docs/restoration_v2_2_selector_geometry.md) 的 policy-free geometry：
把 `n=2,B=2` ceiling 与 `n=4,B=2` primary compression 分开，并分别测量 exact-to-greedy search gap 与
greedy-to-independent objective-projection gap。该步骤不读取 confirm/test，也不运行新 policy forward。

第一次 v1 table replay 的 selector 数值通过两套 independent raw-table 重算，但 contract-completeness audit
发现旧 reducer 没有物化冻结要求的完整 interaction 联合分层，并省略了 analytic random 可解析的
cardinality/match/Jaccard。旧 output 因此未进入 Git。versioned repair contract
[`code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json`](code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json)
固定为 reporting-only：保留所有 selector/utility/bootstrap/shaping 数值，补齐 train/development 共 144 个联合
cells 与 random 集合几何。正式 repaired result 已从 clean pushed `main@9a4eca5a` 生成并逐 byte replay：
primary `n=4,B=2` 上 exact 与 true greedy 在 15/15 states 完全一致；development normalized
objective-projection gap（greedy - budget-conditioned independent）为 `0.159039`，5/5 trajectories 同方向，
90% method-shaping interval `[0.007836, 0.355867]`。冻结规则输出
`set_conditioned_main_candidate + online_greedy_sufficient`，但它不是 paper success gate。OCR/RGB 已闭合；
policy-vision 的 v1/v2 attempts 均在 0 feature 时因版本化 runtime interface mismatch 封存；v3 GPU artifact 的
state projection 已由独立 CPU-only repair exact-byte replay 闭合，正式状态为 `VALID`；result commit 后又从
clean `main@174801112c58d831249fd54f4f8bc9af01524b44` 完成 `REVALIDATED`。

当前关键路径已经闭合到 frozen formal-58 ensemble → fresh-16 valid primary `NO-GO` → formal failure
decomposition `NO_V2_CONDITIONAL_RESCUE` → independent confirm-20 `NO_GO_INDEPENDENT_CONFIRM`。父 v1 与
full-inventory repair 的两次失败均保持永久 `INVALID`；claim-serialization repair 以独立 A/B、namespace 和 HF
identity 完成了 16/48/144/448 denominator、双 H200 `49` processor batches / `97` vision forwards、16-receipt
receipt chain、9+4 direct two-commit publication 与 immutable replay。primary conditional gate 虽显著优于三类
heuristic 的 aggregate mean，但未达到 frozen oracle-proximity、seed-stability 与 trajectory-support gates；
set-conditioned gate 也明显弱于 parameter-matched independent comparator。failure decomposition 进一步证明
true greedy 已接近 exact，但 oracle set-conditioning headroom 只有 `10/16` trajectories 同方向，未达到冻结的
`12/16`；即使 student gap material，也不授权一次 conditional v2 rescue。fresh-16 已消费，不能据此改模型后
再把同一 split 当作 confirm。现有 independent ensemble 未重训，并已在原先冻结且
policy/restoration-output untouched 的 confirm-20 上完成一次性确认；结果未通过 frozen gate。因此本 v1 路径
停止，不实现 paired closed-loop、train-60、matched-NLL 或 sealed-test-75。合同与终态见
[`docs/independent_confirm_closed_loop_v1.md`](docs/independent_confirm_closed_loop_v1.md)。
已发布 child 的冻结分母为
64 trajectories / 192 states，steps 4/5/6 对应
$n=2/3/4$、$B=2$；formal raw table 为 1,792 条 $D(S)$，policy-free 重算 1,856 deployment edges、3,072
full edges、1,984 interactions、576 attributions 与 192 exact oracles。label run 固定 1,984 teacher forwards、
1,792 GPU KL、0 generation，并保留所有 negative marginals。source A、单独 runner freeze B 与 committed
validator 已闭合。唯一 GPU attempt 已完成全部科学计算，但因 source-locked monitor cadence gate 违规而正式
`INVALID`：monitor 首尾覆盖和索引连续均成立，仍有 5/3849 gaps >3 s，max 3.883721 s。原 ledger、root 与
monitor 必须逐 byte 保留，不能 retry、resume 或调阈值。P1 publication protocol 见
[`docs/restoration_v2_2_expansion_labels_invalid_forensic_publication.md`](docs/restoration_v2_2_expansion_labels_invalid_forensic_publication.md)：
它只允许把永久 INVALID 的 P0 archive 与 sidecar 作为同一 private-HF commit 发布，不能直接供 formal loader
消费。唯一 P1 invocation 的 pair bytes 已正确落到 private HF，但 completion 未形成；read-only child 见
[`docs/restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution.md`](docs/restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution.md)。
read-only child 已正式完成，结果见
[`data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_tag_resolution_v1/`](data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_tag_resolution_v1/)；
ledger-neutral scientific-repair core、no-GPU formal runner、正式 local artifact 与只读 revalidation 已闭合，协议见
[`docs/restoration_v2_2_expansion_labels_scientific_repair.md`](docs/restoration_v2_2_expansion_labels_scientific_repair.md)。
结果见
[`data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_v1/`](data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_v1/)。
独立 publication contract 见
[`docs/restoration_v2_2_expansion_labels_scientific_repair_publication.md`](docs/restoration_v2_2_expansion_labels_scientific_repair_publication.md)，
正式结果见
[`data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/`](data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/)。
它绑定新的 private HF immutable identity。formal-58 label-data blocker、train-only gate fit 与 fresh-16 primary
都已闭合；该 historical milestone 当时旧 dev-5、confirm、matched-NLL 与 closed-loop 仍保持 locked。

primary `n=4,B=2` OCR/RGB baseline v1 的 source 与失败边界见
[`docs/restoration_v2_2_ocr_rgb_baseline.md`](docs/restoration_v2_2_ocr_rgb_baseline.md)。contract SHA256 为
`08f57505d71e603d81d6915ef27cf008d0fe097a56d70a05f8b3519211e2e6f9`，固定 15 states、75 unique images 与
60 个 candidate scores，同时报告 at-most-$B$ exact oracle 和 exact-cardinality oracle，并按 trajectory 做 paired
bootstrap。它只允许 CPU reduction；GPU、OCR inference、policy/gate/confirm/test operation 均为 0。Hyper00
首次 formal attempt 在任何 trajectory semantic parse、image decode 或 feature score 之前，因 identity
lexer 不接受同一 JSONL 中 top-level 与 nested metadata 的两个相同 `source_id` occurrence 而 fail closed；
canonical output 与 staging 均不存在，见
[`data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/`](data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/)。
它是零分 implementation failure，不产生 scientific conclusion。versioned replacement 已冻结并从 clean
`main@a9bede85` 在 Hyper00 完成 formal aggregate 与逐 byte replay，见
[`data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/`](data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/) 和
[`causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json`](code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json)，
SHA256 `d68cb032ef3c56c330d57329507d409b20f878b7510bd09d88e2eff1f3f898f3`。它只修 exact-occurrence/equal-value
identity lexer（trajectory 每行两个、OCR 每行一个）；科学契约与 immutable inputs 不变。train/development/
overall mean normalized recovery 为 `0.771189 / 0.019571 / 0.520650`，exact coalition match 为
`3/10 / 0/5 / 3/15`。overall 相对 budget-conditioned independent 的 paired difference 为 `-0.292103`，
90% trajectory interval `[-0.625159,-0.046921]`；相对 recent/random 的 interval 均跨 0。development 中唯一
负 recovery `-2.641163` 来自真实 non-monotone state：`D(empty)=0.000616`，相似度选择 `[3,4]` 后
`D=0.002243`，而 exact `[1,2]` recovery 为 `0.923640`。该小分母样本保留在固定五条 denominator 中，不能
删除或 clamp。结论仅是 OCR/RGB 已闭合但 development 不稳的弱 similarity comparator；它不否定
restoration attribution，也不构成 gate、matched-NLL 或 closed-loop 证据。
独立 raw-to-result 审计未 import 项目 reducer，44/44 checks、0 mismatch；60 scores、15 complete rows 与
7×10,000 bootstrap 均 bit-exact，audit projection SHA256 为 `9e842494...12df`。
exact-three result 与 artifact regression 已 commit/push 为
`main@7e59591573cb31f178dfd07422cc2e3c8aeff573`；Hyper00 随后从该 clean descendant checkout 重建三份
files 并返回 `VALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR`。

最后一个非学习视觉 comparator 的独立协议见
[`docs/restoration_v2_2_policy_vision_baseline.md`](docs/restoration_v2_2_policy_vision_baseline.md)，machine-readable
contract 为
[`code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json`](code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json)，
SHA256 `a2319f8ea52d53fa01487cdbcbfef20b86ac81dcfab1c8a36ce4583b0b503023`。准确方法名是
frozen-policy-backbone vision similarity：events 1--4 post-state 与 event-5/current 都只经过 frozen GUI-Owl
vision tower/final main merger，BF16 token rows 转 FP32 mean/L2 后做 cosine top-2。它不输入 goal/text/OCR，
不调用 language model、LM head 或 generation，不能写成 behavioral policy-aware。正式 schedule 固定双 H200
8/7 parity、15 canonical + 15 same-device replay + 1 cross-device sentinel，共 31 次 vision feature forwards；
feature worker 不接收 (D(S))；GPU 前只验证 raw label archive 的 byte identity，selection 完成后 CPU reducer
才解析并 join labels。唯一 v1 formal attempt 因 PyTorch 返回 `torch._C._CUuuid`、而冻结 parser 只接受
`str/bytes`，在 model/processor load 与 feature forward 前 zero-output fail closed，轻量证据见
[`data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/`](data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/)。
因此这里仍不报告任何 policy-vision recovery 或 comparator conclusion，也不解锁 gate/confirm。versioned v2
repair contract 已冻结为
[`code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json`](code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json)，
SHA256 `23733169ef5ba60a84f4447080ef12e1893858aa375ff50d8d4e3595699e774e`。它只让 pinned
PyTorch `2.11.0+cu130` 的真实 `torch._C._CUuuid` 经过 exact loaded-type identity 后转成字符串；原
`str/bytes` v1 path、UUID regex、expected UUID、`nvidia-smi` 与 PCI binding、15-state denominator、31-forward
schedule、feature/statistics contract 全部不变。v2 使用新 protocol/output identity，不能复用 v1 canonical
path。v1 `run` 已 tombstone；v2 在 model/feature access 前以固定路径 `O_EXCL` durable ledger 锁定唯一 attempt，
并要求 source commit 相对 failure commit 精确匹配冻结 changed-path inventory。唯一 v2 attempt 已跨过 UUID
repair，但 pinned `SizeDict` 可精确转成冻结两键几何、却不实现 `collections.abc.Mapping`，因此又在 0 feature
时 fail closed；证据见
[`data/results/restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/`](data/results/restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/)。
v2 ledger 已永久 claim，不能重跑；下一步只能冻结新的 SizeDict-interface versioned repair。

versioned v3 只新增 `(UUID-v2, exact loaded SizeDict-v3)` runtime profile。它要求
`type(size) is transformers.image_utils.SizeDict`、module/name 精确、不是 `Mapping`、四个 non-edge fields 均为
`None`，再要求 `dict(size)` 恰好只有冻结的 `shortest_edge/longest_edge` 两键；同名伪造类、subclass、generic
iterable、额外键和数值漂移均拒绝。v1/v2 profile 与 runtime ID 保持原样，v2 `run` 永久 tombstone；v3 使用新
protocol、output 和 fixed `O_EXCL` ledger。v3 config SHA256 为
`794474d8bc60463ba10fdd772691461f5910ca5e5501f7cccff4c53542b84b7f`；该 source freeze 只授权一次新的
formal attempt，不是 comparator result，也不解锁 gate、matched-NLL、closed-loop 或 confirm。

唯一 v3 formal attempt 已从 clean pushed `main@a935a3cf` 完成 15 states、75 unique images 与冻结的 31 次
vision feature forwards。same-device 和 cross-device sentinel 的 maximum absolute score difference 均为 0，
policy/LM/generation/gate/matched-NLL/closed-loop/confirm/test operation 均为 0。exact-three artifact 位于
[`data/results/restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair/`](data/results/restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair/)，
scientific payload SHA256 为 `811e59c7...f48`。GPU 输出的 overall/train/development mean normalized recovery 为
`0.404691 / 0.535228 / 0.143615`，exact match 为 `3/15 / 3/10 / 0/5`；独立 CPU repair 已逐 byte 重建
README/state-scores/summary 三份 producer artifact，因此这些数值现已进入 validated comparator lifecycle。overall mean 受小
summary-only distance 的负 outlier 影响，ratio-of-sums recovery 为 `0.703501`，后续报告必须与 mean 并列。

首次 CPU `validate` 因 reconstructor 把 evaluated row 的七键 state 原样送入只接受四键 feature-state 的
provenance check 而失败。只投影回 `index/role/trajectory_id/state_id` 的 bounded diagnostic 已逐 byte 重建三份
artifact，但正式修复必须版本化、纯 CPU、不得修改 artifact bytes 或重跑 GPU。failure binding 见
[`data/results/restoration_v2_2_policy_vision_baseline_v3_cpu_validation_attempt/`](data/results/restoration_v2_2_policy_vision_baseline_v3_cpu_validation_attempt/)。

versioned CPU replay repair v1 已冻结为独立 source contract：producer source 固定为 `a935a3cf`，pending
artifact commit 固定为 `597f0502`，repair source 必须使用新的 clean pushed commit，三者不得混用。contract
[`code/configs/causalcache_restoration_v2_2_policy_vision_v3_validation_repair_v1.json`](code/configs/causalcache_restoration_v2_2_policy_vision_v3_validation_repair_v1.json)
SHA256 为 `64f63ab7563c227423c15ef82d5fd74d8be11579248908c7ec2880137e5d6ddf`。它只允许 exact 7-key→4-key
projection、immutable-label CPU reconstruction 与三份 producer artifact 的逐 byte 比较；model/GPU/feature、
teacher/KL、gate、matched-NLL、closed-loop 与 confirm/test 均为 0。唯一 formal audit 已从
`main@dbb45637cf79c3573bbbc6051b8b480e3f76d69d` 在 Hyper00 CPU-only container
`sglang-omni-jaxan-07170735` 完成，返回
`VALID_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`。exact-two sibling result 的 README/summary
分别为 836 bytes / `3492dbae...ee7e9` 与 10461 bytes / `f7a5ff63...62b1`；attempt ledger SHA256 为
`6b65bef9...b5d3`，completion seal SHA256 为 `837c52f4...a52b`。运行使用
`hongccc/sglang-omni:dev@sha256:6a8f60af...acfa`、Python 3.12.3，container 未配置 GPU DeviceRequests 且
runner 观察到零 NVIDIA device/runtime import。随后在独立 validation-repo 的 clean
`main@174801112c58d831249fd54f4f8bc9af01524b44` 上，以原 validation source
`dbb45637cf79c3573bbbc6051b8b480e3f76d69d` 完成只读 `validate`，runner 返回
`REVALIDATED_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`。postflight 中 exact-two 仍为 mode 0644
且 hash 不变，attempt ledger / completion seal 仍为 mode 0600、SHA256 `6b65bef9...b5d3` / `837c52f4...a52b`；
无 GPU operation。容器已停止但保留用于审计，exit code 137；revalidation 未修改 artifact、config、code 或 test。

新的 v2.1 interface rescue 已在任何 v2.1 policy output 前冻结为独立协议，machine-readable contract 是
[`code/configs/causalcache_restoration_v2_1_pilot.json`](code/configs/causalcache_restoration_v2_1_pilot.json)，
SHA256 为 `9d51a2ed5d6cc382f297c1b8af3100d784090f72800d637136b88982763fdbf7`。它改用 checkpoint
official `tools=` chat template、要求模型原生 `</tool_call>`、删除 `Action:` carrier，并固定先做零 policy
output 的 90-prompt processor preflight，再对 exact 15 个 `v2_development` states 各生成一次。pilot 必须
15/15 strict whole-output parse、15/15 model-emitted closer、15/15 AndroidWorld bridge，任何一次失败即
`NO_GO_V2_1_INTERFACE_PILOT`，不得 retry/top-up/teacher/KL/restoration，也不得向 decoder/processor 暴露
任何 confirm state/prompt/image。完整边界见
[`docs/restoration_v2_1.md`](docs/restoration_v2_1.md)。90-prompt real `AutoProcessor` preflight 已在 Hyper00
CPU-only 正式通过：90/90 official-tools prompts 合法，context overflow 为 0，input token 长度为
3,759--15,013；policy model 未实例化、weights 未 materialize 为 tensors，且没有 forward、generation 或
GPU operation。raw JSON 已上传 private HF，并由 fresh immutable download 逐 byte 复核。随后唯一 fixed-15
attempt 在 clean `main@70724bd` 正式通过：15/15 strict parse、15/15 model-emitted closer、15/15
AndroidWorld bridge，retry/top-up/truncation/teacher/KL/restoration/confirm generation 全为 0。raw 34-file
USTAR 已上传 private HF，并按 immutable revision fresh-download 验证。loader
会验证并 hash 包含 confirm bytes 的完整 artifact，但提供给 decoder/processor 的 confirm state、prompt、image
均为 0，`confirm_processor_prompt_count=0`、confirm generation count 也为 0。
唯一正式 attempt 固定为 `restoration-v2-1-interface-pilot-v1`，持久路径固定为
`/data/experiments/causalcache/restoration-v2-1-interface-pilot-v1`；runner 拒绝 alternate output path，已有
incomplete marker 也不得通过换目录重跑。global ledger/root/run manifest 会在 policy runtime import/model
construction 前 durable claim；constructor/OOM 与中断残留只会落成可打包 `INVALID`，`--resume` 不会继续
generation。raw pilot evidence 使用 deterministic USTAR 上传 private Hugging Face，Git 只保存 immutable
revision、hash 与 compact reduction。

unchanged-interface full-45 已冻结为独立 child contract
[`code/configs/causalcache_restoration_v2_1_full_45.json`](code/configs/causalcache_restoration_v2_1_full_45.json)，
SHA256 为 `0924dd66fab9440bed66585765e9b1f5ab6fb80fbdf65a1492efba7ae81e116b`。它固定 30 个
`v2_label_train` + 15 个 `v2_development` states、每 state 两次 fresh full-history generation、三次
teacher forward 与两次 GPU KL；全 attempt 上限为 90/135/90。45/45 parse、finite logits 与 canonical
agreement 都是硬门槛，且至少 8 个 states 的 summary KL 必须超过 repeat-noise epsilon。唯一正式 run
得到 45/45 strict parse、90/90 closer/bridge 和 32 个 memory-sensitive states，但两次 exact canonical action
完全一致的只有 32/45，低于冻结的 45/45 gate，因此结果为 `NO_GO_V2_1_FULL_45_SUBSTRATE`。13 个 mismatch
均保持 action type，包含 12 个 `click→click` 与 1 个 `swipe→swipe` coordinate jitter；这只是描述性分析，
不能把当前结果事后改写为 PASS。raw evidence 已上传 private HF 并从 immutable revision fresh-download
逐 byte 验证；manifest push 后，clean `main@554c51e` 的 committed-binding validator 返回
`VALID_RESTORATION_V2_1_FULL_45_ARTIFACT` 与同一 NO-GO。轻量结论见
[`data/results/restoration_v2_1_full_45_substrate/`](data/results/restoration_v2_1_full_45_substrate/)，完整契约与
resume/INVALID/artifact 边界见 [`docs/restoration_v2_1_full_45.md`](docs/restoration_v2_1_full_45.md)。

针对 13 个已暴露 coordinate mismatch 的 bounded numerical audit 已完成 source-only 冻结，见
[`docs/spatial_reference_audit_v1.md`](docs/spatial_reference_audit_v1.md) 与
[`code/configs/spatial_reference_audit_v1.json`](code/configs/spatial_reference_audit_v1.json)。它只比较同一 H200 上的
BF16 auto、BF16 eager numerical control 和 4-state FP32 descriptive probe，总上限为 60 次 generation、120 次
teacher forward；confirm、restoration、gate training 与 coordinate radius tuning 均为 0。旧 raw archive、13-state
父投影、source inventory、唯一 no-retry attempt ledger 和 append-only exposure child ledger 都由 hash fail closed。
严格 CUDA determinism 因需要项目禁止的 `CUBLAS_WORKSPACE_CONFIG` 而不作声称；image digest、Python、
PyTorch/CUDA/cuDNN、Transformers、driver、scientific environment absence 和 observed attention backend 都在
durable claim 前 exact 核对。判定分为 eager unstable→semantic、eager stable/auto unstable→eager-specific
recovery、两者均 stable→本次 numerical audit inconclusive；FP32 与 teacher-forced margin 都不控制结论，后者也
不是旧 generation-time margin。唯一 Hyper01 attempt 已完成全部 60 次 generation 与 120 次 teacher forward：
BF16 auto 为 7/13 exact stable、BF16 eager-control 为 13/13、FP32 描述性 probe 为 4/4；confirm、restoration
和 gate 均为 0。原独立 validator 随后因把 processor target `2560` 误当成所有 realized grid 的固定 token 数而
fail closed，且还把 teacher aligned inventory 错写为 `attention_mask + input_ids`。raw profile 与 terminal
ledger 已冻结且未重跑；纯离线 validation repair 只修这两个读取契约，并要求所有 shape metadata 与固定 parent
raw witness exact，见
[`docs/spatial_reference_audit_v1_validation_repair.md`](docs/spatial_reference_audit_v1_validation_repair.md)。
repair、72-member deterministic USTAR 与 private HF immutable fresh-download 已全部闭合，正式归约为
`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；这只支持冻结新的 `v2.2-eager` runtime，不改写 v2.1 NO-GO，
也尚未授权 restoration 或 confirm。轻量结果见
[`data/results/spatial_reference_audit_v1/`](data/results/spatial_reference_audit_v1/)。

[`docs/restoration_v2_2_eager.md`](docs/restoration_v2_2_eager.md) 与
[`code/configs/causalcache_restoration_v2_2_eager.json`](code/configs/causalcache_restoration_v2_2_eager.json)
冻结了新的 eager child。它继承 v2.1 official-tool interface、45-state projection、gate 与
90/135/90 operation ceiling，唯一 scientific delta 是 BF16 eager fixed-seed/TF32-off numerical control；不声称
strict CUDA determinism。正式 attempt 必须 fresh 跑 45 states，并固定在同机同容器的两张 H200 上按偶数
index 23 states / 奇数 index 22 states 分工。global sibling ledger 必须在两个 worker import runtime 前统一
claim；旧 v2.1 raw/ledger/state output 不能复用。唯一 Hyper00 attempt 已正式通过：45/45 parse、45/45 exact
repeat、45/45 finite logits、45 个 memory-sensitive states，90/135/90 calls 精确命中，retry/top-up 为 0。
confirm、restoration 与 gate work 仍全为 0。

正式 source freeze 已在 clean pushed `main@b3a6303d69b1145fbf195e0bd18b9b3065a6f213`
建立：contract SHA256 为
`f473bb8a1657072235dd73bf78a93aff27b438d7baca65b7ef6096cf985effa7`，50-file formal inventory
SHA256 为 `bb6351e4dd5b4470ed1add86dbcbaa714a56063ba8ecf07268b6f13f630f9e54`。执行 source 为
`main@8ae07519f14ac3635f292ee93a7b6d624507427e`；102-file deterministic USTAR 已在 private HF revision
`3577099d505b8c652d764f41269df911128ec767` fresh-download 并逐 byte 复核。轻量结果见
[`data/results/restoration_v2_2_eager_full_45_substrate/`](data/results/restoration_v2_2_eager_full_45_substrate/)。

restoration v2.2 label v1 因错误 model path 在 zero-forward 阶段永久封存为 `INVALID`；replacement v2 已从
`main@5ae40d4aed4eb20b931216776b379bc6ae55629d` 完成唯一双 H200 formal run：45/45 states `PASS`，生成
420 条 canonical $D(S)$、435 条 deployment conditional-marginal labels、45 个 exact-subset oracle 与 465 个
pair interactions，retry、top-up、generation 均为 0。$B=2$ exact oracle 的 normalized recovery 为 overall
0.8735（train 0.9028 / dev 0.8149），44/45 states 获得正 oracle utility；75/435 marginals 为负且分布在
22/45 states，说明后续 gate 需要保留 set conditioning，而不是证明 learned gate 已有效。

raw label USTAR 已在 private HF
[`gavinlaw/causalcache-restoration-labels-mobile@8f6baae5`](https://huggingface.co/datasets/gavinlaw/causalcache-restoration-labels-mobile/tree/8f6baae5c0b23b08915fa1b0fb848dd519b4c8db)
fresh-download 闭合，tag 为 `v2.2-eager-train-dev-exact-v2`。Git compact result 见
[`data/results/restoration_v2_2_eager_labels_v2/`](data/results/restoration_v2_2_eager_labels_v2/)，完整协议与
artifact identity 见 [`docs/restoration_v2_2_labels.md`](docs/restoration_v2_2_labels.md)。这只是 offline
oracle/labels；gate training、matched-NLL、closed-loop 和 confirm 均未运行。

这条路线与评审建议的关键映射已经冻结：reference estimand 是 stable self-behavior，高保真干预只加入
post-state image，八字段 strong low-fidelity summary 已实现；v2.1 只修复 versioned policy interface，不改变
这些科学设计，也不覆盖旧 v2 negative results。

合作方提出的 subset optimality 问题已拆成独立 search ablation，见
[`ablations/subset_search.md`](ablations/subset_search.md)。v1 source/config/CPU runner 已冻结：小规模 exact
subset oracle、true conditional-marginal greedy、2x2 bounded exchange、true-utility beam-$2/4/8$ 分别报告
actual utility、greedy/exact ratio 与全部 coalition query cost；同时把既有 phase-0 的 averaged-score
objective-projection gap 与真正 search regret 分开。首次 CPU attempt 因 tuple/list JSON boundary 在 pre-commit
replay fail closed，output 未保留；修复 push 后，canonical rerun 与独立 pre-commit replay 已逐项通过。正式
结果中 phase-0 averaged-score / true-greedy ratio 为 0.858594/1.0；complementary trap 的 raw greedy/beam-2
为 0.5，exchange/beam-4 为 1.0；旧 v1 cached table 的四个 state/budget cases 中 greedy 全部等于 exact。
这支持保留 set-conditioned greedy 主线与 offline stronger-search diagnostics，但不是新 real-policy generalization
evidence。全程零新 policy/GPU/confirm access，v2.1 不重开。轻量结果见
[`data/results/subset_search_ablation_v1/`](data/results/subset_search_ablation_v1/)。

official Jinja/tojson compatibility 已在输出前冻结：teacher target 与 official assistant `tool_calls` JSON
逐字节一致，包括 canonical insertion order 与原始 Unicode UTF-8；解码后的 text argument 仍须严格满足 NFKC。
interface source SHA256 为 `90cbefc851bed105de6ea0c8f719aae6313a479ca4589e4160fc4ec3e8de3964`。

v2 CPU interface 已独立实现并 hash-pinned，见
[`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md) 与
[`data/manifests/restoration_v2_interfaces.json`](data/manifests/restoration_v2_interfaces.json)。14 个合法
action、23 个非法 action、6,000 个完整标量坐标检查和 decision steps 4/5/6 共 28 个 coalition（含
step-6 全部 16 个）已通过；pinned AndroidWorld `JSONAction` constructor 已在 Aries 对 14/14 payload
通过，证据见
[`data/results/restoration_v2_constructor_preflight/`](data/results/restoration_v2_constructor_preflight/)。
device-side executor dispatch 已在 Aries 正式通过 14/14 cases，negative actuation control 为 HTTP 500，
独立 reducer verdict 为 `PASSED_EXECUTOR_DISPATCH`；证据见
[`data/results/restoration_v2_executor_dispatch/`](data/results/restoration_v2_executor_dispatch/)。因此 action
dependency 已闭合。baseline 公式、
GUI-Owl final-main-merger extractor、完整 snapshot/runtime verifier 与 11-file source manifest 已通过，
dependency 6 已闭合。

OCR/image backend 的 implementation identity 已冻结为 CPU-only
`RapidOCR 3.8.4 + ONNX Runtime 1.24.4 + PP-OCRv5 mobile English`，两个 wheel、det/rec 与虽关闭但
constructor 仍加载的 classifier model 都有 exact SHA；256x256 Pillow bilinear 与 uncapped full-screen
OCR record schema 也已实现。synthetic golden 已从最终 static fixture 两次独立通过；private HF model
`v1.0.0@0dbc766a73ee88d10d52285d434dbfec58617835` 已上传、按 immutable revision fresh re-download 并逐文件
验 hash，Git source/artifact manifest 也已通过 fail-closed validator。6-image policy-blind real-screen
golden 现已完成：pre-output source contract、materializer 与独立 replay validator 先冻结，45 个 screening
states 展开为 180 个
occurrences、75 张 unique images（55 portrait / 20 landscape），confirm 有 97 张 unique images 且 SHA
交集为 0；exact 3+3 screenshots 在未读取 OCR/policy output 时固定。Hyper00 两次独立构建与 replay、HF
immutable re-download 后第三次 replay 全部通过，完整 5-file tree SHA256 为 `605d6396...7e25`。private HF
dataset tag `ocr-real-screen-golden-v1.0.0` 固定到 `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`；OCR
dependency 5 已闭合，该旧 tag 保持可复现。见
[`docs/restoration_v2_ocr.md`](docs/restoration_v2_ocr.md)。

完整 derived dataset 已从 builder commit
`1a01f2323647d092cab67f0531ecb877a4a255de` 在 Hyper00 CPU-only runtime 独立构建两次：35 条
trajectory、175 个 event、65 个 state、210 张图与 210 条 OCR record 的 exact 6-file projection
byte-identical，tree SHA256 为 `475e6cf2...a6e`，OCR aggregate 为 `1e04ddbd...010`。private HF tag
`restoration-v2-derived-v1.0.0` 固定到 immutable revision
`89f136abaff797e14fe758a198996e51032a10a6`；fresh immutable download 后第三次 210-record replay 通过。
dependency 1 已闭合，且三次均未加载 policy 或生成 restoration output。详细证据见
[`data/results/restoration_v2_derived_artifact/`](data/results/restoration_v2_derived_artifact/)。

dependency 8 的 source implementation 前置现已就绪：
[`code/causalcache/restoration_v2_gpu_kl.py`](code/causalcache/restoration_v2_gpu_kl.py) 将 BF16
candidate logits 的 FP32 `log_softmax`、full-vocabulary KL 与 action-token mean 留在同一 CUDA
device；finite/normalization/nonnegative/final-finite predicates 也全部留在 device，非法数值只映射为最终
`NaN` distance，primitive 记录 `validation_scalar_host_reads=0` 与
`full_tensor_host_transfers=0`；
[`code/causalcache/restoration_v2_batching.py`](code/causalcache/restoration_v2_batching.py) 固定
microbatch size 2，按 exact `(image_count, sequence_length)` 分组且禁止自动 OOM
fallback；[`code/causalcache/policy/gui_owl_v2_runtime.py`](code/causalcache/policy/gui_owl_v2_runtime.py)
实现 pinned snapshot/Transformers source 校验、单卡 BF16、每图 target 2560 effective visual tokens、native
batch-1 generation，以及只在 GPU 返回 tool-call distance span logits 的 batch-1/2 teacher forcing。
synthetic-only CUDA audit CLI 也已实现。Hyper00 单张 H200 formal run 已从 clean pushed commit 通过：
batch-1 对独立 float64 CPU oracle 的最大误差为 `3.05e-8`，batch-2 对两次 batch-1 完全一致，
zero-stride/NaN/no-host-intermediate-read 均通过；独立 validator 已从 run commit Git blobs 复核 source 与
全部关键字段。证据见
[`data/results/restoration_v2_gpu_compute_audit/`](data/results/restoration_v2_gpu_compute_audit/)。这仍只是
policy-blind compute audit。真实输入执行底座现已实现：confirm-safe loader 只暴露
label-train/development 的 45 states / 90 images；processor-only audit 会在不实例化 model weights、forward
或 generate 的前提下验证真实 `AutoProcessor` 的 1-image、5-image、nested batch-2、token boundary 与 exact
pixel target；readiness validator 必须同时复核 GPU audit、processor audit、全部 source Git blobs、clean
pushed `main` 和 `CONFIRM_LOCKED`，才返回 `SCREENING_ALLOWED`。production runner 随后先完成 90-prompt
shape sweep，再允许首个 policy output，并以 attempt marker 禁止崩溃后的 hidden retry。上述 source 已通过
本地 tests。正式 Hyper00 processor audit 也已通过：真实 grid 对齐后每图 2,584 effective visual tokens，
1/5-image sequence lengths 为 2,943/13,286，且所有 model/policy/restoration negative declarations 均为 false。
证据见 [`data/results/restoration_v2_processor_audit/`](data/results/restoration_v2_processor_audit/)。首次 GPU-0
execution config/readiness 已在 clean `main` 通过，历史授权见
[`data/results/restoration_v2_readiness/`](data/results/restoration_v2_readiness/)；正式 preflight 随后发现该
physical GPU 已被其他任务占用，因而没有启动 screening。
空闲 GPU 2 上的新单卡 runtime 已重新通过 GPU compute 与 processor audits，证据见
[`data/results/restoration_v2_runtime_reanchor/`](data/results/restoration_v2_runtime_reanchor/)；planned GPU-2
canonical execution config 已改绑新 container/GPU/evidence，SHA256 为 `819cb973...91ca0`。runner 还会在
artifact/model load 前实时核对 GPU UUID、单卡可见性、driver、compute capability、PyTorch/CUDA/cuDNN 与
Transformers。GPU-2 readiness manifest 已绑定 clean implementation commit `14faaa4...452aa`，formal
clean-Git authorization 在 `main@caa4f37` 返回 8/8、`SCREENING_ALLOWED + CONFIRM_LOCKED`；当前结果见
[`data/results/restoration_v2_readiness/`](data/results/restoration_v2_readiness/)。

第一次 fixed 45-state substrate screening 已在 `main@a0001cb`、Hyper00 单张 physical GPU 2 完整执行。
90-prompt shape sweep 通过，45 个 state 均生成一次；但冻结 parser 因 native envelope mismatch 在
`reference_generation_1` 得到 45/45 `PARSE_FAILURE`，正式输出 `NO_GO_V2_SUBSTRATE`。因此 teacher forward、
KL 与 restoration label 均为 0，这不是“logits 不 finite”或“memory 不敏感”的测量。原始 trace 已打成单个
deterministic tar shard 上传 private HF，轻量结论与 immutable revision 见
[`data/results/restoration_v2_substrate_screening/`](data/results/restoration_v2_substrate_screening/)。只读审计中
保守的单动作 envelope-only normalization 上界为 40/45，仍低于 0.99 gate；原 v2 结果不改写，confirm 保持
locked。用于复核该上界的 versioned compatibility parser 与 immutable archive replay 已完成 source
implementation；40 条中只有 25 条在首个 JSON 后 clean EOF，另外 15 条需要丢弃精确白名单中的残余 opener/
extra brace，且 0 条由模型生成 canonical closer，因此不能称为 native well-formed output。auditor 会逐项绑定
HF immutable revision、archive/run-contract/source commit 与 pre-registered per-state classification hash，重新执行
strict parser、保守 adapter、canonical round-trip 与 AndroidWorld bridge。该正式 replay 已在
`main@fc3adf1` 完成，得到 `NO_GO_ADAPTER_ONLY`；完整结果见
[`data/results/restoration_v2_parser_compatibility/`](data/results/restoration_v2_parser_compatibility/)。当时必须先
冻结 versioned interface/generation rescue；该步骤后来已由 v2.1 processor、pilot 与 full-45 milestones
完成并得到独立结果。

exact-ID/exposure materializer 已实现为 policy-blind CPU pipeline：它必须从 pinned 16 个 Parquet 重建
完整 111-trajectory eligible pool，并逐字节复现 frozen pool SHA，不能误从只含 8+15 条 trajectory 的
parent tar 继续抽样。pipeline 固定 `decision_count>=5` 后的首 20 条、step 6、8/15/20 disjoint proof，
同时为 45 个 screening states 和 20 个 confirm states 写 image/action content witnesses。Hyper00 已从
`main@30879c0` 完成两次 byte-identical 全量构建，两次均通过独立 validator；canonical
selection/exposure SHA 分别为 `292c7e52...` / `bc122482...`，见
[`data/results/restoration_v2_selection/`](data/results/restoration_v2_selection/)。confirm 固定 20 条、覆盖
29 个 normalized app labels，未 top-up；confirm 仍没有任何 policy/restoration output。

executor-dispatch 的 live-inspection、negative-control 与 offline-reduction 契约见
[`docs/restoration_v2_executor_dispatch.md`](docs/restoration_v2_executor_dispatch.md)。本次 run 绑定已推送
commit `b6e57c2619e88b8646657b3190bf45853a86c3d2`，未加载 policy 或使用 GPU。

v2 的干预已收窄：所有 memory 始终保留相同 strong low-fidelity summary；恢复 event 时只增加一张
post-action state image，不增加 before image 或额外 action text。confirm 固定每条 trajectory 的 decision
step 6；events 1--4 是四个 visual candidates，event 5 的 post-state 等于 current observation，因而只保留
summary、不重复计图且不能被选择。reference 使用四张历史 post-state 加 current，覆盖每个 non-current
historical event 且不重复 event 5 的 current-equivalent image；
主预算容量为四个候选中最多选两个；confirm 的冻结 report 只使用允许少选的 exact at-most-2
subset oracle，不在看到 confirm 结果后追加 exact-two 或 interaction aggregate。

历史 v1 结果保持有效且不回改：independent UI-TARS reference gate 得到 69/75 parsed、27/75（36.0%）
expert executable match、swipe 0/2，输出 `NO_GO_CURRENT_REFERENCE_STACK`。这否定的是 v1 reference
stack，不是 restoration 假设。raw 结果位于 private HF
`reference-gate-v1@b3e1245c6c6a1723fe2ca3a861148008df39df46`，轻量结论见
[`data/results/independent_reference_gate_v1/`](data/results/independent_reference_gate_v1/)。旧 15-trajectory /
132-decision oracle raw trajectories 已被 builder 读取和打包，但从未产生本项目 policy/restoration output；
v2 只按预先冻结顺序把它们用于 label-train/development screening。

更早的两状态 real-policy diagnostic 为 `INCONCLUSIVE_POSITIVE`：step 8、1024-token cap 下 exhaustive oracle
恢复 84.2%，recent/similarity 为 48.7%，random expectation 为 58.1%。由于它来自已观察 matched subset，
它仍只是一条 selection-biased existence signal，不能成为 paper `GO` 或训练 label。历史 v1 配置、artifact、
H200 anchor 和执行记录全部保留，见 [`docs/go_no_go.md`](docs/go_no_go.md) 与
[`docs/independent_gate_execution.md`](docs/independent_gate_execution.md)。

新合作者按以下顺序阅读：

1. [`docs/restoration_v2.md`](docs/restoration_v2.md)：当前 scientific contract、data roles 与 gates；
2. [`docs/restoration_v2_1.md`](docs/restoration_v2_1.md)：official-tool interface rescue 与固定 15-state pilot；
3. [`docs/restoration_v2_1_full_45.md`](docs/restoration_v2_1_full_45.md)：full-45 stable-reference substrate、gate 与一次性执行边界；
4. [`docs/restoration_v2_2_eager.md`](docs/restoration_v2_2_eager.md)：只改 eager runtime 的 fresh-45 双 H200 source-only contract；
5. [`docs/restoration_v2_2_labels.md`](docs/restoration_v2_2_labels.md)：已闭合的 attribution run、exact subset oracle、conditional-marginal labels 与 artifact identity；
6. [`docs/restoration_v2_2_expansion_exact_labels.md`](docs/restoration_v2_2_expansion_exact_labels.md)：192-state expansion labels 的 source A / runner freeze B、raw-first reduction 与唯一 attempt；
7. [`docs/restoration_v2_2_ocr_rgb_baseline.md`](docs/restoration_v2_2_ocr_rgb_baseline.md)：OCR/RGB v1 source、zero-score invalid attempt 与不可重跑边界；
8. [`docs/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md`](docs/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md)：v2 exact-occurrence repair、正式 OCR/RGB comparator result 与异常点边界；
9. [`ablations/interaction_aware_gate.md`](ablations/interaction_aware_gate.md)：interaction-aware student 的设计、能力边界与 ablation matrix；
10. [`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md)：action、strong LF 与
   post-state-only prompt 的冻结实现；
11. [`docs/execution.md`](docs/execution.md)：跨芯片执行、HF/Git 回写与 Definition of Done；
12. [`docs/restoration_v2_ocr.md`](docs/restoration_v2_ocr.md)：OCR/image identity、schema 与 golden 状态；
13. [`docs/gate_v1_fresh16_evaluation.md`](docs/gate_v1_fresh16_evaluation.md)：formal model seal 后的
    fresh-16 父科学协议、v1 pre-semantic failure 与不可续跑边界；
14. [`docs/gate_v1_fresh16_inventory_repair.md`](docs/gate_v1_fresh16_inventory_repair.md)：精确 15-path
    operational repair、Source-A/B、永久失败与不可续跑边界；
15. [`docs/gate_v1_fresh16_claim_serialization_repair.md`](docs/gate_v1_fresh16_claim_serialization_repair.md)：
    JSON-safe deep-snapshot Source-A、新 identity 与后续唯一执行顺序；
16. [`docs/progress.md`](docs/progress.md)：已完成里程碑、negative results 与下一步；
17. [`docs/experiment_contract.md`](docs/experiment_contract.md)：历史 v0.3 与不变的系统边界；
18. [`docs/go_no_go.md`](docs/go_no_go.md)：历史 v1 和当前 v2 判据；
19. [`code/README.md`](code/README.md) 与 [`data/README.md`](data/README.md)：代码和数据边界；
20. [`paper/main.tex`](paper/main.tex)：AAAI 正文 source。

仓库结构：

```text
README.md          # 总索引与交接状态
AGENTS.md          # 每步 Git/HF/compute 规则
paper/             # AAAI LaTeX package
ablations/         # 未冻结的方法比较、诊断设计与执行前边界
code/              # package、scripts、tests、configs、requirements
data/              # 小 fixture 与轻量 result summaries
docs/              # contract、execution、progress、decisions
```

快速验证：

```bash
make test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces \
  validate-restoration-v2-executor-dispatch validate-restoration-v2-selection \
  validate-restoration-v2-ocr-config validate-restoration-v2-ocr-artifact \
  validate-restoration-v2-baselines paper

cd code && python3 -m scripts.validate_restoration_v2_1_full_45_contract \
  --repository-root .. \
  --config code/configs/causalcache_restoration_v2_1_full_45.json
```

计算 placement：v2 offline substrate/attribution 默认使用 Hyper00 H200，Aries A6000 为 fallback。八项
pre-output dependencies 已闭合并完成第一次正式 screening；v2 保持
`NO_GO_V2_SUBSTRATE + NO_GO_ADAPTER_ONLY`。v2.1 versioned source/contract 已冻结，90-prompt
processor-only preflight 与 fixed-15 interface pilot 均已正式通过；唯一 unchanged-interface full-45 attempt
已在 Hyper00 完成并按冻结 gate 判为 `NO_GO_V2_1_FULL_45_SUBSTRATE`。失败点是 32/45 exact canonical
repeat agreement，而不是 interface parse 或未观测到 summary/full-history behavior distance；按照该 contract，
restoration、confirm 与 gate training 均不得继续。完整 artifact 中的 confirm bytes 只接受 loader
validation/hash，不向 processor 或 decoder 暴露任何 confirm state/prompt/image，也没有生成 confirm output。
若探索 executable/UI-element equivalence，必须先冻结新的 versioned source-only protocol，保留本次 NO_GO，
不能在当前 45 states 上 retroactive relabel。AndroidWorld closed-loop MVP 继续使用已验证的 Aries stack；
Hyper01 当前不参与本轮执行。

## 一句话主张

现有 GUI memory 方法主要根据 recency、similarity、attention 或 learned salience 保存历史。CausalCache 直接测量：**在实际部署预算附近，只给某个 GUI 事件增加 archived post-state image，能在多大程度上恢复冻结策略 parseable、finite、repeat-stable 的 self-behavior**，并把这种昂贵的 restoration attribution 蒸馏成在线 memory selector。

## 核心观察

两个 memory controller 即使获得近似相同的 next-action NLL，也可能产生完全不同的长期任务成功率。局部 action fidelity 无法区分：

- 当前动作暂时用不到、未来却决定任务成败的事件；
- 视觉上相似但对后续控制无关的事件；
- 需要保留精确文本、坐标、开关状态的 dependency-critical event；
- 仅仅获得较高 attention、但删除后并不改变策略行为的事件。

因此，memory quality 不应只由局部 prediction quality 衡量，还应由历史事件对未来 policy behavior 的**可恢复贡献**衡量。

## 问题定义

一条 GUI trajectory 表示为：

$$
H_t=\{e_1,\ldots,e_{t-1}\},\qquad e_j=(o_{j-1},a_j,o_j),
$$

其中每个 event 包含：

- action 前截图；
- 执行的 action；
- action 后截图；
- 可选的结构化 UI delta。

给定任务指令 $g$、当前观测 $o_t$ 和高保真事件预算 $B$，目标是选择：

$$
M_t\subset H_t,\qquad \sum_{j\in M_t}c_j\le B,
$$

使冻结 GUI policy 的长期 executable task success 最大。这里的 $B$ 限制 policy-visible multimodal context，不限制 persistent storage；原始事件保存在 archive，未选事件在当前 query 只暴露 deterministic strong summary，并单独报告实际 text tokens。

## 方法

### 1. Restoration Attribution

首先在 action-contract parse、finite-logit 与 repeat-stability 检查通过的开发状态上运行冻结策略 $\pi_0$，得到 self-behavior reference；untouched confirm 使用固定分母，失败状态不能事后删除：

$$
q_t=\pi_0(\cdot\mid g,o_t,H_t).
$$

将所有历史事件替换为低保真版本，得到 baseline memory。对于已恢复集合 $S$，定义与完整历史行为的距离：

$$
D_t(S)=\sum_l w_l\,
KL\!\left(
q_{t,l}(\cdot\mid a^*_{<l})
\parallel
q^S_{t,l}(\cdot\mid a^*_{<l})
\right),
$$

其中 $a^*$ 是完整历史策略生成的 canonical action，KL 在 teacher-forced action token 上计算。这样无需枚举包含 coordinate 和 text argument 的完整 sequence action space。

恢复事件 $e_j$ 在 coalition $S$ 下的 conditional marginal 为：

$$
\Delta_{j,t}(S)=
D_t(S)-D_t(S\cup\{j\})
$$

正式 independent target 固定为 empty 与 singleton-conditioned marginal 的投影：

$$
G^{\mathrm{ind}}_{j,t}=\frac12\left[
\Delta_{j,t}(\varnothing)+
\frac{1}{|C_t|-1}\sum_{i\in C_t\setminus\{j\}}
\Delta_{j,t}(\{i\})
\right].
$$

它来自会保留 redundancy/complementarity 的 coalition-conditioned policy rerun，但部署时压缩为每个 event
一个 query-dependent 分数，不假设集合 utility 可加。formal labels 使用已有完整 $D(S)$ table；confirm 的
4 个候选精确计算全部 16 个 coalition，并按冻结口径报告 exact at-most-2 subset oracle。完整 coalition
rows 作为 raw evidence 保留，但不把未预注册的 exact-two 或 interaction aggregate 临时加入 confirm 主报告。
回归 target 是 $G^{\mathrm{ind}}_{j,t}/\max(D_t(\varnothing),10^{-12})$；当
$D_t(\varnothing)\le10^{-12}$ 时只从 normalized regression/mean 排除，raw target、正负号、ranking、raw
evaluation 与 state count 全部保留。

### 2. 在线 Memory Gate（当前主方法）

离线 restoration attribution 计算昂贵，因此训练轻量 gate：

$$
s_{j,t}=f_\theta(q_{64,t},h_{64,j},q_{64,t}\odot h_{64,j},g_{8,j,t})
$$

其中 frozen input 是固定 200d `[q64,h64,q64*h64,g8]`：`q64` 编码 instruction/current OCR，`h64`
编码 event summary/candidate OCR，`g8` 编码 age/action/OCR/change/result geometry；不输入 policy-vision
feature 或 requested budget。模型预测 normalized $G^{\mathrm{ind}}_{j,t}$。这里采用 **query-time scoring**：
历史事件在每个决策时刻根据当前状态重新打分，解决 arrival-time label 随未来时刻变化的契约问题。
AndroidWorld live adapter 同样已经在 confirm 前冻结：raw transition 通过 canonical action、pinned OCR、文本
delta 和 256×256 RGB MAD 生成同一 `low_fidelity_v2`；history 只能 oldest→newest，age 固定为 `N-1-index`，
最新 current-equivalent event 被排除，其余任意长度历史全部打分。RGB PNG 使用 generic image contract，且
`foreground_app/executor_result` 继续固定为训练语义中的 `unknown`。

训练目标组合为：

$$
L=L_{\text{regression}}
+0.25L_{\text{pairwise-ranking}}.
$$

推理时等成本 event 使用 5-seed mean score，按 `(-score, step_id)` 排序，只选择严格正分的前 2 个；
阈值在 confirm 前固定为 0，因此实际选择可以少于预算容量。冻结 action policy，只训练 memory gate。

该写法是 interacting set utility 的 independent projection，不表示 event utility 真正可加。已经冻结失败的
set-conditioned v1 永久保持 `NO_V2_CONDITIONAL_RESCUE`，只作为 negative ablation；greedy、exact subset、
interaction 与 pure-complementarity 分析见
[`ablations/interaction_aware_gate.md`](ablations/interaction_aware_gate.md)。

### 3. Mixed-Fidelity Memory

v2 每个 low-fidelity event 固定包含：

```text
step_id
action_type
action_argument
foreground_app
screen_text_added
screen_text_removed
screen_change
executor_result
```

系统由 raw event archive、cheap summary/index 和 policy-visible high-fidelity context 三层组成。所有 memory 中 summary 序列化 byte-identical；high fidelity 只增加一张 post-action state image。当前 formal independent gate 不使用 policy-vision embedding；该信号只属于单独 baseline。第一版不做跨层 KV surgery，而是通过 mixed-fidelity input 重新运行 policy，避免位置编码与上下文依赖导致不合法的 KV 拼接。

## 实验设计

### Benchmarks

- 主要 closed-loop benchmark：AndroidWorld；
- 离线 action prediction 与 memory attribution：GUI-Odyssey、AndroidControl 或同类数据。

### Baselines

- no history、full history、summary only；
- recent top-$B$、random top-$B$；
- visual/text similarity；
- attention-based retention；
- learned salience selector；
- MementoGUI-style memory control；
- AndroTMem/anchor-style memory；
- offline restoration oracle；
- distilled CausalCache gate。

### Metrics

- closed-loop task success；
- action type、target、text argument accuracy；
- successor-action NLL；
- retained restoration mass；
- visual-token budget；
- inference latency；
- success per unit memory/compute。

## 决定论文成败的实验

构造 successor-action NLL 相近的 memory pairs，比较 terminal success：

$$
\operatorname{NLL}(M_1)\approx\operatorname{NLL}(M_2).
$$

当：

$$
\operatorname{RestorationMass}(M_1)>
\operatorname{RestorationMass}(M_2),
$$

检验 $M_1$ 是否仍显著获得更高任务成功率。这是论文最关键的证据：**restoration relevance 捕获了 one-step fidelity 无法解释的长期控制信息。**

正式统计使用 paired closed-loop 两臂的 state union。在每个 state 上以
`M_independent ∪ M_recent`（最多 4 张历史图加 current）生成 repeat-stable pair-union action，同时计算两种
memory 的 action-token NLL 与 raw restoration mass `R=D(empty)-D(M)`。主 caliper 固定为 `0.05`
nats/token。统计层级固定为 origin 内等 state → 两 origin 各 1/2 → template 内固定 instance 等权 →
template 等权；test 的 `[0,1,2]` 三个 instances 必须作为同一个 template cluster。对 matched template 计算
`sign_eps(R_ind-R_recent)*(success_ind-success_recent)`；train-60 只作 development diagnostic，只有
sealed-test-75 至少 15 个 matched template clusters 且冻结的 90% template-bootstrap lower 大于 0 才支持
primary mechanism association。`0.02/0.10` 只作 sensitivity，不能替代 primary，也不解释为 causal
mediation effect。

## Ablations

- permutation 数量与 attribution variance；
- KL、JS、action log-prob recovery；
- event、screenshot、UI element 三种粒度；
- 去掉 normalized regression 或 pairwise ranking；
- query-time gate 对比 arrival-time gate；
- 不同 memory budget；
- success、NLL 与 restoration mass 的独立相关性；
- 跨 app、任务长度和 policy backbone 泛化。
- independent average-value gate 对比 set-conditioned iterative gate，并按 interaction mass 分层报告 oracle gap；
  设计与限制见 [`ablations/interaction_aware_gate.md`](ablations/interaction_aware_gate.md)。

## 预期贡献

1. 提出与部署预算对齐的 restoration-guided GUI memory attribution。
2. 提出将反事实 restoration gain 蒸馏成在线固定预算 memory gate 的方法。
3. 证明 matched next-action fidelity 下，不同 memory 仍产生不同长期成功率。
4. 在相同多模态 memory budget 下，提高 long-horizon GUI task success。

## Claim 边界

这里的 causal 含义限定为：

> 对冻结策略行为进行受控恢复干预得到的 counterfactual attribution。

本项目不声称识别真实环境结构因果，也不把该方法包装成 world model。

## 与最接近工作的差异

- [MementoGUI](https://arxiv.org/abs/2605.18652) 通过训练数据学习 memory selection、compression 和 retrieval；
- [AndroTMem](https://arxiv.org/abs/2603.18429) 使用 causally linked state anchors；
- CausalCache 的事件重要性来自：恢复该事件后，冻结策略行为的边际恢复量，而不是人工 salience、结构规则或相似度。

## Falsification Criteria

满足任一条件就应弱化主张或停止投稿：

- restoration score 不能比 recency、attention 或 similarity 更好地预测 terminal success；
- matched-NLL 后 restoration mass 与成功率不再相关；
- distilled gate 明显无法逼近 restoration oracle；
- 相同预算下 task success 没有稳定提升；
- 方法收益完全来自增加输入 token；
- attribution 成本无法通过少量 permutation 控制。

## Paper Story

> Long-horizon GUI memory selection lacks policy-grounded supervision. CausalCache values an event with a frozen projection of coalition-conditioned behavioral restoration, then distills that signal into a fixed-budget query-time independent gate and validates it through paired executable control.

## 初始路线图

- [x] 固化问题定义、核心机制、实验主线与 claim 边界；
- [x] 建立并验证 AAAI-27 官方 LaTeX anonymous submission 骨架；
- [x] 冻结 action serialization、validated-reference requirements 与 mixed-fidelity experiment contract；
- [x] 固定并评估首个 frozen policy candidate；因 full-history coverage 仅 2/9，拒绝作为主 teacher；
- [x] 冻结 restoration v2 primary policy 与 stable self-behavior reference；GUI-Owl Instruct 仅作为 v2 substrate，不回改其 v1 AndroidWorld rejection；
- [x] 冻结 v2 restricted action、strong LF、post-state-only prompt 与 interface source hashes；
- [x] 在 pinned AndroidWorld 对 14/14 payload 闭合 `JSONAction` constructor；
- [x] 在 Aries 正式闭合 14-case executor dispatch 与 negative actuation control；
- [x] 实现并测试完整 111-pool reconstruction、exact-ID selection 与 append-only exposure materializer；
- [x] 从 pinned source 正式冻结 exact 20-state confirm IDs、65 个 state witnesses 与 exposure ledger；
- [x] 完成 OCR identity、synthetic/real-screen golden 与 immutable HF model/dataset artifact；
- [x] 完成 baseline formulas、policy-vision extractor 与 source hashes；
- [x] 完成完整 derived artifact、immutable HF revision 与 fresh-download replay；
- [x] 完成真实 processor audit，冻结 exact grid/token/tensor evidence；
- [x] 冻结 execution config；
- [x] 物化并正式验证 readiness manifest；
- [x] 实现 trajectory/event schema 与 deterministic low-fidelity summarizer；
- [x] 实现并测试 budget-conditioned restoration attribution 核心；
- [x] 在 synthetic frozen behavior 上验证方差、ranking stability、负 gain 和 interaction error；
- [x] 在已接入的真实轨迹上实现 teacher-forced policy distance 与固定 45-state substrate runner；
- [x] 完成第一次固定 45-state substrate screening；strict parse 0/45，合法输出 `NO_GO_V2_SUBSTRATE`，confirm 未打开；
- [x] 完成 immutable raw-trace parser compatibility replay；40/45 低于 required 45/45，正式为 `NO_GO_ADAPTER_ONLY`；
- [x] 冻结 versioned native-output/generation rescue source，并完成 v2.1 90-prompt processor-only preflight；
- [x] 完成 v2.1 fixed-15 native-output interface pilot；15/15 parse/closer/bridge，保留原 v2 negative result；
- [x] 冻结 unchanged-interface full-45 v2.1 child contract、runner 与 raw artifact chain；
- [x] 执行唯一 full-45 v2.1 substrate attempt；32/45 exact repeat agreement，正式为 `NO_GO_V2_1_FULL_45_SUBSTRATE`；
- [x] 记录 prospective interaction-aware gate ablation；仅 source-only proposal，不重开 v2.1；
- [x] 冻结 subset-search v1 source/config/CPU runner，并完成 formal synthetic + cached-table replay；
- [x] 首次 subset-search CPU attempt 在 pre-commit JSON round-trip validation fail closed；output 未保留；
- [x] push serialization fix 后从新 clean source canonical rerun；pre-commit scientific replay 通过；
- [x] 提交/push subset-search result，并从 clean descendant main 通过 committed full-payload validator；
- [x] 完成 bounded spatial audit、离线 validator repair、deterministic USTAR 与 private HF immutable binding；
- [x] 冻结新的 `v2.2-eager` runtime/source 与双 H200 fresh-45 contract；
- [x] 运行唯一 `v2.2-eager` fresh 45-state substrate并闭合 private HF immutable artifact；45/45 exact repeat，正式 PASS；
- [x] v2.2 已稳定，因此不进入 executable/UI-element equivalence 补救分支；
- [x] 冻结 restoration v2.2 attribution source、exact subset oracle 与 conditional-marginal label contract；source 已 push；
- [x] formal v1 attempt 按 contract 永久封存为 zero-forward `INVALID`；根因是 model snapshot existence 校验晚于 durable claim，而非 restoration estimand 结果；
- [x] 冻结 replacement v2 attempt identity、独立 outcome/HF path 与 pre-claim model snapshot validator；
- [x] 在固定双 H200 23/22 parity、microbatch 1 下完成 v2 formal labels，并闭合 private HF immutable artifact；
- [x] 冻结 v2.2 selector geometry v1 source；preliminary replay 的核心数值已独立复算，但 reporting contract 不完整，旧 output 未提交；
- [x] 从 clean pushed source 执行并验证 v2 reporting-only repair；144 个 interaction joint cells 与 analytic-random 集合几何完整落盘；
- [x] 冻结 primary `n=4,B=2` OCR/RGB source-only contract；
- [x] OCR/RGB v1 首次 Hyper00 attempt 在 zero-score identity scan 阶段 fail closed，并记录 canonical output absent；
- [x] 冻结新 identity 的 exact-occurrence/equal-value lexer repair source；v1 default semantics 与 failure binding 保持不变；
- [x] 在 Hyper00 执行 OCR/RGB v2 aggregate、逐 byte replay与独立数值审计，并锁定 15-row artifact regression；
- [x] 冻结 policy-vision feature-only source/config、双 H200 worker isolation 与 replay contract；
- [x] 执行 policy-vision v1 formal attempt；在 0 feature 时因 pinned PyTorch UUID object type drift fail closed并留证；
- [x] 冻结只修 `torch._C._CUuuid` normalization 的 policy-vision v2 source；v1 default 与 failure bytes 保持不变；
- [x] 执行唯一 policy-vision v2 attempt；UUID repair 通过，但在 0 feature 的 SizeDict-interface check fail closed并留证；
- [x] 冻结新的 exact loaded SizeDict-interface v3 repair；v1/v2 identity 与失败证据保持不变；
- [x] 执行唯一 policy-vision v3 formal attempt并提交 exact-three artifact；GPU schedule/replay 完成且不允许重跑；
- [x] 冻结 state-projection-only CPU validation repair source；producer/artifact/repair commit 与 exact-byte边界已分离；
- [x] 执行唯一 CPU-only validation-repair audit并锁定 exact-two sibling result；三份 producer artifact 逐 byte 相同；
- [x] 从 clean pushed `main@174801112c58d831249fd54f4f8bc9af01524b44` 执行只读 `validate`，返回 `REVALIDATED`；
- [x] 根据 geometry 结论冻结 gate v1 training/evaluation contract：set-conditioned 为主方法、parameter-matched independent 为 comparator，fresh-16 是唯一 formal GO slice；
- [x] 实现 gate v1 label-blind feature、conditional/independent trainer、train-only OOF、严格 ensemble provenance 与 fresh-16/combined-21 evaluation source；synthetic-only CPU smoke 不产生 checkpoint 或 paper metric；
- [x] 从 clean pushed main 物化并验证 48/16 structural manifest；192 states / 1792 distance rows / 1856 edges；
- [x] 物化 pre-output exposure ledger；expansion-64 与 prior-output-23 / confirm-20 的六组交集全部为空；
- [x] 构建并上传 policy-blind expansion derived artifact；64 trajectories / 192 decision views / 384 images，private HF immutable download 已完成第三遍 OCR replay；
- [x] 冻结 192-state substrate source-only contract：双 H200 96/96 parity、384 generation / 576 teacher forward / 384 KL、逐 excluded-event action canary 与完整 transitive source lock；
- [x] 从 clean pushed `main@6bf3f8c` 物化 canonical substrate config；SHA256 `42144f33...a6e5`，source-only validator 明确不授权 GPU；
- [x] 实现实际 processor serializer、双 worker runner、raw artifact manager 与 source-locked monitor sidecar；
- [x] push runner source commit A、物化并 push 67-file runner freeze commit B，并完成唯一双 H200 substrate 与 immutable HF 闭环；
- [x] 执行唯一 expansion exact-label v1 GPU attempt；192/192 scientific payload 完整，但 monitor cadence gate
  违规，formal attempt 永久 `INVALID`；
- [x] 冻结 source-only invalid-forensic transport：双 ledger namespace、strict USTAR、406 members、永久
  formal-ineligible；
- [x] 从真实 v1 bytes 生成并 readback 验证 local invalid-forensic archive；
- [x] 将 invalid-forensic archive 发布到独立 private HF repo，并由 read-only tag-resolution child 完成 immutable
  fresh replay、P0 strict readback 与幂等复验；
- [x] 实现 stdlib-only independent expansion math audit；从 raw $D(S)$ 独立重算 edges、interactions、Shapley
  与 exact oracle，并在 synthetic 192-state denominator 上逐字段匹配现有 reducer；
- [x] 冻结并执行 invalid-evidence read-only transport child；annotated-tag object 与 resolved commit 分开绑定，
  remote mutation count 为 0；
- [x] 实现并审计 ledger-neutral CPU scientific-repair core；formal/test-only tier、captured production replay、
  exact P0 gate 与 structural monitor rule 均已覆盖；
- [x] 冻结 no-GPU scientific-repair runner/config/bootstrap/state machine；29/29 focused、173/173 expansion
  regression 通过；source-freeze 时未执行 formal run，现已由下一项闭合；
- [x] 在 Hyper00 执行 no-GPU scientific-repair runner 与完整只读 revalidation；local repaired payload 为
  `VALID + REVALIDATED`，原 producer 仍永久 `INVALID`；
- [x] 冻结 repaired-label private-HF publication contract；annotated-tag object/resolved commit、exact-pair
  provenance、response-loss recovery 与 completion-last 已由 25/25 focused tests 覆盖；
- [x] 发布 repaired-label private-HF artifact，完成 immutable 幂等 replay 与独立只读 postflight；只解除
  formal-58 label-data prerequisite；
- [x] 冻结 formal-58 train-only cache Source-A：selective feature/label readers、分阶段 state machine、exact-three
  HF bundle 与 Source-A/Execution-B 边界均已实现；本项不读取 formal semantics 或生成 cache；
- [x] 从 clean pushed Source-A 机械生成并单独 push runner freeze B；首次 Hyper00 run 在 pre-semantic feature
  transport binding 处 fail-closed，旧 claim 与失败证据已封存，HF destination 仍不存在；
- [x] 冻结只修复一个 SHA leaf、使用独立 local/HF namespace 的 transport-repair Source-A；它在 token/Hub/new
  claim 前绑定旧 claim、旧输出缺失和 producer 三重 witness；
- [x] 从 clean pushed repair Source-A 机械生成唯一 Execution-B，并在 Hyper00 no-GPU runtime materialize +
  immutable replay formal-58 feature/label cache；v1 claim 保留，repair cache 已绑定 private HF immutable commit；
- [x] 冻结 formal-train Source-A：精确绑定 repair cache/preregistration、train-only OOF/final-fit、十个输出
  checkpoint 与 private HF model destination；source-only validator 不授权执行；
- [x] 从 clean pushed Source-A 机械生成唯一 Execution-B，完成 formal-58 OOF/final fit，封存十个
  checkpoint 并完成 private HF immutable model replay；
- [x] 完成 fresh-16 evaluation Source-A 的 focused regression、全量回归审计与 source-only validator；当前冻结
  16/48/144/448 denominator、variable-`n` heuristics、49/97 policy schedule、all-selections-before-label firewall
  与 9+4 two-commit publication，不代表已读取 fresh data；
- [x] 将上述 fresh-16 Source-A 作为 focused commit push 到 canonical `main`；
- [x] 从 clean pushed A 机械生成并单独 push v1 runner-freeze B；唯一 Hyper00 attempt 在 exact-tree preflight
  pre-semantic fail closed，旧 receipts/evidence 与 destination absence 已封存，v1 不可续跑；
- [x] 完成 full-inventory repair Source-A 的实现、回归、source-only freeze，并随本 milestone commit/push；
- [x] 从 clean pushed repair A 重放 validator，机械生成并单独 push 唯一 repair B；唯一 Hyper00 attempt 在
  label-blind semantic/GPU phase 后、label claim 持久化前因 `mappingproxy` serialization 永久 fail closed；
- [x] 冻结 claim-serialization repair Source-A：只修 JSON-safe deep snapshot，绑定旧 repair
  evidence/successor absence 与全新 namespace；57-path source inventory、focused/full regression 已闭合，B
  在该历史 milestone 当时 absent、未执行；Source-A 随后成为 canonical；
- [x] 从 clean pushed claim-repair A 重放 source validation，机械生成唯一 direct-child B 并单独 commit/push；
- [x] 对已冻结 ensemble 执行一次性 fresh-16 primary，完成 13-target private-HF publication 与独立 immutable
  validate；结果为 selector `NO-GO`、set-conditioning `NO-GO`；
- [x] 完成只读 fresh-16 failure-decomposition Source-A/Execution-B、private-HF exact-three publication 与
  immutable replay；route=`NO_V2_CONDITIONAL_RESCUE`，项目选择不执行 post-primary combined-21 compatibility；
  confirm、matched-NLL 与 closed-loop 的 post-GO 权限继续保持 locked；
- [x] 冻结 independent confirm restoration-only continuation Source-A：只读采用既有 exact payload，加入
  isolated snapshot verifier、fresh-exec CUDA tripwire、durable attempt terminal 与 partial-publication reconciliation；
  112 项 focused regression 通过；
- [x] 从 clean pushed continuation Source-A 机械生成唯一 Execution-B，并在 Hyper00 四张 H200 上完成 fresh
  topology、原冻结 restoration/report、private-HF report/tag 与 immutable replay；有效结果为
  `NO_GO_INDEPENDENT_CONFIRM`，因此 paired closed-loop、matched-NLL 与 sealed test 未执行；
- [x] 完成 consumed confirm-20 的 oracle-independent `J` failure decomposition；`J` 显著高于 learned `I`，
  但 raw-primary 仍低于 OCR/RGB，正式为 `CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB`；
- [ ] 整理论文与复现实验配置。

## Source of Truth

### Code and Documentation

- GitHub: <https://github.com/luojiaxuan/CausalCache>
- Canonical branch: `main`
- Paper source: [`paper/main.tex`](paper/main.tex)
- Code layout and commands: [`code/README.md`](code/README.md)
- Small-data policy: [`data/README.md`](data/README.md)
- Cross-chip execution and handoff: [`docs/execution.md`](docs/execution.md)
- Independent confirm continuation protocol: [`docs/independent_confirm_continuation_v1.md`](docs/independent_confirm_continuation_v1.md)
- Independent confirm continuation result:
  [`data/results/independent_confirm20_continuation_v1/`](data/results/independent_confirm20_continuation_v1/)
- Independent confirm failure-decomposition protocol/result:
  [`docs/independent_confirm_failure_decomposition_v1.md`](docs/independent_confirm_failure_decomposition_v1.md),
  [`data/results/independent_confirm20_failure_decomposition_v1/`](data/results/independent_confirm20_failure_decomposition_v1/)
- Independent confirm failure-decomposition Source-A/Execution-B:
  [`code/configs/causalcache_independent_confirm20_failure_decomposition_v1.json`](code/configs/causalcache_independent_confirm20_failure_decomposition_v1.json),
  [`code/configs/causalcache_independent_confirm20_failure_decomposition_runner_v1.json`](code/configs/causalcache_independent_confirm20_failure_decomposition_runner_v1.json)
- Independent confirm continuation contract and source validator:
  [`code/configs/causalcache_independent_confirm_continuation_v1.json`](code/configs/causalcache_independent_confirm_continuation_v1.json),
  [`code/scripts/validate_independent_confirm_continuation.py`](code/scripts/validate_independent_confirm_continuation.py)
- Ablation index: [`ablations/README.md`](ablations/README.md)
- Interaction-aware gate proposal: [`ablations/interaction_aware_gate.md`](ablations/interaction_aware_gate.md)
- Subset-search contract and interpretation: [`ablations/subset_search.md`](ablations/subset_search.md)
- Frozen subset-search config: [`code/configs/subset_search_ablation_v1.json`](code/configs/subset_search_ablation_v1.json)
- Subset-search canonical CPU result: [`data/results/subset_search_ablation_v1/`](data/results/subset_search_ablation_v1/)
- Spatial reference audit protocol: [`docs/spatial_reference_audit_v1.md`](docs/spatial_reference_audit_v1.md)
- Spatial validator-repair contract: [`docs/spatial_reference_audit_v1_validation_repair.md`](docs/spatial_reference_audit_v1_validation_repair.md)
- Spatial validator-repair config: [`code/configs/spatial_reference_audit_v1_validation_repair_v1.json`](code/configs/spatial_reference_audit_v1_validation_repair_v1.json)
- Spatial reference audit canonical result: [`data/results/spatial_reference_audit_v1/`](data/results/spatial_reference_audit_v1/)
- Restoration v2.2-eager source-only contract: [`docs/restoration_v2_2_eager.md`](docs/restoration_v2_2_eager.md)
- Restoration v2.2-eager machine-readable config: [`code/configs/causalcache_restoration_v2_2_eager.json`](code/configs/causalcache_restoration_v2_2_eager.json)
- Restoration v2.2-eager canonical result: [`data/results/restoration_v2_2_eager_full_45_substrate/`](data/results/restoration_v2_2_eager_full_45_substrate/)
- Restoration v2.2 label protocol: [`docs/restoration_v2_2_labels.md`](docs/restoration_v2_2_labels.md)
- Restoration v2.2 label machine-readable contract: [`code/configs/causalcache_restoration_v2_2_labels.json`](code/configs/causalcache_restoration_v2_2_labels.json)
- Restoration v2.2 label v2 repair contract: [`code/configs/causalcache_restoration_v2_2_labels_v2_repair.json`](code/configs/causalcache_restoration_v2_2_labels_v2_repair.json)
- Restoration v2.2 label source validator: [`code/scripts/validate_restoration_v2_2_label_contract.py`](code/scripts/validate_restoration_v2_2_label_contract.py)
- Restoration v2.2 formal label runner: [`code/scripts/run_restoration_v2_2_labels.py`](code/scripts/run_restoration_v2_2_labels.py)
- Restoration v2.2 label artifact manager: [`code/scripts/manage_restoration_v2_2_label_artifact.py`](code/scripts/manage_restoration_v2_2_label_artifact.py)
- Restoration v2.2 label v1 zero-forward failure: [`data/results/restoration_v2_2_eager_labels_v1_attempt/`](data/results/restoration_v2_2_eager_labels_v1_attempt/)
- Restoration v2.2 label v2 canonical result: [`data/results/restoration_v2_2_eager_labels_v2/`](data/results/restoration_v2_2_eager_labels_v2/)
- Gate v1 preregistration: [`docs/gate_v1_preregistration.md`](docs/gate_v1_preregistration.md)
- Gate v1 machine-readable contract: [`code/configs/causalcache_gate_v1_preregistration.json`](code/configs/causalcache_gate_v1_preregistration.json)
- Gate v1 source validator: [`code/scripts/validate_gate_v1_contract.py`](code/scripts/validate_gate_v1_contract.py)
- Gate v1 formal-58 cache protocol: [`docs/gate_v1_formal_cache.md`](docs/gate_v1_formal_cache.md)
- Gate v1 formal-58 cache Source-A config: [`code/configs/causalcache_gate_v1_formal_cache_v1.json`](code/configs/causalcache_gate_v1_formal_cache_v1.json)
- Gate v1 formal-58 cache manager: [`code/scripts/manage_gate_v1_formal_cache.py`](code/scripts/manage_gate_v1_formal_cache.py)
- Gate v1 formal-58 transport-repair protocol: [`docs/gate_v1_formal_cache_transport_repair.md`](docs/gate_v1_formal_cache_transport_repair.md)
- Gate v1 formal-58 transport-repair Source-A config: [`code/configs/causalcache_gate_v1_formal_cache_transport_repair_v1.json`](code/configs/causalcache_gate_v1_formal_cache_transport_repair_v1.json)
- Gate v1 formal-58 transport-repair Execution-B: [`code/configs/causalcache_gate_v1_formal_cache_transport_repair_runner_v1.json`](code/configs/causalcache_gate_v1_formal_cache_transport_repair_runner_v1.json)
- Gate v1 formal-58 transport-repair manager: [`code/scripts/manage_gate_v1_formal_cache_transport_repair.py`](code/scripts/manage_gate_v1_formal_cache_transport_repair.py)
- Gate v1 formal-58 transport-repair result: [`data/results/gate_v1_formal58_cache_transport_repair_v1/`](data/results/gate_v1_formal58_cache_transport_repair_v1/)
- Gate v1 formal-train Source-A protocol: [`docs/gate_v1_formal_train.md`](docs/gate_v1_formal_train.md)
- Gate v1 formal-train Source-A config: [`code/configs/causalcache_gate_v1_formal_train_v1.json`](code/configs/causalcache_gate_v1_formal_train_v1.json)
- Gate v1 formal-58 training completion: [`data/results/gate_v1_formal58_train_v1/`](data/results/gate_v1_formal58_train_v1/)
- Gate v1 formal-train source validator: [`code/scripts/validate_gate_v1_formal_train_contract.py`](code/scripts/validate_gate_v1_formal_train_contract.py)
- Gate v1 formal-train manager: [`code/scripts/manage_gate_v1_formal_train.py`](code/scripts/manage_gate_v1_formal_train.py)
- Gate v1 trainer/evaluator execution: [`docs/gate_v1_execution.md`](docs/gate_v1_execution.md)
- Gate v1 fresh-16 evaluation protocol: [`docs/gate_v1_fresh16_evaluation.md`](docs/gate_v1_fresh16_evaluation.md)
- Gate v1 fresh-16 Source-A config: [`code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json`](code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json)
- Gate v1 fresh-16 source validator: [`code/scripts/validate_gate_v1_fresh16_evaluation_contract.py`](code/scripts/validate_gate_v1_fresh16_evaluation_contract.py)
- Gate v1 fresh-16 inventory-repair protocol: [`docs/gate_v1_fresh16_inventory_repair.md`](docs/gate_v1_fresh16_inventory_repair.md)
- Gate v1 fresh-16 inventory-repair Source-A config: [`code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json`](code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json)
- Gate v1 fresh-16 inventory-repair manager: [`code/scripts/manage_gate_v1_fresh16_inventory_repair.py`](code/scripts/manage_gate_v1_fresh16_inventory_repair.py)
- Gate v1 fresh-16 inventory-repair v1 failure: [`data/results/gate_v1_fresh16_inventory_repair_v1_attempt/`](data/results/gate_v1_fresh16_inventory_repair_v1_attempt/)
- Gate v1 fresh-16 claim-serialization repair protocol: [`docs/gate_v1_fresh16_claim_serialization_repair.md`](docs/gate_v1_fresh16_claim_serialization_repair.md)
- Gate v1 fresh-16 claim-serialization repair Source-A config: [`code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json`](code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json)，SHA256 `3979573be235d630ee2f46dc23be8747a843190c9b57ee81e1b3a17b4416d8c7`；57-path source inventory SHA256 `572992cf5ac75142cdf8f0e82bdfc2caad43ba37632c741eae277285db54d689`
- Gate v1 fresh-16 primary result: [`data/results/gate_v1_fresh16_claim_serialization_repair_v1/`](data/results/gate_v1_fresh16_claim_serialization_repair_v1/)；private HF tag `gate-v1-fresh16-claim-serialization-repair-v1` → `3541fe1ea2c46e555c29cc53483e6f3b809f8f81`
- Gate v1 fresh-16 failure-decomposition protocol: [`docs/gate_v1_fresh16_failure_decomposition.md`](docs/gate_v1_fresh16_failure_decomposition.md)
- Gate v1 fresh-16 failure-decomposition Source-A config: [`code/configs/causalcache_gate_v1_fresh16_failure_decomposition_v1.json`](code/configs/causalcache_gate_v1_fresh16_failure_decomposition_v1.json)，SHA256 `fa2cd3759150f838fce78b72a987d7a889ef23f5eec5ec41286ae69090104f1f`
- Gate v1 fresh-16 failure-decomposition result: [`data/results/gate_v1_fresh16_failure_decomposition_v1/`](data/results/gate_v1_fresh16_failure_decomposition_v1/)；private HF tag `gate-v1-fresh16-failure-decomposition-v1` → exact-three commit `9aa2540088c575f5c34dfb208e4f429fa53d355b`；route `NO_V2_CONDITIONAL_RESCUE`
- Gate v1 synthetic-only smoke: [`code/scripts/run_gate_v1_trainer_smoke.py`](code/scripts/run_gate_v1_trainer_smoke.py)
- Label-expansion exposure protocol: [`docs/restoration_v2_2_label_expansion_exposure.md`](docs/restoration_v2_2_label_expansion_exposure.md)
- Label-expansion exposure materializer: [`code/scripts/materialize_restoration_v2_2_label_expansion_exposure.py`](code/scripts/materialize_restoration_v2_2_label_expansion_exposure.py)
- Label-expansion derived-artifact protocol: [`docs/restoration_v2_2_label_expansion_derived.md`](docs/restoration_v2_2_label_expansion_derived.md)
- Label-expansion derived builder: [`code/scripts/build_guiodyssey_restoration_v2_expansion.py`](code/scripts/build_guiodyssey_restoration_v2_expansion.py)
- Label-expansion derived validator: [`code/scripts/validate_guiodyssey_restoration_v2_expansion.py`](code/scripts/validate_guiodyssey_restoration_v2_expansion.py)
- Label-expansion derived completion: [`data/results/restoration_v2_2_label_expansion_derived/`](data/results/restoration_v2_2_label_expansion_derived/)
- Label-expansion substrate source freeze: [`docs/restoration_v2_2_label_expansion_substrate.md`](docs/restoration_v2_2_label_expansion_substrate.md)
- Label-expansion substrate materializer: [`code/scripts/materialize_restoration_v2_2_expansion_substrate_contract.py`](code/scripts/materialize_restoration_v2_2_expansion_substrate_contract.py)
- Label-expansion substrate source validator: [`code/scripts/validate_restoration_v2_2_expansion_substrate_contract.py`](code/scripts/validate_restoration_v2_2_expansion_substrate_contract.py)
- Label-expansion substrate canonical config: [`code/configs/causalcache_restoration_v2_2_expansion_substrate_v1.json`](code/configs/causalcache_restoration_v2_2_expansion_substrate_v1.json)
- Label-expansion substrate runner protocol: [`docs/restoration_v2_2_expansion_substrate_runner.md`](docs/restoration_v2_2_expansion_substrate_runner.md)
- Label-expansion substrate runner: [`code/scripts/run_restoration_v2_2_expansion_substrate.py`](code/scripts/run_restoration_v2_2_expansion_substrate.py)
- Label-expansion substrate canonical result: [`data/results/restoration_v2_2_label_expansion_substrate_v1/`](data/results/restoration_v2_2_label_expansion_substrate_v1/)
- Expansion exact-label v1 invalid attempt: [`data/results/restoration_v2_2_expansion_exact_labels_v1_attempt/`](data/results/restoration_v2_2_expansion_exact_labels_v1_attempt/)
- Expansion exact-label invalid-forensic protocol: [`docs/restoration_v2_2_expansion_labels_invalid_forensic.md`](docs/restoration_v2_2_expansion_labels_invalid_forensic.md)
- Expansion exact-label invalid-forensic result: [`data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_v1/`](data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_v1/)
- Invalid-forensic P1 publication protocol: [`docs/restoration_v2_2_expansion_labels_invalid_forensic_publication.md`](docs/restoration_v2_2_expansion_labels_invalid_forensic_publication.md)
- Invalid-forensic P1 fail-closed result: [`data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_publication_v1_attempt/`](data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_publication_v1_attempt/)
- Invalid-forensic read-only tag-resolution protocol: [`docs/restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution.md`](docs/restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution.md)
- Invalid-forensic read-only tag-resolution result: [`data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_tag_resolution_v1/`](data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_tag_resolution_v1/)
- Expansion-label ledger-neutral scientific-repair core: [`code/causalcache/restoration_v2_2_expansion_labels_scientific_repair.py`](code/causalcache/restoration_v2_2_expansion_labels_scientific_repair.py)
- Expansion-label scientific-repair protocol: [`docs/restoration_v2_2_expansion_labels_scientific_repair.md`](docs/restoration_v2_2_expansion_labels_scientific_repair.md)
- Expansion-label scientific-repair runner contract: [`code/configs/causalcache_restoration_v2_2_expansion_labels_scientific_repair_runner_v1.json`](code/configs/causalcache_restoration_v2_2_expansion_labels_scientific_repair_runner_v1.json)
- Expansion-label scientific-repair runner: [`code/causalcache/restoration_v2_2_expansion_labels_scientific_repair_runner.py`](code/causalcache/restoration_v2_2_expansion_labels_scientific_repair_runner.py)
- Expansion-label scientific-repair direct bootstrap: [`code/scripts/run_restoration_v2_2_expansion_labels_scientific_repair.py`](code/scripts/run_restoration_v2_2_expansion_labels_scientific_repair.py)
- Expansion-label scientific-repair local result: [`data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_v1/`](data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_v1/)
- Expansion-label repaired publication protocol: [`docs/restoration_v2_2_expansion_labels_scientific_repair_publication.md`](docs/restoration_v2_2_expansion_labels_scientific_repair_publication.md)
- Expansion-label repaired publication contract: [`code/configs/causalcache_restoration_v2_2_expansion_labels_scientific_repair_publication_v1.json`](code/configs/causalcache_restoration_v2_2_expansion_labels_scientific_repair_publication_v1.json)
- Expansion-label repaired publication manager: [`code/scripts/manage_restoration_v2_2_expansion_labels_scientific_repair_publication.py`](code/scripts/manage_restoration_v2_2_expansion_labels_scientific_repair_publication.py)
- Expansion-label repaired publication result: [`data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/`](data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/)
- Label-expansion substrate raw-artifact manager: [`code/scripts/manage_restoration_v2_2_expansion_substrate_artifact.py`](code/scripts/manage_restoration_v2_2_expansion_substrate_artifact.py)
- Frozen label-expansion structural manifest: [`data/manifests/restoration_v2_2_label_expansion_selection.json`](data/manifests/restoration_v2_2_label_expansion_selection.json)
- Frozen label-expansion exposure ledger: [`data/manifests/restoration_v2_2_label_expansion_exposure.json`](data/manifests/restoration_v2_2_label_expansion_exposure.json)
- Restoration v2.2 selector-geometry protocol: [`docs/restoration_v2_2_selector_geometry.md`](docs/restoration_v2_2_selector_geometry.md)
- Restoration v2.2 selector-geometry machine-readable contract: [`code/configs/causalcache_restoration_v2_2_selector_geometry.json`](code/configs/causalcache_restoration_v2_2_selector_geometry.json)
- Restoration v2.2 selector-geometry validator: [`code/scripts/validate_restoration_v2_2_selector_geometry_contract.py`](code/scripts/validate_restoration_v2_2_selector_geometry_contract.py)
- Restoration v2.2 selector-geometry v2 reporting-repair contract: [`code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json`](code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json)
- Restoration v2.2 selector-geometry v2 source validator: [`code/scripts/validate_restoration_v2_2_selector_geometry_v2_contract.py`](code/scripts/validate_restoration_v2_2_selector_geometry_v2_contract.py)
- Restoration v2.2 selector-geometry v2 runner: [`code/scripts/run_restoration_v2_2_selector_geometry_v2.py`](code/scripts/run_restoration_v2_2_selector_geometry_v2.py)
- Restoration v2.2 selector-geometry v2 canonical result: [`data/results/restoration_v2_2_selector_geometry_v2_repair/`](data/results/restoration_v2_2_selector_geometry_v2_repair/)
- Restoration v2.2 OCR/RGB source-only protocol: [`docs/restoration_v2_2_ocr_rgb_baseline.md`](docs/restoration_v2_2_ocr_rgb_baseline.md)
- Restoration v2.2 OCR/RGB v1 zero-score failure: [`data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/`](data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/)
- Restoration v2.2 OCR/RGB machine-readable contract: [`code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json`](code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json)
- Restoration v2.2 OCR/RGB source validator: [`code/scripts/validate_restoration_v2_2_ocr_rgb_contract.py`](code/scripts/validate_restoration_v2_2_ocr_rgb_contract.py)
- Restoration v2.2 OCR/RGB formal reducer: [`code/scripts/run_restoration_v2_2_ocr_rgb_baseline.py`](code/scripts/run_restoration_v2_2_ocr_rgb_baseline.py)
- Restoration v2.2 OCR/RGB v2 identity-repair protocol: [`docs/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md`](docs/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md)
- Restoration v2.2 OCR/RGB v2 identity-repair contract: [`code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json`](code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json)
- Restoration v2.2 OCR/RGB v2 validator: [`code/scripts/validate_restoration_v2_2_ocr_rgb_contract_v2.py`](code/scripts/validate_restoration_v2_2_ocr_rgb_contract_v2.py)
- Restoration v2.2 OCR/RGB v2 runner: [`code/scripts/run_restoration_v2_2_ocr_rgb_baseline_v2.py`](code/scripts/run_restoration_v2_2_ocr_rgb_baseline_v2.py)
- Restoration v2.2 OCR/RGB v2 canonical result: [`data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/`](data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/)
- Restoration v2.2 OCR/RGB v2 artifact regression: [`code/tests/test_restoration_v2_2_ocr_rgb_v2_artifact.py`](code/tests/test_restoration_v2_2_ocr_rgb_v2_artifact.py)
- Material-run metadata schema: [`code/configs/run_manifest.schema.json`](code/configs/run_manifest.schema.json)
- Current restoration v2 contract: [`docs/restoration_v2.md`](docs/restoration_v2.md)
- Machine-readable v2 config: [`code/configs/causalcache_restoration_v2.json`](code/configs/causalcache_restoration_v2.json)
- Frozen v2 interface semantics: [`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md)
- Frozen v2 interface hashes: [`data/manifests/restoration_v2_interfaces.json`](data/manifests/restoration_v2_interfaces.json)
- Pinned AndroidWorld constructor preflight: [`data/results/restoration_v2_constructor_preflight/`](data/results/restoration_v2_constructor_preflight/)
- Executor-dispatch contract: [`docs/restoration_v2_executor_dispatch.md`](docs/restoration_v2_executor_dispatch.md)
- Pinned AndroidWorld executor-dispatch result: [`data/results/restoration_v2_executor_dispatch/`](data/results/restoration_v2_executor_dispatch/)
- Restoration-v2 selection materializer: [`code/causalcache/data/restoration_v2_selection.py`](code/causalcache/data/restoration_v2_selection.py)
- Frozen restoration-v2 exact selection: [`data/manifests/restoration_v2_selection.json`](data/manifests/restoration_v2_selection.json)
- Frozen restoration-v2 exposure ledger: [`data/manifests/restoration_v2_exposure.json`](data/manifests/restoration_v2_exposure.json)
- Restoration-v2 selection result: [`data/results/restoration_v2_selection/`](data/results/restoration_v2_selection/)
- Restoration-v2 OCR/image contract: [`docs/restoration_v2_ocr.md`](docs/restoration_v2_ocr.md)
- Restoration-v2 OCR backend config: [`code/configs/restoration_v2_ocr_backend.json`](code/configs/restoration_v2_ocr_backend.json)
- Restoration-v2 OCR implementation and validator: [`code/causalcache/restoration_v2_text_backend.py`](code/causalcache/restoration_v2_text_backend.py), [`code/scripts/validate_restoration_v2_ocr_backend.py`](code/scripts/validate_restoration_v2_ocr_backend.py)
- Restoration-v2 OCR runtime lock and synthetic fixture: [`code/requirements/restoration_v2_ocr_lock.txt`](code/requirements/restoration_v2_ocr_lock.txt), [`data/fixtures/restoration_v2_ocr_golden.json`](data/fixtures/restoration_v2_ocr_golden.json)
- Restoration-v2 OCR source/artifact manifest: [`data/manifests/restoration_v2_ocr_backend.json`](data/manifests/restoration_v2_ocr_backend.json)
- Restoration-v2 OCR evidence: [`data/results/restoration_v2_ocr_backend/`](data/results/restoration_v2_ocr_backend/)
- Restoration-v2 real-screen materializer: [`code/causalcache/data/restoration_v2_real_screen.py`](code/causalcache/data/restoration_v2_real_screen.py), [`code/scripts/materialize_restoration_v2_real_screen.py`](code/scripts/materialize_restoration_v2_real_screen.py)
- Restoration-v2 real-screen source/artifact validator: [`code/scripts/validate_restoration_v2_real_screen.py`](code/scripts/validate_restoration_v2_real_screen.py)
- Frozen real-screen pre-output source contract: [`data/manifests/restoration_v2_real_screen_source.json`](data/manifests/restoration_v2_real_screen_source.json)
- Passed real-screen run/HF summary: [`data/results/restoration_v2_ocr_backend/real_screen_summary.json`](data/results/restoration_v2_ocr_backend/real_screen_summary.json)
- Restoration-v2 deterministic baseline formulas: [`code/causalcache/restoration_v2_baselines.py`](code/causalcache/restoration_v2_baselines.py)
- Restoration-v2 policy-vision extractor: [`code/causalcache/policy/gui_owl_v2_vision.py`](code/causalcache/policy/gui_owl_v2_vision.py)
- Restoration-v2.2 policy-vision protocol: [`docs/restoration_v2_2_policy_vision_baseline.md`](docs/restoration_v2_2_policy_vision_baseline.md)
- Restoration-v2.2 policy-vision source contract: [`code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json`](code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json)
- Restoration-v2.2 policy-vision source validator: [`code/scripts/validate_restoration_v2_2_policy_vision_contract.py`](code/scripts/validate_restoration_v2_2_policy_vision_contract.py)
- Restoration-v2.2 policy-vision formal runner: [`code/scripts/run_restoration_v2_2_policy_vision_baseline.py`](code/scripts/run_restoration_v2_2_policy_vision_baseline.py)
- Restoration-v2.2 policy-vision UUID repair contract: [`code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json`](code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json)
- Restoration-v2.2 policy-vision UUID repair validator: [`code/scripts/validate_restoration_v2_2_policy_vision_v2_contract.py`](code/scripts/validate_restoration_v2_2_policy_vision_v2_contract.py)
- Restoration-v2.2 policy-vision v2 formal runner: [`code/scripts/run_restoration_v2_2_policy_vision_baseline_v2.py`](code/scripts/run_restoration_v2_2_policy_vision_baseline_v2.py)
- Restoration-v2.2 policy-vision SizeDict repair contract: [`code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair.json`](code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair.json)
- Restoration-v2.2 policy-vision SizeDict repair validator: [`code/scripts/validate_restoration_v2_2_policy_vision_v3_contract.py`](code/scripts/validate_restoration_v2_2_policy_vision_v3_contract.py)
- Restoration-v2.2 policy-vision v3 formal runner: [`code/scripts/run_restoration_v2_2_policy_vision_baseline_v3.py`](code/scripts/run_restoration_v2_2_policy_vision_baseline_v3.py)
- Restoration-v2.2 policy-vision reducer: [`code/causalcache/restoration_v2_2_policy_vision.py`](code/causalcache/restoration_v2_2_policy_vision.py)
- Restoration-v2.2 policy-vision regressions: [`code/tests/test_restoration_v2_2_policy_vision.py`](code/tests/test_restoration_v2_2_policy_vision.py)、[`code/tests/test_run_restoration_v2_2_policy_vision_baseline.py`](code/tests/test_run_restoration_v2_2_policy_vision_baseline.py)
- Restoration-v2.2 policy-vision v1 invalid attempt: [`data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/`](data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/)
- Restoration-v2.2 policy-vision v2 invalid attempt: [`data/results/restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/`](data/results/restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/)
- Restoration-v2.2 policy-vision v3 producer artifact: [`data/results/restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair/`](data/results/restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair/)
- Restoration-v2.2 policy-vision v3 CPU validation failure: [`data/results/restoration_v2_2_policy_vision_baseline_v3_cpu_validation_attempt/`](data/results/restoration_v2_2_policy_vision_baseline_v3_cpu_validation_attempt/)
- Restoration-v2.2 policy-vision v3 CPU repair contract: [`code/configs/causalcache_restoration_v2_2_policy_vision_v3_validation_repair_v1.json`](code/configs/causalcache_restoration_v2_2_policy_vision_v3_validation_repair_v1.json)
- Restoration-v2.2 policy-vision v3 VALID CPU audit: [`data/results/restoration_v2_2_policy_vision_baseline_v3_validation_repair_v1/`](data/results/restoration_v2_2_policy_vision_baseline_v3_validation_repair_v1/)
- Restoration-v2.2 policy-vision v3 audit regression: [`code/tests/test_restoration_v2_2_policy_vision_v3_validation_repair_artifact.py`](code/tests/test_restoration_v2_2_policy_vision_v3_validation_repair_artifact.py)
- Restoration-v2.2 feature-only runtime: [`code/causalcache/policy/gui_owl_v2_2_vision_runtime.py`](code/causalcache/policy/gui_owl_v2_2_vision_runtime.py)
- Restoration-v2 baseline source manifest: [`data/manifests/restoration_v2_baselines.json`](data/manifests/restoration_v2_baselines.json)
- Restoration-v2 derived dataset builder: [`code/scripts/build_guiodyssey_restoration_v2.py`](code/scripts/build_guiodyssey_restoration_v2.py)
- Restoration-v2 derived dataset validator: [`code/scripts/validate_guiodyssey_restoration_v2.py`](code/scripts/validate_guiodyssey_restoration_v2.py)
- Restoration-v2 derived artifact completion index: [`data/manifests/restoration_v2_derived_artifact.json`](data/manifests/restoration_v2_derived_artifact.json)
- Restoration-v2 derived artifact evidence: [`data/results/restoration_v2_derived_artifact/`](data/results/restoration_v2_derived_artifact/)
- Historical experiment contract v0.3: [`docs/experiment_contract.md`](docs/experiment_contract.md)
- Frozen policy selection: [`docs/policy_selection.md`](docs/policy_selection.md)
- AndroidWorld benchmark-native stack: [`docs/androidworld_stack.md`](docs/androidworld_stack.md)
- AndroidWorld frozen task partition: [`docs/androidworld_task_partition.md`](docs/androidworld_task_partition.md)
- Progress record: [`docs/progress.md`](docs/progress.md)
- Synthetic estimator validation: [`data/results/synthetic_phase0/README.md`](data/results/synthetic_phase0/README.md)
- Qwen3-VL real-policy smoke test: [`data/results/qwen_policy_smoke/README.md`](data/results/qwen_policy_smoke/README.md)
- Qwen3-VL full-history coverage: [`data/results/qwen_policy_coverage/README.md`](data/results/qwen_policy_coverage/README.md)
- UI-TARS full-history coverage: [`data/results/ui_tars_policy_coverage/README.md`](data/results/ui_tars_policy_coverage/README.md)
- Independent gate artifact index: [`data/manifests/independent_reference_gate_v1_artifact.json`](data/manifests/independent_reference_gate_v1_artifact.json)
- Independent UI-TARS reference rejection: [`data/results/independent_reference_gate_v1/README.md`](data/results/independent_reference_gate_v1/README.md)
- OpenCUA-7B pinned snapshot manifest: [`code/configs/open_cua_7b_snapshot.json`](code/configs/open_cua_7b_snapshot.json)
- OpenCUA-7B pinned runtime dependency: [`code/requirements/opencua.txt`](code/requirements/opencua.txt)
- OpenCUA-7B logits and mixed-fidelity smoke: [`data/results/open_cua_policy_smoke/README.md`](data/results/open_cua_policy_smoke/README.md)
- OpenCUA-7B full-history coverage: [`data/results/open_cua_policy_coverage/README.md`](data/results/open_cua_policy_coverage/README.md)
- ShowUI-2B pinned snapshot manifest: [`code/configs/showui_2b_snapshot.json`](code/configs/showui_2b_snapshot.json)
- ShowUI-2B logits and mixed-fidelity smoke: [`data/results/showui_policy_smoke/README.md`](data/results/showui_policy_smoke/README.md)
- ShowUI-2B full-history coverage: [`data/results/showui_policy_coverage/README.md`](data/results/showui_policy_coverage/README.md)
- GUI-Owl-1.5-8B pinned snapshot manifest: [`code/configs/gui_owl_1_5_8b_snapshot.json`](code/configs/gui_owl_1_5_8b_snapshot.json)
- GUI-Owl-1.5-8B-Think pinned snapshot manifest: [`code/configs/gui_owl_1_5_8b_think_snapshot.json`](code/configs/gui_owl_1_5_8b_think_snapshot.json)
- Replacement teacher preregistration: [`code/configs/androidworld_replacement_teacher_v1.json`](code/configs/androidworld_replacement_teacher_v1.json)
- AndroidWorld stack preregistration: [`code/configs/androidworld_stack.json`](code/configs/androidworld_stack.json)
- AndroidWorld task partition manifest: [`code/configs/androidworld_task_partition.json`](code/configs/androidworld_task_partition.json)
- AndroidWorld validation execution plan: [`code/configs/androidworld_validation_plan.json`](code/configs/androidworld_validation_plan.json)
- GUI-Owl native logits/history smoke: [`data/results/gui_owl_native_smoke/README.md`](data/results/gui_owl_native_smoke/README.md)
- GUI-Owl model-default native-resolution smoke: [`data/results/gui_owl_native_resolution_smoke/README.md`](data/results/gui_owl_native_resolution_smoke/README.md)
- AndroidWorld environment/reward smoke: [`data/results/androidworld_environment_smoke/README.md`](data/results/androidworld_environment_smoke/README.md)
- GUI-Owl AndroidWorld validation smoke: [`data/results/gui_owl_androidworld_validation_smoke/README.md`](data/results/gui_owl_androidworld_validation_smoke/README.md)
- GUI-Owl configuration-invalid validation audit: [`data/results/gui_owl_androidworld_validation_attempt2/README.md`](data/results/gui_owl_androidworld_validation_attempt2/README.md)
- GUI-Owl native-resolution validation rejection: [`data/results/gui_owl_androidworld_validation/README.md`](data/results/gui_owl_androidworld_validation/README.md)
- GUI-Owl Think strict-parser smoke: [`data/results/gui_owl_1_5_8b_think_smoke_strict/README.md`](data/results/gui_owl_1_5_8b_think_smoke_strict/README.md)
- GUI-Owl Think passing native smoke: [`data/results/gui_owl_1_5_8b_think_smoke/README.md`](data/results/gui_owl_1_5_8b_think_smoke/README.md)
- GUI-Owl Think AndroidWorld validation rejection: [`data/results/gui_owl_1_5_8b_think_androidworld_validation/README.md`](data/results/gui_owl_1_5_8b_think_androidworld_validation/README.md)
- Restoration v2 real processor audit: [`data/results/restoration_v2_processor_audit/README.md`](data/results/restoration_v2_processor_audit/README.md)
- Restoration v2 execution config: [`code/configs/restoration_v2_execution_hyper00_v1.json`](code/configs/restoration_v2_execution_hyper00_v1.json)
- Restoration v2 readiness authorization: [`data/results/restoration_v2_readiness/README.md`](data/results/restoration_v2_readiness/README.md)
- Restoration v2 GPU-2 runtime re-anchor: [`data/results/restoration_v2_runtime_reanchor/README.md`](data/results/restoration_v2_runtime_reanchor/README.md)
- Restoration v2 fixed 45-state substrate result: [`data/results/restoration_v2_substrate_screening/README.md`](data/results/restoration_v2_substrate_screening/README.md)
- Restoration v2 parser replay golden contract: [`data/manifests/restoration_v2_parser_compatibility_golden.json`](data/manifests/restoration_v2_parser_compatibility_golden.json)
- Restoration v2 formal parser compatibility replay: [`data/results/restoration_v2_parser_compatibility/README.md`](data/results/restoration_v2_parser_compatibility/README.md)
- Restoration v2.1 frozen pilot contract: [`code/configs/causalcache_restoration_v2_1_pilot.json`](code/configs/causalcache_restoration_v2_1_pilot.json)
- Restoration v2.1 interface and execution boundary: [`docs/restoration_v2_1.md`](docs/restoration_v2_1.md)
- Restoration v2.1 passed processor preflight: [`data/results/restoration_v2_1_processor_preflight/README.md`](data/results/restoration_v2_1_processor_preflight/README.md)
- Restoration v2.1 passed fixed-15 interface pilot: [`data/results/restoration_v2_1_interface_pilot/README.md`](data/results/restoration_v2_1_interface_pilot/README.md)
- Restoration v2.1 frozen full-45 contract: [`code/configs/causalcache_restoration_v2_1_full_45.json`](code/configs/causalcache_restoration_v2_1_full_45.json)
- Restoration v2.1 full-45 execution boundary: [`docs/restoration_v2_1_full_45.md`](docs/restoration_v2_1_full_45.md)
- Restoration v2.1 full-45 NO-GO result: [`data/results/restoration_v2_1_full_45_substrate/README.md`](data/results/restoration_v2_1_full_45_substrate/README.md)
- Restoration v2.1 processor-only audit CLI: `cd code && python3 -m scripts.audit_gui_owl_v2_1_processor --help`
- Restoration v2.1 fixed-15 no-retry runner: `cd code && python3 -m scripts.run_restoration_v2_1_interface_pilot --help`
- Restoration v2.1 raw evidence manager: `cd code && python3 -m scripts.manage_restoration_v2_1_pilot_artifact --help`
- Restoration v2.1 full-45 runner: `cd code && python3 -m scripts.run_restoration_v2_1_full_45_substrate --help`
- Restoration v2.1 full-45 artifact manager: `cd code && python3 -m scripts.manage_restoration_v2_1_full_45_artifact --help`
- Build command: `make paper`
- Test command: `make test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces validate-restoration-v2-executor-dispatch validate-restoration-v2-selection validate-restoration-v2-ocr-config validate-restoration-v2-ocr-artifact validate-restoration-v2-baselines`；v2.1 pilot/full-45 contracts 分别用 `cd code && python3 -m scripts.validate_restoration_v2_1_contract --repository-root .. --config code/configs/causalcache_restoration_v2_1_pilot.json` 与 `cd code && python3 -m scripts.validate_restoration_v2_1_full_45_contract --repository-root .. --config code/configs/causalcache_restoration_v2_1_full_45.json`
- 当前状态：v1 UI-TARS reference 以 27/75、swipe 0/2 判负；v2 第一次固定 45-state screening 在
  clean `main` 完成，strict parse 0/45，正式为 `NO_GO_V2_SUBSTRATE + CONFIRM_LOCKED`；事后 immutable
  replay 的保守上界也仅 40/45，正式为 `NO_GO_ADAPTER_ONLY`。已有 45 个 native policy outputs，但没有
  teacher forward、KL、restoration label 或 CausalCache 方法效果结果。v2.1 official-tool contract、
  90-prompt processor preflight 与唯一 fixed-15 pilot 均已正式通过；后者 15/15 parse/closer/bridge，raw
  artifact 已绑定 private HF immutable revision。唯一 full-45 formal attempt 得到 45/45 parse、32/45 exact
  canonical repeat agreement、32/45 finite-logit coverage 与 32 个 memory-sensitive states，正式为
  `NO_GO_V2_1_FULL_45_SUBSTRATE`。bounded spatial audit 随后得到 auto 7/13、eager 13/13，并在不重跑
  profile 的离线 repair 后正式归约为 `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；private HF immutable
  artifact 已 fresh-download 复核。v2.2-eager 随后在两张 H200 上按 23/22 parity 完成唯一 fresh-45 attempt：
  45/45 parse、45/45 exact repeat、45/45 finite logits、45 个 memory-sensitive states，正式为
  `PASS_V2_2_EAGER_FULL_45_SUBSTRATE`；raw artifact 已绑定 private HF immutable revision。restoration v2.2
  label v1 source 已在 clean pushed `main@3942d687d03bf63ea683fe8ad906a161eb10dc27` 冻结，contract SHA256
  为 `56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87`，29-file inventory SHA256
  为 `8d0ecf6b0df75f9fa7753631b8fca5efd98c71a7953007e684393e6e8fe52d9b`。formal v1 attempt 因指定 model
  snapshot directory 不存在而在 0/45 attempted states、0 teacher forward、0 KL 时 fail closed；原 ledger/root
  保留且不可重跑，没有 raw labels 或 HF artifact。replacement v2 identity 与 pre-claim full snapshot validator 已
  冻结；下一步只运行固定双 H200 schedule。gate、matched-NLL、closed-loop 与 confirm 仍未运行且不属于本步。

### Data and Models

| Artifact | Canonical location | Revision/status | Notes |
| --- | --- | --- | --- |
| GUIOdyssey pilot trajectory | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-pilot-mobile> | `1de9c34ff029d4c01665cdaca74436ae24bff276`，private | schema v0.3；10 screenshots、9 events、9 decisions |
| Independent GUIOdyssey gate artifact | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-independent-mobile> | `v0.1.0` / `84c9f5a335e9612ccb4bd566f977574f359b2485`，private | schema v0.4；reference 8 trajectories/75 decisions；oracle 15/132；immutable re-download verified |
| Independent UI-TARS reference run | 同一 private independent dataset repo | `reference-gate-v1` / `b3e1245c6c6a1723fe2ca3a861148008df39df46` | 69/75 parsed、27/75 match、swipe 0/2；`NO_GO_CURRENT_REFERENCE_STACK`；oracle 未运行 |
| Rejected policy candidate | <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct> | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | full-history executable-match 2/9；不作为主 teacher |
| Rejected GUI-tuned candidate | <https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B> | `683d002dd99d8f95104d31e70391a39348857f4e` | parsed 9/9、executable-match 4/9；未通过预注册 50% gate |
| Rejected computer-use candidate | <https://huggingface.co/xlangai/OpenCUA-7B> | `a2efb7d2b104d477a4a2666a357e79550a28aafc` | parsed 7/9、executable-match 1/9；未通过预注册 gate |
| Rejected GUI navigation candidate | <https://huggingface.co/showlab/ShowUI-2B> | `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60` | parsed 9/9、executable-match 2/9；未通过预注册 gate |
| GUI-Owl-1.5-8B-Instruct | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct> | `06d5faecff74840bab2be2425e9c42667a5d04fc` | v1 AndroidWorld success teacher 被拒；v2 首次 fixed screen 因 strict envelope parse 0/45 判 `NO_GO_V2_SUBSTRATE` |
| Rejected replacement candidate | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Think> | `afe3707fc84caebc4d7046118b34493ecf8bb060` | 512/513 parsed；official-success 上界 29/62，未通过 50% gate |
| AndroidWorld native validation traces | <https://huggingface.co/datasets/gavinlaw/causalcache-androidworld-validation-mobile> | `v0.2.0` / `0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`，private | 42 Think traces；deterministic gzip JSONL；`v0.1.0` Instruct artifact 保持不变 |
| Restoration v2 OCR models | <https://huggingface.co/gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en> | `v1.0.0` / `0dbc766a73ee88d10d52285d434dbfec58617835`，private | 三份 ONNX、model card 与 manifest；fresh immutable re-download 后 6/6 file hashes verified |
| Restoration v2 derived dataset | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-restoration-v2-mobile> | `restoration-v2-derived-v1.0.0` / `89f136abaff797e14fe758a198996e51032a10a6`，private | exact 6-file derived projection 已 fresh re-download 并第三次 replay；旧 OCR golden tag 仍固定到 `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b` |
| Label-expansion derived dataset | 同一 private restoration-v2 dataset repo | `restoration-v2-label-expansion-v1.0.0` / `630363a6adb692d72774f16dd0653a50216313ff` | 64 trajectories / 192 views / 384 images；exact-six tree `9394b369...94abc`，immutable fresh-download OCR replay 384/384 |
| Label-expansion substrate v1 | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile> | `v2.2-label-expansion-substrate-v1` / `25ac19cf6ef98adc243d421cd0039ac104ddb539`，private | 192/192、0 failed；185 memory-sensitive；384/576/384 exact counts；406-file USTAR SHA256 `4e77a38b...ff47d`，fresh immutable byte replay verified |
| Expansion exact-label v1 attempt / invalid forensic | [private HF invalid-attempt dataset](https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-invalid-attempts-mobile)；[failure binding](data/results/restoration_v2_2_expansion_exact_labels_v1_attempt/) / [transport result](data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_tag_resolution_v1/) | `v2.2-expansion-exact-labels-v1-attempt-1-forensic-v1` / `5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416`，private | 192/192 raw body complete；406-member forensic USTAR SHA256 `8e205d73...ca489` immutable replay verified；producer 仍 `INVALID` 且 formal-ineligible |
| Expansion exact-label scientific-repair child | [private HF repaired-label dataset](https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile)；[scientific result](data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_v1/) / [publication result](data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/) | `v2.2-expansion-exact-labels-scientific-repair-v1` / annotated object `6a907ba2...a4a57` / resolved commit `7a6c254b...5cef3`，private | 4-member USTAR SHA256 `1a9fdcc0...e01` 已 `VALID + REVALIDATED` 并 immutable replay/postflight；原 producer 不重分类；仅 formal-58 label-data prerequisite cleared，gate 未训练 |
| Gate v1 formal-58 train-only cache | v1 [failure evidence](data/results/gate_v1_formal58_cache_v1_attempt/)；repair [private HF dataset](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile)；[completion record](data/results/gate_v1_formal58_cache_transport_repair_v1/) | v1 A=`990f015` / B=`079c095` permanently invalid；repair A=`4f8c01b` / B=`f96c197`，tag `gate-v1-formal58-cache-transport-repair-v1` → `a61b31b...cc386` | repair 只更正证实错误的 SHA leaf，旧 claim/旧输出缺失先验证；cache 已 immutable-revalidated 并由 formal training 消费 |
| Restoration v2 first substrate trace | 同一 private restoration-v2 dataset repo | `restoration-v2-substrate-screening-v1.0.0` / `c073e143b935a79befd8ab1fd7123796792efad8` | fixed 45 states；strict 0/45、conservative recovery 40/45；raw shard + manifest fresh-download verified；`NO_GO_V2_SUBSTRATE` / `NO_GO_ADAPTER_ONLY` |
| Restoration v2.1 processor preflight | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile> | `v2.1-processor-preflight-v1` / `85576161b7cb8bbae14e46a482c42b5be5bf1d7e`，private | 90-prompt CPU-only PASS；raw SHA256 `5349ffc6...499191`、7,609,803 bytes；fresh immutable download verified |
| Restoration v2.1 interface pilot trace | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-1-interface-pilot-mobile> | `v2.1-interface-pilot-v1` / `bdff8ca71f150afd80d6291b4ecec76cbf9e7432`，private | fixed-15 PASS；raw USTAR SHA256 `f71d5fd5...32064`、133,120 bytes；fresh immutable download verified |
| Restoration v2.1 full-45 substrate trace | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile> | `v2.1-full-45-substrate-v1` / `814506ef1450838d4bc6ed3d89fe53e0773d92fb`，private | 45/45 parse、32/45 exact repeat agreement、32 memory-sensitive；`NO_GO_V2_1_FULL_45_SUBSTRATE`；raw USTAR SHA256 `8cd53d6e...f4fa4`、962,560 bytes；fresh immutable download verified |
| Spatial reference audit trace | <https://huggingface.co/datasets/gavinlaw/causalcache-spatial-reference-audit-mobile> | `spatial-reference-audit-v1` / `d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`，private | auto 7/13、eager 13/13、FP32 4/4 descriptive；`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；72-member USTAR SHA256 `d62ad05f...e5ecc`；fresh immutable canonical rebuild verified |
| Restoration v2.2-eager fresh-45 trace | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile> | `v2.2-eager-full-45-substrate-v1` / `3577099d505b8c652d764f41269df911128ec767`，private | 45/45 parse/repeat/finite logits、45 memory-sensitive；`PASS_V2_2_EAGER_FULL_45_SUBSTRATE`；raw USTAR SHA256 `b22827e6...09fb5`、fresh immutable download verified |
| Restoration v2.2 exact labels | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-labels-mobile> | `v2.2-eager-train-dev-exact-v2` / `8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`，private | v1 zero-forward `INVALID`；v2 已完成 45 states、420 raw `D(S)` rows、45 exact-subset oracle、435 deployment conditional marginals；fresh immutable download verified |
| Gate v1 formal-58 selector ensemble | [private HF model](https://huggingface.co/gavinlaw/causalcache-gate-v1-formal58-selector-mobile)；[completion record](data/results/gate_v1_formal58_train_v1/) | tag `gate-v1-formal58-train-v1` → manifest commit `23f6786075c7bff91f93fd7e8a878e070efb72a9`；annotated tag `fa85e746...b4d6` | 5 conditional + 5 independent checkpoints、2 full OOF reports、4 manifests；immutable replay 0 mutation；independent 5-seed ensemble 已冻结为新 confirm 主线，不重训 |
| Gate v1 fresh-16 primary evaluation | [protocol](docs/gate_v1_fresh16_claim_serialization_repair.md)；[completion record](data/results/gate_v1_fresh16_claim_serialization_repair_v1/)；[private HF dataset](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile) | claim repair A=`f0dd53b` / B=`ce523ff`；tag `gate-v1-fresh16-claim-serialization-repair-v1` → report commit `3541fe1ea2c46e555c29cc53483e6f3b809f8f81`；annotated tag object `34d5928...c4339` | `COMPLETED + REVALIDATED`；conditional normalized/exact=`0.6942`、raw/exact=`0.9431`、vs strongest heuristic delta=`+0.3261`；selector `NO-GO`、set-conditioning `NO-GO`；当时 confirm/matched-NLL/closed-loop locked，后续 confirm 终态见下行 |
| Independent confirm-20 v1 attempt | [contract](code/configs/causalcache_independent_confirm_closed_loop_v1.json)；[protocol](docs/independent_confirm_closed_loop_v1.md)；[failure evidence](data/results/independent_confirm20_v1_attempt/)；[private HF dataset](https://huggingface.co/datasets/gavinlaw/causalcache-independent-confirm20-mobile) | A=`e1cc8b3`，B=`f1e9196`；payload `6d0cd959...9f52d`；旧 attempt tombstone 时 tag/report absent | 20-state label-blind payload 已 seal/replay；restoration 在 reference output 0 时因 CUDA-after-fork 失效。永久 execution `INVALID`，无科学结论；当时 closed-loop locked，只允许采用 exact payload 的 versioned continuation |
| Independent confirm-20 restoration continuation v1 | [contract](code/configs/causalcache_independent_confirm_continuation_v1.json)；[protocol](docs/independent_confirm_continuation_v1.md)；[result](data/results/independent_confirm20_continuation_v1/)；[private HF report](https://huggingface.co/datasets/gavinlaw/causalcache-independent-confirm20-mobile/tree/a0b408e58d629299be334a74ecbd0ec2fa2ed1fc/independent-confirm20/v1/report) | A=`8c0d3ae`，B=`68e71fd`；payload `6d0cd959...9f52d` → report `a0b408e5...d1fc`；tag `independent-confirm20-v1` | `COMPLETED_AND_PUBLISHED`；20/20 memory-sensitive，但 independent/exact raw=`0.8213 < 0.85` 且弱于三个 heuristic；有效 `NO_GO_INDEPENDENT_CONFIRM`，closed-loop/matched-NLL/sealed test 未执行 |
| Independent confirm-20 oracle-independent failure decomposition | [protocol](docs/independent_confirm_failure_decomposition_v1.md)；[result](data/results/independent_confirm20_failure_decomposition_v1/)；[private HF child](https://huggingface.co/datasets/gavinlaw/causalcache-independent-confirm20-failure-decomposition-mobile/tree/31aa5e22c08a3d56bc729fcbc88d790cce4249c0) | A=`d3451db`，B=`9f20e4b`；tag `independent-confirm20-failure-decomposition-v1` → report commit `31aa5e22...49c0`；annotated tag object `a7d6838d...d450` | `CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB`；exact/OCR/`J`/`I` raw=`0.8564/0.7833/0.7653/0.7034`；student gap 存在但 projection ceiling 已低于 OCR，停止同 objective rescue；closed-loop/matched-NLL 继续 locked |

Pilot 的生成配置见 [`code/configs/guiodyssey_pilot.json`](code/configs/guiodyssey_pilot.json)，independent
artifact 见 [`code/configs/independent_reference_gate_v1.json`](code/configs/independent_reference_gate_v1.json)。
v2 完整 derived dataset、OCR 三模型、6-image real-screen golden 与第一次 screening raw trace 已成为 private
HF dataset/model canonical artifacts。Hyper00 model cache 仍可重建；native outputs 不留在 Git，而由完成态
Git manifest/result 绑定 HF immutable revision、逐文件 hashes 与 fresh-download evidence。

## Citation

项目仍处于研究与实验阶段，正式 citation 将在论文公开后补充。
