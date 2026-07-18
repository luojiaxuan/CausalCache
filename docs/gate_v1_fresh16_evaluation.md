# Gate v1 fresh-16 primary evaluation

> 当前状态：formal-58 gate 已完成训练、private-HF model seal 与只读 immutable replay。fresh-16 v1
> Source-A/Execution-B 已冻结，但 2026-07-18 的首次 formal attempt 在 derived repo exact-tree preflight
> fail closed。该 parent 失败发生在任何 fresh semantic decode 前；versioned inventory-repair v1 随后已完成
> label-blind trajectory/OCR/image semantics、checkpoint replay 与双 H200 workers，但在 label claim 落盘前因
> `mappingproxy` serialization 永久 `INVALID`。**fresh label decode、paper report、HF mutation、旧 dev-5、
> confirm、matched-NLL 与 closed-loop 仍为 0；两个旧 attempts 都不可续跑，也没有 fresh-16 GO/NO-GO。**

本协议只回答一个预注册问题：冻结的 conditional / independent ensemble 在从未参与训练或选择的
`fresh_development` 16 条 trajectory 上，是否同时通过 `GO_SELECTOR` 与 `GO_SET_CONDITIONING`。阈值、bootstrap
与 aggregation 仍以
[`causalcache_gate_v1_preregistration.json`](../code/configs/causalcache_gate_v1_preregistration.json) 为上位
科学契约；本阶段不能调整 architecture、LR、epoch、seed、budget、threshold 或 heuristic 定义。

## Source of Truth

- Git canonical repo/branch：<https://github.com/luojiaxuan/CausalCache> / `main`；
- formal gate completion：[`data/results/gate_v1_formal58_train_v1/`](../data/results/gate_v1_formal58_train_v1/)；
- formal gate private HF model：
  [`gavinlaw/causalcache-gate-v1-formal58-selector-mobile`](https://huggingface.co/gavinlaw/causalcache-gate-v1-formal58-selector-mobile)，
  tag `gate-v1-formal58-train-v1`，manifest commit
  `23f6786075c7bff91f93fd7e8a878e070efb72a9`；
- fresh split source：
  [`causalcache_restoration_v2_2_label_expansion_v1.json`](../code/configs/causalcache_restoration_v2_2_label_expansion_v1.json)；
- policy-blind derived input：private HF dataset
  [`gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-restoration-v2-mobile)
  immutable revision `630363a6adb692d72774f16dd0653a50216313ff`；
- repaired label input：private HF dataset
  [`gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile)
  immutable revision `7a6c254b8cec0dd3d8111dfc9c080de357e5cef3`；
- fresh-16 Source-A machine-readable contract：
  [`causalcache_gate_v1_fresh16_evaluation_v1.json`](../code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json)，
  SHA256 `c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e`；
- fresh-16 planned private HF destination：
  `gavinlaw/causalcache-gate-v1-fresh16-evaluation-mobile`，tag
  `gate-v1-fresh16-evaluation-v1`。v1 failure 后该 destination 仍不存在，不能把预期 URL 或 tag 写成已经发布；
- v1 pre-semantic failure evidence：
  [`data/results/gate_v1_fresh16_evaluation_v1_attempt/`](../data/results/gate_v1_fresh16_evaluation_v1_attempt/)；
- versioned full-inventory repair：
  [`gate_v1_fresh16_inventory_repair.md`](gate_v1_fresh16_inventory_repair.md) 与
  [`causalcache_gate_v1_fresh16_inventory_repair_v1.json`](../code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json)，
  SHA256 `5ba1b2d433c01defa0faae92a163dc9a9a916d967218abdc7607bd3724829a44`；
- inventory-repair v1 pre-label failure evidence：
  [`data/results/gate_v1_fresh16_inventory_repair_v1_attempt/`](../data/results/gate_v1_fresh16_inventory_repair_v1_attempt/)。

## v1 pre-semantic inventory failure

Source-A=`97694eff052ecbdc5f12f58b6f9ee10f4dd616ab`，机械 direct-child
Execution-B=`a8bb27ccf9b8820c1d8487f63883f813e6845645`。Hyper00 上的 formal run 已先验证 exact Docker receipt、
双 H200 与 14-file GUI-Owl projection；随后在 `_download_primary_inputs()` 的第一次 derived-repo inventory
检查中退出。冻结 revision `630363a6adb692d72774f16dd0653a50216313ff` 有 15 个 remote paths：4 个本阶段
消费的 label-expansion paths、`.gitattributes`/`README.md` 与 9 个历史 artifact paths。v1 validator 只把前 4
个和两个 base paths 纳入允许集合，因此错误地把同一 immutable revision 中的历史 paths 判为 drift。

旧 namespace 只留下 mode-0600 `runtime_receipts` 与 `global_claim`；artifact 目录为空，label claim、policy worker、
report、HF destination/tag 全部不存在。该 failure 不是 selector NO-GO，也没有泄露 fresh labels。repair 必须绑定
旧两条 receipt 与旧 successor absence，显式冻结完整 15-path remote inventory，并切换到新的 local namespace；
不能删除旧目录后重新调用 v1。

## 固定 denominator 与 selective input

fresh roster 是 policy-blind split 已冻结的 exact 16 个 `source_id`，digest 为
`1c37cbf6b67b0ddee61b3efe27d33b30c8fbfb471ced12f12e49624a10b82454`。每条 trajectory 只取 decision
step `4/5/6`，因此固定得到：

| 项目 | 固定数量 |
| --- | ---: |
| trajectories | 16 |
| states | 48 |
| candidate feature occurrences | 144 |
| `D(S)` distance values | 448 |
| deployment conditional edges | 464 |
| selected OCR/image paths | 80 |

derived trajectories 只能 semantic-decode rows `[48,64)`，repaired raw labels 只能 semantic-decode rows
`[144,192)`；禁止调用会把 64 trajectories 或 192 raw label states 全量解码的 generic reader。Source-A
validator 自身固定 network、write、PyTorch import、fresh semantic decode、model load/forward 与 report count
全部为 0。

## Variable-`n` comparator

fresh-16 同时包含 `n=2/3/4`，budget 固定为 `B=2`。Source-A 将历史只接受 `n=4` 的 comparator 推广为：

- `dynamic_recent`：取最后 `min(B,n)` 个 candidate event；
- `ocr_rgb_v2`：复用既有 OCR/RGB similarity、score tie 与 event-step tie，选前 `min(B,n)`；
- `policy_vision_v3`：复用 frozen GUI-Owl vision embedding、cosine 与 tie rule，选前 `min(B,n)`；
- learned conditional gate 每选一个 event 后基于新 coalition rescore；independent gate 只在空 coalition
  score 一次。两者都保持冻结的 `tau=0` stop。

`n=4,B=2` compatibility regression 必须证明新路径与已封存 comparator 的 score/order/selection 完全一致；
small-`n` 是自然 budget ceiling，不允许 padding、重复 event 或特殊阈值。

## 双 H200 schedule

formal Execution-B 的 policy-vision phase 固定使用两张 H200、两个 worker，各处理 24 states。48 states 各做
两次同设备 feature replay，再加 1 个跨设备 `n=4` sentinel，因此全局 operation count 是：

- processor batches：`48 + 1 = 49`；
- vision feature forwards：`48 × 2 + 1 = 97`；
- cosine scalar transfers：`292`。

这里的 `80` 是唯一截图 identity 数，不是 PIL open 次数。CPU feature materialization 解码 `80` 次；48 个
primary variable-image batches 共解码 `192` 次，cross-device sentinel 再解码 `5` 次，所以 execution receipt
必须同时记录 `80` unique identities、`80` feature decodes、`197` policy decodes 与 `277` total decodes。primary
report 与 state-record materialization 各自 replay exact-subset oracle，故 primary generation 记录 `48` unique
oracle states / `96` oracle invocations；两条 bootstrap interval 各 `10,000` resamples，总 draw count 为
`20,000`。report commit 前的 sealed-output local replay 会再执行一组 `96 + 2×10,000`；report commit 后的
model-backed immutable replay 还会重新下载 12 个 formal model files、加载 10 个 checkpoint，并再执行一组
`96 + 2×10,000`。三段 count 分开记录，不能把 primary-generation planned map 误写成整个进程的 observed total。

这是 planned operation contract，不是 observed execution receipt，也不是已经完成的运行记录。Source-A 不加载
GUI-Owl，也不占用 GPU。正式 B 仍须
遵守 GPU preflight、显式两卡 allocation、runtime receipt 和 startup utilization 检查；finalize/report 阶段回到
CPU 单线程且禁止 optimizer step。

## Label firewall

Execution-B 必须按以下顺序推进：

1. 先验证 immutable transport bytes，只 selective-decode 16 条 trajectory、80 条 OCR 与 80 张 image；
2. 在不接受任何 `D(S)` object 的进程边界中生成 48 feature states、三组 comparator selections，以及
   conditional/independent 的 ensemble 与全部五 seed selections；conditional artifact 还必须保存每轮完整
   candidate score vector、add/stop decision 与 coalition transition；
3. 把八个 label-blind payload files durable seal，逐 byte 重放 score/ranking/selection/trace，确认所有 learned 与
   heuristic decision bytes 不再可变；
4. 之后才允许创建 label-access claim，并 selective-decode 48 label states / 448 distances；
5. label 解封后的 evaluator 只能接受 sealed decisions 与 true-`D(S)` table；不得再接收 checkpoint、score function
   或调用 selector。true-nonpositive addition rate 只按封存 trace 中实际执行的 add 与其 true conditional marginal
   计算，terminal stop 不进入分母；
6. report commit 前先对本地 13-file candidate 做完整 sealed-output replay；远端 immutable replay 必须重新加载
   10 个 checkpoint，重建 learned bytes、formal provenance、primary report 与 state records 后逐 byte 比较。

旧 dev-5、confirm-20、matched-NLL 与 closed-loop 在整个 fresh primary B 中计数必须为 0。任何提前 label access、
全 transport semantic decode、selection seal 后 mutation 或 operation-count 漂移都永久 fail closed，不能用换目录
或 top-up 修复。

## 两段 immutable publication

HF publication 不是一个可部分解释的单 commit。固定 direct chain 为：

1. payload commit：精确 9 个 targets，包括 feature/label cache、3 个 heuristic selections、2 个 score JSONL
   和 conditional/independent learned selections；
2. report commit：必须是 payload 的 direct child，只新增精确 4 个 targets，即 primary state records、primary
   report、run manifest 与 bundle manifest；
3. annotated tag 只指向 report commit，随后从 immutable revision fresh-download，对 13 个 targets 逐 byte
   replay；completed replay 的 remote mutation count 必须为 0。

primary report 中所有 feature/label/heuristic bindings 指向 payload commit；manifest 不自嵌尚未存在的 report
commit。只有 primary report 已写入、hash、tag、fresh-readback 并完成 local completion seal，后续独立阶段才可
打开旧 dev-5 做 combined-21 compatibility guard。dev-5 不能与 fresh-16 在同一次 execution 中读取，也不能影响
fresh primary GO。

## GO 与 claim 边界

正式 B 通过 [`gate_v1_evaluation.py`](../code/causalcache/gate_v1_evaluation.py) 的
`evaluate_primary_slice_from_sealed_decisions` 复用同一冻结 aggregation core：trajectory-equal aggregation、
10,000 次 trajectory bootstrap、seed `271828`、90% percentile 与 Hyndman-Fan type-7 quantile 均不能覆盖。
最终分别报告 `GO_SELECTOR` 和
`GO_SET_CONDITIONING`；train-only OOF 不替代这两个 fresh generalization verdict。

即使 fresh primary 为 GO，本协议也不自动授权 confirm、matched-NLL 或 closed-loop。它们必须等待另立 post-GO
contract；若 fresh primary NO-GO，也必须原样保存 report，而不能先看 dev-5 后修改 gate。

## 当前完成度与下一步

本 v1 Source-A 与其唯一 Execution-B 已成为不可重跑的历史输入。唯一 formal attempt 的状态是
`INVALID_PRE_SEMANTIC_INPUT_INVENTORY_FAILURE`；它只证明旧 inventory allowlist 不完整，不产生 selector 或
set-conditioning 结论。旧 37-path source-only receipt、A/B commits、两条 local receipts 与空 successor/output
必须原样保留。

inventory-repair v1 A=`6fb3e868e293bce191ce30a5c6a15ecc544c4591` 与唯一 B=
`c22734ffc85935882f57ddb081c9194d6dae92d0` 已 commit/push；唯一 attempt 精确验证 15-path tree、只下载原 4 个
consumed files，并完成 16 trajectory / 80 OCR decode、48 feature states、144 candidates、10 checkpoint loads、
2 policy workers / 97 vision forwards。它在 heuristic seal 后、label claim 写入前发生 claim-serialization failure；
fresh label decode/report/HF mutation 均为 0，因此仍无 GO verdict 与 combined-21/post-GO authorization。

下一步只能另立 versioned claim-serialization repair，先验证 repair v1 的 ordinal `0..7` receipts、seal、artifact、
logs/runtime 与 successor absence，再使用全新的 local/HF identities。完整失败边界见
[`gate_v1_fresh16_inventory_repair.md`](gate_v1_fresh16_inventory_repair.md)。

## 执行入口

原 `scripts.manage_gate_v1_fresh16_evaluation run` 与 inventory-repair v1 的 `run`/`validate` 都不得再次调用。
claim-serialization repair 尚未冻结；在其独立 Source-A/B 与新 namespace 写入 Git 前没有授权执行入口。
