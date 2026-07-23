# Restoration v2.2-eager label protocol

## 当前状态

本协议是 `v2.2-eager fresh-45` substrate 通过后的独立 child protocol。它只授权 30 个
`v2_label_train` 和 15 个 `v2_development` states 的 restoration attribution，不打开 confirm，不训练 gate，
也不运行 matched-NLL 或 closed-loop evaluation。

历史 v1 contract 是
[`code/configs/causalcache_restoration_v2_2_labels.json`](../../code/configs/causalcache_restoration_v2_2_labels.json)；
当前唯一可执行的 replacement 是
[`code/configs/causalcache_restoration_v2_2_labels_v2_repair.json`](../../code/configs/causalcache_restoration_v2_2_labels_v2_repair.json)。
任何 restoration teacher forward 前必须先 commit/push source，并从 clean `main` 验证 contract、source inventory、
immutable v2.2 parent raw 和 immutable derived artifact。

v1 source freeze 已完成并推送：clean `main@3942d687d03bf63ea683fe8ad906a161eb10dc27`，contract SHA256
`56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87`，29-file source inventory
SHA256 `8d0ecf6b0df75f9fa7753631b8fca5efd98c71a7953007e684393e6e8fe52d9b`。完整回归为 575 tests
OK（skipped 11），`make paper` 通过。该状态只证明 source、schema、schedule 与 fail-closed lifecycle 已冻结；
formal v1 随后因指定 model snapshot directory 不存在，在两个 worker 都尚未写 state marker、teacher forward 或
KL 时 fail closed：attempted/completed states 为 0/0，原 identity 的 retry/resume 均永久禁止。没有 raw `D(S)`、
exact-subset oracle、conditional-marginal rows 或 HF repo/tag/revision；compact failure binding 位于
`data/results/archive/restoration_v2_2_eager_labels_v1_attempt/`。confirm、gate training、matched-NLL 与 closed-loop 仍未
运行，且本协议禁止把它们混入 label attempt。

v2 replacement 已冻结为 `restoration-v2-2-eager-labels-v2` / `v2_preclaim_repair`，config SHA256 为
`7fc96883b905a5f026c18e65c3c7e377972559c775ac7ec95030ea1f04c2f94c`。它不改变 estimand、state denominator、
coalition enumeration 或 operation schedule，只修复 execution identity 和 pre-claim model validation。正式 v2
现已完成闭环：45/45 states `PASS`，even/odd workers 分别完成 23/23 与 22/22，retry、top-up 和 generation 均为
0。执行窗口是 `2026-07-16T17:13:13.360318Z`--`2026-07-16T17:27:46.202730Z`；run-contract SHA256
为 `b78ca1e70652c7eef68efc3472adf78cec4510e507a0c00cfcee2c2cf05285d9`，source 与 packaging commit 均为
`5ae40d4aed4eb20b931216776b379bc6ae55629d`。Git 中的 compact binding 和科学归约见
[`data/results/archive/restoration_v2_2_eager_labels_v2/`](../../data/results/archive/restoration_v2_2_eager_labels_v2/)；raw labels 不进入 Git。

## Reference identity

parent 是已经闭合的 v2.2 artifact：

- private HF：`gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile`；
- immutable revision：`3577099d505b8c652d764f41269df911128ec767`；
- raw SHA256：`b22827e6e2d8d33b03686fc177dc8f9c55c5133470fbe40f9fb3e33cce809fb5`；
- parent run-contract SHA256：`397f8da2b89b495c93ef3a034dfe16fb6d883ab4fdb7b830003397c9cc9a2727`；
- 45/45 states 的两次 native generation canonical action 完全一致。

parent tar 不保存 reference logits，因此 labels 不是从旧 scalar KL “离线导出”。新 run 固定：

1. 从 validated parent state envelope 读取唯一 canonical action；
2. 不做新 generation；
3. 在相同 `GUIOwlV22EagerRuntime` 和 immutable derived images 上 fresh rerun full-history teacher forward，得到
   reference log-probs；
4. 对全部 mixed-fidelity coalition 做 fresh teacher forward，并在 GPU 上计算 KL。

因此 reference mode 是
`parent_canonical_action_frozen_fresh_teacher_recompute`。若 parent action、state projection、worker member、raw
hash 或 derived crosswalk 任一不一致，attempt 在 runtime import 前 fail closed。

## Canonical distance table

每个 decision state 有 $n\in\{2,3,4\}$ 个 candidate events。虽然部署主预算固定为两个 event slots，raw
scientific truth 枚举完整 power set：

$$
\mathcal{T}_t=\{(S,D_t(S)):S\subseteq H_t^{\mathrm{candidate}}\}.
$$

这样一次 run 同时支持 exact permutation attribution、完整 interaction analysis 和不同预算的离线诊断，不需要
再次访问 policy。step 4/5/6 分别产生 4/8/16 rows；15 条 trajectory 共 420 rows：

| Split | States | Raw $D(S)$ rows |
| --- | ---: | ---: |
| `v2_label_train` | 30 | 280 |
| `v2_development` | 15 | 140 |
| Total | 45 | 420 |

full-history coalition 的 fresh forward 同时作为 reference，按定义写入 $D(H_t)=0$。另外每 state 再做一次
full-history repeat teacher forward；repeat KL 必须 finite、non-negative 且不超过 `1e-4`。所有其他 375 个
coalition 各产生一个 scalar KL。完整 schedule 是 0 generation、465 teacher forwards、420 KL measurements。

## Primary deployment labels

主预算使用 event-slot capacity $B=2$，每个 event 的 slot cost 为 1。实际 image count、visual-token count 和
latency 仍保存在 raw metadata 中，但不在本次 source freeze 后改写 primary selection objective。token-budget
版本只能作为新的预注册 ablation。

定义：

$$
U_t(S)=D_t(\varnothing)-D_t(S),
$$

$$
\Delta_{j,t}(S)=D_t(S)-D_t(S\cup\{j\}).
$$

部署可达 edge 只包含 $|S|<2$ 且 $|S\cup\{j\}|\le2$。step 4/5/6 分别是 4/9/16 edges，总计：

| Split | Conditional-marginal labels |
| --- | ---: |
| `v2_label_train` | 290 |
| `v2_development` | 145 |
| Total | 435 |

负 utility 和负 marginal 必须原样保存，不 clamp、不筛 state。edge label 必须由同一 canonical `D(S)` table
policy-free 相减，不能为每条 edge 单独跑两次 policy。full hypercube 的 720 edges、465 个 pair-interaction rows
与 135 个 exact permutation event attribution rows 是 analysis payload；gate 如何采样或加权这些 rows 留给独立
gate-training contract。

## Exact subset oracle

primary exact oracle 定义为：

$$
S_t^*=\arg\min_{|S|\le2}D_t(S).
$$

空集和 singleton 都在候选域中，因此不是“强制选满两个”。tie epsilon 固定为 0；顺序是：

1. 更低 `D(S)`；
2. 更小 cardinality；
3. lexicographically 更小的 sorted `event_step_id` tuple。

当 `D(empty)<=1e-12` 时 normalized recovery 写为 `null`，不能写成 1 并进入均值。oracle 只定义上界；线上
greedy、learned gate 和 search/distillation gap 都不在本协议中运行。

## 正式结果与科学归约

正式 run 严格执行冻结 schedule：0 generation、465 teacher forwards、420 KL measurements，并生成 420 个
canonical $D(S)$ rows、435 个 deployment conditional-marginal labels、45 个 exact-subset oracle rows 和 465 个
pair-interaction rows。45/45 state denominator 全部保留，没有 retry、top-up、state filtering 或 confirm 访问。

在 event-slot capacity $B=2$ 下，exact oracle 的 mean normalized recovery 为：

| Split | States | Mean normalized recovery |
| --- | ---: | ---: |
| `v2_label_train` | 30 | 0.9028414853 |
| `v2_development` | 15 | 0.8148528628 |
| Overall | 45 | 0.8735119444 |

45 个 states 中有 44 个获得正 oracle utility，oracle 选择 cardinality 为 0/1/2 的 state 数分别是 1/3/41。
deployment conditional marginals 的严格正/负计数为 360/75，22 个 states 至少包含一条负 marginal。raw
restoration utilities 的严格正/零/负计数为 338/45/37；零值包括每个 state 按定义得到的 empty-coalition
utility。pair interactions 的严格正/负计数为 188/277，absolute mass 为 `6.009230253776877`，最大绝对值为
`0.1697198525071144`。

这些数字表明 frozen policy 的 restoration utility 存在 interaction 与 non-monotonicity，因此支持把后续 student
设计为 set-conditioned iterative gate，而不是把 coalition-dependent teacher 压缩成独立 event score。它们不证明
learned gate 有效，也不构成 matched-NLL 或 closed-loop success 证据：当前完成的只是 offline exact oracle 与
conditional-marginal labels；gate training、matched-NLL、closed-loop evaluation 均未运行，confirm 仍然 locked。

## Algebraic validation

raw `D(S)` 是唯一 canonical scientific truth。artifact validator 必须从 raw table 独立重算：

- $U(\varnothing)=0$；
- 每条 $\Delta=D(S)-D(S\cup\{j\})$；
- edge utility difference 与 distance difference 一致；
- 任意 permutation path telescoping 到 $D(\varnothing)-D(H)$；
- pair interaction 对交换 $i,j$ 对称；
- exact subset oracle 由 raw feasible table 重新求解。

任何 missing/extra/duplicate coalition、non-finite/negative distance、derived row drift、split leakage 或 operation
count drift 都使整个 attempt `INVALID`，不得删 state、retry 或 top-up。科学上 interaction 很弱、oracle recovery
很低或负 marginal 很多不是 `INVALID`；这些是需要如实报告的结果。

worker process、spawn/join 或 aggregate validation 任一失败时，global ledger 必须原地 durable seal 为
`LABEL_ATTEMPT_INVALID`，保存两个 worker 的 marker/state high-water 和 failure provenance，并显式写入
`retry_allowed=false`、`resume_allowed=false`。失败 attempt 不得删除后重跑；需要修改 protocol 时只能冻结新的
attempt identity。

## Two-GPU execution

正式 run 固定在同一 Hyper H200 host/container 的两张 GPU 上：even worker 处理 23 states，odd worker 处理
22 states。两个 worker 都在 runtime import 前由 global/sibling ledger prebind；worker 内每个 state 先写 marker，
再运行完整 table。microbatch 固定为 1，避免不同 coalition prompt shape 或 batch numerical path 成为新变量。

正式命令形态：

```bash
cd /data/CausalCache/code
python -m scripts.run_restoration_v2_2_labels \
  --contract configs/causalcache_restoration_v2_2_labels_v2_repair.json \
  --repository-root .. \
  --scientific-config configs/causalcache_restoration_v2.json \
  --selection-manifest ../data/manifests/restoration_v2_selection.json \
  --ocr-backend-config configs/restoration_v2_ocr_backend.json \
  --snapshot-manifest configs/gui_owl_1_5_8b_snapshot.json \
  --derived-artifact-root /data/tmp/causalcache-restoration-labels-v2-derived \
  --parent-v22-evidence /data/tmp/causalcache-restoration-labels-v2-parent/raw/restoration-v2-2-eager-full-45-substrate-v1.tar \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --output-dir /data/experiments/causalcache/restoration-v2-2-eager-labels-v2 \
  --global-ledger /data/experiments/causalcache/.restoration-v2-2-eager-labels-v2.attempt.json \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id 39749a3bd0875f3c15216211f875220a4934fe62313487b537c1116e82040ff0 \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa
```

容器必须只暴露 preflight 选出的两张 GPU；正式 launch 后要核对 `torch.cuda.device_count()==2` 并启动 GPU
utilization monitor。

本次正式执行位于 `hyper00` / `node-radixark-16-0000`，容器只暴露 physical GPU 0/1，两路 runtime
分别看到 `cuda:0/1` 的 NVIDIA H200。runtime 固定为 Python 3.12.3、PyTorch 2.11.0+cu130、CUDA 13.0、
cuDNN 91900、Transformers 5.6.0、`torch.bfloat16`、seed 0、eager attention、TF32 off；不声称 strict CUDA
determinism。容器 image digest 是
`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`。完整 argv、两个 GPU UUID、
driver 570.172.08 与逐 state runtime metadata 都保存在 immutable raw archive 的 `run_manifest.json` 和
`workers/*/runtime_identity.json`。

在 global ledger 创建前，v2 runner 必须对上述 canonical model projection 执行 full snapshot validator：root 不是
symlink，repo slug/basename、`.snapshot.json` revision、14-file inventory、17,545,907,171 bytes、每文件 SHA 与
Transformers source 全部一致。失败时不得留下 v2 root/ledger。

## Artifact lifecycle

v1 formal attempt 仍是 zero-forward `INVALID`，其 identity 不得重试。v1 canonical local output、ledger 与 archive 分别是
`/data/experiments/causalcache/restoration-v2-2-eager-labels-v1`、
`/data/experiments/causalcache/.restoration-v2-2-eager-labels-v1.attempt.json` 和
`/data/experiments/causalcache/restoration-v2-2-eager-labels-v1.raw.tar`；前两个现作为不可重试的 failure evidence
保留，archive 不存在。

replacement v2 的 output、ledger、archive 分别是
`/data/experiments/causalcache/restoration-v2-2-eager-labels-v2`、
`/data/experiments/causalcache/.restoration-v2-2-eager-labels-v2.attempt.json`、
`/data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar`；三者均已生成并通过完整 raw reducer。
deterministic USTAR 包含 101 files、大小为 3,747,840 bytes，SHA256 为
`99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e`，tree-inventory SHA256 为
`c5104594b0741810f3d49d007a63a74f16ee4236dd137d1dea92b9373065c45f`。

raw artifact 已上传到 private HF dataset
`gavinlaw/causalcache-restoration-labels-mobile` 的
`raw/v2.2-eager-train-dev-exact-v2.tar`。immutable revision 是
`8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`，tag `v2.2-eager-train-dev-exact-v2` 已验证解析到同一 revision；
从该 immutable revision 下载到独立路径后的 archive 已重新通过 reducer，并与 source archive 逐 byte 相同。

Git 只保存
[`compact artifact binding 与 scientific summary`](../../data/results/archive/restoration_v2_2_eager_labels_v2/)，不保存 raw labels。
confirm artifact、gate checkpoint、matched-NLL pairs 和 closed-loop episodes 都必须使用后续独立 contract。
