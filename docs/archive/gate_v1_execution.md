# Gate v1 训练与评估执行逻辑

> 当前状态：trainer/evaluator source、formal-58 OOF/final fit、10-checkpoint private-HF publication、fresh-16
> primary 与各自 immutable replay均已闭合。fresh-16 有效 verdict 为 selector `NO-GO`、set-conditioning
> `NO-GO`；旧 dev-5、confirm-20、matched-NLL 与 closed-loop 仍未读取或执行。
> repaired expansion labels 已在 private HF immutable revision 闭合，因此 formal-58 的 label-data prerequisite
> 已满足；formal-58 train-only feature/label cache 也已经 transport-repair Source-A/Execution-B 完成
> private-HF publication 与只读 immutable replay，`formal58_training_input_eligible=true`。formal-train
> Source-A=`e20f004…b9`、Execution-B=`bad28b7…a2f2` 与正式 model seal 已完成；不能用临时 Python 重跑
> `run_formal_oof` 或据 train-only OOF 调整模型。fresh-16 parent A/B 与 full-inventory repair A/B 也均已
> commit/push；parent v1 在 pre-semantic inventory fail closed，repair v1 则跨过 label-blind semantic/GPU phase，
> 在 label claim 落盘前因 `mappingproxy` serialization 永久 `INVALID`，两个旧 namespace 都不得续跑。独立
> claim-serialization repair 只修 JSON-safe deep snapshot，并已由 A=`f0dd53b…0ed1`、B=`ce523ff…a634`
> 完成 13-target private-HF publication 与 immutable replay。

本文件是 [`gate_v1_preregistration.md`](gate_v1_preregistration.md) 的执行说明。冻结阈值、roster、模型、loss、
OOF 与访问顺序仍以
[`causalcache_gate_v1_preregistration.json`](../../code/configs/causalcache_gate_v1_preregistration.json) 为唯一
machine-readable contract；本次实现不改变任何已冻结选择。

## 代码边界

- [`gate_v1_data.py`](../../code/causalcache/gate_v1_data.py)：label-blind signed-hash feature、330/200 维输入、
  conditional/independent target、层级权重、ranking pair 与两个 selector；
- [`gate_v1_training.py`](../../code/causalcache/gate_v1_training.py)：CPU FP32 deterministic MLP、SmoothL1 + ranking、
  AdamW、固定五折 OOF、LR/epoch 选择、formal-58 final refit 与无持久输出的 two-step smoke；
- [`gate_v1_provenance.py`](../../code/causalcache/gate_v1_provenance.py)：immutable artifact binding、完整 OOF
  selection digest、五 seed checkpoint/model-state digest 与 formal ensemble replay；
- [`gate_v1_evaluation.py`](../../code/causalcache/gate_v1_evaluation.py)：trajectory-equal 指标、type-7 paired
  bootstrap、fresh-16 GO checks，以及只有冻结 fresh-16 report 才能调用的 combined-21 guard；
- [`run_gate_v1_trainer_smoke.py`](../../code/scripts/run_gate_v1_trainer_smoke.py)：只构造 synthetic records，不接受
  data path 或 output path，因此不能误读 formal artifact，也不能遗留 checkpoint。
- [`gate_v1_formal_train_contract.py`](../../code/causalcache/gate_v1_formal_train_contract.py)：绑定 cache/preregistration、
  source inventory、access firewall、预期 OOF/final-fit output 与 private HF model destination；
- [`gate_v1_formal_train.py`](../../code/causalcache/gate_v1_formal_train.py)：只从已冻结 feature/label cache join 构造
  formal-58 training records，并组织 conditional/independent OOF 与 final refit；
- [`gate_v1_formal_train_runner.py`](../../code/causalcache/gate_v1_formal_train_runner.py) 与
  [`manage_gate_v1_formal_train.py`](../../code/scripts/manage_gate_v1_formal_train.py)：只允许在唯一机械
  Execution-B 上进入正式 CPU-only state machine；
- [`validate_gate_v1_formal_train_contract.py`](../../code/scripts/validate_gate_v1_formal_train_contract.py)：Source-A-only
  config/source/hash validator，不读 formal semantics、不访问 HF、不写 state 且不授权训练。
- [`gate_v1_fresh16_data.py`](../../code/causalcache/gate_v1_fresh16_data.py)：只接受 derived rows `[48,64)` 与
  repaired-label rows `[144,192)` 的 selective reader，拒绝全 transport semantic decode；
- [`gate_v1_fresh16_heuristics.py`](../../code/causalcache/gate_v1_fresh16_heuristics.py)：`n=2/3/4,B=2` 的
  deterministic recent/OCR-RGB/policy-vision selection 与 `n=4` compatibility；
- [`gate_v1_fresh16.py`](../../code/causalcache/gate_v1_fresh16.py) 与
  [`gate_v1_fresh16_policy_vision.py`](../../code/causalcache/gate_v1_fresh16_policy_vision.py)：48-state label-blind
  artifact builder、80-image plan 与 variable-image-count GUI-Owl vision runtime；
- [`gate_v1_fresh16_evaluation_contract.py`](../../code/causalcache/gate_v1_fresh16_evaluation_contract.py) 与
  [`validate_gate_v1_fresh16_evaluation_contract.py`](../../code/scripts/validate_gate_v1_fresh16_evaluation_contract.py)：
  fresh Source-A config/source/hash/operation/publication validator，零网络、零写入且不授权执行。

`source_id` 与 `state_id` 只用于 cache join、roster firewall 和聚合；它们不进入输入 tensor。feature cache
不含 `D(S)`、marginal、oracle、role 或 split；label cache 不含 instruction、OCR、low-fidelity semantic feature。
两侧分别验证完才可通过 `(source_id, state_id)` join。Formal roster 还必须严格按冻结 source 顺序逐条排列为
`step-4 / step-5 / step-6`，并验证 canonical `state_id`、decision step、candidate prefix、table event IDs 与全局
唯一 state ID；仅仅拥有相同的 source set 或 `n=2/3/4` 计数不够。

## Feature replay 的确定定义

1. text 先做 Unicode NFKC、whitespace collapse、strip、casefold，再按 whitespace token；已经是 token array 的
   OCR 与 screen delta 对每个元素执行同一归一化；
2. 每个 atom 严格为 `UTF8(namespace + NUL + token)`，SHA256 前 8 bytes big-endian 取 64-bin，第 9 byte
   奇偶决定正负，聚合后非零向量做 L2 normalization；
3. `candidate_age_over_history_length = (history_length - candidate_step) / history_length`；其余 g8 按冻结
   cap、ordinal 与 exact order 构造，并逐值检查 finite `[0,1]`；
4. `phi(S)` 是已选 event `h64` 的 sum；每轮 conditional selection 都重新生成 330 维输入。Independent 只在
   空 coalition 生成一次 200 维输入，禁止 rescoring；
5. prediction tie 取较小 numeric event step，最大 prediction `<= 0` 时停止。

## Formal 执行顺序

### 0. Expansion labels 闭合前（历史阶段，已完成）

只运行 synthetic smoke。旧 train-10 即使可用，也不能用于 metric、模型选择或持久 checkpoint；当前实现的
默认 smoke 完全不读取旧 artifact。

```bash
cd code
PYTHONPATH=. python -m scripts.run_gate_v1_trainer_smoke
```

预期 status 是 `VALID_GATE_V1_SYNTHETIC_TWO_STEP_SMOKE`，并且
`paper_metric_count=0`、`persistent_checkpoint_count=0`、`development_semantic_access_count=0`。它检查：

- conditional/independent 25,409/25,609 parameter MLP；
- 330/200 输入、regression weight sum、finite forward/backward 与负 target 未 clipping；
- 两步 deterministic replay；
- memory-only 临时 checkpoint reload 后 score exact equal；
- conditional rescore、`tau=0` stop 与 independent one-shot。

Synthetic distance table 显式包含 `event-1/event-2` complementarity，因此 smoke 不只检查 selector callback 是否
重打分，还会确认训练 batch 中同一 candidate 在不同 coalition 下获得不同 conditional target。

### 1. Train-only cache join

48+16 expansion label artifact 已上传 private HF、绑定 immutable resolved commit
`7a6c254b8cec0dd3d8111dfc9c080de357e5cef3`，并完成幂等 immutable replay 与独立只读 postflight；正式记录见
[`../../data/results/archive/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/`](../../data/results/archive/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/)。
因此现在可以构造 formal train，但只能加载旧 train-10 与新 train-48，共 58 trajectories / 174 states。
`validate_formal_training_roster`
会同时验证：

- source order 与 frozen formal-58 完全一致；
- 每条 trajectory 恰有一个 `n=2/3/4` state；
- 12/12/12/11/11 五折的每个 fold digest 与 assignment digest 和 preregistration 完全一致。

任何 development 或 confirm cache 都不得在此进程提前打开。

四个 parent artifacts 都把 train/development records 共置在同一 transport 中；因此已完成的 formal cache
materializer 先验证 immutable bytes，再只 semantic-decode formal train allowlist，从未调用会全量 decode
legacy/expansion artifact 的 generic readers。canonical repair bundle 固定为：

| Binding | Frozen value |
| --- | --- |
| feature cache SHA256 | `81fded50c4450700220742d3e0be9a5585d1bc51086150515b463bbdf4b4df8e` |
| label cache SHA256 | `4f9ef172aaa94c3ea8ce53aa43336c9fcb7800c24e181e1462d8239e31053cee` |
| bundle manifest SHA256 | `15c8bf56ddad4f6f278599c32aaadcd813e0e016db0d523b4892eb47f8d9d144` |
| HF tag | `gate-v1-formal58-cache-transport-repair-v1` |
| HF immutable commit | `a61b31bf2e69be00f94469f4a2f2d6b336fcc386` |
| formal-58 join audit SHA256 | `551e70b7e99f7a761f9c50d2adae3be72b0933044982a015a65e93372e41d77b` |
| gate v1 preregistration SHA256 | `37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b` |

feature/label cache 是两个独立 artifact，只有 exact-three HF commit、annotated tag、fresh immutable replay 与
final completion seal 都已完成后，formal-train Source-A 才允许将其声明为未来 trainer 的输入。该
cache milestone 已经闭合且不重跑；它的 optimizer、OOF、model forward 与 development metric 计数均为 0。

### 1.5 Formal-train Source-A → Execution-B

Source-A 的 machine-readable config 是
[`causalcache_gate_v1_formal_train_v1.json`](../../code/configs/causalcache_gate_v1_formal_train_v1.json)，SHA256
`bff920266b3f691618005b239c8a0aaa99369b45c0612ae628eec1a8f7eebf2f`。它只冻结 source、config、上表 immutable inputs、CPU FP32 training
contract、expected outputs 与 private HF model destination，不执行 formal loader 或 optimizer。Source-A validator 必须
返回：

```text
training_executed = false
execution_authorized = false
```

唯一 Execution-B 已由 clean pushed Source-A 机械生成；B 是 A 的 direct single-parent child，只新增
`causalcache_gate_v1_formal_train_runner_v1.json`，SHA256 为
`a89bb081c6a98e4da7da178e8930357773e7ee23cea211600cffea4500ba7532`。正式 manager 已验证
`HEAD == origin/main == live origin/main == B`、source inventory 与 runner-freeze 后进入训练。

训练运行是 CPU-only/no-GPU；source 固定 full-batch FP32 deterministic PyTorch，不应为这个 25K-parameter
selector 申请 GPU。B 必须先消费 host 侧 mode-0600 Docker inspect receipt，证明
`DeviceRequests=[]`、`Privileged=false`、`Runtime=runc`、container/image identity 和 `/data` bind mount；
容器内无 CUDA 只是第二层检查。整个 Source-A/B 与 formal training process 只允许 formal-58 semantic decode；以下计数必须
一直为 0：

```text
fresh-16 semantic decode
legacy dev-5 semantic decode
confirm-20 access
matched-NLL
closed-loop
```

### 2. Train-only OOF 与 final fit

Conditional 与 parameter-matched independent 分开运行：

1. 对每个 LR `3e-4 / 1e-3`、seed `0..4`，五个 fold model 同 epoch 推进；
2. 每 epoch 只在 held-out trajectory 的唯一 `n=4,B=2` state 做 rollout；
3. raw-utility ratio 改进必须严格大于 `1e-4`，否则保留更早 epoch；连续 50 epoch 不改进即停，最多 500；
4. 以五 seed best OOF mean 选择 LR；两者差不超过 `1e-4` 固定取 `3e-4`；
5. 每 seed 用该 seed 的 selected epoch 在完整 formal-58 从相同 seed 初始化重新训练。

每个 family 的 `2 LR × 5 seed` 是 10 个 OOF trials，每个 trial 同时训练五个 held-out-fold
model；conditional 与 independent 合计因此是 100 条 fold-training track。五折大小必须保持
`12 / 12 / 12 / 11 / 11 trajectories`。held-out fold 只在每条 trajectory 的唯一 `n=4,B=2` state
选 epoch；它不是 formal-58 in-sample evaluation，也不产生 fresh/development claim。

OOF 输出必须保留完整 `2 LR × 5 seed` grid，重放 earliest-epoch、50-epoch patience 与 LR tie rule 后生成
selection SHA256。正式 checkpoint 为每个 seed 记录 immutable artifact SHA256 和 pickle 无关的 canonical
model-state SHA256；五个 model object、model-state digest 与 checkpoint artifact digest 都必须不同。Conditional
与 independent provenance 必须绑定相同 formal-58 feature/label caches、冻结 gate config SHA256 和各自 training
report，评估前逐个与内存模型重放一致。
Ensemble manifest 固定 `schema_version=0.1.0` 与
`protocol_id=causalcache_gate_v1_frozen_ensemble`，所有 object 使用 exact-key validation；额外字段、mutable
revision、路径逃逸、seed/epoch 顺序或嵌套 digest 漂移都会 fail closed。

训练 tensor 全部为 CPU FP32 full batch。AdamW 参数、SmoothL1 beta、ranking coefficient、gradient clipping 与
deterministic algorithms 均由 source 固定，不允许 development override。正式 checkpoint 已上传 private
Hugging Face model repo；Git 只记录 config、manifest、immutable revision 与轻量 summary。

冻结输出约定至少包含：

- conditional 和 independent 的完整 OOF grid；
- 每 family 的 selected learning rate、每 seed selected epoch 和 selection SHA256；
- 5 conditional + 5 independent final checkpoints；
- 每个 checkpoint 的 canonical model-state SHA256 与 checkpoint artifact SHA256；
- family training reports、ensemble manifests、top-level run manifest 与 operation counts；
- cache/preregistration/source/runtime binding 与 private HF immutable model receipt。

private model repo 是 `gavinlaw/causalcache-gate-v1-formal58-selector-mobile`，tag
`gate-v1-formal58-train-v1`。Source-A 时已验证该 repo/tag 不存在；B 已完成 10 checkpoints、2 full OOF
reports 与 4 manifests 的 strict-readback。payload commit=`a6c9e7f…2889`，其 direct-child manifest
commit=`23f6786…72a9`，annotated-tag object=`fa85e74…b4d6`；fresh immutable replay remote mutation 为 0。
Git completion 见 [`../../data/results/archive/gate_v1_formal58_train_v1/`](../../data/results/archive/gate_v1_formal58_train_v1/)。

### 3. Fresh-16 一次性 GO

五 seed final model 与 train-only selection 先冻结，再单独启动只加载 fresh-16 / 48 states 的 evaluation。
`evaluate_primary_slice` 强制精确 source roster、每 trajectory 三个 states、五 seed ensemble 与固定三个 heuristic
顺序，输出带 SHA256 binding 的 `FROZEN_GATE_V1_FRESH16_PRIMARY_EVALUATION`。它计算全部 preregistered：

- normalized/raw exact recovery；
- 相对每个 heuristic 与 strongest heuristic 的 paired trajectory 统计；
- 10,000 次 seed 271828、90% percentile、Hyndman-Fan type-7 bootstrap lower；
- 五 seed ratio、population std、hard `n=3/n=4` ratio 与 true nonpositive addition rate；
- conditional 相对 independent 的 primary set-conditioning checks。

Formal API 不接受裸 callable scorer，也不暴露 bootstrap override。报告固定写入 trajectory bootstrap、10,000
resamples、seed `271828`、90% percentile 与 Hyndman-Fan type-7，并绑定：gate config、formal training
provenance、十个 checkpoint、fresh-16 feature/label artifact、三个 heuristic artifact 及其 selection digest。
缩短 bootstrap 的入口只存在于显式 test-only private helper，其 status 不能进入 combined-21。

该 report 不授权 confirm、matched-NLL 或 closed-loop。

已冻结并 commit/push 的 fresh-16 Source-A 将上述 variable-`n` 自然推广固定为：recent 取最后 `min(B,n)` 个 event；OCR/RGB 与
policy-vision 对全部 candidate 使用原 similarity/tie rule 后取 `min(B,n)`，并用 `n=4` compatibility test
证明与旧 artifact 完全一致。fresh selective loader 只解析 expansion rows `[48,64)` / label rows
`[144,192)`，不能复用会读取 train 或全 transport semantics 的 generic reader。精确 denominator 是 16
trajectories、48 states、144 candidate feature occurrences、448 `D(S)` values 与 464 deployment conditional
edges。

Source-A 还冻结 B 的双 H200 policy phase：2 workers 各 24 states；48 states 各做两次 same-device
feature replay，并加入 1 个 cross-device `n=4` sentinel，所以全局恰为 49 processor batches 与 97 vision
forwards。80 个唯一截图会产生 80 次 CPU feature decode、192 次 primary policy decode 和 5 次 sentinel decode，
即 277 次 PIL open；exact-subset 记录为 48 个唯一 states / 96 次实际 invocation，bootstrap 为 2 条 interval ×
10,000 resamples。这些数字是 primary generation/publication 的 planned contract。inventory-repair v1 已实际完成
2 workers / 97 forwards，但在 label claim 前失败；因此 GPU phase 已运行，不等于 report completion 或 paper metric。

label firewall 要求先完成 48 feature states、三组 heuristic selection、conditional/independent learned
selection 与 score records 的 local durable seal，之后才可创建 label-access claim、读取 48 label states 与 448
distances。publication 必须先产生精确 9-target payload commit，再产生其 direct-child、只新增 4 targets 的
report commit，tag 指向 report commit 并对全部 13 files 做 immutable fresh-download replay。完整协议见
[`gate_v1_fresh16_evaluation.md`](gate_v1_fresh16_evaluation.md)。

唯一 parent v1 attempt 的 pre-semantic inventory failure 见
[`../../data/results/archive/gate_v1_fresh16_evaluation_v1_attempt/`](../../data/results/archive/gate_v1_fresh16_evaluation_v1_attempt/)；
inventory-repair v1 A=`6fb3e868e293bce191ce30a5c6a15ecc544c4591`、B=
`c22734ffc85935882f57ddb081c9194d6dae92d0` 的 pre-label claim-serialization failure 见
[`../../data/results/archive/gate_v1_fresh16_inventory_repair_v1_attempt/`](../../data/results/archive/gate_v1_fresh16_inventory_repair_v1_attempt/)。
repair v1 完成 16 trajectory / 80 OCR decode、48 feature states、144 candidates、10 checkpoint loads、2 policy
workers / 97 vision forwards；fresh label decode、label claim write、report 与 HF mutation 均为 0。独立
claim-serialization repair 随后完成全部 16 receipts、primary report 与 immutable replay；轻量结果见
[`../../data/results/archive/gate_v1_fresh16_claim_serialization_repair_v1/`](../../data/results/archive/gate_v1_fresh16_claim_serialization_repair_v1/)。
selector 与 set-conditioning 都未 GO，因此项目选择不执行 post-primary combined-21 compatibility；真正的
post-GO stages 继续保持 locked。

### 4. 旧 dev-5 compatibility guard

上位 preregistration 只允许在 fresh-16 report 写成不可变结果并通过 replay 后，由独立阶段打开旧 dev-5。
当前 fresh-16 selector/set-conditioning 已双 `NO-GO`，项目按 falsification boundary 不执行该可选 compatibility
guard；旧 dev-5 继续 locked。`evaluate_combined21_compatibility` 保留为历史冻结实现，不代表当前授权。

Combined-21 必须再次验证 conditional/independent ensemble digest 与 fresh-16 report 完全相同；换 checkpoint、
换 training selection 或换 artifact binding 会在 selector evaluation 前 fail closed。Final report 另外记录 combined
feature/label artifact 与 small-`D(empty)` excluded state/trajectory counts。

### 5. GO 之后

只有 `GO_SELECTOR && GO_SET_CONDITIONING` 才能另立 post-GO contract。confirm-20、matched-NLL、closed-loop
均不属于本 trainer 的权限范围，不能因 GO 自动读取或执行。

## 当前验证

```bash
cd code
PYTHONPATH=. python3 -m unittest \
  tests.test_gate_v1_contract \
  tests.test_gate_v1_pipeline -v
PYTHONPATH=. python3 -m scripts.validate_gate_v1_contract --repository-root ..
PYTHONPATH=. python3 -m scripts.manage_gate_v1_formal_train validate-source \
  --repository-root .. \
  --contract code/configs/causalcache_gate_v1_formal_train_v1.json

# fresh-16 Source-A；零网络/零写入 config-only validation
PYTHONPATH=. python3 -m scripts.validate_gate_v1_fresh16_evaluation_contract \
  --repository-root .. \
  --contract code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json
PYTHONPATH=. python3 -m unittest \
  tests.test_gate_v1_fresh16_evaluation_contract \
  tests.test_gate_v1_fresh16_data \
  tests.test_gate_v1_fresh16_heuristics \
  tests.test_gate_v1_fresh16 \
  tests.test_gate_v1_fresh16_policy_vision \
  tests.test_gate_v1_fresh16_evaluation_runner \
  tests.test_gate_v1_pipeline -v
```

上述 source-only 输出是历史 A 的零执行证明。正式 B `run` 已返回
`VALID_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1`；同一 B 的完成态 `validate` 返回
`REVALIDATED_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1`，remote mutation count 为 `0`。conditional / independent
train-only OOF mean 分别为 `0.8925353801368878` / `0.9063764691683989`，都选择 LR `3e-4`；这些数值不构成
fresh-16 GO。

本机基础 Python 没有 PyTorch 时，schema、feature、weight、selector、OOF decision 与 GO aggregation tests 仍会
运行，唯一需要 PyTorch 的 two-step optimizer test 会明确 skip。正式 smoke 必须在已安装 PyTorch 的 CPU
runtime 再执行；它不需要 GPU，也不能作为 paper result。

formal-train Source-A validator 也不需要 PyTorch；它只验证 config/source/hash、上游 immutable bindings、
Execution-B absence 与访问边界；repo/tag absence 是单独的只读 Hub preflight，不由这个零网络 validator 声称。
该历史 formal-train A 中唯一合法状态仍是 `training_executed=false` 与 `execution_authorized=false`；完成态以
formal B 的 immutable model seal 为准。

历史 claim-serialization repair Source-A 只能返回 `evaluation_executed=false`、
`fresh16_access_authorized=false` 与 `execution_authorized=false`。config SHA256 为
`3979573be235d630ee2f46dc23be8747a843190c9b57ee81e1b3a17b4416d8c7`；57-path source inventory SHA256 为
`572992cf5ac75142cdf8f0e82bdfc2caad43ba37632c741eae277285db54d689`；focused/full regression 分别为
99 passed + 7 subtests 与 1263 passed、16 skipped、4 deselected、617 subtests，4 个 deselect 均为已完成历史
A/B 的 B-absence lifecycle tests。后续 B 已按 direct-child 单文件 diff 生成并完成 formal execution；这些
source-only 计数不改写，但当前完成态以 `REVALIDATED_GATE_V1_FRESH16_PRIMARY_EVALUATION_V1` 为准。这不是
修改或重跑 formal trainer。
