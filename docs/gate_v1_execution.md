# Gate v1 训练与评估执行逻辑

> 当前状态：trainer/evaluator source 与 synthetic/unit smoke 已实现；**尚未运行 formal-58 训练，尚未读取
> fresh-16、旧 dev-5 或 confirm-20，也没有正式 checkpoint、GO metric、matched-NLL 或 closed-loop 结果。**
> repaired expansion labels 已在 private HF immutable revision 闭合，因此 formal-58 的 label-data prerequisite
> 已满足；这不等于 gate 已训练。

本文件是 [`gate_v1_preregistration.md`](gate_v1_preregistration.md) 的执行说明。冻结阈值、roster、模型、loss、
OOF 与访问顺序仍以
[`causalcache_gate_v1_preregistration.json`](../code/configs/causalcache_gate_v1_preregistration.json) 为唯一
machine-readable contract；本次实现不改变任何已冻结选择。

## 代码边界

- [`gate_v1_data.py`](../code/causalcache/gate_v1_data.py)：label-blind signed-hash feature、330/200 维输入、
  conditional/independent target、层级权重、ranking pair 与两个 selector；
- [`gate_v1_training.py`](../code/causalcache/gate_v1_training.py)：CPU FP32 deterministic MLP、SmoothL1 + ranking、
  AdamW、固定五折 OOF、LR/epoch 选择、formal-58 final refit 与无持久输出的 two-step smoke；
- [`gate_v1_provenance.py`](../code/causalcache/gate_v1_provenance.py)：immutable artifact binding、完整 OOF
  selection digest、五 seed checkpoint/model-state digest 与 formal ensemble replay；
- [`gate_v1_evaluation.py`](../code/causalcache/gate_v1_evaluation.py)：trajectory-equal 指标、type-7 paired
  bootstrap、fresh-16 GO checks，以及只有冻结 fresh-16 report 才能调用的 combined-21 guard；
- [`run_gate_v1_trainer_smoke.py`](../code/scripts/run_gate_v1_trainer_smoke.py)：只构造 synthetic records，不接受
  data path 或 output path，因此不能误读 formal artifact，也不能遗留 checkpoint。

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
[`../data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/`](../data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/)。
因此现在可以构造 formal train，但只能加载旧 train-10 与新 train-48，共 58 trajectories / 174 states。
`validate_formal_training_roster`
会同时验证：

- source order 与 frozen formal-58 完全一致；
- 每条 trajectory 恰有一个 `n=2/3/4` state；
- 12/12/12/11/11 五折的每个 fold digest 与 assignment digest 和 preregistration 完全一致。

任何 development 或 confirm cache 都不得在此进程提前打开。

### 2. Train-only OOF 与 final fit

Conditional 与 parameter-matched independent 分开运行：

1. 对每个 LR `3e-4 / 1e-3`、seed `0..4`，五个 fold model 同 epoch 推进；
2. 每 epoch 只在 held-out trajectory 的唯一 `n=4,B=2` state 做 rollout；
3. raw-utility ratio 改进必须严格大于 `1e-4`，否则保留更早 epoch；连续 50 epoch 不改进即停，最多 500；
4. 以五 seed best OOF mean 选择 LR；两者差不超过 `1e-4` 固定取 `3e-4`；
5. 每 seed 用该 seed 的 selected epoch 在完整 formal-58 从相同 seed 初始化重新训练。

OOF 输出必须保留完整 `2 LR × 5 seed` grid，重放 earliest-epoch、50-epoch patience 与 LR tie rule 后生成
selection SHA256。正式 checkpoint 为每个 seed 记录 immutable artifact SHA256 和 pickle 无关的 canonical
model-state SHA256；五个 model object、model-state digest 与 checkpoint artifact digest 都必须不同。Conditional
与 independent provenance 必须绑定相同 formal-58 feature/label caches、冻结 gate config SHA256 和各自 training
report，评估前逐个与内存模型重放一致。
Ensemble manifest 固定 `schema_version=0.1.0` 与
`protocol_id=causalcache_gate_v1_frozen_ensemble`，所有 object 使用 exact-key validation；额外字段、mutable
revision、路径逃逸、seed/epoch 顺序或嵌套 digest 漂移都会 fail closed。

训练 tensor 全部为 CPU FP32 full batch。AdamW 参数、SmoothL1 beta、ranking coefficient、gradient clipping 与
deterministic algorithms 均由 source 固定，不允许 development override。正式 checkpoint 属于 reusable model
artifact，完成后应上传 private Hugging Face model repo；Git 只记录 config、manifest、immutable revision 与轻量
summary。

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

### 4. 旧 dev-5 compatibility guard

只有 fresh-16 report 已写成不可变轻量结果并通过自身 SHA256 replay 后，另一个阶段才能打开旧 dev-5，和
fresh-16 组成 combined-21。`evaluate_combined21_compatibility` 会先验证 primary report binding，再验证
21 trajectories / 63 states，最后只加入 preregistered combined delta guard。最终
`FINAL_GATE_V1_GO_DECISION` 仍将三项 authorization 全部保持为 false。

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
```

本机基础 Python 没有 PyTorch 时，schema、feature、weight、selector、OOF decision 与 GO aggregation tests 仍会
运行，唯一需要 PyTorch 的 two-step optimizer test 会明确 skip。正式 smoke 必须在已安装 PyTorch 的 CPU
runtime 再执行；它不需要 GPU，也不能作为 paper result。
