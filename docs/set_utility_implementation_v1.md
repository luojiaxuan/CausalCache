# Set Utility Predictor v1 实现与执行交接

> 状态：budget-agnostic model、variable-`n` features、cardinality-capped label table、group-aware split
> audit、joint search、training loss 与 true-utility evaluator 已实现并通过本机 CPU tests。long-pool census、
> 新 restoration labels、正式 PyTorch smoke、训练 checkpoint 和离线 development 结果尚未产生。

## 数据与控制流

```text
policy-blind long-pool census
        │ source/group hashes only
        ▼
Freeze-B roster + group-disjoint roles
        │
        ├── label-blind OCR/RGB/low-fidelity features
        └── exact cardinality-capped D(S) tables
                         │
                         ▼
               validated feature/label join
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      pairwise-additive          DeepSets
              └──────────┬──────────┘
                         ▼
       joint argmax over all |S| <= B, including empty
                         │
                         ▼
       selected subsets rescored with true U(S)
```

`B` 只出现在最后两步的 search/evaluation API。feature state、padded batch 和两个 predictor 的输入都不含
budget、remaining budget 或 selection order。集合 cardinality 是 `S` 自身的属性，不是外部预算特征。

## 实现模块

- `set_utility_features.py`：复放旧 q64/h64，并显式加入 current-event OCR/RGB、recency 和 event-pair
  interaction features；支持 `n=1..16`、zero padding 和 permutation-equivariant event/pair axes；拒绝
  budget、`D(S)`、utility 与 group key 等额外字段；
- `set_utility_label_table.py`：验证全部且仅有 `|S|<=K` 的 finite non-negative `D(S)`，其中
  `K∈{2,3,4}`；机械计算允许正负的 `U(S)=D(empty)-D(S)`，不 clipping；
- `set_utility_split_validation.py`：验证 trajectory/source 唯一、instruction+app group 不跨 development
  partition、formal-58 仅为 `legacy_train_only`，并显式拒绝 old-dev5/fresh16/confirm20 source identities；
- `set_utility_data.py`：只允许 canonical feature state 与 validated exact table join；训练 state 再次要求
  cardinality-capped table 完整，并对 state/cardinality/subset 做等权；
- `set_utility_models.py`：pairwise-additive 和 DeepSets；两者精确约束 `U(empty)=0`；
- `set_utility_training.py`：raw/normalized SmoothL1 与 within-state ranking primitives。具体启用哪些项、权重、
  seed、epoch 和 early stop 仍必须由 Freeze-B 一次性绑定；
- `set_utility_baselines.py`：冻结 recent、版本化 OCR/RGB positive-top-`B` 与 oracle-independent `J`；
- `set_utility_search.py`：完整枚举 empty、singleton、…、`B`-set，tie-break 固定为 utility 高、cardinality
  小、event tuple 字典序小；
- `set_utility_evaluation.py`：所有选择重新用真实 `U(S)` 评分，并按 trajectory 等权报告 utility、exact
  recovery、W/T/L、cardinality、regression 和 ranking。

## 计算量与数据生产

phase-1 每个 state 必须完整生成：

\[
N_2(n)=1+n+\binom n2.
\]

因此 `n=4/8/16` 分别为 `11/37/137` 个 subset rows；加一次 full-history reference 和一次 canonical
repeat 后，teacher forward 预算分别约为 `13/39/139`。不能把抽样 pair 宣称为 exact B2 table。

phase-2 只在 phase-1 进入条件通过后另立 freeze：`n=6,K=3` 为 42 rows，`n=8,K=4` 为 163 rows。
phase-2 calibration 与 evaluation 仍须 trajectory/group-disjoint。

## 跨机器执行逻辑

1. long-pool census 是 CPU-only，优先 Taurus/Aries；不需要 GPU preflight，也不能读取 policy/OCR/
   restoration output；
2. exact label generation 是 frozen GUI-Owl inference，优先 Hyper H200。按 state shard 并行，先执行 GPU
   preflight；在有有效并行时使用当前空闲卡，单任务最多 4 张。每个 worker 只写独立聚合 shard，最终按
   state inventory 合并，避免大量小文件；
3. Pairwise/DeepSets 本身很轻，单卡能装下时默认单卡训练；只有经过吞吐测量确认 state/subset batching
   能有效扩展时才增加 GPU。启动 warmup/steady-state 窗口检查利用率，之后无需持续监控；
4. 不同芯片不要求 bitwise checkpoint 一致，但同一正式 run 必须固定 runtime、dtype、seed 和 feature bytes。
   Hyper/A6000 至少通过相同 prediction/search behavioral smoke。

## Source of Truth

- Git：本目录中的 contract、代码、tests、轻量 manifest 与 aggregate；
- planned private HF dataset：`gavinlaw/causalcache-set-utility-new-development-mobile`，用于 feature cache、
  exact/capped `D(S)` shards、split manifest 与 provenance；当前尚未创建或绑定 immutable revision；
- planned private HF model：必须在 Freeze-B 中另行命名并绑定，用于 checkpoints、optimizer-independent model
  manifest 和 evaluation prediction tables；当前未创建。

## 当前验证与阻塞

本机 `code/tests/test_set_utility*.py` 为 `100 passed, 11 skipped, 5 subtests passed`。11 个 skip 都是本机
没有 PyTorch；source/config、features、labels、split、baselines、search 和 evaluator 的纯 CPU tests 已通过。

Freeze-B 还必须明确 student representation：当前实现是可执行的 q64/h64 + low-fidelity + OCR/RGB/recency
基线，不含高层 policy-vision embedding。旧 student 已暴露 representation bottleneck，因此正式训练前应在
不查看新 evaluation labels 的条件下二选一并冻结：加入现有 GUI-Owl final-main 4096-d normalized visual
embedding（只在 HF feature cache 保存），或把当前轻量表示明确列为 v1 主模型并把 richer embedding 作为预注册
ablation。不能等 evaluation 失败后再换表示。

真实 P0 尚未运行。本地受限执行环境不能解析 Taurus SSH host，Codex remote handoff 也未找到 Taurus 上的
matching saved project；因此没有把历史的“81 条因旧 cap 被排除”误写成“81 条新 eligible”。获得可访问的
Taurus/Hyper checkout 后，下一步严格按以下顺序执行：

1. 从 pushed `main` 运行 CPU census 并 commit/push manifest；
2. 根据真实 pool inventory 冻结 Freeze-B 的 roster、group-disjoint split、query states、label counts、
   hyperparameter grid 和 HF destinations；
3. 单独冻结 label Execution-B，再做 GPU preflight 和 exact `D(S)` production；
4. 在有 PyTorch 的固定 runtime 先跑 tensor/model/optimizer/search smoke，再训练两个 family；
5. 只有新 evaluation 上至少一个 learned family 超过 OCR/RGB，才打开 phase-2 B3/B4 transfer freeze。
