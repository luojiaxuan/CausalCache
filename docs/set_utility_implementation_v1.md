# Set Utility Predictor v1 实现与执行交接

> 当前结论：路线已切换为“先扩全量 source pool，再生产 `|S|<=2` labels，最后训练 Set Transformer”。
> budget-agnostic model/label/trainer 的 source core、610-shard metadata contract 与 107-identity consumed
> firewall 已实现；真实 P-1 inventory、P0 semantic census 与 Freeze-B v2 roster/query repair 已完成；原
> Freeze-B v1 因 terminal decision off-by-one 永久 invalid；新
> labels、训练 checkpoint 和 offline method delta 尚未产生。
> closed-loop、matched-NLL 与 AndroidWorld sealed test 继续 locked。
>
> Canonical SHA256：predictor=`9548159b219795b1c258c28f772f53351256e0d728b88dd409cb333bd2100fe4`；
> P-1 source=`1b2b4374d1653bcf22444d8e708c71bc41ac87fa956243ddcb9bd963eeca7e96`；
> P-1 manifest=`e892e7e8f226e9500d978147a9698ad206a70ad9c303ebd918350f9e10ae6c5e`；
> P0 Execution-A=`f01beae98432bae19d02f7811d94dc5fa569263b0d917188f2489e15edbf371f`；
> P0 census manifest=`729d1e1046761177d53d0f320139331224c9f77f7add5097d04bce479566189b`；
> Freeze-B v2 config=`7d7dad8580939be67b56bb7ad5a9e06b771e6d0f25ffd3665da84d14466cbf75`；
> Freeze-B v2 manifest=`915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30`；
> consumed ledger=`b6f44c603b99d2f954b981e01818cf0afa028ce3a410ed935203532a097bb4ad`；
> P0 source-only=`7e65227e710009d3626bd0063d425c831dfc59e9a6bbe871b9d3e4d15e085e8b`。focused suite
> 为 `213 passed, 15 skipped, 24 subtests passed`。

## 最终接口

模型预测：

\[
\hat U_\theta(q_t,C_t,m_S),
\qquad
U_t(S)=D_t(\varnothing)-D_t(S).
\]

`C_t` 是冻结 candidate universe，`m_S` 是 selected/unselected type bit。模型始终看见全部有效候选；只有真实
padding 被 mask。外部预算不进入 feature、模型或 checkpoint：

\[
\hat S_B=\arg\max_{S\subseteq C_t,\ |S|\le B}\hat U_\theta(q_t,C_t,m_S).
\]

因此 empty、singleton、pair 可以全局比较，不再依赖 conditional greedy 的逐轮误差与 stop threshold。
`U(empty)=0` 由同一个 `C_t` 上的 empty baseline 差分精确约束。

## 执行链

```text
P-1: 610-shard metadata inventory
        │ path + bytes + LFS SHA only
        ▼
canonical consumed ledger (58 train-only + 49 forbidden)
        ▼
P0: policy-blind full-pool semantic census
        │ exclude 107 identities; no split/query/label
        ▼
Freeze-B: group-disjoint roster + query states + feature/grid/runtime/HF
        ▼
processor-only candidate freeze (exclude current-equivalent; recent <=16; fit 32k)
        ▼
exact capped D(S) production, phase 1: |S| <= 2
        ▼
Set Transformer main + DeepSets/Pairwise comparators
        ▼
one-shot offline evaluation vs recent/OCR-RGB/J/exact
        ▼
small identity-disjoint B3/B4 transfer study
        ▼
only then decide whether to open a new closed-loop contract
```

阶段不能合并：P-1 完成不授权 P0；P0 完成不授权 label forward；phase-1 offline GO 也不直接授权
closed-loop。

## 已实现模块

### Source inventory 与数据防火墙

- `set_utility_full_pool_inventory_v1.py`：严格验证 metadata-only P-1 config，要求完整 610 个连续 shard、
  positive byte size 与 lowercase LFS SHA256，exclusive 写 canonical manifest，CLI 返回的 SHA256 与实际
  落盘 bytes（含末尾换行）完全一致；
- `set_utility_consumed_ledger_contract.py` / `set_utility_consumed_ledger.py`：从六个 byte-pinned historical
  inputs 机械恢复 107 个 source identity，严格分成 `legacy_train_only=58` 与
  `forbidden_consumed=49`；
- `set_utility_full_pool.py`：P0 source core 直接消费 P-1 schema 和 canonical ledger，复用旧 parser 的所有
  non-length eligibility；科学上仅保留 `decision_count>=6`，不再沿用 12/64 上限。它从 new eligible pool
  排除 107 个 identity，同时为 consumed rows 生成无 instruction 原文的 group-hash audit；
- `set_utility_split_validation.py`：trajectory/source/group 三层不跨 role，formal-58 只能
  `legacy_train_only`，已消费 evaluation identities 不可训练或调参；它还要求 Freeze-B
  显式传入 historical legacy/forbidden group SHA256 集合，禁止新 source identity 用近重复
  instruction+app group 绕过 firewall。

P0 runner 与 source-only skeleton 已存在，且 source config 已绑定 canonical consumed ledger。真实 P-1 manifest
现已产生，独立 P0 Execution-A 也已绑定其 SHA256。Execution-A 只授权读取本地 pinned shards、row decode、
policy-blind semantic census 和一次 output write；source-only config 本身仍不授权任何 execution，role/split、query、
OCR/model、labels、training、GPU、closed-loop 与 sealed test 继续为 0。

正式 P0 已完成：扫描 8,146 rows，排除 1,213 rows，得到 6,933 条新 eligible trajectory / 6,928 个
instruction-app group。candidate-capacity strata 为 `6–9=1,736`、`10–17=3,601`、`18+=1,596`；decision
count 范围 6–54。107 个历史 consumed identity 全部被观测并隔离。该规模足以冻结新的
train/tune/one-shot evaluation roster，但不能把 P0 pool 直接全部视为训练集。

### Label 生产

- `set_utility_label_schedule.py`：支持 `n=1..16`、`K∈{2,3,4}` 的完整 capped subset inventory；按 state
  deterministic LPT 分 worker，并对完成 rows 做 missing/duplicate/unexpected 检查；
- `set_utility_label_producer.py`：冻结 query/source/group/provenance identity；排除 current-equivalent history
  event；先取最近 16 个 eligible candidates，再用真实 processor input length 逐个丢弃最老 event，直到
  `input_tokens+256<=32768`；最终 candidate ids 在 generation/teacher forward 前冻结；batch validator
  强制接收 Freeze-B 中预注册的 `maximum_reference_repeat_kl`，超限 state fail closed，
  validated batch 也回写该阈值供 result manifest 审计；
- `set_utility_label_table.py`：只接受完整、finite、non-negative 的 `D(S)` capped table，机械计算可正可负的
  `U(S)`，不 clipping。

第一阶段每 state 的 raw rows 为：

\[
R(n,K)=\sum_{k=0}^{K}\binom nk.
\]

设 `I=1[n<=K]`，一次 state 需要 2 次 canonical generations、1 次 reference teacher forward、1 次
identical-reference repeat forward，以及 `R-I` 次 capped-candidate teacher forwards；teacher forwards 总数为
`2+R-I`，KL measurements 为
`1+R-I`。对 `K=2`，`n=4/8/16` 的 row 数分别为 `11/37/137`。不能把 sampled pairs 声称为 exact B2 table。

### Features、模型、训练与评估

- `set_utility_features.py`：variable-`n` query/context/event/pair features；拒绝 budget、utility、group key
  等泄漏字段；
- `set_utility_data.py`：validated feature-label join、padding、state/cardinality/subset balanced weights；
- `set_utility_models.py`：
  - Pairwise-additive：显式一阶/二阶 interaction；
  - DeepSets：分别聚合 selected 与 unselected candidates；
  - Set Transformer（main）：所有 candidate token 可见，selection embedding 区分选/未选，无 positional
    embedding，padding-only attention mask；
- `set_utility_trainer_runner.py`：trajectory-uniform training、trajectory-equal tune selection、early stop、
  deterministic CPU checkpoint core；Set Transformer 的 `num_heads/num_layers/dropout` 必须通过严格
  `UtilityModelConfig` 显式给出，checkpoint 保存 config 与 hash；
- `set_utility_baselines.py` / `set_utility_search.py` / `set_utility_evaluation.py`：recent、positive-top-B
  OCR/RGB、oracle-independent `J`、exact search、true-utility rescore 与 trajectory-equal reducers。

当前 trainer 是 dependency-clean 的 CPU reference runner，不是正式 GPU execution contract。Freeze-B 还需固定
device/runtime、feature schema、architecture grid、optimizer/schedule、seed、checkpoint/HF layout 与完整 argv。
同一 Freeze-B 还必须固定 reference-repeat KL stability threshold 和 historical consumed-group
firewall inputs，不能看到 labels 后调整。

## 数据与 feature 决策

不能把 610 shards 全部当训练集。P0 先报告真实 eligible pool 与 `6–9 / 10–17 / >=18` strata；Freeze-B
随后以 trajectory + normalized instruction/app group 为单位一次性分 train/tune/one-shot evaluation。旧
formal-58 可加到 train；reference8、old-dev5、fresh16、confirm20 永远不能进入新训练、
调参或评估。

Source-A 暂不根据旧小样本决定最终 student representation。Freeze-B 必须在新 evaluation label access 前二选一：

1. q64/h64 + low-fidelity + OCR/RGB/recency；
2. 上述轻量特征再加 frozen GUI-Owl final-main normalized visual embedding。

Set Transformer、DeepSets、Pairwise 必须共享同一 feature bytes、labels、split 与 search/evaluation contract。
主模型可以有更高容量，但不能在 evaluation 失败后再换 feature 或扩大 grid。

## Source of Truth 与跨机器执行

- Git：source/config/tests、P-1/P0/Freeze manifests、轻量 aggregate、本文档；
- planned private HF dataset：`gavinlaw/causalcache-set-utility-new-development-mobile`，保存 source shards 的
  reusable projection、feature cache、exact `D(S)` shards、split/query manifests 与 predictions；当前未创建/
  未绑定 revision；
- planned private HF model：Freeze-B 另命名，保存 Set Transformer/DeepSets/Pairwise checkpoints；当前未创建。

P-1/P0 是 CPU/network/data 工作，不需要 GPU preflight。exact label production 优先 Hyper H200，在正式
GPU preflight 后按 state shard 并行；非 Taurus/Aries 单任务最多 4 张。每 worker 写聚合 shard，避免大量小文件。
训练只有在 labels 与 Freeze-B 都提交后启动；先在固定 runtime 跑 PyTorch behavioral smoke，再按有效并行度
选卡。不同芯片不要求 bitwise logits，但必须复用相同 feature/label/checkpoint bytes，并分别记录 latency、显存和
runtime metadata。

## 当前验证、失败与下一步

本机没有 PyTorch，因此 torch-dependent tests 被明确 skip；source/config/inventory/ledger/label/split/search/
evaluator 的纯 CPU tests 可运行。focused suite 为 `213 passed, 15 skipped, 24 subtests passed`。
全仓回归为 `1738 passed, 23 skipped, 38 failed, 644 subtests`；38 个失败来自已有 lifecycle
互斥测试、sandbox 下的 git worktree 操作与 full-suite import-order，不是 focused set-utility 回归。
真实 P-1 已完成 610-shard metadata inventory；manifest SHA256=
`e892e7e8f226e9500d978147a9698ad206a70ad9c303ebd918350f9e10ae6c5e`。详见
[P-1 记录](set_utility_full_pool_inventory_v1.md) 与
[consumed ledger](set_utility_consumed_ledger_v1.md)。

下一步严格是：

1. commit/push Freeze-B v2 terminal-index repair roster/query plan；
2. 从 clean pushed v2 另立 execution，只读取 1,200 条 frozen row，执行 processor-only candidate freeze 并
   生成 exact operation budget；
3. processor freeze 完成后另立 train-only policy throughput pilot contract，再立 label Execution-B 生产 phase-1
   `|S|<=2` tables；
4. 训练三类 predictor，one-shot offline evaluation；只有 learned family 超过 OCR/RGB 且不弱于 `J`，才打开
   identity-disjoint B3/B4 transfer study。

截至本 commit，没有本 full-pool 路线新产生的 restoration label、predictor checkpoint、offline method
delta、closed-loop episode、
matched-NLL 或 sealed-test result。
