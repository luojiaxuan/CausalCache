# Gate v1 formal-58 train-only training

> 当前状态：Source-A=`e20f004ab79afaba4a04a108e70779d49087b2b9` 与唯一
> Execution-B=`bad28b74c421ccf6be1ab2f4407f7bad3414a2f2` 已冻结并 push。Hyper00 CPU-only formal-58
> OOF/final fit、private HF publication 和只读 immutable replay 均已完成；`gate_trained=true`，10 个
> checkpoint 已封存。fresh-16、legacy dev-5、confirm-20、matched-NLL 与 closed-loop 仍是零访问/零执行。

## 目标与边界

历史 Source-A 把已闭合的 formal-58 train-only cache 连接到已预注册的 conditional/independent gate
trainer，并在 source-freeze 阶段保持零训练。machine-readable contract 是
[`causalcache_gate_v1_formal_train_v1.json`](../code/configs/causalcache_gate_v1_formal_train_v1.json)。该 config
与下列 source 一起构成 Source-A：

- [`gate_v1_formal_train_contract.py`](../code/causalcache/gate_v1_formal_train_contract.py)；
- [`gate_v1_formal_train.py`](../code/causalcache/gate_v1_formal_train.py)；
- [`gate_v1_formal_train_runner.py`](../code/causalcache/gate_v1_formal_train_runner.py)；
- [`validate_gate_v1_formal_train_contract.py`](../code/scripts/validate_gate_v1_formal_train_contract.py)；
- [`manage_gate_v1_formal_train.py`](../code/scripts/manage_gate_v1_formal_train.py)。

临时 Python 直接调用 `run_formal_oof`、从工作区手写 checkpoint，或绕过 Source-A/Execution-B 边界都不是
合法 formal execution。

## 冻结输入

Source-A 必须精确绑定下表值；任一路径、size、hash、tag 或 revision 漂移都 fail closed。

| Input | Frozen identity |
| --- | --- |
| formal feature cache | `formal58-transport-repair/v1/feature-cache-v1.tar`，SHA256 `81fded50c4450700220742d3e0be9a5585d1bc51086150515b463bbdf4b4df8e`，993,280 bytes |
| formal label cache | `formal58-transport-repair/v1/label-cache-v1.tar`，SHA256 `4f9ef172aaa94c3ea8ce53aa43336c9fcb7800c24e181e1462d8239e31053cee`，163,840 bytes |
| cache bundle manifest | `formal58-transport-repair/v1/cache-bundle-manifest-v1.json`，SHA256 `15c8bf56ddad4f6f278599c32aaadcd813e0e016db0d523b4892eb47f8d9d144` |
| private HF dataset | `gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile` |
| cache tag | `gate-v1-formal58-cache-transport-repair-v1` |
| tag-resolved immutable commit | `a61b31bf2e69be00f94469f4a2f2d6b336fcc386` |
| annotated tag object | `c603397b9b1b1c3ad472f49125f827b8a096997d` |
| formal-58 join audit | SHA256 `551e70b7e99f7a761f9c50d2adae3be72b0933044982a015a65e93372e41d77b` |
| gate v1 preregistration | [`causalcache_gate_v1_preregistration.json`](../code/configs/causalcache_gate_v1_preregistration.json)，SHA256 `37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b` |
| cache completion result | [`summary.json`](../data/results/gate_v1_formal58_cache_transport_repair_v1/summary.json)，SHA256 `67d0bb81737b4755925ae2744d4af59e2638f69fe0bf9ba1c19c3ba3b89c38b2` |

trainer 只允许分别 strict-readback feature 与 label cache，两者各自验证后再按
`(source_id, state_id)` join。禁止重新读取共置 development records 的 parent artifacts，也禁止使用
generic parent readers。

## Semantic 访问防火墙

Source-A-only validation 不读任何 semantic record；formal-58 transport bytes 的存在不等于 semantic
decode。未来 Execution-B 也只允许 formal-train roster：

| Operation | Source-A | Execution-B planned fixed count |
| --- | ---: | ---: |
| formal trajectory semantic decode | 0 | 58 |
| formal feature-state semantic decode | 0 | 174 |
| formal label-state semantic decode | 0 | 174 |
| cache join validation | 0 | 174 |
| fresh-16 semantic decode | 0 | 0 |
| legacy dev-5 semantic decode | 0 | 0 |
| confirm-20 access | 0 | 0 |
| matched-NLL evaluation | 0 | 0 |
| closed-loop episode | 0 | 0 |

role/split、`source_id`、`state_id`、$D(S)$、oracle subset 与 policy-vision feature 不得进入 training
tensor。十个 final checkpoint 和全部 train-only provenance 完成前，不得打开任何 development
semantic content。

## 冻结训练逻辑

运行时为 CPU-only/no-GPU、FP32 full-batch deterministic PyTorch，`torch` intra/inter-op threads 都固定为
1。B 在任何 token、cache semantic decode 或 optimizer 之前还必须读取 host 侧生成的 mode-0600 Docker
inspect receipt；receipt 固定 `DeviceRequests=[]`、`Privileged=false`、`Runtime=runc`、运行中 container/image
identity 与唯一 `/data` bind mount。仅靠容器内 `torch.cuda.is_available()==false` 不足以通过 runtime boundary。
模型和选择规则继承 gate v1 preregistration：

- family 顺序固定为 conditional、independent；
- LR grid 固定为 `3e-4`、`1e-3`，seed 固定为 `0..4`；
- 五折大小固定为 `12 / 12 / 12 / 11 / 11 trajectories`；
- 每 family 有 10 个 OOF trials，每 trial 同时训练 5 个 fold model；两个 family 合计 100 条
  fold-training track；
- 每 epoch 只在 held-out fold 每条 trajectory 的唯一 `n=4,B=2` state 计算
  raw-utility/oracle ratio；改进必须严格超过 `1e-4`，50 epochs 无改进停止，最多 500 epochs；
- 用五 seed OOF mean 选 LR，差不超过 `1e-4` 时固定取 `3e-4`；
- 每 family/seed 从相同 seed 初始化，在全部 formal-58 上训练该 seed 选定的 epoch 数，最终
  生成 5 conditional + 5 independent checkpoints。

这一 OOF 只用于 train-only epoch/LR selection；任一被评估 trajectory 都没有参与对应 fold model
的训练。它不是 formal-58 in-sample paper metric。

## 正式输出与 model destination

Execution-B 已一次性形成：

- conditional 和 independent 的完整 `2 LR × 5 seed` OOF reports，不允许只保留 winner；
- 两个 selected learning rates、每 seed selected epoch 和两个 selection SHA256；
- 十个 metadata-free safetensors checkpoints，每个都有独立 canonical model-state SHA256 与 checkpoint
  artifact SHA256；
- conditional/independent ensemble manifests、run manifest、bundle manifest 与完整 operation counts；
- source/cache/runtime provenance，payload commit、manifest commit、annotated tag 和 fresh immutable replay evidence。

private HF model destination 固定为：

| Field | Formal value/status |
| --- | --- |
| repo | `gavinlaw/causalcache-gate-v1-formal58-selector-mobile` |
| repo type | `model`, private |
| tag | `gate-v1-formal58-train-v1` |
| Source-A preflight | repo/tag/revision/artifact verified absent |
| payload commit | `a6c9e7f6dab6bc27794438b5b66da07fd59b2889`；10 checkpoints + 2 full OOF reports |
| manifest commit | `23f6786075c7bff91f93fd7e8a878e070efb72a9`；payload 的 direct child；2 ensemble + run + bundle manifests |
| annotated tag object | `fa85e74685d5ab509e60b469c6c8cae61efab4d6` |
| immutable model revision | `23f6786075c7bff91f93fd7e8a878e070efb72a9` |

Source-A 没有创建该 repo、tag 或任何 checkpoint。Execution-B 的训练 artifact strict-readback、两阶段 remote
commit、annotated tag 与 fresh immutable replay 已全部闭合，final completion 将 `gate_trained` 设为 true；完整
SHA、OOF selection 与 operation counts 见
[`../data/results/gate_v1_formal58_train_v1/`](../data/results/gate_v1_formal58_train_v1/)。

## Source-A → Execution-B

Source-A 包含 scientific config、contract、trainer adapter、runner source、manager、validator、tests 和本文档，但不包含
generated runner freeze。Source-A commit/push 后已机械生成：

```text
code/configs/causalcache_gate_v1_formal_train_runner_v1.json
```

该文件 SHA256 为 `a89bb081c6a98e4da7da178e8930357773e7ee23cea211600cffea4500ba7532`；B 是 A 的 direct
single-parent child，且它是唯一 source-tree diff。

Execution-B 必须是 A 的 direct single-parent child，上述文件是唯一 source-tree diff。B 不得修改
trainer、config、docs、tests 或任何 input/output binding。正式运行前还必须验证 clean Git、
`HEAD == origin/main == live origin/main == B`、runner-bound Source-A inventory 和 loaded-module inventory。

## Source-A-only 验证

```bash
cd code
PYTHONPATH=. python3 -m scripts.manage_gate_v1_formal_train validate-source \
  --repository-root .. \
  --contract code/configs/causalcache_gate_v1_formal_train_v1.json
```

validator 不读 HF token、不访问 network、不 import PyTorch、不 semantic-decode cache、不写 claim/state/artifact，
并要求 generated Execution-B runner 不存在。model destination 的 absent 状态由独立只读 Hub preflight 记录，
不伪装成 source-only validator 的网络检查。validator 的
合法状态必须明确包含：

```text
training_executed = false
execution_authorized = false
formal58_cache_access_authorized = false
formal58_training_authorized = false
checkpoint_publication_authorized = false
gate_trained = false
fresh16_access_authorized = false
legacy_dev5_access_authorized = false
confirm20_access_authorized = false
matched_nll_authorized = false
closed_loop_authorized = false
```

因此 Source-A freeze 只证明正式训练的输入、算法、输出和执行边界已经闭合，不是 learned
gate 有效性证据。

## B 生成与 runtime receipt（历史执行契约，已完成）

以下命令不属于 Source-A；必须等 A clean push 后执行。第一条只生成唯一 runner-freeze B，随后 B 必须作为
A 的 direct single-parent child 单独 commit/push：

```bash
cd code
PYTHONPATH=. python3 -m scripts.manage_gate_v1_formal_train materialize-runner-freeze \
  --repository-root .. \
  --contract code/configs/causalcache_gate_v1_formal_train_v1.json \
  --source-a-git-commit <SOURCE_A_COMMIT>
```

B push 后，在 Hyper00 host、正式 no-GPU container 仍在运行时生成 runtime receipt。`<HOST_DATA_ROOT>` 是
container `/data` 的真实 host bind source；receipt 写入该目录下的固定 experiment path，不进入 Git：

```bash
cd /data02/jaxan/CausalCache/code
PYTHONPATH=. python3 -m scripts.manage_gate_v1_formal_train capture-runtime \
  --repository-root .. \
  --contract code/configs/causalcache_gate_v1_formal_train_v1.json \
  --execution-b-git-commit <EXECUTION_B_COMMIT> \
  --container-name <sglang-omni-jaxan-MMDDHHMM> \
  --host-data-root <HOST_DATA_ROOT>
```

只有 receipt、clean pushed B、CPU/thread environment、private cache identity 全部通过后，`run` 才会创建
global claim 并开始 formal-58 semantic decode。`validate` 只接受已存在的 terminal hard-link completion，重新
下载 base/payload/manifest 三个 immutable revisions、重放 10 个 checkpoint，并要求 remote mutation 为 0。

## Formal result 与只读 replay

2026-07-18，B 在 pinned no-GPU CPU runtime 的正式 `run` 返回
`VALID_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1`。conditional/independent 都选择 LR `3e-4`；五 seed
mean OOF raw-utility/oracle ratio 分别为 `0.8925353801368878` / `0.9063764691683989`，final epochs 分别是
`[62,59,98,4,13]` / `[60,51,6,56,54]`。两阶段 HF identity 是：

```text
payload commit:   a6c9e7f6dab6bc27794438b5b66da07fd59b2889
manifest commit:  23f6786075c7bff91f93fd7e8a878e070efb72a9
annotated tag:    fa85e74685d5ab509e60b469c6c8cae61efab4d6
```

完成态只读 replay 使用同一 B、token path、data root 与 thread/runtime binding：

```bash
cd code
PYTHONPATH=. python3 -m scripts.manage_gate_v1_formal_train validate \
  --repository-root .. \
  --contract code/configs/causalcache_gate_v1_formal_train_v1.json \
  --execution-b-git-commit bad28b74c421ccf6be1ab2f4407f7bad3414a2f2 \
  --hf-token-file /data/.secrets/hf_key.txt \
  --data-root /data \
  --fresh-download-parent /data/tmp
```

它返回 `REVALIDATED_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1`，逐个重放 10 个 checkpoint，remote
mutation count 为 `0`。完整 completion binding 见
[`../data/results/gate_v1_formal58_train_v1/`](../data/results/gate_v1_formal58_train_v1/)。OOF 仍只是
train-only LR/epoch selection evidence，不是 fresh-16 泛化或 set-conditioning GO。下一步必须另立 fresh-16
evaluation Source-A；本 trainer 不得直接打开 fresh-16、旧 dev-5、confirm、matched-NLL 或 closed-loop。
