# Gate v1 formal-58 training：COMPLETED + REVALIDATED

## 结论

formal-58 train-only OOF、model selection 与 final refit 已从唯一
Source-A=`e20f004ab79afaba4a04a108e70779d49087b2b9`、Execution-B=`bad28b74c421ccf6be1ab2f4407f7bad3414a2f2`
在 Hyper00 的 CPU-only runtime 完成。正式 `run` 返回
`VALID_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1`；随后同一 B 的只读 immutable replay 返回
`REVALIDATED_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1`，remote mutation count 为 `0`。

训练产出 5 个 conditional 与 5 个 parameter-matched independent checkpoint。两个 family 都由完整
`2 LR × 5 seed × 5 folds` train-only OOF 选择 `3e-4`；conditional 与 independent 的五 seed mean OOF
raw-utility/oracle ratio 分别为 `0.8925353801368878` 和 `0.9063764691683989`。这些数值只用于冻结 LR 与
每 seed epoch，不是 fresh-development 或 paper-level 泛化结果。尤其是 independent 的 train-only OOF 较高，
不能据此判定 set-conditioning 成败；该问题只由下一阶段一次性 fresh-16 primary GO 回答。

## Canonical artifacts

| Artifact | Canonical location | Immutable identity |
| --- | --- | --- |
| 10 checkpoints、2 full OOF reports、4 manifests | private HF model [`gavinlaw/causalcache-gate-v1-formal58-selector-mobile`](https://huggingface.co/gavinlaw/causalcache-gate-v1-formal58-selector-mobile) | tag `gate-v1-formal58-train-v1` → manifest commit `23f6786075c7bff91f93fd7e8a878e070efb72a9`；annotated-tag object `fa85e74685d5ab509e60b469c6c8cae61efab4d6` |
| compact completion record | Git: [`summary.json`](summary.json) | config SHA256 `bff92026…bf2f`；B=`bad28b7…a2f2`；completion SHA256 `8e31acba…264d` |

HF payload commit 为 `a6c9e7f6dab6bc27794438b5b66da07fd59b2889`，manifest commit 是其 direct child。
checkpoint artifact/model-state SHA、selection SHA、完整 operation counts、runtime receipt 与 local hard-link seal
均在 [`summary.json`](summary.json) 中。`/data/artifacts/...` 与 `/data/experiments/...` 仅是 staging/audit 副本，
reusable model 的 source of truth 是上述 private HF immutable revision。

## 访问边界与下一步

本次 execution 只 semantic-decode formal-58 的 58 trajectories / 174 feature states / 174 label states；
fresh-16、legacy dev-5、confirm-20、matched-NLL 与 closed-loop 的访问/执行计数均为 `0`，相应 authorization
仍全部为 `false`。`gate_trained=true` 只表示训练与模型封存完成，不表示 learned selector 已通过 GO。

下一步是另立 fresh-16 evaluation Source-A/Execution-B，先绑定本目录与 HF ensemble，再进行唯一一次
16-trajectory / 48-state primary GO。只有该报告 immutable 封存并 replay 后，才能打开旧 dev-5 做
combined-21 compatibility guard；confirm、matched-NLL 与 closed-loop 仍需新的 post-GO contract。
