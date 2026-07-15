# Restoration v2.1 full-45 substrate

## 结论

唯一正式 attempt 的结果是：

```text
NO_GO_V2_1_FULL_45_SUBSTRATE
```

失败点不是 official-tool interface；在进入 teacher forcing 的 32 个 exact-agreement states 中，也都观测到
summary/full-history behavior distance。45/45 states 的两次 generation 都能 strict parse，90/90 outputs 都有
model-emitted closer 并能转换为 AndroidWorld action；但只有 32/45
states 的两次 canonical action 完全一致，低于冻结的 45/45 gate。其余 13 个 states 全部是相同 action type
下的 coordinate jitter：12 个 `click→click`、1 个 `swipe→swipe`。

因此本结果否定的是：

> 在 exact coordinate-level canonical equality 下，当前 GUI-Owl v2.1 full-history reference 足够稳定，可直接
> 进入 restoration attribution。

它不证明 restoration hypothesis 错误，也不证明 summary/history 没有作用。按照冻结 contract，v2.1
restoration confirm、coalition construction 和 gate training 必须停止；不能事后给这 13 个 states 加容差并把
本结果改写成 PASS。

## Frozen reduction

| 指标 | 正式结果 | Gate/说明 |
| --- | ---: | --- |
| fixed denominator | 45 | 30 `v2_label_train` + 15 `v2_development` |
| strict parse states | 45/45 | PASS |
| native outputs / closer / bridge | 90 / 90 / 90 | 两次 generation 均保留 |
| repeat canonical agreement | 32/45 = 0.7111 | 要求 45/45，FAIL |
| finite-logit states | 32/45 = 0.7111 | mismatch states 按 schedule 不进入 teacher；没有 non-finite failure |
| teacher forwards / KL measurements | 96 / 64 | 32 eligible states × 3 / × 2 |
| repeat KL | min/median/max = 0 | 32 个 agreed states |
| repeat-noise epsilon | 0.0001 | `max(1e-4, 10 × mean repeat KL)` |
| memory-sensitive states | 32 | 要求至少 8，PASS |
| summary-reference KL | min 0.000783；median 0.031472；mean 0.044022；max 0.196197 | 32/32 高于 epsilon |
| retry / top-up | 0 / 0 | 无补样 |
| confirm / expert / restoration / baseline / gate work | 全部 0 | 仍 locked |
| duration | 327.912 s | Hyper00，single H200，BF16，batch-1 |

13 个 mismatch 中，10 个来自 `v2_label_train`、3 个来自 `v2_development`；decision steps 4/5/6 分别为
3/5/5 个。完整 per-state outputs、teacher metadata 与 GPU KL audits 只保存在 private HF raw archive；Git 中的
[`summary.json`](summary.json) 是 raw archive 内 `aggregate.json` 的逐 byte 副本。

## Source of Truth

- source Git commit：`7a5b6d5710fe4d054936b5aa474648149f725edb`；
- frozen config SHA256：`0924dd66fab9440bed66585765e9b1f5ab6fb80fbdf65a1492efba7ae81e116b`；
- run-contract SHA256：`094b059fc706b5d0f510e0f47a751e82162b6d9c9abe732060e68b499c014382`；
- canonical attempt：`restoration-v2-1-full-45-substrate-v1`；
- runtime：`hyper00` / `node-radixark-16-0000`，container-local `cuda:0` 映射 physical H200 GPU 2，
  BF16、greedy batch-1，PyTorch `2.11.0+cu130`、Transformers `5.6.0`；
- UTC bracket：`2026-07-15T22:59:20.590509Z`--`2026-07-15T23:04:48.502942Z`；
- private HF dataset：<https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile>；
- tag：`v2.1-full-45-substrate-v1`；
- immutable revision：`814506ef1450838d4bc6ed3d89fe53e0773d92fb`；
- HF path：`raw/restoration-v2-1-full-45-substrate-v1.tar`；
- raw canonical USTAR：962,560 bytes，SHA256
  `8cd53d6e56d5ad509da2af91d73d9e83b4db989ffc26aa18e1bca84e4c4f4fa4`；
- tree inventory SHA256：`6c5e515ad6c6323bd14984e107931b5d3f2dd738bb8eb2391dea87c55e119281`；
- aggregate SHA256：`f385baf261510dd82e510c8b40a05e5c2a5b91ef7e2329b4a4771b938644c6b8`；
- sibling ledger SHA256：`b6a2a58ef58c2acb7b3cee83c49984d7c8d7fed2a4a006088a00705fc978b0b8`；
- run manifest SHA256：`34425730ef1b09a0e1b201889601cf9795600d8300206356e828aa8d9c155acc`；
- Git artifact manifest：[`artifact.json`](artifact.json)，SHA256
  `9aa7fd6bdc20173532649a4133faf565e99975af51e38d57cf9612f29af996be`。

HF tag 已通过 API 解析到上述 immutable revision；随后从该 40-hex revision 强制 fresh-download，得到同样的
962,560 bytes 与 raw SHA256。正式 Git manifest 由这个 fresh download 创建，而不是由 upload 前的本地 tar
创建。当前 result milestone 首次提交前，committed-binding validator 仍为 pending；必须先 commit/push
manifest，再从 clean descendant `main` 对该 fresh archive 验证，才能闭合完整证据链。

## 下一步边界

当前 v2.1 路线到此 `NO_GO`，不运行 restoration 或 confirm。若继续探索，必须把 coordinate-level
equivalence 作为新的 versioned protocol：先给出不依赖本次成功率调参的 executable/UI-element equivalence，
保留本次 NO_GO，不复用本 attempt，不对当前 45 states 做 retroactive PASS。该方向是否值得继续，应在新的
source-only design review 后再决定。
