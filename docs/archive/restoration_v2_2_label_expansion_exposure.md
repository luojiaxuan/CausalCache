# Restoration v2.2 label expansion exposure ledger

> Canonical ledger 已从 clean pushed `main@50a3f73` 物化，SHA256 为
> `e5f7e7b5ffeb71ac117b5cb4a470eb31648978a734edb9a521125d3bda29a323`；6/6 required intersections
> 为空，structural manifest binding 已启用。它仍不授权任何 policy/restoration/gate/confirm output。

本协议在 64 条 expansion trajectory 的任何 policy/restoration output 前，冻结它们与既有 output exposure、
sealed confirm 的身份边界。source-only implementation 位于
[`code/causalcache/restoration_v2_2_label_expansion_exposure.py`](../../code/causalcache/restoration_v2_2_label_expansion_exposure.py)。

## 审计对象

既有 parent roles 分成两类：

- prior output-exposed：reference 8 + old train 10 + old development 5，共 23 条；
- sealed confirm：20 条，保持与 output-exposed cohort 分离。

Expansion cohort 固定为 48 train + 16 fresh development，共 64 条。账本机械验证 expansion 与三个 prior
roles、prior-23 union、confirm-20、all-reserved-43 共六个交集都为空。

账本不写出 source IDs、instruction、action、image/OCR identity 或 policy/restoration value，只记录 cohort count、
ordered-ID digest 与 empty-intersection digest。它绑定：

- 48/16 split config 与 parent selection；
- 原始 16 个 Parquet 的 source manifest；
- 历史 pre-output exposure ledger；
- spatial audit 前已经更新的 prior-output exposure ledger；
- 若 structural split manifest 已落盘，则动态绑定其 SHA256 与 generator revision。

## Claim 边界

Identity disjointness 由 bound inputs 机械重算。`未访问 expansion policy/restoration output`、`未 top-up`、
`未读取 confirm output` 属于 pre-output process declaration，不包装成 cryptographic proof。原始 GUIOdyssey
source 的结构读取也不等于 policy/restoration output exposure。

Selection process 固定为 parent salted order：排除 prior 23 后剩余 88，按 `decision_count>=5` 得到 84；前 20
保持 sealed confirm，tail 64 再以前 48 / 后 16 划分。membership 不使用 instruction/action/image/OCR、memory
sensitivity、oracle recovery 或 model performance。

## Materialize 与验证

source commit/push 后，从 clean canonical `main` 运行：

```bash
cd code
python3 -m scripts.materialize_restoration_v2_2_label_expansion_exposure \
  --config configs/causalcache_restoration_v2_2_label_expansion_v1.json \
  --parent-selection ../data/manifests/restoration_v2_selection.json \
  --source-file-manifest ../data/manifests/independent_reference_gate_v1_source_files.json \
  --parent-exposure ../data/manifests/restoration_v2_exposure.json \
  --prior-output-exposure ../data/manifests/spatial_reference_audit_v1_exposure.json \
  --expansion-selection ../data/manifests/restoration_v2_2_label_expansion_selection.json \
  --git-revision <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --output ../data/manifests/restoration_v2_2_label_expansion_exposure.json

python3 -m scripts.validate_restoration_v2_2_label_expansion_exposure \
  --config configs/causalcache_restoration_v2_2_label_expansion_v1.json \
  --parent-selection ../data/manifests/restoration_v2_selection.json \
  --source-file-manifest ../data/manifests/independent_reference_gate_v1_source_files.json \
  --parent-exposure ../data/manifests/restoration_v2_exposure.json \
  --prior-output-exposure ../data/manifests/spatial_reference_audit_v1_exposure.json \
  --expansion-selection ../data/manifests/restoration_v2_2_label_expansion_selection.json \
  --ledger ../data/manifests/restoration_v2_2_label_expansion_exposure.json
```

Materializer 要求 clean、已 push 的 `main` 和 exclusive output；validator 重新绑定当前与 generator commit 中的
所有 source bytes，并严格拒绝 duplicate JSON keys 与 NaN/Infinity。
