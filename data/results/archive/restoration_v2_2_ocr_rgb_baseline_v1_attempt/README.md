# Restoration v2.2 OCR/RGB baseline v1 attempt

本目录只保存第一次 formal OCR/RGB baseline attempt 的轻量 failure binding。该 attempt 已 fail closed，不能在
`v1` identity 下重跑、resume 或覆盖；后续修复必须使用新的 protocol、contract 和 canonical output directory。

## 结论

`INVALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V1`

clean pushed `main@aa5898f25e2e7d647363fe701ac90134bf744a5c` 通过 Git/source、input identity 与 geometry
preflight。reducer 随后在选择 allowlisted trajectory records 时，于第 0 行 identity lexer 失败：每条 immutable
trajectory 的同一个 `source_id` 会同时出现在 top-level 与 nested selection metadata，两个值完全相同；v1 lexer
却错误地要求整行只能出现一次该字段。失败发生在 feature materialization 与正式 score 产生之前：

- selected trajectory / OCR record semantic parse：`0 / 0`；
- image payload extract / image decode-resize：`0 / 0`；
- OCR / RGB / combined candidate score：`0 / 0 / 0`；
- selected-coalition distance lookup / formal state row / aggregate：`0 / 0 / 0`；
- GPU、OCR inference、policy/policy-vision、teacher/KL、gate、matched-NLL、closed-loop：全部 `0`；
- confirm/test semantic access 与 feature score：全部 `0`。

label 45-state / 420-row semantic validation 位于失败点之后，因此本 attempt 也尚未执行。Git/source、input file hash
与 geometry primary-record preflight 已执行，不能混称为“所有 operation 都为 0”。本 attempt 不产生任何 scientific
conclusion，也没有 OCR/RGB recovery、match rate 或 paired delta 可报告。Hyper00 上 canonical output 与 staging
directory 均确认不存在。

## 根因与修复边界

根因是 identity 的 byte-level lexer 错误地把“字段在整行恰好出现一次”当作 invariant，而 immutable schema 合法地
重复了相同 identity。post-failure byte-only forensic 确认：35/35 trajectory lines 各有两个相同 `source_id`，
210/210 OCR lines 各有一个 `image_member_path`，unique identities 分别为 35 和 210。修复必须精确冻结这两个
per-file occurrence denominator，并要求同一行所有 occurrence 的 UTF-8 decoded value 完全一致；零 occurrence、
escaped identity、数量漂移或不一致值仍 fail closed。allowed identities、
opaque nonselected semantics、feature、selection、statistics、operation ceiling 与所有 immutable inputs 必须保持
不变。replacement 必须显式绑定本 failure record，并使用新的 attempt identity；不得把 v1 写成成功或负
scientific result。该 forensic 只扫描 raw identity literal，没有 semantic parse record content。

## 运行与证据绑定

- protocol：`causalcache_restoration_v2_2_ocr_rgb_baseline_v1`；
- source：`main@aa5898f25e2e7d647363fe701ac90134bf744a5c`；
- contract SHA256：`08f57505d71e603d81d6915ef27cf008d0fe097a56d70a05f8b3519211e2e6f9`；
- host / container：`node-radixark-16-0000` /
  `39749a3bd0875f3c15216211f875220a4934fe62313487b537c1116e82040ff0`；
- container image digest：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- Python / Pillow：`3.12.3` / `12.2.0`；
- raw label archive SHA256：`99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e`；
- derived tree SHA256：`475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e`；
- exact exception：`ValueError: derived trajectories line 0 has invalid identity encoding`；
- post-failure forensic observation：`2026-07-16T19:54:16Z`。

该 CPU reducer 没有 durable attempt ledger；Git 记录与 output-absence check 是当前 failure provenance。没有 reusable
result artifact，因此没有创建新的 Hugging Face repo/version。Git 中的 [`summary.json`](summary.json) 是该
zero-score failure 的 canonical compact record。
