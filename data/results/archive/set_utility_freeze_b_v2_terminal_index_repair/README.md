# Set Utility Freeze-B v2 terminal-index repair

本目录记录 Freeze-B v1 terminal off-by-one 的 policy-blind v2 repair。v1 manifest 原样保留，但其 query
states 状态为 `INVALID_QUERY_PLAN_TERMINAL_OFF_BY_ONE`，不得供 processor、labels 或 training 使用。

Canonical v2 manifest 是
[`../../manifests/set_utility_freeze_b_v2_terminal_index_repair.json`](../../../manifests/set_utility_freeze_b_v2_terminal_index_repair.json)，
SHA256=`915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30`。本步只读 parent v1 与 P0
manifest；raw source、OCR、processor/model、KL/utility、label、optimizer、evaluation label 和 closed-loop 操作均为
0。

完整根因、修复范围和下一执行边界见
[`../../../docs/archive/set_utility_freeze_b_v2_terminal_index_repair.md`](../../../../docs/archive/set_utility_freeze_b_v2_terminal_index_repair.md)。
