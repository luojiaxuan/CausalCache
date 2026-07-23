# Restoration v2 readiness authorization

本目录保存 dependency 8 的 clean-Git authorization 结果。`summary.json` 是当前 GPU-2 canonical
authorization；首次 GPU-0 结果已归档为 `gpu0-summary.json`。

- state：`SCREENING_ALLOWED`；
- confirm state：`CONFIRM_LOCKED`；
- dependencies：8/8 passed；
- summary：`summary.json`；
- summary SHA256：`36bd183f9fb4e0c3440d9ca8bc5c01e97cd1d25ca0c42e391961026fb8bcaa02`；
- execution config：`code/configs/restoration_v2_execution_hyper00_v1.json`，SHA256
  `819cb973d9211a5a9b3b4d7c109520605e35b07ad008bf125353d33d0bf91ca0`；
- implementation commit：`14faaa44cf1b2044b1f1bcb3c9dcfce36eb452aa`；
- validation commit / `origin/main`：`caa4f376026d13acd21db1f88b893bc92dd482b1`；
- allowed roles：`v2_label_train`、`v2_development`；
- confirm role：`v2_confirm_primary` 仍禁止访问。

历史 GPU-0 summary：`gpu0-summary.json`，SHA256
`20b9e81050e56622cfe9bfe4dd1343f33513ae7f37fe25716d13b86ea8115964`；它不再授权当前 execution runtime。

validator 从 implementation commit 的 Git blobs 复核 config 与 14 个 source roles，从当前 Git 文件复核全部
dependency evidence，并要求 clean worktree、canonical remote、`HEAD == origin/main`。本次运行明确记录
`policy_imported_by_validator=false`、`policy_output_generated=false`、
`restoration_output_generated=false`。

当前 GPU-2 结果正式重新关闭 restoration v2 dependency 8，只授权固定 45-state development substrate
screening，不授权 confirm。后续 screening runner 每次启动仍会重新运行 validator，并在 artifact/model load 前
实时核对 GPU/software identity；若 Git、evidence、runtime 或 role 发生漂移，必须在 policy import 前 fail
closed。
