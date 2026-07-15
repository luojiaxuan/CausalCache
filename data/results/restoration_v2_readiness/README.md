# Restoration v2 readiness authorization

本目录保存 dependency 8 的首次正式 clean-Git authorization 结果。它绑定已经 superseded 的 GPU-0 runtime，
因此是历史证据，不是当前 GPU-2 execution authorization。

- state：`SCREENING_ALLOWED`；
- confirm state：`CONFIRM_LOCKED`；
- dependencies：8/8 passed；
- summary：`summary.json`；
- summary SHA256：`20b9e81050e56622cfe9bfe4dd1343f33513ae7f37fe25716d13b86ea8115964`；
- execution config：`code/configs/restoration_v2_execution_hyper00_v1.json`，SHA256
  `f2b6521ed8b1d65d4b6170c94c5c4cf8e46d135e352bdb52b21bf4e8f64173a5`；
- implementation commit：`a2aeb7f1930d40cb569c8e2adfc0dc39e950d131`；
- validation commit / `origin/main`：`429c4584ba7c18eee0b96741b6c1514bd4d4d7ec`；
- allowed roles：`v2_label_train`、`v2_development`；
- confirm role：`v2_confirm_primary` 仍禁止访问。

validator 从 implementation commit 的 Git blobs 复核 config 与 14 个 source roles，从当前 Git 文件复核全部
dependency evidence，并要求 clean worktree、canonical remote、`HEAD == origin/main`。本次运行明确记录
`policy_imported_by_validator=false`、`policy_output_generated=false`、
`restoration_output_generated=false`。

该结果在当时正式关闭 restoration v2 dependency 8，只授权固定 45-state development substrate screening，
不授权 confirm。GPU-2 re-anchor 后，canonical manifest 已主动切到
`SCREENING_LOCKED_RUNTIME_REANCHOR_PENDING_RESIGN`；新的 readiness summary 将在 GPU-2 clean-Git authorization
通过后单独记录。后续 screening runner 每次启动仍会重新运行 validator；若 Git、evidence、runtime 或 role
发生漂移，必须在 policy import 前 fail closed。
