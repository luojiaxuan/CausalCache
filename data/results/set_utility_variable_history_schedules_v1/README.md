# Variable-history broad schedules v1

- 完成：256/256 logical shards；
- train/tune states：11,746（10,680 / 1,066）；
- coalition labels：461,040；
- 需要 policy forward 的非 full-anchor coalitions：449,294；
- 每个 `n_t>5` state 为 40 个 label-blind stratified subsets，`n_t=5` 为全部 32 个 subsets；
- candidate universe 始终是完整 `C_t`，没有 recent-`n` 截断；
- Hyper00/Hyper01 persistent path：`/data02/jaxan/artifacts/causalcache-set-utility-variable-history-schedules-v1-9267e39`；
- artifact 状态：`PENDING_HF_UPLOAD`。

schedule 只使用 frozen hidden-state similarity 与 OCR 做采样覆盖，不读取 restoration target 或 evaluation 结果。
