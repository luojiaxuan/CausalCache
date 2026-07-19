# Variable-history source 与 512-token context census

- Hyper00 已完成 256/256 resumable Parquet shards：1,200 trajectories、13,160,599,795 bytes；
- source manifest SHA256：`a37ce0b5076e427a6d2b02f86994468e8d456c6df9e2132dd33e4a255042cde4`；
- persistent path：`/data02/jaxan/artifacts/causalcache-set-utility-variable-history-source-v1-a7213db`；
- HF 状态：`PENDING_HF_UPLOAD`，目标 `gavinlaw/causalcache-set-utility-variable-history-mobile`；
- 512 visual-token census 覆盖 12,792/12,792 states，12,791 fit；唯一不 fit state 为
  `5238570709976982:decision:046`，prompt+256 reserve 为 33,236，超过 32,768 共 468 tokens；
- verdict：`BLOCK_FULL_HISTORY_CONTEXT_CENSUS`。不截断候选，改用新的 480-token v2 profile 重跑 census。

该结果只否定 512-token reference profile，不是否定 variable-history data definition 或 restoration 方法。
