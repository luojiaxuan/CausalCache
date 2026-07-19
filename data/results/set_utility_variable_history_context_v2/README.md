# Variable-history 480-token context census

480-token v2 profile 在 tokenizer/chat-template census 上覆盖 12,792/12,792 states，全部 fit。最大
prompt+256 action reserve 为 31,764，低于 32,768 共 1,004 tokens。

这是启动 visual-token extraction 的 PASS；正式 label forward 前还会用真实 `image_grid_thw` 逐 observation
重算 exact context postflight，避免把 target token 数误当成每张图实际 token 数。

Persistent result：Hyper00 `/data02/jaxan/runs/causalcache-variable-history-context-v2-1285ba8`。
