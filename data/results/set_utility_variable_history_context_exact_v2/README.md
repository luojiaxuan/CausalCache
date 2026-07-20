# Variable-history exact-grid context postflight v2

- 结论：`PASS_FULL_HISTORY_EXACT_GRID_CONTEXT_POSTFLIGHT`；
- states：12,792/12,792 fit，0 unfit；
- 最大真实 prompt：30,036 tokens；加 256 action reserve 后为 30,292，低于 32,768；
- token shards：256/256，约 67GB，失败为 0；
- token source revision：`47161cac5b26c3466e89343c43b3e75a05f11b46`；
- 远端完整输出：Hyper00 `/data02/jaxan/runs/causalcache-variable-history-context-exact-47161ca`；
- 大 artifact 状态：`PENDING_HF_UPLOAD`。

该结果只解除完整历史 reference 的 context-fit blocker，不包含 restoration labels 或 predictor 结果。
