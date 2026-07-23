# Dense per-step census

旧版每条 trajectory 只取 anchor/terminal，现改为从 decision step 6 起逐 step 建 state，并保持 trajectory-level split。

- 1,200 trajectories；
- 12,792 eligible states：10,680 train / 1,066 tune / 1,046 evaluation；
- 旧 artifact 已覆盖 12,635 states；
- 20 条超长 trajectory 的中段缺 157 states 所需截图，已从 pinned raw source 补回 77 张 PNG；四个 processor shards 的完整展开校验精确通过 12,792 states。

机器可读结果见 [`census.json`](census.json) 和 [`input_validation.json`](input_validation.json)。补图暂存于 Hyper00 `/data02/jaxan/artifacts/causalcache-set-utility-dense-v1-backfill-d43a15c`（42MB），状态 `PENDING_HF_UPLOAD`，将与 dense labels 一起发布到现有 CausalCache dataset repo。

Label generation 使用 [`set_utility_dense_v1_rollout.json`](../../../manifests/set_utility_dense_v1_rollout.json) 的 8×H200 logical shards。Predictor 的正式 selector 只报 at-most-`B`；dense labels 完成后，先缓存 frozen GUI-Owl visual hidden states，再训练能消费 visual token sequence 的 Set Transformer。旧 64-d signed-hash、RGB histogram 与 OCR overlap 只作为 cheap-feature baseline，不再用于判断 Set Transformer 上限。
