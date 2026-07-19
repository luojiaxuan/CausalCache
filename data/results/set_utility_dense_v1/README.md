# Dense per-step census

旧版每条 trajectory 只取 anchor/terminal，现改为从 decision step 6 起逐 step 建 state，并保持 trajectory-level split。

- 1,200 trajectories；
- 12,792 eligible states：10,680 train / 1,066 tune / 1,046 evaluation；
- 旧 artifact 已覆盖 12,635 states；
- 20 条超长 trajectory 的中段缺 157 states 所需截图，下一步从 pinned raw source 补齐。

机器可读结果见 [`census.json`](census.json)。
