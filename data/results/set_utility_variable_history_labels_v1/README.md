# Variable-history formal label rollout v1

状态：`RUNNING`。

- code/source revision：`969f2b9aaa516295b03643aedf3f16af73c631c3`；
- scientific config SHA256：`90c72283e14a65ad234658b0687aab29dd10ac20a5608c9ebc241f0697243566`；
- schedule：11,746 train/tune states、461,040 coalition labels；
- execution：Hyper00 GPU 0–3 + Hyper01 GPU 2–5，24 workers，每卡 3 个 in-flight processes；
- persistent output：两台机器均为 `/data02/jaxan/runs/causalcache-set-utility-variable-history-labels-v1-969f2b9`；
- logs：两台机器均为 `/data02/jaxan/logs/causalcache-variable-history-labels-v1-969f2b9`；
- resume unit：coalition microbatch；reference、microbatch 与 terminal state 均单独原子落盘；
- initial validation：86 completed states / 3,232 labels，`n_t=5..9`，所有距离 finite 且非负；
- 2026-07-20 01:43 UTC checkpoint：2,332 / 11,746 terminal state records；从 24-worker 重启后净增
  2,270 states / 80.3 minutes，约 28.3 states/min；按当前速度剩余约 5.5 小时，考虑后段长历史与尾部
  shard 不均衡，执行 ETA 记为 5.5--7 小时；
- 2026-07-20 03:06 UTC，旧 runner 的 6 个 partitions 各遇到 reference tool-call parse failure；该异常原本
  未进入 state-level admissibility，导致 Hyper00/Hyper01 各一个三 worker 容器退出。repair revision
  `415f608` 只将 `GUIOwlV21GenerationParseError` 原子记录为 skipped state，其余异常仍 fail-fast；已有
  terminal state 与 coalition microbatch 全部复用；
- 用户授权每台 Hyper 最多使用 6 张 GPU 后，repair execution 改为 Hyper00 GPU 0/1/2/3/5/6 与 Hyper01
  GPU 2--7，每卡 2 workers，共 12xH200 / 24 workers。partition 继续保持原 host affinity，不改变 frozen
  schedule 或 state identity；
- 2026-07-20 03:28 UTC checkpoint：5,317 / 11,746 terminal states（45.3%），12/12 containers healthy；
  最近 6 分钟约 26.8 states/min，线性剩余约 4 小时，考虑 long-history slowdown 后 ETA 为 4--5.5 小时；
- worker expansion revision `7680e40` 为每个原 partition 增加 SHA256 state lanes；同一 state 只属于一个
  lane，state identity 与已有 terminal/microbatch 均不变化。36-worker immutable mapping 位于
  `code/configs/causalcache_set_utility_variable_history_labels_workers_36_v1.json`（commit `c7c6966`）；
- 2026-07-20 03:51 UTC 在 5,924 states 处从 24 workers 切换到 36 workers：每台 Hyper 6 张卡、每卡
  3 processes；23 个未完成 base partitions 中 13 个按两条 lane 拆分，已完成 partition 6 不再启动；
- 2026-07-20 03:54 UTC checkpoint：5,983 / 11,746 states，12/12 containers healthy，单卡显存约
  87--103GB。首个稳定分钟约 27 states/min，尚未显示相对 24-worker 的显著吞吐提升，说明当前更接近
  GPU forward saturation；保留 36 workers 继续运行，当前 ETA 约 3.5--5 小时；
- 36-worker burst 在 long-history tail 上证明 3 processes/H200 不安全：Hyper00 的 partitions
  `8:lane0`、`9:lane0`、`9:lane1`、`16` 分别因峰值显存超过约 139.8GB OOM；其他 Hyper00 lanes 已
  正常结束，Hyper01 6 个 containers 继续运行。所有 OOM 均发生在原子 microbatch 外层，已有结果保留；
- 2026-07-20 06:59 UTC checkpoint：10,619 / 11,746 states（90.4%）。四条未完成 lanes 随后各自绑定
  Hyper00 GPU 1--4 单进程恢复；2026-07-20 07:01 UTC 为 10,643 states，4/4 tail containers 与
  Hyper01 6/6 containers healthy。剩余 1,103 states，保守 ETA 为 1--1.5 小时；
- reusable artifact：完成后上传 `gavinlaw/causalcache-set-utility-variable-history-mobile`，当前 `PENDING_HF_UPLOAD`。

首次 `ee77f47` attempt 因 inherited official-tools encoder 的 batch 上限仍硬编码为 2，在 16 个
empty-coalition microbatches 后 fail-fast，0 terminal states；它不进入正式数据。修复只让 variable-history
subclass 读取自己的 max batch=16，historical runtime 的默认上限仍为 2。

首次 12-GPU remap 错误地跨 host 移动 partition，而两台机器只保有原 assignment 对应的 source/schedule
shards；同时 Hyper00 GPU 4 在 preflight 后被新进程占满。该 attempt 在缺 shard/OOM 时 fail-fast，未覆盖已有
terminal records；容器已删除。正式 repair remap 保持 partition 的原 host 归属，并以 GPU 6 替换 GPU 4。
