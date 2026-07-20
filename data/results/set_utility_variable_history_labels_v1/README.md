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
- reusable artifact：完成后上传 `gavinlaw/causalcache-set-utility-variable-history-mobile`，当前 `PENDING_HF_UPLOAD`。

首次 `ee77f47` attempt 因 inherited official-tools encoder 的 batch 上限仍硬编码为 2，在 16 个
empty-coalition microbatches 后 fail-fast，0 terminal states；它不进入正式数据。修复只让 variable-history
subclass 读取自己的 max batch=16，historical runtime 的默认上限仍为 2。
