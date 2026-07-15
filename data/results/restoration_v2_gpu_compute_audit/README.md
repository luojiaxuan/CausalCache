# Restoration v2 GPU compute audit

本目录保存 dependency 8 当前 canonical 的 policy-blind GPU compute primitive 证据。GPU-2 re-anchor run
在 Hyper00 的单张 NVIDIA H200 上从 clean detached
`main@47062741a950b7c6050a6223b91f4bbae65332e7` 执行；它只使用固定
synthetic logits，没有导入 policy module、加载 GUI-Owl、生成 policy output 或生成 restoration output。

## 结论

- formal outcome：`PASSED_RESTORATION_V2_GPU_COMPUTE_AUDIT`；
- independent validator outcome：
  `PASSED_INDEPENDENT_RESTORATION_V2_GPU_COMPUTE_AUDIT_VALIDATION`；
- batch-1 GPU 对独立 float64 CPU oracle 的最大绝对误差：`3.047375374265471e-08`；
- batch-2 对两次 batch-1 的最大绝对误差：`0.0`；
- BF16 logits 对 pre-normalized FP32 log-probabilities 的 distance 误差：`0.0`；
- zero-stride expanded reference 与 batch-1 reference 共享 storage，实际 compute batch 为 1；
- invalid candidate logits 只变成最终 `NaN` distance；primitive validation scalar host reads 为 0，
  full-tensor host transfers 为 0；
- planner 固定 `planning_scope=single_decision_state`、microbatch size 2、无自动 OOM fallback。

这只证明 GPU KL 与 deterministic batching primitives 可执行且符合冻结数值契约，不证明真实 GUI-Owl
processor/model runtime 已通过，也不关闭 dependency 8。`validation.json` 明确保持
`dependency_8_closed=false` 与 `screening_unlocked=false`。

## 身份与证据

- UTC：`2026-07-15T17:59:09.694330Z--2026-07-15T17:59:11.299909Z`；
- host：`hyper00` / `node-radixark-16-0000`；
- GPU：physical GPU 2，`NVIDIA H200`，UUID
  `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81`；container 内只可见 `cuda:0`；
- NVIDIA driver：`570.172.08`；PyTorch：`2.11.0+cu130`；CUDA build：`13.0`；
- container：`69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df`；
- image digest：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- [`summary.json`](summary.json) SHA256：
  `dce797694194cd88ac749dcc357c3259c41b69a5bde7a99fc93c675cab2e9dac`；
- [`validation.json`](validation.json) SHA256：
  `ad53e18691424bc173919bdd578f7aff5a412c7fafb201816d2ae8576cd4f2a6`。

run 前按 repository 规则执行 host/GPU/disk/container preflight 和两轮 10 秒 utilization sampling；没有
旧 GPU 0 已被其他任务占用，因此没有共享该卡或启动 screening；10 秒 preflight 选择空闲 GPU 2。host-side
`hostname` 与 `docker inspect` 分别复核 summary 中的 hostname、container full ID、image digest、
`privileged=false` 和只暴露 physical GPU 2 的 DeviceIDs。

independent validator 不 import audit runner、GPU KL 或 batching module；它从 run commit 的 Git blobs
重新计算三份 source size/SHA，独立复核 standard JSON、argv、UTC、runtime identity、全部 equivalence
records、NaN fail-closed、host-read counters、zero-stride storage 与 planner semantics。validator source SHA256
为 `237a1e598402035d0a7d0bc7410d9e9e0726ed8727290f620893b88d9aa5d9d1`。

本目录只有 13 KiB 左右的轻量审计 evidence，canonical source of truth 是 Git，不需要新建 Hugging Face
artifact。旧 GPU-0 canonical evidence 保留在 Git history；同一份 re-anchor 原始副本与迁移原因见
`../restoration_v2_runtime_reanchor/`。完整 argv 已保存在 `summary.json`；执行与回写契约见
[`docs/execution.md`](../../../docs/execution.md)。
