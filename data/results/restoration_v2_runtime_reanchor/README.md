# Restoration v2 Hyper00 runtime re-anchor

2026-07-15 的正式 screening preflight 发现原 execution config 锁定的 physical GPU 0 被其他任务占用
112+ GiB 显存并持续 86%--87% utilization；10 秒 preflight 选择了空闲 physical GPU 2。为避免共享繁忙 GPU，
screening 未启动，先在任何 policy output 前完成 runtime re-anchor。

## 新 runtime identity

- host：Hyper00 / `node-radixark-16-0000`；
- physical GPU：2；container-visible device：`cuda:0`；visible GPU count：1；
- GPU：NVIDIA H200，UUID `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81`；
- container：`sglang-omni-jaxan-07160158`，ID
  `69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df`；
- image digest：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- container isolation：`privileged=false`、Docker device request `device=2`、PyTorch 只见一张 GPU。

首次新容器错误使用 `--privileged`，检查时发现它会绕过单卡隔离并使 PyTorch 看见 8 张 GPU；该空容器在
任何审计或 policy load 前被删除。完成态容器去掉 `--privileged`、保留 `SYS_PTRACE`，单卡隔离检查通过。

## Policy-output-free evidence

- `gpu-summary.json`：formal CUDA compute audit passed，SHA256
  `dce797694194cd88ac749dcc357c3259c41b69a5bde7a99fc93c675cab2e9dac`；
- `gpu-validation.json`：independent validator passed，SHA256
  `ad53e18691424bc173919bdd578f7aff5a412c7fafb201816d2ae8576cd4f2a6`；
- `processor-summary.json`：real pinned `AutoProcessor` audit passed，SHA256
  `69bb8ddb578bcb8019fda05fd3a4a2be600043b9b2d58db052078f5107bbd119`；
- run commit：`47062741a950b7c6050a6223b91f4bbae65332e7`，clean detached worktree；
- GPU audit UTC：`17:59:09.694330Z--17:59:11.299909Z`；processor audit UTC：
  `17:59:55.707871Z--18:00:22.778372Z`。

GPU numerical equivalence、NaN/no-host-read、microbatch=2/no-fallback 与 processor exact grids/tokens/tensors 均
复现通过。三份 evidence 均声明 `policy_output_generated=false`、
`restoration_output_generated=false`；processor audit 还声明 model weights 未 materialize 为 tensors，未调用
forward/generate。

## 状态边界

这些文件是 re-signing 输入，不会自动改写已冻结 config/readiness。旧 GPU-0 authorization 保留为历史证据；
当前 planned GPU-2 screening 保持锁定，直到 canonical evidence、execution config 与 readiness manifest 在
后续 clean pushed commits 中重新绑定并通过正式 authorization。
