# OSWorld terminal-s60 transfer pilot v1

## 结论

本轮不能支持“Android/mobile 侧训练得到的 terminal-s60 LoRA 能零样本提高 OSWorld desktop
task performance”：

| Arm | Mean OSWorld score | Score > 0 | Policy HTTP 500 | Mean completed steps |
|---|---:|---:|---:|---:|
| frozen GUI-Owl | 0.13333 | 4/30 | 10/30 | 37.87 |
| terminal-s60 | 0.07037 | 3/30 | 1/30 | 20.13 |

成对差值 `terminal-s60 - frozen = -0.06296`，task-level paired bootstrap 95% CI 为
`[-0.22963, 0.10370]`；s60 获得 3 wins、4 losses、23 ties。因此本 pilot 的点估计为负且
CI 跨 0：不能声称 s60 优于 frozen，也不能据此证明其必然有害。

s60 明显减少了 endpoint HTTP 500，并更频繁地输出 terminal action；但它平均只执行 20.13
步，frozen 为 37.87 步。可执行性改善没有转化成更高 task score，行为更像 terminal repair
带来的提前停止偏置。

## 固定设置

- 任务：official OSWorld no-Google-Drive roster 中 evenly-spaced 30 tasks，覆盖 10 domains；
- 两臂使用完全相同的 task identities；
- memory：`recent`，at-most `B=4`；
- `max_steps=50`，每臂 6 个并行 environments、1 个 policy replica；
- frozen arm：GUI-Owl-1.5-8B-Instruct；
- adapted arm：terminal-s60，rank 16、alpha 32，checkpoint SHA256
  `cba455cbfede5e091c06d8714ef000d5582adb46c4a5343fa8c0ab8c4ab47b18`；
- 任务级 policy/runner failure 不从 denominator 删除，按 score 0 计入。

本结果只评估 policy LoRA 的 cross-platform transfer，不包含 learned CausalCache selector，也不是
361-task 完整 OSWorld benchmark。

## Source of Truth

- 轻量结果：[`summary.json`](summary.json)，SHA256
  `42aff280638a8e70eacc0855837310449a2ecb72cbc5c25bf3e1624173f608d6`；
- 执行 config：
  [`code/configs/causalcache_osworld_transfer_pilot_v1.json`](../../../code/configs/causalcache_osworld_transfer_pilot_v1.json)；
- episode runner Git commit：`78428f8aed6eff254bfae3ffbae1c6fe3eb52cbd`；
- reducer Git commit：`27db19a`；
- OSWorld revision：`b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`；
- raw diagnostic mirror：Hyper01
  `/data01/jaxan/osworld-runner/transfer-pilot-v1/raw`，715,756,714 bytes / 1,925 files；
  raw screenshots/checkpoints 未进入 Git，不是 reusable published dataset；
- 详细解释与限制：[`docs/osworld_transfer_pilot_v1.md`](../../../docs/osworld_transfer_pilot_v1.md)。

