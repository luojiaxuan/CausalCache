# Set Utility AndroidWorld v1

## 当前状态

本版本只是 **post-GO validation runner skeleton**，不是已获准启动的 closed-loop
结果。它把 `winner`、`recent`、`ocr_rgb` 三种 memory arm 接入同一 GUI-Owl
mixed-fidelity policy loop，并提供 Mac/Taurus/Aries/Hyper 的 loopback relay 规划与
health/RTT canary。

真实 launch 仍被以下 gate 阻塞：

- held-out selector 与 native policy replay 必须先产生完整、签名且 SHA 绑定的 GO；
- AndroidWorld relay 和 winner selector transport 必须通过 Hyper 侧 loopback health/RTT
  canary，不能假定 Hyper 可直接连接 Aries；
- online OCR 必须具备 frozen config、manifest 和对应 model artifacts；缺失或 drift 时
  fail closed，不允许伪造 OCR token 或静默 fallback；
- policy runtime 必须保持 H200 的每图 480 effective visual-token profile。

## 最小 validation scope

首次真实验证只允许使用 outcome-exposed `validation12`，最小范围为：

- `winner_B2`、`recent_B2`、`ocr_rgb_B2`；
- 12 个冻结 task identities，共 36 个独立 episode；
- decision 1--5 使用相同 summary-only policy；其后使用完整 prior candidate history；
- 三臂保持相同 high-fidelity budget 和 policy prompt，winner 只能调用 GO-authorized
  predictor，两个 baseline 只能本地确定性选择；
- 先运行 topology canary，再按 source/episode 串行执行。

禁止打开或运行 sealed AndroidWorld test split。validation12 只能用于
outcome-exposed engineering/validation evidence，不能表述为最终 test result。

## Episode durability

每个 episode 使用独立 output path。terminal JSON 通过同目录临时文件 fsync 后原子、
独占发布，不覆盖已有文件。已有 terminal 默认在启动前 fail-fast；只有显式传入
`--resume-completed` 且 arm、budget、episode、task 和 terminal status 全部一致时，才
复用已完成结果且不再次访问远程服务。

