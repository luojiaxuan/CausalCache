# GUI-Owl AndroidWorld validation attempt 2（configuration-invalid）

## 结论

本次运行禁止用于接受或拒绝 GUI-Owl policy。45 个 episode 后 success gate 已出现数学失败上界，
但随后的 pinned-source 审计发现本次强制使用 256 visual tokens/图，而上游 native preprocessing
实际产生 2,550 effective visual tokens/图。视觉输入只保留约 1/10，因此本次不是有效的 native
policy reproduction。

## Early-stop 观测

- 冻结 plan 与 threshold 未改变；
- 45/62 checkpoint，12 official success；
- 38 个 completed episode、7 个 infrastructure exceptions；
- 485 model steps、484 parsed actions；
- termination：26 policy terminated、11 step-budget exhausted、1 parse error、7 exception；
- 即使剩余 17 个全部成功，上界也只有 29/62，低于 31/62；
- 这些数字仅用于发现配置问题，不进入 policy-selection 表或论文结果。

## Root cause

pinned adapter 的 `coordinate_resize` 先把 1080×2400 screenshot resize 到 1092×2408；pinned
GUI-Owl processor 对原图和 resize 图都产生 `[1,150,68]` grid。模型 `merge_size=2`，所以每图
有效视觉 token 为 `150×68/4=2550`。本次命令的 `--visual-tokens-per-image 256` 明显偏离上游。

## 处理

runtime 改为显式选择 model-default 或 fixed-token 模式，并记录实际 grid 与 visual token count。
下一步先做 model-default 1/5-image memory smoke；只有 smoke 通过后才从空 checkpoint 目录重启
同一 validation plan。prompt、action equivalence、threshold、task split 和模型权重不变。
