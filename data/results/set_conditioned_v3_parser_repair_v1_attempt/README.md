# parser-repair v1 pre-label attempt

v1 repair 在 Hyper00 通过 exact producer protocol check 后，因错误要求 historical record 与 feature JSONL
逐位置同序而 fail closed。两边均为 48 个唯一 state，state-id set 完全相同；historical artifact 按 state-id
字典序排列，feature JSONL 使用 trajectory roster 顺序，因此 45/48 个位置不同。

该 dry-run 在读取或解析 repair label 之前停止，所以 fresh-label semantic decode 总数仍是父 attempt 的 1 次。
没有 metric/report，也没有 checkpoint、prediction、selector 或阈值变化。

后续 v2 只允许用唯一 `state_id` 做 exact join，并继续要求两边 set 完全相等；其余 schema、seed feasibility、
selection digest 与 sealed bytes 检查不得放宽。
