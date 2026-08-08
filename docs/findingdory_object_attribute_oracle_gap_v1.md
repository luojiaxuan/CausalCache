# FindingDory Object Attributes exact-B oracle gap v1

## 目的

验证 FindingDory 是否满足扩展为 multimodal-agent memory 场景的三个前置条件：公开 benchmark、
早期视觉证据在文本压缩后丢失、同一冻结 policy 与同一 exact-$B$ 预算下存在可消费的 oracle gap。

## 冻结协议

- 数据：公开 `yali30/findingdory-subsampled-96` validation，先跑前 20 个 episode pilot；通过后扩到
  100 episode 的全部 `Object Attributes`（491 tasks）。
- 模型：`Qwen/Qwen2.5-VL-7B-Instruct@cc594898...`，summary 与 policy 都冻结、greedy decode。
- 压缩：每 8 帧生成一条 task-independent JSON event summary；summary prompt 不接收 question、
  task id 或 answer。
- policy 输入：完整压缩摘要、task、exact-$B$ 张恢复历史图、固定 current frame 95。current 不计入
  $B$；$B\in\{1,4\}$。
- Recent-$B$：current 之前最后 $B$ 帧。Oracle-$B$：PDDL-valid 历史区间最长连续段的中心帧，剩余
  槽位只用最近且不重复的历史帧补满。两臂均严格为 $B$ 张图。
- policy 可输出 0--95 中任意原始 frame id，并不被恢复图集合限制；成功当且仅当输出落入公开
  PDDL-valid answer frames。因此 Recent 仍可依赖文本摘要成功，不会被机械置零。
- 统计：task-level paired difference，按 episode 做 10,000 次 cluster bootstrap，并报告 exact
  McNemar $p$。

## Go gate

至少一个 $B$ 同时满足：Oracle HL-SR $\ge 0.30$、Oracle$-$Recent $\ge 0.10$、episode-clustered
95% CI 下界 $>0$、两臂选择集合不同的 task coverage $\ge 0.50$。Pilot 通过只触发 full run；论文
层面的 `GO` 只由 491-task full run 判定。

完整 scientific config：
[`code/configs/findingdory_object_attribute_oracle_gap_v1.json`](../code/configs/findingdory_object_attribute_oracle_gap_v1.json)。
