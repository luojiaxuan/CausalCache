# FindingDory Object Attributes exact-B oracle gap v1

## 目的

验证 FindingDory 是否满足扩展为 multimodal-agent memory 场景的三个前置条件：公开 benchmark、
早期视觉证据在文本压缩后丢失、同一冻结 policy 与同一 exact-$B$ 预算下存在可消费的 oracle gap。

## 冻结协议

- 数据：公开 `yali30/findingdory-subsampled-96` validation，先跑前 20 个 episode pilot；通过后扩到
  100 episode 的全部 `Object Attributes`（491 tasks）。
- 模型：`Qwen/Qwen2.5-VL-7B-Instruct@cc594898...`，summary 与 policy 都冻结、greedy decode。
- 压缩：每 8 帧生成一条 task-independent content summary；summary prompt 不接收 question、
  task id 或 answer。公开视频像素内含原长轨迹的 legacy frame/time overlay，与 0--95
  subsampled answer namespace 不同；VLM 输出中的所有数字和 frame/time 字段确定性移除，只有代码
  外挂的 canonical chunk range 可见于 policy。所有 summary/restored/current 图在视觉编码前还统一
  遮掉官方代码写入 `(10,50)/(10,70)` 的左上角 legacy `Frame/Time` 区域；两臂处理相同。
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

## 结果

最终 masked run 为 `GO`。100 episodes / 491 tasks / 1,964 paired policy rows：

| B | Recent HL-SR | Oracle HL-SR | Oracle − Recent | episode-bootstrap 95% CI | gate |
|---:|---:|---:|---:|---:|:---:|
| 1 | 7.54% | 82.89% | +75.36 pp | [+70.47,+80.00] pp | PASS |
| 4 | 5.50% | 44.20% | +38.70 pp | [+33.81,+43.64] pp | PASS |

B1/B4 的 exact McNemar p 分别为 `7.75e-110` / `6.15e-56`，selection-difference coverage
为 100% / 97.96%。运行 commit `fdf4a4a573bc4f10da0ca25d2b0c8f4b8492cb51`；完整结果见
[`data/results/findingdory_object_attribute_oracle_gap_v1/`](../data/results/findingdory_object_attribute_oracle_gap_v1/README.md)。

## 有效性审计

100 summary files、1,200 chunks、1,964 unique result keys 全量通过：numeric/frame/time
namespace、canonical chunk ranges、exact-$B$、selection rule、Oracle GT-valid evidence、answer
groups 与 success 重算均为 0 violation。最终 run 的真正 out-of-domain prediction 为 0；B4
的 `-1` 是 policy abstention（Recent 154 / Oracle 105），不是 legacy frame id。B4 因多图输入
显著弱于 B1，因此本结果不支持 utility 随 B 单调增长。

公开 metadata 也满足场景筛选：58.66% tasks 至少一个 valid frame 位于前半段，29.12% 的全部
valid frames 都在前半段；Recent-B 完全没有 valid frame 的比例为 B1=99.39%、B4=89.41%。
`ep_1` 的 frame 72 可见紫色圆柱纸卷，而 recent frame 94 与 current 95 只见墙面；purple/paper
两题均为 Recent-B1 失败、Oracle-B1 成功。

这个 Oracle 使用公开 PDDL-valid answer groups，是 privileged offline upper bound。`GO` 只说明
FindingDory 存在同一冻结 policy 可消费的 early-image headroom；部署方法仍须训练不看 answer 的
budget-conditioned selector，并与 Recent/random/text-only/GT Oracle 比较。

## Invalidated run

1. 首个 20-episode pilot 曾把像素中的 legacy frame number 写进压缩摘要，而 policy 输出域与
   answer 使用 subsampled 0--95。它标记为 `INVALID_FRAME_ID_NAMESPACE`，不得引用。
2. content-summary namespace 修复后的 100-episode run 仍把带 legacy overlay 的恢复图直接送入
   policy，导致 B4 的 out-of-domain prediction 明显增加。该 run 标记为
   `INVALID_PIXEL_OVERLAY_NAMESPACE`，不得引用。最终版在所有视觉输入上统一 mask overlay，并从
   summary 重新生成。
