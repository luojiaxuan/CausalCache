# GUI-Odyssey 官方切分交叉审计 v1

状态:`COMPLETED_GUIODYSSEY_OFFICIAL_SPLIT_AUDIT`。

把冻结的 1,200 条 trajectory(train/tune/evaluation = 1,000/100/100,assignment manifest SHA256=
`915892ef...`)与官方 `OpenGVLab/GUI-Odyssey` 的四个切分文件
(`splits/{random,task,device,app}_split.json`)逐条比对。完整数字见 [`audit.json`](audit.json),
生成命令:

```bash
PYTHONPATH=code python3 code/scripts/audit_guiodyssey_official_splits.py \
  --assignment-manifest data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json \
  --output data/results/guiodyssey_official_split_audit_v1/audit.json
```

## 关键事实

- 本项目切分与官方任何一种切分都**不对齐**:以 random_split 为例,我们的 train 含 258 条官方 test
  episode,我们的 evaluation 有 77/100 落在官方 train;task/device/app split 同模式。
- 1,200 条中 63 条(train 49 / tune 10 / evaluation 4)在官方当前 revision 的任何切分名单中
  **均不存在**,疑似 `cua-lite/GUIOdyssey` 镜像与官方后续更新之间的版本差;ID 清单在
  `audit.json` 的 `missing_trajectory_ids`。
- 官方四切分内部 train/test 无重叠(脚本校验)。

## Paper 立场(预注册的辩护口径)

1. 本项目**不做** GUI-Odyssey benchmark 任务:标签(restoration distance)、指标(restoration
   recovery)、估计量(冻结 GUI-Owl 的 self-behavior 恢复)均为本项目新定义,不与官方 leaderboard
   比较,故不受官方切分约束。
2. 本项目切分在对 memory-selection 重要的泄漏轴上**严于**官方 random split:按
   `instruction_app_group` 去重 + 零重叠程序化校验 + decision-count 分层;官方 random split 允许
   同任务同 App 组合的近重复 episode 跨 train/test 出现。
3. **Limitations 必须声明**:evaluation 角色的 77--88% episode 落在官方 train 名单内,GUI-Owl 训练
   语料无法排除包含官方 train;本项目只保证 *selector* 的训练/评测隔离,不声称对 policy 预训练
   语料去污染(与 `docs/go_no_go.md` 中既有措辞一致)。
4. 可选加固:evaluation 角色中有 19 条(random-split test)/ 8 条(task-split test)属于官方 test;
   untouched evaluation 解锁后可附"限官方 test 子集"的方向性 robustness 切片。
