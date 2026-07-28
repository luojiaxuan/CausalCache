# MobileWorld official-faithful frozen GUI-Owl B0–B4 budget curve

> **状态：`COMPLETE`。** 五个 history-image budget 均按 frozen strict-117
> denominator 完成；B1/B2/B3 在 Aries GPU0+1 上严格顺序运行。

## 结果

共同合同：frozen GUI-Owl、MobileWorld `8ae5064`、117-task roster、`59+58`
确定性 shards、`max_step=50`、2,560 visual tokens/image、官方多轮 prompt/parser
和确定性 decoding。唯一科学变量是历史图预算。

| budget | success | Wilson 95% CI | 相对 B0 | paired 95% CI | McNemar p |
|---:|---:|---:|---:|---:|---:|
| B0 | 33/117 = 28.21% | [20.84, 36.95]% | — | — | — |
| B1 | 31/117 = 26.50% | [19.34, 35.15]% | −1.71 pp | [−7.69, +4.27] pp | 0.7744 |
| B2 | 31/117 = 26.50% | [19.34, 35.15]% | −1.71 pp | [−7.69, +4.27] pp | 0.7905 |
| B3 | 34/117 = 29.06% | [21.60, 37.85]% | +0.85 pp | [−5.13, +6.84] pp | 1.0000 |
| B4 | 35/117 = 29.91% | [22.36, 38.74]% | +1.71 pp | [−5.98, +9.40] pp | 0.8238 |

**简短结论：Recent-B 的曲线不是单调剂量效应，所有相对 B0 的 paired CI
都跨 0。** B4 点估计最高，但现有 117 个任务不能支持“用满更多 recent history
images 会稳定提高成功率”。这正好说明论文应把重点放在**固定预算内选择哪些图、
何时少用槽位**，而不是宣称更大 history budget 本身有效。

B1/B2/B3 的 policy failures 均为 0，实际 maximum history images 为 1/2/3；
冻结 parser 的 parse→wait 次数为 46/1/22。B2 唯一
`MattermostIncidentEscalationTask` emulator initialization failure 使用相同科学
配置在健康环境定点修复一次，最终三臂都是 117/117 observed。

## 运行与清理

- execution Git：`7ff6cad806ab9c36bd2fc9b8b9d8fff5fdf1ed57`；
- GUI-Owl：`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faec…d04fc`；
- B1/B2/B3 双卡 makespan：约 `3:05:51 / 3:48:56 (+7:46 repair) / 4:26:57`；
- Aries run root：
  `/data/runs/mw-official-b123-sequential-v2-7ff6cad`；
- 16 个 manifest 指定 emulator containers、B3 两路 policy 和 nested dockerd
  已停止；canonical `sglang-omni-jaxan` 保留，GPU0/1 均回到 5 MiB、0%。

raw trajectories 保留为 Aries persistent storage 上的
`LOCAL_PRIVATE_RAW_TRACE`，不进入 Git，也没有生成需要上传 Hugging Face 的
reusable dataset/model。

## Artifacts

- [`summary.json`](summary.json)：B0–B4 per-arm CI、相对 B0/相邻预算 paired
  delta、bootstrap CI 和 McNemar；
- [`task-level-matrix.csv`](task-level-matrix.csv)：117-task 配对矩阵；
- `b{1,2,3}-strict-summary.json` 与 `b{1,2,3}-task-scores.json`：逐臂严格归约；
- [`provenance.json`](provenance.json)：revision、完整命令、hash、wall time、
  repair 与 cleanup 记录；
- [`B0/B4 v2`](../mobileworld_frozen_gui_owl_official_b0_b4_v2/README.md)：
  先前两端点的执行记录。
