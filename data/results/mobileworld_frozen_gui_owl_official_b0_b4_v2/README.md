# MobileWorld official-faithful frozen GUI-Owl B0/B4 v2

> **状态：`COMPLETE_VALID_PAIRED_NO_SIGNIFICANT_DIFFERENCE`。**
>
> 在冻结的 117-task GUI-only roster 上，Recent-B4 比 B0 多成功 2 个任务，
> 但配对区间跨 0，不能据此声称 B4 优于 B0，也不能声称 B0 优于 B4。

## 主结果

两臂使用相同 GUI-Owl snapshot、MobileWorld revision、117-task roster、50-step
上限、retry、visual-token 配置、确定性 decoding、官方多轮 prompt 与宽松
UNKNOWN/wait parser；唯一科学变量是历史图预算：

| arm | observed | strict missing-as-zero | missing |
|---|---:|---:|---:|
| B0（当前图，无历史图） | 33/115 = 28.70% | **33/117 = 28.21%** | 2 |
| Recent-B4（当前图 + 至多 4 张 recent） | 35/115 = 30.43% | **35/117 = 29.91%** | 2 |

strict-117 配对结果：

- `B4 − B0 = +1.71 pp`；
- 20,000 次 task-level paired bootstrap 95% CI：
  `[-5.98, +9.40] pp`；
- 9 个 B0-only、11 个 B4-only、97 个 ties；
- exact two-sided McNemar `p=0.823803`。

只比较两臂都有 evaluator 结果的 115 个任务时，`B4 − B0 = +1.74 pp`，
95% CI `[-6.09, +9.57] pp`，9/11 discordant 与 McNemar p 值不变。

因此本轮证据支持：**在冻结 GUI-Owl 上，把最近历史图预算从 0 增至 4，
平均成功率没有可辨别的稳定变化。** 这不等于历史图从不改变动作；
20 个任务出现了 arm-specific success，只是净差很小且不确定。

## Memory split

| split | B0 | B4 | B4 − B0 | B0-only / B4-only |
|---|---:|---:|---:|---:|
| cross-app memory candidate（62） | 14 | 14 | 0.00 pp | 3 / 3 |
| single-app control（55） | 19 | 21 | +3.64 pp | 6 / 8 |

cross-app memory candidate 上没有出现 B4 的净提升。该结果只评估
“固定 recent-B4 vs 无历史图”，不评估 learned selector、target-specific
restoration 或“不总用满 budget、不总存 recent”的方法主张。

## 官方协议与 B0 语义

- prompt：`mobile_agent_v3_5_gui_owl_official_faithful`；
- `chat_template_tools_kwarg=false`，没有额外把 `tools=` 注入
  `apply_chat_template`；
- history 使用 pre-action screenshot、模型原始完整 response 与
  `Action:` 文本，按 user/assistant 多轮交替；
- B0：`memory_budget=0`、`last_image=1`，5,461 个 primary policy
  requests 中实际 `maximum_history_images=0`；
- B4：`memory_budget=4`、`last_image=5`，3,421 个 primary policy
  requests 中实际 `maximum_history_images=4`；
- 两臂 policy request failures 均为 0。B0/B4 分别有 13/2 次模型输出
  解析失败，均按预先冻结的 `unknown_wait_step` 规则消耗一步并继续，
  没有升级为 task attempt failure。

MobileWorld 官方 leaderboard 对同一
`GUI-Owl-1.5-8B-Instruct` 报告 GUI-only `38.2%`、`runs=3`；
上游 agent 默认 `history_n=1`（包含当前图，因此为 0 张历史图）。
我们的单次 strict B0 低约 10.0 pp。该 absolute gap 仍是 open discrepancy，
不能用官方多次运行值替换本轮同 harness 的 paired baseline。

## 两个共同 evaluator failure

两臂都缺少：

- `MattermostReadingGroupTask`；
- `MattermostShiftCoverageTask`。

两任务均完成全部三次 attempt 后在 `get_task_score` 失败。环境日志显示
Mattermost 初始化在服务 ready 前执行 login/create-channel，初始化接口仍返回
HTTP 200；evaluator 随后读取不到 `channel_info`。B0 另用不同 fleet member
各做三次独立 repair，仍复现同一失败。因此不修改 pinned upstream、不改变
117 分母，两臂都按 strict missing-as-zero 计 0。共同缺失不会制造
B4/B0 的 discordant pair。

## 时间与并发

Aries 使用 2 × RTX A6000；每张 GPU 一个冻结 policy service，各并发驱动
8 个 emulator：

| arm | 双卡 makespan |
|---|---:|
| B0 | 8,839.692 秒（2:27:19.692） |
| B4 | 11,915.087 秒（3:18:35.087） |

B4 比 B0 多 3,075.396 秒（51:15.396，约 34.8%）。任务动作长度和最后两个
Mattermost failure 的耗时也不同，因此该 wall-time 差是部署容量实测，不单独
解释为历史图 token 数的纯因果开销。B0 正式开始至 B4 正式结束的顺序 campaign
总历时 22,532.756 秒（6:15:32.756，含两臂间归约/切换）。

## 冻结身份与 provenance

- scientific Git：
  `71caac97ea0204946462295da0255e46e276c513`；
- MobileWorld：
  `8ae506487bf87785292d6cad101c49955d704d39`；
- environment image：
  `ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240`；
- GUI-Owl：
  `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；
- runtime：PyTorch `2.11.0+cu130`、Transformers `5.6.0`、
  bfloat16、2,560 visual tokens/image；
- B0/B4 config SHA256：
  `67566f95...1d8d78` / `81f0e815...5b1196`；
- frozen shard SHA256：
  `4f0c63c1...6fcb31` / `8167ff61...fcb31`；
- full roster SHA256：
  `d11e0d93...f97fce`；
- reducer：seed `20260726`、20,000 bootstrap iterations。

完整 runner/reducer argv、GPU UUID、container/image identity、config/manifest
hash、起止时间、failure classification 与 cleanup 记录见
[`provenance.json`](provenance.json)。

## Artifacts 与 cleanup

- machine-readable paired result：
  [`summary.json`](summary.json)，SHA256
  `0aa205785ec32b9fab97dc9993fb5f79071d0a0827da7ac10c7814309a489227`；
- provenance：
  [`provenance.json`](provenance.json)，SHA256
  `39f55c228a107e82b8518415c6d5687537b11dea88e1890c2c131ca06407ea28`；
- raw trajectories：Aries
  `/mnt/data6/jiaxuanluo/runs/mw-official-sequential-v2-71caac9/`，
  状态 `LOCAL_PRIVATE_RAW_TRACE`，不进入 Git；
- 旧 concurrent partial run：
  `/mnt/data6/jiaxuanluo/runs/mw-official-v2-71caac9/`，已标记
  `ABORTED_BY_USER_SEQUENTIAL_REALLOCATION`，不计入本结果。

正式结束后，按两份 fleet manifest 精确删除了 16 个临时 emulator
containers，停止两路 policy、runner 与 nested dockerd；canonical
`sglang-omni-jaxan` 保留。Aries GPU 0/1 均回到 5 MiB、0% utilization。
临时容器 writable layer 不可恢复，Git metadata 与 persistent run outputs 保留。
