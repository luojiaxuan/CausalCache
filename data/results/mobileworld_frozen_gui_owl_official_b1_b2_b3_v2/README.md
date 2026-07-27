# MobileWorld official-faithful frozen GUI-Owl B1/B2/B3 v2

> **状态：`B1_B2_COMPLETE_B3_RUNNING`。**
>
> B1、B2 已在冻结 strict-117 denominator 上完成；B3 仍在 Aries
> `sglang-omni-jaxan` 内运行。本文档是 B3 完成前的中间 source-of-truth，
> 不能替代最终 B0–B4 paired budget-curve 归约。

## 当前结果

三臂继续使用与
[`B0/B4 official-faithful v2`](../mobileworld_frozen_gui_owl_official_b0_b4_v2/README.md)
相同的 frozen GUI-Owl、MobileWorld revision、117-task roster、59/58
确定性 shards、50-step/retry、2,560 visual tokens/image、官方多轮 prompt/parser
与确定性 decoding；唯一科学变量是历史图预算：

| arm | memory budget | strict result | policy failures | parse→wait | max history observed |
|---|---:|---:|---:|---:|---:|
| Recent-B1 | 1 | **31/117 = 26.50%** | 0 | 46 | 1 |
| Recent-B2 | 2 | **31/117 = 26.50%** | 0 | 1 | 2 |
| Recent-B3 | 3 | **RUNNING** | 0（快照） | 0（快照） | 3 |

`parse→wait` 是 frozen policy 生成 malformed action 后由预先冻结的
`unknown_wait_step` 路径映射为 `wait`；没有据此重跑或修改 task 结果。

B2 两个 primary shards 最初得到 116/117 个 result files；唯一缺失
`MattermostIncidentEscalationTask` 是 emulator initialization failure。该任务随后在
相同 B2 config、官方 prompt、冻结 policy 与健康 fleet member 上做一次定点
infrastructure repair，得到 `score=1.0`，因此 B2 最终 union 为 117/117。
失败的首次 repair launch 因缺少 `PYTHONPATH` 在 agent import 前退出，保留于
run root 作为 infrastructure provenance，不计入科学结果。

## B3 运行状态

截至 `2026-07-27T17:54:03.895985+00:00`：

- shard0：`19/59` 个 result files；
- shard1：`18/58` 个 result files；
- 两路 policy requests=`986/964`，policy failures=`0/0`，
  parse failures=`0/0`；
- `maximum_history_images=3`、`chat_template_tools_kwarg=false`、
  prompt protocol=`mobile_agent_v3_5_gui_owl_official_faithful`。

以上 `37/117` 只是运行时已落盘的结果文件数量，不是 B3 success 数或最终分数。
B3 runner/policy 仍在 Aries GPU0+GPU1 上运行；高频 heartbeat scheduler 已按用户要求
删除，不影响远端既有进程。

## 冻结身份与位置

- execution Git：
  `7ff6cad806ab9c36bd2fc9b8b9d8fff5fdf1ed57`；
- branch：`luojiaxuan/mobileworld-memory-osworld2`；
- MobileWorld：
  `8ae506487bf87785292d6cad101c49955d704d39`；
- GUI-Owl：
  `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；
- B1/B2/B3 config SHA256：
  `b8be5d94…759bab` / `c12eb2a0…bd6206` / `90121b42…752ad`；
- frozen shard SHA256：
  `4f0c63c1…6fcb31` / `8167ff61…9d63dd`；
- Aries canonical compute container：
  `sglang-omni-jaxan`；
- container run root：
  `/data/runs/mw-official-b123-sequential-v2-7ff6cad`；
- Aries persistent host root：
  `/mnt/data6/jiaxuanluo/runs/mw-official-b123-sequential-v2-7ff6cad`。

原始 trajectories 是 `LOCAL_PRIVATE_RAW_TRACE`，不进入 Git。B1/B2 的 exact
task-score 文件与 execution records 仍在上述 persistent run root；最终 B3 完成后，
统一生成 B0–B4 per-arm CI、相对 B0/相邻预算的 paired delta/CI/McNemar 与
task-level paired matrix，再将最终轻量结果回写 Git。

## Git artifacts

- 当前机器可读快照：
  [`partial-summary.json`](partial-summary.json)；
- execution/runtime provenance：
  [`provenance.json`](provenance.json)。

