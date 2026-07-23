# AndroidWorld task partition

## 结论

116 个 pinned AndroidWorld task templates 已在任何 validation rollout 之前完成冻结。机器可读
manifest 位于 `code/configs/androidworld_task_partition.json`，其 sorted task-name registry hash 为：

```text
185ae2019706693bd32ecc25ffd0c8f87be87331cae6f7d7e31c91674c962b89
```

## Assignment

每个 task type 使用以下唯一规则分桶：

```text
int(sha256(task_type).hexdigest(), 16) % 10
```

| Split | Buckets | Templates | Combinations/template | Instances | Seed |
| --- | --- | ---: | ---: | ---: | ---: |
| train | 0--5 | 60 | 3 | 180 | 314159 |
| validation | 6--7 | 31 | 2 | 62 | 271828 |
| test | 8--9 | 25 | 3 | 75 | 161803 |

hash partition 的数量不完全均匀；项目不做事后 rebalance，也不会因 app setup、policy success 或
task length 移动 templates。Final test 在 validation gate 通过前保持 sealed。

## Validation execution plan

`code/configs/androidworld_validation_plan.json` 使用 pinned suite seed 实例化 validation-only 参数，
冻结每个实例的 goal、template、complexity、`start_on_home_screen` 与官方
`int(10 * complexity)` step budget。62 个 instance records 的 SHA256 为：

```text
8204e7f832f1d70becd51f299977f0e4a322a080bbfab64010345c48f901b0e8
```

所有 62 个实例都要求 episode 前 reset 到 home。step budget 分布为 10--60，不能为了降低推理
成本统一截断；单任务链路 smoke 通过后，完整 validation runner 必须逐实例 checkpoint，允许安全
续跑但不改变实例顺序或预算。

完整 runner 使用多个相互隔离的 emulator worker 并行推进 episode，但只加载一个冻结 policy
runtime；所有 generation 通过进程内 lock 串行执行，避免在同一张 GPU 上复制模型或并发改变
推理语义。任务按 plan index 对 worker 做确定性 round-robin 分配。每个 episode 完成或抛出异常后，
都会先原子写入 `episodes/<plan-index>-<task-type>-<task-index>.json`，再继续下一项；`--resume`
只复用 instance record 与冻结 plan 完全相同的 checkpoint。

最终 `summary.json` 以全部 62 个实例为 official-success denominator。setup/infrastructure、parse、
executor 与 terminal failure 分开计数；parse coverage 只衡量实际生成的 action outputs，不通过删除
失败 episode 来提高 success rate。

## Validation templates

validation gate 固定使用以下 31 个 templates、每个 2 个动态实例：

```text
ClockStopWatchPausedVerify
ExpenseAddMultiple
ExpenseAddMultipleFromGallery
ExpenseDeleteMultiple
FilesDeleteFile
MarkorAddNoteHeader
MarkorChangeNoteContent
MarkorCreateNoteAndSms
MarkorDeleteNote
OsmAndFavorite
RecipeDeleteDuplicateRecipes
RecipeDeleteDuplicateRecipes3
RecipeDeleteMultipleRecipesWithConstraint
RecipeDeleteSingleWithRecipeWithNoise
RetroPlayingQueue
RetroSavePlaylist
SimpleCalendarAddOneEvent
SimpleCalendarDeleteEvents
SimpleCalendarEventsInTimeRange
SimpleCalendarNextEvent
SimpleCalendarNextMeetingWithPerson
SimpleSmsResend
SimpleSmsSend
SimpleSmsSendClipboardContent
SimpleSmsSendReceivedAddress
SystemBluetoothTurnOffVerify
SystemBrightnessMax
SystemBrightnessMaxVerify
SystemWifiTurnOn
TurnOnWifiAndOpenApp
VlcCreatePlaylist
```

Contacts 首次 setup warning 不直接影响 validation templates；VLC 的 x86_64 fallback 已安装成功，
但 `VlcCreatePlaylist` 仍必须在 rollout 中接受独立 task-initialize 检查。setup failure、parse
failure、executor failure 与 terminal failure 必须分开计数，不能从 success denominator 删除。

MobileAgent fork 的 HTTP server 原始代码错误导入旧 `android_world.env.json_action`，无法接收
GUI-Owl 官方 converter 产生的四坐标 swipe。构建 Docker context 时必须先运行
`scripts.prepare_androidworld_server`，严格替换为同一 pinned fork 的
`android_world.agents.new_json_action`；这只闭合 transport schema，不改变 agent action。

## Reproduction

在 pinned AndroidWorld server ready 后运行：

```bash
python3 -m scripts.build_androidworld_task_partition \
  --base-url http://127.0.0.1:5000 \
  --stack-config code/configs/androidworld_stack.json \
  --expected-task-types 116 \
  --output code/configs/androidworld_task_partition.json
```

单元测试会从 committed task names 与 stack config 完整重建 manifest，验证 hash assignment、
全覆盖、无重复和无 bucket overlap；validation plan 另行验证 instance hash、每 template 两个实例
和官方 complexity budget。

构建 pinned server image 前还需执行：

```bash
python3 -m scripts.prepare_androidworld_server \
  --source server/android_server.py
```

完整 validation gate 使用下面的入口；每个 `--base-url` 必须对应一个独立、health-ready 且来自同一
pinned image digest 的 AndroidWorld server：

```bash
python3 -m scripts.run_gui_owl_androidworld_validation \
  --base-url http://172.17.0.1:5000 \
  --base-url http://172.17.0.1:5001 \
  --base-url http://172.17.0.1:5002 \
  --base-url http://172.17.0.1:5003 \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Think \
  --validation-plan /data/causalcache-think-validation/code/configs/androidworld_validation_plan.json \
  --device cuda:0 \
  --use-model-default-visual-resolution \
  --maximum-visible-images 5 \
  --max-new-tokens 256 \
  --minimum-parse-coverage 0.95 \
  --minimum-official-success 0.5 \
  --early-stop-when-success-is-mathematically-impossible \
  --server-image causalcache-androidworld:11cea575-executor1 \
  --server-image-sha256 542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486 \
  --run-git-commit <FULL_GIT_COMMIT> \
  --output-dir /data/experiments/gui_owl_1_5_8b_think_androidworld_validation \
  --resume
```

端口 `5000–5003` 来自已记录的正式 validation summary；早期文档中的 `5001–5004` 是
off-by-one 书写错误，不得用来改变 worker assignment。新 Think run 必须使用独立空目录；
`--resume` 只用于同一 run contract 的中断恢复。首个正式 episode checkpoint 同时验证闭环链路；不得
先单独运行、观察并随后重复一个 validation instance。

2026-07-14 的 Think 正式 run 使用上述 exact plan 与 ports，在 Git
`36526c58997be5409f65c5976fb35e17aa007ad7` 正常退出。40 checkpoints 时 success gate 已数学不可能，
两个在途 worker 收尾后 summary 保存 42 checkpoints、9 successes、20 unobserved，上界 29/62；candidate
被拒绝。final-test plan 未实例化或运行。结果与 immutable HF revision 见
`data/results/archive/gui_owl_1_5_8b_think_androidworld_validation/`。
