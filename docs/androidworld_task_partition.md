# AndroidWorld task partition

## 结论

116 个 pinned AndroidWorld task templates 已在任何 validation rollout 之前完成冻结。机器可读
manifest 位于 `configs/androidworld_task_partition.json`，其 sorted task-name registry hash 为：

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

`configs/androidworld_validation_plan.json` 使用 pinned suite seed 实例化 validation-only 参数，
冻结每个实例的 goal、template、complexity、`start_on_home_screen` 与官方
`int(10 * complexity)` step budget。62 个 instance records 的 SHA256 为：

```text
8204e7f832f1d70becd51f299977f0e4a322a080bbfab64010345c48f901b0e8
```

所有 62 个实例都要求 episode 前 reset 到 home。step budget 分布为 10--60，不能为了降低推理
成本统一截断；单任务链路 smoke 通过后，完整 validation runner 必须逐实例 checkpoint，允许安全
续跑但不改变实例顺序或预算。

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

## Reproduction

在 pinned AndroidWorld server ready 后运行：

```bash
python3 -m scripts.build_androidworld_task_partition \
  --base-url http://127.0.0.1:5000 \
  --stack-config configs/androidworld_stack.json \
  --expected-task-types 116 \
  --output configs/androidworld_task_partition.json
```

单元测试会从 committed task names 与 stack config 完整重建 manifest，验证 hash assignment、
全覆盖、无重复和无 bucket overlap；validation plan 另行验证 instance hash、每 template 两个实例
和官方 complexity budget。
