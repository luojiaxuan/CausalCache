# 32B 跨骨干闭环:8B 教出的 selector 能否直接迁移

## 问题

论文的 selector 是用 8B 策略的标签训出来的。换成 GUI-Owl-1.5-32B 冻结策略后,
把这个现成 selector 直接挂上去,还能不能拿到增益?

## 结果(配对,n=51 交集任务)

| 臂 | 成功率 |
|---|---|
| 32B + Recent-4 | 49.0% |
| 32B + 8B 教出的 selector | 39.2% |
| **配对差** | **−9.8pp**,CI [−21.6, +2.0],不一致 2:7 |

区间跨零,但**方向在整个采集过程中六次采样全部一致**(−5.3 / −12.5 / −10.9 /
−11.8 / −10.2 / −9.8),不一致对始终偏向 Recent-4。

## 覆盖率与封盘(必读)

- recent 臂 **111/117**,selector 臂 **57/117**,交集 51。
- selector 臂在采集中途被主动暂停,把资源让给 recent 臂(用户指令"先把
  recent-4 结果跑出来"),之后未恢复。
- recent 臂最后 6 个任务(GoogleMapsAlibabaPhoneContact、MastodonAddBookmark、
  MastodonCreateMemo、MastodonOpenAutomatedDeletion、
  MattermostResourceConflictResolution、ReadQwen3PaperTask2)连续 3 个 attempt
  失败,错误全是 `TimeoutError`(模拟器内部卡死,容器与 HTTP 服务仍存活)。
  换过两批模拟器无效,判定为基建故障而非模型失败,遂封盘。

**因此这不是全量口径**,引用时必须写明 n=51 配对交集。若论文要用它支撑
"跨骨干迁移"的说法,建议补跑 selector 臂到全量。

## 与 8B 的对照

| 骨干 | Recent-4 | + selector |
|---|---|---|
| 8B(论文全量) | 30.2% | 36.8% |
| 32B(n=51) | 49.0% | 39.2% |

32B 的 Recent-4 基线比 8B 高约 19pp——大骨干靠 recent 窗口就能解决大部分
8B 需要记忆调度才做对的任务。这也解释了负迁移:**8B 教师标注的是 8B 的短板**,
那些"值得恢复的远端帧"对 32B 而言未必稀缺。

## 结论

8B 教出的 selector 不能免费迁移到 32B。要在 32B 上复现增益,需要用 32B 自身
打分产出标签重训 selector(与 `../teacher_ab_v1` 里"教师须与部署策略匹配"的
发现一致)。
