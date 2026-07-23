# Exploratory AndroidWorld closed-loop validation-12 v1

## 为什么另立这个协议

confirm-20 的有效结论 `NO_GO_INDEPENDENT_CONFIRM` 保持不变。它证明当前 learned selector 没有在 20 个
untouched policy/restoration states 上稳定泛化 **single-state restoration objective**：independent / exact raw
ratio 未过门槛，并且 learned independent selector 输给三类冻结 heuristic。

该结论不能直接回答另一个 estimand：在一条真实长程交互中，每一步都重新选择 memory 后，哪一种 controller
获得更高的 AndroidWorld executable terminal success。single-state restoration proxy 与 repeated closed-loop
control 可能相关，但不是同一个量。因此本协议把两类证据改为并列关系，而不是把历史 confirm 改写为 PASS：

```text
offline confirm NO-GO
    └─ 当前 learned selector 没有稳定学好 restoration proxy

development-only closed loop
    └─ 同一冻结弱 backbone 下，反复 memory selection 的 task-level directional signal
```

本步骤是 outcome-exposed AndroidWorld validation split 上的 12-template exploratory probe。该 split 以前已用于
backbone validation，所以它不是新 holdout、不能进入论文 primary table。正结果只允许另立新的 train-60
development contract；sealed test-75、matched-NLL 与任何 paper-level claim 都继续 locked。负结果也不能反向
修改 confirm 阈值、数据、checkpoint、prompt 或 comparator。

## 固定 roster

父计划固定为
[`androidworld_validation_plan.json`](../../code/configs/androidworld_validation_plan.json)，文件 SHA256
`6429599d9d351c396cf0da12939ed82fb84c9acb9c07b051ae92f9d2832e3b42`，内部 instance roster SHA256
`8204e7f832f1d70becd51f299977f0e4a322a080bbfab64010345c48f901b0e8`，suite seed `271828`。

只允许 `task_index=0`。使用 arm-independent 的冻结 `max_steps` 做 pre-treatment strata：

- short：`max_steps <= 12`；
- medium：`13 <= max_steps <= 22`；
- long：`max_steps >= 23`。

每层按
`SHA256(protocol_id \0 validation \0 271828 \0 task_type \0 0)` 升序机械取前 4 个，不手挑 app、任务或
已知 backbone success。exact 12-instance records、selection digest 与 identity hash 位于
[`exploratory_closed_loop_validation12_v1.json`](../../data/manifests/exploratory_closed_loop_validation12_v1.json)，
文件 SHA256 `8e68e6eb1adde5627e83ce07cf4e8d7cef4843250a2e1b58baa247af57a07a42`。固定分母是 12 templates、5 arms、
60 episodes。

`selected_instance_identity_sha256` 的 canonical payload 是 manifest 顺序下每条 record 的
`horizon_stratum`、`max_steps`、`task_index`、`task_type` 四字段投影；以 UTF-8、sorted keys、compact
separators、禁止 NaN 的 canonical JSON 编码后取 SHA256。机械复算与 pool-size/top-4 检查由
`causalcache.exploratory_closed_loop_roster` 完成。

## 五个 arms

1. `independent_B2`：当前 paper method 的唯一 primary arm；
2. `recent_B2`：primary baseline；
3. `ocr_rgb_B2`：confirm 上的 strongest heuristic，对应 key challenger；
4. `conditional_B2`：预注册 negative ablation，不能在看到结果后替换 primary method；
5. `summary_B0`：无高保真历史的 sanity control，不属于等预算主比较。

不加入 random、policy-vision、full-history 或 B4，避免把小型 probe 扩成新的 model-selection sweep。所有 B2 arm
共享 at-most-2 高保真 event capacity；当前 observation 免费且始终可见；与当前 observation 等价的最新
post-state 只保留摘要，不作为候选；所有更早 event 都保留相同的 fixed low-fidelity summary。learned gate 使用
formal-58 已封存的 5-seed checkpoints，raw score 算术平均，threshold 固定 `0`。Independent 一次性对全部候选
排序；conditional 每加入一个 event 后对余下候选重新打分；二者只选严格正分。Recent/OCR-RGB 选
`min(2,n)`；summary 选择空集。所有 tie 最终按较小 `step_id`。

相同 B 是 capacity 相同，不保证实际输入 token 相同。必须逐 decision 报告 high-fidelity count、policy-visible
text/image tokens、latency 和显存；不能把 learned gate 的 early stop 写成与 recent/OCR 相同的 token budget。

## 真正的 repeated online selection

每个 `(task_type, task_index, arm)` 都独立 `suite reinitialize → task initialize → reset → rollout → score →
tear-down`，不能让五个 arm 共享一条轨迹。每个 policy decision 必须执行：

1. 从环境取得唯一一次 current screenshot，并确定性编码；
2. 若上一动作已执行，用 before/action/after 与 pinned OCR 构造新的 event 和 low-fidelity summary；
3. 在本 arm 自己产生的完整历史上重新计算 eligible candidates 和 selector；
4. 依时间顺序放入所有摘要，仅为 selected events 插入 post-state image，再附 current image；
5. 使用同一 frozen GUI-Owl v2.1 official-tools policy 生成、严格 parse、映射并执行一个 action；
6. 记录 selector scores、selected ids、budget/current-equivalence audit、action、截图/OCR identity、visual tokens、
   latency、executor response 和 official task score。

禁止用 recent trajectory 对其他 arm 做固定 replay。不同 controller 导致不同 state distribution 正是 task-level
estimand 的一部分。Actual episode steps、成功后提前结束和第一次 action divergence 都是 post-treatment，只能
做描述，不能重新定义 horizon strata。

五臂顺序用 protocol/task identity 的 SHA256 决定 cyclic rotation；每个 template 的 base order 相同、rotation
不同。不能总让某个 arm 先跑，也不能根据已完成 arm 的结果改变后续顺序。formal run 不允许 outcome-based early
stop、hidden retry、top-up 或 template/arm deletion。缺失 episode record 使 execution `INVALID`，不是自动补零；
已经形成且通过 identity 检查的 parse/selector/executor/infrastructure failure record 在 ITT 中 success=0。

## 统计与诊断

primary contrast 只有 `independent_B2 - recent_B2`。同时固定报告：

- 每臂 official terminal success count/rate；
- independent 相对 recent、OCR/RGB、summary 的 paired mean 与 win/tie/loss；
- conditional 相对 recent/OCR 的 negative-ablation contrast；
- overall 12 templates 与 short/medium/long 各 4 templates 的相同统计；
- template-paired 10,000 bootstrap，`random.Random(271828)`、90% percentile、Hyndman-Fan type 7，仅描述；
- discordant pairs 的 exact two-sided sign test，仅描述；
- 达到 `candidate_count>B` 的 memory-binding decisions、selector 与 recent 首次 selection 分歧、各 arm 首次
  action 分叉、consecutive action/state repetition、history length、实际 visual budget、latency 与失败分类。

每个 horizon stratum 只有 4 templates，不能以 stratum bootstrap lower 或 p-value 声称显著 long-horizon
improvement。真正需要观察的是：restoration gate 相对 recent/OCR 是否出现方向一致、且在 long stratum 不消失的
task-level signal。

## Execution validity 与资源路由

正式 execution validity 必须全部满足：

- 60/60 exact episode records，五臂 paired inventory 完整；
- hidden retry/top-up/deletion 为 0；
- budget、selection、current-equivalence、OCR/feature fallback audit coverage 为 100%；
- 每臂 action parse coverage 至少 95%；
- 每臂至少 11/12 episode 没有 infrastructure failure；
- 至少 10/12 templates 的五臂都完成正常环境执行链。

未通过则输出 `INVALID_EXPLORATORY_CLOSED_LOOP_EXECUTION_V1`，不能解释成方法失败；修复必须另立 version。

MVP 使用已经验证的 Aries 同机 policy + AndroidWorld emulator stack，不把尚未验证的 Aries↔Hyper transport
混入科学实验。按 GPU preflight 的空闲和有效并行度，最多并行 4 个 emulator/policy workers；每个 worker 固定
一个显式 A6000 device。正式 GPU job 前执行 10 秒 idle cleanup，启动后在代表性 generation 窗口验证每张卡
利用率至少 80%。

## 仅用于下一步资源决策的 advance rule

`ADVANCE_TO_NEW_TRAIN60_PROTOCOL` 不是显著性或 paper claim。它要求 execution valid，且：

- independent-vs-recent overall `wins-losses >= 2/12`；
- long stratum `wins-losses >= 1/4`；
- independent-vs-OCR/RGB 在 overall 和 long 均 `wins >= losses`；
- independent-vs-summary overall 不为负；
- 确实存在 memory-binding decisions 和 independent-vs-recent selector disagreement。

如果 independent 相对 recent 与 OCR/RGB 在 overall 和 long 四个方向均不为正，则输出
`STOP_CURRENT_INDEPENDENT_CLOSED_LOOP_DIRECTION`；其余为 `INCONCLUSIVE_DEVELOPMENT_ONLY`。Conditional
即使胜出，也只能支持另立新方法版本，不能事后替换 primary arm 或救回当前 independent claim。

## 当前状态

当前只冻结 P0 scientific contract、roster、纯 selector/prompt/evaluator 语义和 source-only validation。尚未访问
HF checkpoints、OCR model 或 AndroidWorld live server，尚未加载 policy/gate，GPU/episode/HF publication 数均为
0。执行前还需完成 live RGB OCR、artifact loader、multi-arm episode runner、scheduler/package 的 Source-A，
单独提交并 push 唯一 Execution-B；随后才可进行 Aries preflight 和 formal 60-episode run。
