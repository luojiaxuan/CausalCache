# Set Utility Freeze-B v2 terminal-index repair

## 结论

Freeze-B v1 的 roster/query materialization 存在 terminal decision off-by-one，因此 v1 的 2,400 个
query state 不得用于 processor candidate freeze、restoration labels 或 predictor training。v2 repair 已在
不读取 raw shard、截图、instruction、processor、model 或 utility 的前提下重新物化 policy-blind roster 和
query plan。

v2 config canonical SHA256 为
`7d7dad8580939be67b56bb7ad5a9e06b771e6d0f25ffd3665da84d14466cbf75`；manifest SHA256 为
`915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30`。

## 根因

GUIOdyssey parser 把：

```text
decision_count = len(recorded_actions) - 1
```

定义为初始 recorded action 之后仍需预测的 action 数量。`build_pilot_manifest()` 生成的有效
`decision_step_id` 却是：

```text
2, 3, ..., len(recorded_actions)
```

因此最后一个有效 state 是：

```text
terminal_decision_step_id = decision_count + 1
```

v1 错把 terminal 写成 `decision_count`，实际选中倒数第二个 state；同时把
`decision_count=6/10/18` 的边界轨迹错误排除为“无法提供两个 distinct queries”。

## 修复范围

v2 保留并绑定 v1 bytes 作为失败证据，不覆盖旧 manifest。它复用相同 P0、historical consumed-group
firewall、selection salt、role quotas、feature/training grid 和 generic operation ceiling，但在正确的可用 state
范围上重新执行：

1. 每个 instruction-app group 的 deterministic representative；
2. corrected eligible universe 上的 role assignment；
3. stratum anchor `6/10/18` 与 terminal `decision_count+1`；
4. recent-16 pre-processor candidate plan；
5. 仅用于上界审计的初始 `|S|<=2` exact schedule。

这不是结果驱动的数据重抽：本步访问 outcome、utility、KL、OCR、截图、instruction 和 policy output 的计数均为
0。v1 与 v2 roster 有 1,008/1,200 source overlap；192 条退出、192 条进入，变化来自修复后的边界轨迹进入
deterministic ordering。

## 完成态

| 项目 | 数值 |
| --- | ---: |
| trajectories | 1,200 |
| train / tune / evaluation | 1,000 / 100 / 100 |
| query states | 2,400 |
| stratum trajectories | 400 / 400 / 400 |
| assignment inventory SHA256 | `ac2bd077...65cec2` |
| query-plan inventory SHA256 | `e38f3889...0e881a` |
| pre-processor `|S|<=2` rows | 171,730 |
| generic raw-row ceiling | 328,800 |

初始 candidate histogram 为：

```text
n=4:400, 5:56, 6:66, 7:106, 8:572, 9:82, 10:63,
11:54, 12:49, 13:59, 14:42, 15:32, 16:819
```

171,730 只是 processor freeze 前的精确上界。真实 label budget 必须在 full-reference prompt 上执行
processor-only recent-suffix fit 后重新生成。

## 下一边界

下一步另立 processor-only Execution-CF contract。它只允许：

1. 按 v2 roster 读取 1,200 个 pinned raw rows；
2. 生成 OCR/low-fidelity substrate；
3. 用 pinned GUI-Owl `AutoProcessor` 在 CPU 上测 full-reference prompt length；
4. 逐个丢弃最老 candidate，直到 `input_tokens + 256 <= 32768`；
5. 固定最终 candidates、四 worker schedule 和 exact operation budget。

本阶段不混入 12-state policy throughput pilot。真实 policy forward 的 throughput pilot 必须在 processor freeze
结果 commit/push 后另立 train-only contract；evaluation label bytes 也必须与 train/tune 物理分区并保持封存。

复现：

```bash
PYTHONPATH=code python3 -m scripts.validate_set_utility_freeze_b_v2 \
  --repository-root .

PYTHONPATH=code python3 -m scripts.materialize_set_utility_freeze_b_v2 \
  --repository-root .

PYTHONPATH=code .venv/bin/python -m pytest -q code/tests/test_set_utility_*.py
```
