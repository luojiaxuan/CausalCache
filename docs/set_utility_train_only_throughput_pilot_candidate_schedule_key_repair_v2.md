# Set-utility throughput pilot candidate-schedule key repair v2

## 当前结论

本 repair 是 v1 唯一 formal attempt 在 `0/84` native calls 处 fail closed 后的新 source identity。它只修复
candidate schedule 的 producer serialization reconstruction，不改变 processor artifact、candidate schedule bytes、
12-state roster、worker mapping、mb1→mb2 顺序、84-call ceiling、80% reserved-memory threshold 或 0.95 teacher-wall
ratio。

canonical source config 为
[`../code/configs/causalcache_set_utility_train_only_throughput_pilot_v2_candidate_schedule_key_repair.json`](../code/configs/causalcache_set_utility_train_only_throughput_pilot_v2_candidate_schedule_key_repair.json)：

- config SHA256：`b2e224147a70dbec14c389098a8fdcdd80b789b4f7f7cbafca15857102eb41ae`；
- 44-file transitive source inventory SHA256：
  `ea3ca6dfda529ccf2cde50e7d1c577a6a8714fbbeb90415faa3c32a16456a669`；
- status：`SOURCE_ONLY_TRAIN_THROUGHPUT_PILOT_V2_CANDIDATE_SCHEDULE_KEY_REPAIR_FROZEN`；
- source validation：
  `VALID_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_SOURCE_V2_CANDIDATE_SCHEDULE_KEY_REPAIR`。

source config 仍禁止 policy/model load、GPU/CUDA、pilot result、labels、training、HF mutation 与 tune/evaluation/test
访问。只有本 source commit push 后，以 direct-child commit 新增 exact v2 execution envelope，才授权新的唯一
formal attempt。

## Parent failure lineage

repair 逐 byte 绑定 v1 failure summary（4,975 bytes，SHA256=
`d4fde65ff8c9fc1c9b67a38c980de1edd89c99b0d6958d238b6aa0417aa3f45c`）：

- parent source A：`5158f2ad04e456fd088765973da6a99064b4efcf`；
- parent execution B：`c7d5b31f8235124699ecbee5beb5240c72eb8977`；
- parent envelope SHA256：`adf50363d9b3623b84b9229d709f796d7b1971faf7e4056c468cfe3b55ce228e`；
- parent actual native calls：`0`；
- parent retry：禁止；旧 run root、attempt 与 terminals 必须原样保留。

旧 v1 source/execution JSON 继续留在 Git，不删除、不覆盖。v2 使用新 source config 与新 execution config path：

```text
code/configs/causalcache_set_utility_train_only_throughput_pilot_v2_candidate_schedule_key_repair.json
code/configs/causalcache_set_utility_train_only_throughput_pilot_v2_candidate_schedule_key_repair_execution.json
```

## Exact byte diagnosis

Hyper00 上的 immutable candidate schedule 为 18,718,642 bytes，SHA256=
`186f2952108273672c6cdbf963094754298d23693fd6268a72b6223e99c2299d`，与 processor publication 完全一致。
strict JSON parse 成功；naive `json.loads`→canonical pretty round-trip 与 raw 有 70 个 byte positions 不同。

唯一差异来自：

```text
$.exact_label_schedule.state_count_by_candidate_count
```

producer 在写 JSON 前使用整数 keys `4..11`，按数值排序；JSON 读取后 keys 变为字符串，naive consumer 会按
字典序把 `"10"` 排在 `"4"` 前。把这个唯一注册路径上的 canonical positive decimal string keys 恢复为 int，
再执行同一 pretty serialization，得到的 bytes 与 raw 逐字节相同，SHA256 重新得到 `186f...2299d`。

## Repair contract

`canonical_candidate_schedule_producer_bytes_v2()` 只允许上述路径做 typed-key reconstruction：

1. raw artifact 仍先通过 envelope 的 exact SHA256、byte count、formal inventory 与 stat identity；
2. parser 仍拒绝 invalid UTF-8、duplicate JSON keys、non-finite values 和非 object root；
3. registered mapping 的 key 必须匹配 canonical positive base-10 text，不允许 leading zero、非数字、空 mapping
   或 int conversion collision；
4. 仅浅复制 top-level 与 `exact_label_schedule`；其他任何 numeric-looking mapping 不做转换；
5. producer-typed reconstruction 必须与 raw bytes 逐字节相同；
6. 后续 candidate-inventory SHA、train-only assignment、processor tar、strict join 与 state roster 验证保持原样。

这不是放宽为“任意 strict JSON 都接受”，也不修改或重新发布 HF artifact。

## Execution boundary

新 formal run 必须使用新的 run root、fresh 10-second preflight、四张同构 H200、四个独立 GPU UUID 和新的
source-A→envelope-B lifecycle。四个 worker 仍同时启动，各处理三个预注册 state；禁止重用 v1 claims、retry、
top-up、换 state 或合并异构 host metric。成功路径仍精确为 48 generation + 36 teacher = 84 native calls。

在 v2 aggregate 完成前，restoration labels、predictor training、matched-NLL 与 closed-loop 继续 locked。
