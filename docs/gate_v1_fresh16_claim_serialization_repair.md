# Gate v1 fresh-16 claim-serialization repair v1

> 当前状态：formal-58 training 已完成、封存并完成 private-HF immutable replay。本文件描述的
> claim-serialization repair **Source-A source-only 已冻结**；本 milestone commit/push 后成为 canonical。
> Execution-B runner freeze 当前 absent，`evaluation_executed=false`、`execution_authorized=false`。尚未重跑
> fresh-16，没有新的 fresh label semantic
> access、label claim、primary report、HF mutation、GO/NO-GO、旧 dev-5、confirm、matched-NLL 或 closed-loop
> 结果。

本 repair 是
[`gate_v1_fresh16_inventory_repair.md`](gate_v1_fresh16_inventory_repair.md) 的独立 versioned operational child。
它只修复 `LabelAccessClaim.claim` 返回 `mappingproxy`、无法交给 canonical JSON serializer 的接口错误：property
改为返回 canonical-JSON round-trip 后的 deep snapshot。这个 snapshot 可序列化、与内部 state 不共享可变嵌套
对象，且预期生成的 canonical claim bytes 与把原 plain mapping 直接 canonicalize 的 bytes 完全相同。

## 为什么必须另立版本

inventory-repair v1 的唯一 Hyper00 attempt 已验证 full-15 derived tree、只下载 4 个 consumed files，并完成
16 trajectory / 80 OCR semantic decode、80 selected images、48 feature states、144 candidates、10 checkpoint
loads、双 H200 2 workers / 97 vision forwards。随后在 `heuristic_local_seal` 已持久化、fresh labels 尚未打开时，
表达式 `pretty_json_bytes(claim.claim)` 抛出：

```text
TypeError: Object of type mappingproxy is not JSON serializable
```

旧 attempt 因此永久为 `INVALID_PRE_LABEL_CLAIM_SERIALIZATION_FAILURE`，不能删除 state 后续跑、不能复用旧
artifact 继续、也不能重新调用 inventory-repair v1 的 `run`/`validate`。失败证据已经由 Git commit
`7fbfe1b8314ea61d7d646a47be902fdf1c6d4af9` 封存：

- [`summary.json`](../data/results/gate_v1_fresh16_inventory_repair_v1_attempt/summary.json)；
- [`README.md`](../data/results/gate_v1_fresh16_inventory_repair_v1_attempt/README.md)。

该 evidence 绑定 ordinal `0..7` receipts、独立 heuristic seal、88-file / 51,508,707-byte artifact tree、run
log/start/exit、Docker receipt 与所有 successor absence。新 repair 在读取 token 前只读验证 Execution-B identity、
这些 local retained bytes/successor absence，以及新 state/artifact roots absent；新的 runtime receipt 必须已由
clean B 单独捕获并通过 binding。之后才可读取 token
并构造 HF API；API 构造后、任何 fresh semantic access、新 root 创建或 HF mutation 前，必须先验证 HF owner
`gavinlaw` 与当前凭据 write role，之后才能把 private 404 解释为 parent v1、inventory-repair v1、本 repair 三个
repo/tag identities absent。

## Source of Truth

- Git canonical repo/branch：<https://github.com/luojiaxuan/CausalCache> / `main`；
- formal selector completion：
  [`data/results/gate_v1_formal58_train_v1/`](../data/results/gate_v1_formal58_train_v1/)；
- parent scientific protocol：
  [`causalcache_gate_v1_fresh16_evaluation_v1.json`](../code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json)；
- full-inventory parent repair：
  [`causalcache_gate_v1_fresh16_inventory_repair_v1.json`](../code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json)，
  Source-A=`6fb3e868e293bce191ce30a5c6a15ecc544c4591`、Execution-B=
  `c22734ffc85935882f57ddb081c9194d6dae92d0`；
- claim-serialization failure evidence commit：
  `7fbfe1b8314ea61d7d646a47be902fdf1c6d4af9`；
- 本 repair Source-A contract：
  [`causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json`](../code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json)；
- 本 repair Source-A commit：本 material step commit/push 后成为 canonical；本文件不预写未知 commit；
- 本 repair Execution-B：当前必须 absent，不能把预期 runner-freeze 写成已生成或已授权；
- 本 repair planned private HF dataset：
  `gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile`，tag
  `gate-v1-fresh16-claim-serialization-repair-v1`；当前只是 destination identity，Source-A 不联网，也不把其
  remote absence 作为 source-only validator 结论。

### Source-A freeze evidence

最终 Source-A source tree 的冻结证据为：

| 项目 | 冻结值 |
| --- | --- |
| config SHA256 | `3979573be235d630ee2f46dc23be8747a843190c9b57ee81e1b3a17b4416d8c7` |
| source inventory | 57 paths |
| source inventory SHA256 | `572992cf5ac75142cdf8f0e82bdfc2caad43ba37632c741eae277285db54d689` |
| focused regression | 99 passed，7 subtests passed；其中新增 repair coverage 为 30 passed + 7 subtests passed |
| full regression | 1263 passed，16 skipped，4 deselected，617 subtests passed |

4 个 deselected tests 都是已完成历史 Source-A/Execution-B 中专门验证 runner-freeze B absence 的 lifecycle tests；
它们与当前 repair Source-A 的 B-absence contract 冲突，因此在全仓审计中显式 deselect，不代表测试失败或未覆盖
当前 repair。

## 唯一允许的代码修复

允许变化只有：

1. `LabelAccessClaim.claim` 返回 JSON-safe deep snapshot，而不是暴露 `MappingProxyType`；
2. 新增只验证旧 failure evidence 与新 namespace 的 versioned contract/runner/manager/tests；
3. 新 Source-A/Execution-B lineage、local state/artifact/runtime receipt 与 planned HF identity。

回归必须证明：

- snapshot 可以由 `pretty_json_bytes` 直接 canonical serialize；
- 两次读取返回彼此独立的对象；修改任意一份 snapshot 的顶层或 nested container 不改变内部 claim；
- deep snapshot 的 canonical bytes 等于输入 plain mapping 的 canonical bytes；
- claim payload 的字段、值、排序语义和 SHA 不因这个 repair 改变。

## 科学与数据不变项

| 冻结对象 | 本 repair 规则 |
| --- | --- |
| formal training | 已完成；不重训、不改 checkpoint、LR、epoch、seed 或 ensemble |
| scientific protocol | 与 fresh-16 parent 完全相同 |
| derived inventory | 继续使用 inventory-repair v1 冻结的 exact full-15 tree；仍只下载原 4 个 consumed files |
| model input | 同一 14-file / 17,545,907,171-byte GUI-Owl verified projection 与同一 10 checkpoints |
| labels | 同一 repaired immutable label input；Source-A semantic decode 为 0 |
| geometry | 同一 16 trajectories / 48 states / 144 candidates / 448 distances，`n=2/3/4,B=2` |
| evaluation | 同一 conditional/independent + recent/OCR-RGB/policy-vision、bootstrap 与 aggregation |
| output | 同一 9-target payload + direct-child 4-target report chain；只改变 namespace，不改变相对 target paths |
| thresholds | Selector GO、set-conditioning GO 与所有 stop/ratio/std/bootstrap 阈值完全不变 |
| label firewall | all selections durable seal 后才允许创建 label claim；Source-A 不得打开 labels |

因此本 repair 不增加 input token、不改变 memory budget、不换 prompt/roster/loss/model，不把旧失败解释成科学
NO-GO，也不依据失败执行中已经观察到的 label-blind selections 调参。

## 新运行与发布 identity

本 repair 不复用 parent v1 或 inventory-repair v1 的任何 success path：

| 项目 | claim-serialization repair v1 identity |
| --- | --- |
| state root | `/data/experiments/causalcache/gate-v1-fresh16-evaluation-claim-serialization-repair-v1` |
| execution namespace | `gate-v1-fresh16-evaluation-claim-serialization-repair-v1` |
| artifact directory | `/data/artifacts/causalcache/gate-v1-fresh16-evaluation-claim-serialization-repair-v1` |
| Docker receipt | `/data/experiments/causalcache/.gate-v1-fresh16-evaluation-claim-serialization-repair-v1.docker-inspect.json` |
| private HF dataset | `gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile` |
| planned tag | `gate-v1-fresh16-claim-serialization-repair-v1` |

以下旧对象必须原样保留且在新 run 前继续验证：

- parent v1 state/artifact/runtime/log evidence；
- inventory-repair v1 state root、88-file artifact、runtime receipt 与 run log/start/exit；
- ordinal `0..7` receipts、heuristic seal、`label-access-claim` 和 ordinal `8..15` successor absence；
- 新 state/artifact roots 在 token access 前 absent；新 runtime receipt 已由 clean B 的 `capture-runtime` 预先
  捕获并绑定，不能与 run state root 混为一谈；
- token 读取并构造 HF API 后、任何 fresh semantics/new-root/HF mutation 前，先验证 HF owner `gavinlaw` 与
  当前凭据的 write role；只有这两项通过，private repo 的 404 才能解释为 absence。随后验证
  `causalcache-gate-v1-fresh16-evaluation-mobile`、
  `causalcache-gate-v1-fresh16-inventory-repair-mobile`、
  `causalcache-gate-v1-fresh16-claim-serialization-repair-mobile` 三个 repo/tag identities absent。

## Source-A-only validation

Source-A validator 只能检查 committed source/config/hash、failure evidence bindings、runner-freeze absence 与
effective-delta proof；它不能联网、写文件、导入 PyTorch/Hugging Face、读取 fresh semantics、加载 model 或授权
执行。预期命令形状是：

```bash
PYTHONPATH=code python3 -m scripts.validate_gate_v1_fresh16_claim_serialization_repair_contract \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json

cd code
PYTHONPATH=. python3 -m pytest -q \
  tests/test_gate_v1_fresh16_claim_serialization_repair_contract.py \
  tests/test_gate_v1_fresh16_claim_serialization_repair_runner.py \
  tests/test_gate_v1_fresh16_evaluation_runner.py
```

Source-A 的合法输出边界固定为：

```text
network_call_count = 0
file_write_count = 0
torch_import_count = 0
fresh16_semantic_access_count = 0
label_semantic_decode_count = 0
model_load_count = 0
model_forward_count = 0
primary_report_count = 0
hf_mutation_count = 0
evaluation_executed = false
execution_authorized = false
runner_freeze_b_present = false
```

当前没有 Execution-B，也没有合法 `run` 权限；上面的 source-only validation 不是实验运行，不产生 fresh result。

## 后续唯一执行顺序

1. 把已通过 focused/full regression 的 Source-A 作为 focused commit push 到 canonical `main`；
2. 从 clean pushed A 运行 `validate-source`，绑定完整 A SHA；
3. 机械 materialize 唯一 runner-freeze B；B 必须是 A 的 direct single-parent child，且 tree diff 只能新增
   `code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_runner_v1.json`；
4. 将 B 作为独立 commit push，再从 clean B 验证 exact Git/source inventory；
5. 在 Hyper 执行 host/GPU/disk/container preflight 与 10 秒 idle cleanup，选择精确两张 H200，捕获新的
   mode-0600 Docker receipt；token 前验证 Execution-B、local retained evidence 与新 local roots absence；
6. 读取 token、构造 HF API，随即在任何 fresh semantic access、新 root 创建或 HF mutation 前验证 owner 与
   write role，再验证三个 repo/tag identities absent；private 404 只有在 owner/write-role check 通过后才能解释为
   absence。之后才从全新的 state/artifact/download roots 运行唯一 formal attempt，并在
   representative steady-state window 验证两张卡各自至少 80% utilization；
7. run 完成后只读验证 local completion、13-target HF publication、tag-resolved immutable fresh-download replay，
   再把轻量 result 写回 Git；
8. 只有 sealed fresh-16 report 自身完成后，才能另立 combined-21；即使 primary GO，也不能自动打开 confirm、
   matched-NLL 或 closed-loop。

在第 4 步之前，任何 token access、fresh semantic decode、GPU model load/forward、label claim、output root 或 HF
mutation 都是未授权操作。
