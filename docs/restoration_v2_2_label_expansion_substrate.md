# Restoration v2.2 label-expansion substrate source freeze

## 当前边界

本步骤只建立 64 条 expansion trajectory、192 个 decision state 的 **source-only substrate
契约**。它不运行 GUI policy，不生成任何 policy/restoration output，不读取 confirm20，不构造 coalition，
也不开始 gate training、matched-NLL 或 closed-loop。

canonical config 已在 policy-blind derived artifact 的 preupload standalone validation、private Hugging Face
upload、immutable revision 固定与 fresh-download clean projection replay 全部闭合后，从 clean、已 push 的
`main@6bf3f8c85e4f3be2ba40f1a64c3e2e05542b977d` 唯一物化：

```bash
cd code
python3 -m scripts.materialize_restoration_v2_2_expansion_substrate_contract \
  --source-git-commit 6bf3f8c85e4f3be2ba40f1a64c3e2e05542b977d
```

物化输出固定为
`code/configs/causalcache_restoration_v2_2_expansion_substrate_v1.json`，SHA256 为
`42144f33e2473c787b3648c0ace18b22902fa3376615732aef8fd3aa5780a6e5`。物化器本身只写 config，输出中
`policy_or_gpu_execution_authorized=false`；config commit/push 仍不自动授权 GPU run，runner/source inventory
还必须作为单独 material step 冻结。

## 固定 denominator 与计算预算

| split | trajectory | state（step 4/5/6） |
| --- | ---: | ---: |
| `gate_train_expansion` | 48 | 144 |
| `gate_development_expansion` | 16 | 48 |
| 总计 | 64 | 192 |

每个 state 的 substrate 仍沿用 v2.2 eager 的三路比较：

1. full-history native generation repeat 1；
2. full-history native generation repeat 2；
3. 对 reference 1、reference 2、summary-only 做 teacher forward；
4. 计算 repeat-reference KL 与 summary-reference KL。

因此本阶段的 planned 和 maximum counts 同时固定为：

| operation | per-state | 192-state total |
| --- | ---: | ---: |
| mixed-fidelity/full-history generation | 2 | **384** |
| teacher forward | 3 | **576** |
| KL | 2 | **384** |

这里的 384 generation 是 substrate reference/repeat 运行，不是 restoration coalition generation。后续 exact
subset / conditional-marginal label 阶段的 1,792 个 `D(S)`、1,856 条 deployment edge 与 1,984 次 teacher
forward 只作为已冻结 workload 记录；本 config 明确不授权它们执行。

## 最重要的 decision-view 防泄漏约束

derived trajectory JSONL 为复用存储而物理保存 event 1--5，且这些 event 带各自 action。对 step 4 或 step 5
直接消费整条 `events` 会看到当前或未来 action，因此是标签泄漏。
契约只声称经过严格 slice 的 **per-decision view** 不含 current expert action；不会把 shared trajectory storage
误写成不含 later-event actions。

每个 state 的 loader 必须调用
`restoration_v2_2_expansion_substrate_inputs.build_decision_view_input()`；该函数内部强制调用：

```python
decision_view_events(trajectory["events"], decision)
```

或者执行逐字节等价、并被 validator 证明等价的逻辑。返回 event step 必须严格等于
`history_event_step_ids == range(1, decision_step_id)`；任何 `event.step_id >= decision_step_id` 都必须
fail closed。正式 runner 还需为 192 个 state 逐一记录 slicing witness。禁止把完整 `events` 直接传给 prompt
builder，也禁止使用 event 4/5 action 为较早 decision 构造输入。

runner source freeze 还必须实现一个 **processor-byte canary**：只修改被排除的 current/future event action，
最终 processor/prompt bytes 必须完全不变；修改一个合法 history action，bytes 必须变化。canary 必须在第一次
generation 前通过。raw request manifest 记录实际 included event step IDs，不能只复述声明字段。canonical
action 只能来自 frozen full-history policy native generation，禁止从当前/相邻 event 的 expert action 回填。

loader 还必须验证 `state_id == f"{source_id}:decision_step:{decision_step_id:03d}"`。runner 不能只相信
request manifest 自报的 step IDs/hash：它必须把实际 included event objects 一并交给
`validate_request_manifest()`，由 validator 重算 exact step sequence 与 canonical payload SHA256，并在构造实际
processor request 前通过。

被排除的 event 必须逐个独立 mutation，不能只测每个 state 的第一个 excluded event。step 4 有 event 4/5、
step 5 有 event 5、step 6 没有 excluded event，因此固定为 128 个有 excluded canary 的 state、共 **192 次**
excluded-event mutation；另有 192 次 included-history mutation。canary 对 shared event action 的机械读取/修改
只用于证明 processor bytes 不受影响，不计作 expert target 的语义消费；current/future expert target read、semantic
consumption 与 action backfill 仍严格为 0。

## derived artifact provenance

source freeze 绑定：

- structural selection manifest path/SHA256；
- pre-policy exposure ledger path/SHA256 及六个空交集；
- derived completion manifest path/SHA256；
- private HF repo、tag、immutable revision 与 payload prefix；
- artifact exact-six 六项完整 path/size/SHA256 inventory、total bytes 与 tree SHA256；
- inventory 中真实存在的根 `.gitattributes`、根 `README.md` 与新 prefix 下四个 payload；
- `derived/restoration-v2-label-expansion-v1/manifest.json` 的真实 size/SHA256；
- preupload standalone validator outcome；
- fresh-download standalone validator outcome；
- preupload/fresh-download tree、file count、bytes、payload manifest SHA 全相等。

不存在的 `metadata.json` 不属于 artifact，也不得出现在 artifact witness。build、standalone validation 与 HF
链路的审计信息单独放 completion `execution`：host/container/image、`DeviceRequests=[]`、UTC bracket、完整但
不含 secret 的 argv、exit code、log size/SHA256、runtime versions、16-file rehash、64-row reload、exposure
ledger SHA、private repo 声明，以及旧 immutable tag 未被改写的 witness。

fresh download 必须先从 HF immutable revision 下载，再建立只含新 prefix exact-six 的 clean projection；禁止
直接把可能同时含旧 prefix 的整仓库 snapshot 当成待验证 artifact。

completion manifest canonical path：
`data/results/restoration_v2_2_label_expansion_derived/artifact.json`。其 negative declarations 必须逐项为：
字段 schema 固定在
`code/configs/causalcache_restoration_v2_2_label_expansion_derived_completion.schema.json`。

- confirm state/output 未访问；
- policy module 未加载、policy forward 未调用、policy output 未生成；
- restoration output 未生成；
- gate training、matched-NLL、closed-loop 均未开始。

## 执行拓扑与失败语义

正式 substrate 预注册为同一 Hyper H200 host/container 内两个 worker，even/odd parity 各固定 96 个 state。
不允许 state stealing、retry、resume、top-up、replacement 或 sample mutation。任一 worker failure、attempt
marker 缺 terminal record、state union 不等于固定 192，均使整个 attempt invalid。

substrate gate 沿用既有 v2.2 语义：parse coverage 至少 0.99（需 191/192）、finite KL 与 repeat canonical
action agreement 均为 192/192；memory-sensitive 定义仍为
`summary_reference_kl > max(1e-4, 10*mean_repeat_kl)`，沿用原冻结的最低 8 个 state，不新造事后比例阈值，
同时完整报告实际 fraction。

source lock 不是只绑定新 loader：它完整继承 v2.2 eager formal source inventory，包括 transitive prompt builder、
parser、runtime、vision preprocessing、teacher/KL、model snapshot/config 与 parent evidence，再追加 expansion
loader、derived-artifact builder/validator 和本契约 source。两个 worker 必须报告相同 scientific runtime/container
digest，只允许 device、GPU UUID/PCI bus id 和 logical/nvidia-smi index 按 parity worker 不同；同一 state 的两次
generation 和三次 teacher forward 必须留在同一个 worker/device。

## source-only 验证

canonical config 落盘并 commit/push 后运行：

```bash
cd code
python3 -m scripts.validate_restoration_v2_2_expansion_substrate_contract
```

validator 只读取 Git-tracked source/config/manifests，重建 deterministic contract，并检查每个 source file 与
冻结 parent commit 的 Git blob 相同。它不会 import policy runtime，不会访问 GPU/HF/raw artifact，也不会读取
任何 policy/restoration output。
