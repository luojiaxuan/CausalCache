# Gate v1 fresh-16 inventory repair v1

> 当前状态：repair Source-A=`6fb3e868e293bce191ce30a5c6a15ecc544c4591` 与唯一 Execution-B=
> `c22734ffc85935882f57ddb081c9194d6dae92d0` 已 commit/push 并被唯一 Hyper00 attempt 消费。该 attempt 已跨过
> full-15 inventory、label-blind semantic materialization、checkpoint replay 与双 H200 workers，但在
> `label-access-claim` 落盘前因 `mappingproxy` serialization 永久 `INVALID`。label decode、report、HF mutation
> 均为 0，本 attempt 没有 fresh-16 GO；旧 namespace 不得续跑。独立 claim-serialization repair 后续已在
> 全新 A/B、namespace 与 HF identity 上完成 valid primary，结果为 selector/set-conditioning 双 `NO-GO`；旧
> dev-5、confirm、matched-NLL 与 closed-loop 仍未执行。

本协议是
[`gate_v1_fresh16_evaluation.md`](gate_v1_fresh16_evaluation.md) 的 versioned operational repair。父协议的
scientific evaluation、fresh roster、gate checkpoints、model snapshot、label bytes、threshold、bootstrap、
aggregation、output target paths 与 label firewall 全部不变；repair 只关闭 immutable derived repo 的远端
inventory contract，并使用新的 local/HF namespace。

## 为什么必须另立 repair

原 v1 Source-A=`97694eff052ecbdc5f12f58b6f9ee10f4dd616ab`、Execution-B=
`a8bb27ccf9b8820c1d8487f63883f813e6845645`。2026-07-18 的唯一 Hyper00 formal attempt 在
`derived_input_inventory_validation_before_download` fail closed：固定 revision
`630363a6adb692d72774f16dd0653a50216313ff` 实际有 15 个 remote paths，而 v1 只声明 4 个 consumed paths，
并允许 `.gitattributes` 与 `README.md`；另外 9 个同 revision 历史 artifact paths 因未绑定而被误判为 drift。

该失败发生在任何 download、trajectory/OCR/image/label semantic decode、checkpoint load、policy worker、GPU
forward 或 HF mutation 之前。旧 state namespace 只含 mode-0600 `runtime_receipts` 与 `global_claim`，旧
artifact directory 为空，原 planned HF repo/tag 仍不存在。Git 中的轻量证据见
[`data/results/gate_v1_fresh16_evaluation_v1_attempt/`](../data/results/gate_v1_fresh16_evaluation_v1_attempt/)；
旧 namespace、receipt、run log 与 absence 必须原样保留，不能删除后续跑 v1。

## Source of Truth

- 父 scientific contract：
  [`causalcache_gate_v1_fresh16_evaluation_v1.json`](../code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json)，
  SHA256 `c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e`；
- repair Source-A contract：
  [`causalcache_gate_v1_fresh16_inventory_repair_v1.json`](../code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json)，
  SHA256 `5ba1b2d433c01defa0faae92a163dc9a9a916d967218abdc7607bd3724829a44`；
- derived private HF dataset：
  [`gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-restoration-v2-mobile)，
  immutable revision `630363a6adb692d72774f16dd0653a50216313ff`；
- v1 failure evidence Git commit：`2d85d088591fa1c7196f2a8e220439120f64f600`；
- repair Source-A commit：`6fb3e868e293bce191ce30a5c6a15ecc544c4591`；
- repair Execution-B commit：`c22734ffc85935882f57ddb081c9194d6dae92d0`，其唯一新增文件为
  `code/configs/causalcache_gate_v1_fresh16_inventory_repair_runner_v1.json`；
- repair v1 failure evidence：
  [`data/results/gate_v1_fresh16_inventory_repair_v1_attempt/`](../data/results/gate_v1_fresh16_inventory_repair_v1_attempt/)；
- repair v1 failure evidence commit：`7fbfe1b8314ea61d7d646a47be902fdf1c6d4af9`；
- claim-serialization repair completion：
  [`gate_v1_fresh16_claim_serialization_repair.md`](gate_v1_fresh16_claim_serialization_repair.md) 与
  [`../data/results/gate_v1_fresh16_claim_serialization_repair_v1/`](../data/results/gate_v1_fresh16_claim_serialization_repair_v1/)，
  A=`f0dd53b…0ed1`、B=`ce523ff…a634`；
- repair planned private HF dataset：
  `gavinlaw/causalcache-gate-v1-fresh16-inventory-repair-mobile`，tag
  `gate-v1-fresh16-inventory-repair-v1`。唯一 attempt 后仍未创建，不能写成已发布 artifact。

## 永久执行结果

唯一 repair v1 formal attempt 于 `2026-07-18T04:27:04Z` 启动、exit `1`。它已完成 16 个 trajectory 与
80 个 OCR semantic decode、80 selected-image payload、48 feature states、144 candidate features、10 个 frozen
checkpoint loads、2 个 policy workers 与 97 次 vision forward。随后
`pretty_json_bytes(claim.claim)` 因 `claim.claim` 为 `mappingproxy` 抛出
`TypeError: Object of type mappingproxy is not JSON serializable`。失败发生在独立 heuristic local seal 与 ordinal
`0..7` receipts 已持久化之后、`label-access-claim` 写入之前。

因此 fresh label semantic decode、label access claim、primary report、HF mutation、旧 dev-5、confirm、
matched-NLL 与 closed-loop operation 全为 0；本 attempt 没有合法 GO/NO-GO。Hyper00 原始 mode-0600 evidence
保留 8 条 ordered receipts、独立 heuristic seal、88-file / 51,508,707-byte artifact tree、run log/start/exit 与
Docker receipt。失败执行器冻结/报告的 artifact canonical inventory SHA256 为
`5443db6df16234c29b32501bc9f766670d428141fb02c4a665be936e1ca2587c`；其输入是按 relative POSIX path 排序的
`{path, mode=stat.S_IMODE, size_bytes, sha256}` records 的 compact/sort-keys UTF-8 JSON。完整逐文件 binding 见
上述轻量 result。
原 v1 与 repair v1 的 planned HF repositories 都不存在，remote mutation 为 0。

本协议以下 source/runtime argv 仅作为已消费 A/B 的历史冻结形状，**不得再次调用 `run` 或 `validate`**。
独立 versioned claim-serialization repair 绑定了本 attempt 的 receipts、seal、artifact、logs、runtime 与 successor
absence，并使用全新的 state/artifact/runtime-receipt/HF identities；唯一代码语义变化是 JSON-safe deep
snapshot。它已完成唯一 B、formal run 与 immutable validate；本 inventory-repair v1 的失败状态和 planned HF
repo absence 不被追改。

## 精确 15-path 远端树

repair 对固定 revision 要求 exact tree equality：不能缺少路径，也不能多出路径。完整 path-list digest 为
`f881b67d028dfe7ce147df0611213a2b9147b2a1432563180e68ebfaee47a202`。15 个路径分为：

- 4 个 consumed files：
  `derived/restoration-v2-label-expansion-v1/{images-00000-of-00001.tar,manifest.json,ocr-records-00000-of-00001.jsonl,trajectories-00000-of-00001.jsonl}`；
- 2 个 repo base paths：`.gitattributes`、`README.md`；
- 9 个 historical auxiliary paths：
  `derived/restoration-v2-v1/{images-00000-of-00001.tar,manifest.json,ocr-records-00000-of-00001.jsonl,trajectories-00000-of-00001.jsonl}`、
  `golden/real-screen-v1/{images-00000-of-00001.tar,manifest.json,ocr-records-00000-of-00001.jsonl}`、
  `runs/restoration-v2-substrate-screening-v1/artifact_manifest.json` 与
  `runs/restoration-v2-substrate-screening-v1/restoration-v2-substrate-screening-20260715T182823Z.tar.gz`。

remote metadata preflight 验证完整 15-path tree；真正 download 与 byte verification 仍只处理原来的 4 个
consumed files。两个 base paths 与 9 个 historical paths 不下载、不 semantic-decode，也不进入 selector 输入。
因此它们只关闭 immutable revision 的 tree identity，不增加输入 token 或科学数据。

## 不变项与允许变化

contract 的 effective-delta proof 只允许以下 operational fields 变化：

- Source-A lineage 与唯一 repair runner-freeze binding；
- derived input 的 base/auxiliary/full remote-tree metadata；
- local state namespace、artifact directory 与 Docker receipt path；
- planned HF destination identity。

以下对象必须与父协议 canonical JSON 完全相同：`evaluation_contract`、`formal_model_input`、`label_input` 与
`output_contract`。这意味着 exact 16 trajectories / 48 states / 144 candidate features / 448 distances、
`n=2/3/4,B=2`、conditional/independent ensemble、five-seed decisions、GO thresholds、label firewall、9-target
payload + 4-target report chain 与双 H200 49-batch/97-forward schedule 均不变。repair 不重训 gate，不替换
checkpoint，不改 prompt、roster、loss、seed、bootstrap 或 claim boundary。

## 新运行与发布命名空间

repair 不复用任何 v1 success path：

| 项目 | repair identity |
| --- | --- |
| execution namespace | `gate-v1-fresh16-evaluation-inventory-repair-v1` |
| artifact directory | `/data/artifacts/causalcache/gate-v1-fresh16-evaluation-inventory-repair-v1` |
| Docker receipt | `/data/experiments/causalcache/.gate-v1-fresh16-evaluation-inventory-repair-v1.docker-inspect.json` |
| private HF dataset | `gavinlaw/causalcache-gate-v1-fresh16-inventory-repair-mobile` |
| annotated tag | `gate-v1-fresh16-inventory-repair-v1` |

`run` 必须先验证当前 checkout 是 clean pushed repair Execution-B，之后才允许读取 HF token。delegate 给父
scientific runner 之前还必须 read-only 验证旧 v1 两条 receipt、14 个 successor absence、空 artifact
directory、Docker receipt 与 run log/start/exit bytes，并确认旧 HF destination 和新 repair destination 均不存在。
任何 B 身份、旧证据或 namespace 漂移都 fail closed。

创建新的 artifact/state roots 前还会完成三项无 model forward 的 mechanical preflight：精确验证 non-label HF
repo metadata（包括 derived full-15 tree）、hash GUI-Owl 本地 verified projection 的全部 14 files /
17,545,907,171 bytes，以及把 logical `cuda:0/cuda:1` 逐一绑定到两张 `NVIDIA H200` 的 canonical GPU UUID。
新 artifact/state roots 必须同时不存在；通过后的 B、旧 failure/destination absence、remote metadata、本地模型与
GPU 映射摘要会写入第一条 durable `runtime_receipts`。

## Source-A validation（历史冻结记录）

冻结 Source-A 时从仓库根目录运行了 config-only validator；它不能访问 network、导入 PyTorch/Hugging Face、
写文件、读取 fresh semantics 或授权执行：

```bash
PYTHONPATH=code python3 -m scripts.validate_gate_v1_fresh16_inventory_repair_contract \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json
```

Source-A 冻结时 validator 返回
`VALID_SOURCE_ONLY_GATE_V1_FRESH16_INVENTORY_REPAIR_V1`、47-path source inventory，且
`runner_freeze_b_present=false`、`evaluation_executed=false`、`execution_authorized=false`；所有 network/write/
torch/fresh semantic/model/report/HF mutation counts 均为 0。冻结时 focused regression 为 73 passed、8 subtests；
当时在 `code/` 运行：

```bash
cd code
PYTHONPATH=. python3 -m pytest -q \
  tests/test_gate_v1_fresh16_inventory_repair_contract.py \
  tests/test_gate_v1_fresh16_inventory_repair_runner.py \
  tests/test_gate_v1_fresh16_evaluation_runner.py
```

Source-A commit/push 后，必须从 clean canonical checkout 再绑定完整 Git SHA：

```bash
PYTHONPATH=code python3 -m scripts.manage_gate_v1_fresh16_inventory_repair validate-source \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json \
  --source-a-git-commit <FULL_CLEAN_PUSHED_REPAIR_SOURCE_A_SHA>
```

## 唯一 Execution-B 与 formal run（历史冻结形状，已失效）

在 clean pushed A 上机械生成 B：

```bash
PYTHONPATH=code python3 -m scripts.manage_gate_v1_fresh16_inventory_repair materialize-runner-freeze \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json \
  --source-a-git-commit <FULL_CLEAN_PUSHED_REPAIR_SOURCE_A_SHA>
```

B 必须是 A 的 direct single-parent child，且 source-tree diff 只能新增上述 runner-freeze JSON；验证后单独
commit/push。随后在 Hyper host 完成 GPU/disk/container preflight 与 10 秒 idle cleanup，选择精确两张 H200，
在 unprivileged container 中固定 Docker receipt：

```bash
PYTHONPATH=code python3 -m scripts.manage_gate_v1_fresh16_inventory_repair capture-runtime \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json \
  --execution-b-git-commit <FULL_CLEAN_PUSHED_REPAIR_EXECUTION_B_SHA> \
  --container-name <SGLANG_OMNI_TIMESTAMP_CONTAINER> \
  --host-data-root /data02/jaxan
```

同一个 container 内的唯一 `run` 形状为：

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
BLIS_NUM_THREADS=1 TZ=UTC LC_ALL=C.UTF-8 \
PYTHONPATH=code python3 -m scripts.manage_gate_v1_fresh16_inventory_repair run \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json \
  --execution-b-git-commit <FULL_CLEAN_PUSHED_REPAIR_EXECUTION_B_SHA> \
  --hf-token-file /data/.secrets/hf_key.txt \
  --data-root /data \
  --fresh-download-parent /data/tmp/gate-v1-fresh16-inventory-repair \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
  --docker-inspect-receipt /data/experiments/causalcache/.gate-v1-fresh16-evaluation-inventory-repair-v1.docker-inspect.json \
  --device cuda:0 --gpu-uuid <GPU_UUID_0> \
  --device cuda:1 --gpu-uuid <GPU_UUID_1>
```

启动后的 representative steady-state window 中两张卡都应达到仓库要求的至少 80% utilization。正式 run 完成后，
再以相同 clean B、Docker receipt 与 runtime 执行只读 immutable replay；`validate` 不接收 model/device 参数：

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
BLIS_NUM_THREADS=1 TZ=UTC LC_ALL=C.UTF-8 \
PYTHONPATH=code python3 -m scripts.manage_gate_v1_fresh16_inventory_repair validate \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json \
  --execution-b-git-commit <FULL_CLEAN_PUSHED_REPAIR_EXECUTION_B_SHA> \
  --hf-token-file /data/.secrets/hf_key.txt \
  --data-root /data \
  --fresh-download-parent /data/tmp/gate-v1-fresh16-inventory-repair-validate \
  --docker-inspect-receipt /data/experiments/causalcache/.gate-v1-fresh16-evaluation-inventory-repair-v1.docker-inspect.json
```

原设计要求 13-target publication、tag-resolved immutable replay、local completion seal 与轻量 Git result 全部
闭合后才可报告 fresh-16 verdict；repair v1 未到达这些阶段。任何后续 claim-serialization repair 仍不得自动
授权旧 dev-5、confirm、matched-NLL 或 closed-loop，这些需要独立 contract。
