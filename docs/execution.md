# 跨芯片执行与团队交接

## 目标

合作方可以使用不同 GPU，但必须执行同一套实验语义。硬件 adapter 只允许改变吞吐相关参数；policy、
data、preprocessing、decoding、budget、seed 与 gate 均由 Git/HF revision 固定。

## 三层 source of truth

1. Git `main`：代码、config、脚本、论文、轻量 summary、progress 与本执行文档；
2. Hugging Face：全部 reusable datasets、raw data artifacts、model weights、checkpoints、adapters；
3. Mac/Hyper00/Hyper01/Aries 本地盘：cache、staging 和 active run，可随时重建，不是 canonical source。

Mac 上的 HF token 位于 `~/hf_key.txt`。该文件不得复制进仓库、命令日志、result JSON 或 Docker
image；`.gitignore` 也显式拒绝 `hf_key.txt` 与 `.secrets/`。优先在 Mac 已认证的 `hf` CLI 上执行
HF create/upload/tag。远端确需访问 private repo 时，由操作者通过安全通道单独登录，不把 token
作为实验参数或 shell 环境变量记录。

## 执行位置

| 工作 | 默认位置 | 备选 | 原因 |
| --- | --- | --- | --- |
| 文档、Git、LaTeX、单测、轻量数据处理 | Mac | Aries/Taurus | 不占共享 GPU |
| restoration v2 OCR/image materialization | Hyper00 CPU | Aries CPU | exact wheel/model SHA 和单线程 ONNX Runtime；不申请 GPU |
| restoration v2 offline inference、attribution | Hyper00 H200 | Aries A6000 | Hyper01 当前有其他任务；v2 canonical runtime 最终由 execution config 冻结 |
| Gate v1 formal selector training | Hyper00 CPU-only/no-GPU | Mac/Aries CPU | 25K-parameter FP32 full-batch deterministic MLP；不申请 GPU |
| RL、大模型训练或大规模重训练 | B200 | Hyper H200 | 按实际并行度与当次空闲卡分配，非 Taurus/Aries 默认最多 4 卡 |
| AndroidWorld emulator + policy rollout | Aries A6000 | Hyper01 H200（待解锁） | Aries stack 已验证；Hyper01 先解决 Docker root 容量并重做 environment smoke |
| 小模型 smoke、sample-level debug | Aries/Taurus A6000 | Hyper01 | 避免为小任务占用 H200 |

2026-07-14 实测 Hyper01：8×NVIDIA H200（每卡 143,771 MiB）、x86_64、`/dev/kvm` 可用；
`/data01` 约 1.8T 可用，`/data02` 约 975G 可用，而根分区只剩约 7G。Docker root 是
`/var/lib/docker`，当前有 `hongccc/sglang-omni:dev`，但没有约 13.4G 的
`causalcache-androidworld` image。禁止用全局 Docker prune 腾空间；在 Docker root 被管理员迁移或
明确释放足够空间前，Hyper01 只承担已有 image 可完成的 policy/offline 工作，closed-loop MVP 留在
Aries。任何 repo、virtualenv、log、checkpoint 与 cache 都不得写入根分区或容器层。

## 芯片无关的实验逻辑

以下字段发生变化时必须视为新实验 config，不能解释成“换机器”：

- Git commit、HF dataset/model repo 与 exact revision；
- task split、instance records hash、seed、step budget；
- prompt、action grammar、executable equivalence、parser；
- dtype、processor revision、image resize/grid、visual token budget、history packing；
- decoding 参数、teacher validation、attribution sampling、selector threshold；
- benchmark/server image digest 与 reward implementation。

允许作为 host adapter 改变的只有：物理 GPU id、按实际并行度选择的 device mapping、batch/concurrency、emulator worker
数、cache/staging 绝对路径。改变这些参数后，语义输出仍需通过固定 smoke。restoration v2 confirm 的
microbatch size 虽不改变科学 estimand，也必须在 execution config 中于 confirm output 前冻结，不能按
结果或 OOM 选择性变化。

## 每次 GPU job 的 preflight

restoration v2 当前从 Mac 非交互检查 Hyper00：

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no hyper00 '
hostname
uname -m
test -e /dev/kvm && ls -l /dev/kvm
nvidia-smi
df -hT / /data01 /data02
docker ps -a --format "{{.Names}}" | sort
'
```

启动任何 GPU job 前必须额外执行项目约定的 10 秒 continuous-zero idle cleanup，最多选择 2 张
空闲 GPU，并将选中的 id 显式写进 Docker `--gpus` 或 Python `--device cuda:N`。长任务同时启动
GPU utilization monitor；低于 90% 时检查 batching、I/O、ADB 等待或减少 GPU 数，不能无人值守地
低效运行。

## Hyper00 v2 容器约定

默认 image 为 `hongccc/sglang-omni:dev`。下面示例使用 GPU 0；必须用 preflight 的实际选择替换。
科学参数不通过环境变量传递；固定的 cache 环境变量只负责把基础设施写入持久盘。

```bash
ssh hyper00

docker run -itd \
  --shm-size 32g \
  --gpus '"device=0"' \
  -v /data01/cache/huggingface:/root/.cache/huggingface \
  -v /data02/jaxan:/data \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e XDG_CACHE_HOME=/data/.cache \
  -e PIP_CACHE_DIR=/data/.cache/pip \
  -e WANDB_DIR=/data/wandb \
  -e TMPDIR=/data/tmp \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name sglang-omni-jaxan-$(date +%m%d%H%M) \
  hongccc/sglang-omni:dev \
  /bin/zsh
```

容器内统一使用：

```bash
git clone https://github.com/luojiaxuan/CausalCache.git /data/repo
git -C /data/repo fetch origin main
git -C /data/repo checkout --detach <GIT_COMMIT>
test -z "$(git -C /data/repo status --porcelain)"
python3 -m venv /data/.venv/causalcache
/data/.venv/causalcache/bin/python -m pip install -e /data/repo
cd /data/repo
/data/.venv/causalcache/bin/python -m scripts.validate_contract \
  --config code/configs/phase0_contract.json \
  --decision data/fixtures/validated_decision.json
/data/.venv/causalcache/bin/python -m scripts.validate_restoration_v2_contract \
  --config code/configs/causalcache_restoration_v2.json
/data/.venv/causalcache/bin/python -m scripts.validate_restoration_v2_interfaces \
  --contract code/configs/causalcache_restoration_v2.json \
  --action-fixture data/fixtures/gui_owl_v2_action_roundtrip.json \
  --prompt-fixture data/fixtures/restoration_v2_prompt_low_fidelity.json \
  --interface-manifest data/manifests/restoration_v2_interfaces.json
```

上面的 interface 命令只做 CPU prompt/parser/bridge 检查，正常状态明确为
`androidworld_json_action_constructor_validation.status=not_run`。在任何 policy output 前，还要在 pinned AndroidWorld
source 上同时显式传 `--androidworld-source-root <PATH>`、`--androidworld-source-revision <FULL_SHA>`、
`--container-image-digest sha256:<DIGEST>` 与 `--run-git-commit <FULL_SHA>`。validator 会 fail closed 检查
两个 checkout、module origin 与 revision，再确认全部 14 个合法 payload 被真实 `JSONAction` 接受；轻量
constructor preflight summary 随后 commit/push。`JSONAction(**payload)` 仅证明 schema construction，不是
device-side executor；还必须补 executor dispatch/behavioral smoke。此后才允许构建/冻结 derived HF
artifact；同时还必须闭合 exact confirm IDs、exposure ledger、OCR identity、baseline source hashes，并
冻结引用全部 artifact/interface identity 的 execution config。其中 exact IDs/exposure 与 OCR identity 已在
下述 formal runs 闭合；baseline source hashes 也由 `make validate-restoration-v2-baselines` 闭合。完整
derived artifact 已由 2026-07-15 formal runs 闭合；当前只剩 execution config pending。
`docs/restoration_v2.md` 所列八项全部完成
后，才允许执行 GPU substrate screening。

### Restoration v2 GPU reduction 与 microbatch 语义

正式 v2 distance 不得调用历史 `QwenPolicyRuntime.teacher_forced_action_log_probs`
的 CPU full-tensor 返回路径。reference 必须作为 normalized FP32 `[1,T,V]` tensor 留在选定的
CUDA device，对 batch 使用零拷贝 `.expand(B,-1,-1)`；candidate BF16 logits 在 GPU 上转 FP32
`log_softmax`，随后做 vocabulary sum 和 distance-token mean。production 函数返回 GPU 上的
FP32 `[B]` scalar tensor。finite input、normalization、nonnegative per-token KL 与 finite output predicates
全部留在 GPU；invalid example 变为 `NaN` final distance。KL primitive 不做 validation scalar read 或
full-tensor host transfer；调用方只读取最终 distance scalar(s)，遇到 nonfinite 必须判为 invalid。Python
生成的小型 static audit metadata 可序列化，但不属于 device tensor value transfer。runtime 可为 shape/
accounting 搬运 `image_grid_thw` 等小型 metadata，也可解码生成 token；reference、candidate、full logits 与
intermediate KL tensor 不得搬到 CPU。

coalition planner 只接受显式 `microbatch_size=2`，按 exact `(image_count, sequence_length)`
分组，组内按 frozen `input_index` 排序。任何自动 OOM fallback、按运行结果调 batch，或把
不同 shape padding 到同一正式 batch 都是 contract violation。Hyper00 还必须验证：

- batch 1 GPU KL 对独立 float64 CPU oracle；
- batch 2 对两次 batch 1；
- cached reference 的 zero-stride expansion 没有 materialized copy；
- invalid numeric input 只产生 `NaN` final distance，kernel validation host read 为 0；
- `atol=1e-6, rtol=1e-5`、finite output、FP32 output 和同 device 约束全部通过。

GUI-Owl v2 runtime 与 synthetic CUDA audit runner source 已有 Mac unit-test 验证。formal audit 必须在
GPU preflight 后从 pushed clean commit 运行，显式传入
`--device`、`--run-git-commit`、container image digest/id、host alias/hostname 与不存在的 exclusive output
path。该 runner 只使用 synthetic logits，summary 必须保持 `policy_output_generated=false` 与
`restoration_output_generated=false`。

2026-07-15 Hyper00 GPU-2 re-anchor formal compute audit 已按上述契约通过，canonical summary SHA256 为
`dce797694194cd88ac749dcc357c3259c41b69a5bde7a99fc93c675cab2e9dac`；独立 validator 不 import
runner/compute modules，而从 run commit Git blobs 重新验证 source 与关键字段。证据位于
`data/results/restoration_v2_gpu_compute_audit/`；旧 GPU-0 结果保留在 Git history。该结果不是 GUI-Owl
processor/model runtime pass，也不单独授权 policy inference。

### Restoration v2 real processor 与 screening CLI 顺序

当前 GPU-2 canonical 记录从 clean pushed commit
`47062741a950b7c6050a6223b91f4bbae65332e7` 运行
`scripts.audit_gui_owl_v2_processor`。该命令是 CPU processor audit，
不会实例化 model weights 或调用 forward/generate，因此不是 v2 policy output；但会读取完整 model snapshot
做 SHA，并 import pinned Transformers modules 复核 source。正式命令必须显式传 model/cache、Git、host、
container 与 exclusive output path：

```bash
cd /data/worktrees/<clean-causalcache-commit>/code
python3 -m scripts.audit_gui_owl_v2_processor \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --expected-snapshot-manifest configs/gui_owl_1_5_8b_snapshot.json \
  --output-summary /data/tmp/restoration-v2-processor-audit/summary.json \
  --repository-root .. \
  --run-git-commit <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id <FULL_64_HEX_CONTAINER_ID> \
  --container-image-digest sha256:<FULL_IMAGE_DIGEST>
```

summary 必须证明真实 `AutoProcessor` 的 1-image、5-image、nested batch-2、no-padding、token boundary、
pixel target、tensor dtype/shape 与实际 visual grid；顶层与 nested flags 都必须声明只调用
`AutoProcessor.from_pretrained`、model weights 未 materialize、policy forward/generate/output 与 restoration
output 均为 false。pinned Transformers `5.6.0` 会把 `min_pixels/max_pixels` 构造参数保存为
`image_processor.size.shortest_edge/longest_edge`；审计必须验证这一 exact `SizeDict` 表示，不能误要求运行时
仍存在 direct attributes。随后把轻量 summary commit/push，再用其 exact SHA 和实际 geometry 生成
execution config。

canonical summary 位于 `data/results/restoration_v2_processor_audit/summary.json`，SHA256
`69bb8ddb578bcb8019fda05fd3a4a2be600043b9b2d58db052078f5107bbd119`。实际 portrait/landscape grids 为
`[1,152,68]` / `[1,68,152]`，每图 2,584 effective visual tokens；1-image、5-image、nested batch-2 sequence
lengths 分别为 2,943、13,286、2,946。后续 execution config 必须引用这些实测值，不能回写构造 target 2,560
冒充实际 accounting。

GPU-2 execution config 已物化为 `code/configs/restoration_v2_execution_hyper00_v1.json`，SHA256
`819cb973d9211a5a9b3b4d7c109520605e35b07ad008bf125353d33d0bf91ca0`。它绑定 8 项 exact evidence、14 个
source roles、processor summary/actual geometry、Hyper00 runtime、新 H200 UUID、single-device CUDA、固定
microbatch=2 与 no-OOM-fallback；当前 `_validate_execution_config` 已通过。production runner 在 artifact/model
load 前还会用 PyTorch、`nvidia-smi` 与 package metadata 逐字段核对 GPU UUID、visible count、driver、compute
capability、SM count、Python、PyTorch/CUDA/cuDNN 和 Transformers，Torch 与 `nvidia-smi` UUID 必须直接一致。
该 config 本身不授权 policy inference。

正式 GPU preflight 后旧 runtime 的 physical GPU 0 被其他任务占用，10 秒采样选择 physical GPU 2。为避免
共享繁忙 GPU，screening 未启动；新 non-privileged 单卡容器已通过 CUDA compute、独立 validator 与 real
processor audit。re-anchor evidence 位于 `data/results/restoration_v2_runtime_reanchor/`。GPU-2 execution
config 已完成 GPU-2 改绑；readiness manifest 已绑定 clean implementation commit
`14faaa44cf1b2044b1f1bcb3c9dcfce36eb452aa` 并恢复 `SCREENING_ALLOWED` schema。旧 authorization 只作历史
记录，不用于启动新 runtime。manifest commit `caa4f376026d13acd21db1f88b893bc92dd482b1` push 后，正式
CLI 在 clean `HEAD == origin/main` 返回 `SCREENING_ALLOWED + CONFIRM_LOCKED`；canonical summary SHA256 为
`36bd183f9fb4e0c3440d9ca8bc5c01e97cd1d25ca0c42e391961026fb8bcaa02`，且仍记录零 policy/restoration
output。

readiness manifest commit/push 并通过 `scripts.validate_restoration_v2_readiness` 后，production screening 才能
运行。CLI 的固定顺序是：CPU readiness 8/8 + confirm lock → canonical Git input/hash binding → derived
artifact/selection witness validation → runtime import/model load → 全部 90 prompts processor-only shape sweep →
首个 policy output。任一前置失败都不得触发后续阶段。正式 screening 参数全部显式传入：

```bash
cd /data/worktrees/<clean-readiness-commit>/code
python3 -m scripts.run_restoration_v2_substrate_screening \
  --repository-root .. \
  --execution-config code/configs/restoration_v2_execution_hyper00_v1.json \
  --readiness-manifest data/manifests/restoration_v2_readiness.json \
  --derived-artifact-root /data/artifacts/<immutable-derived-materialization> \
  --scientific-config code/configs/causalcache_restoration_v2.json \
  --selection-manifest data/manifests/restoration_v2_selection.json \
  --ocr-backend-config code/configs/restoration_v2_ocr_backend.json \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --device cuda:0 \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id <FULL_64_HEX_CONTAINER_ID> \
  --container-image-digest sha256:<FULL_IMAGE_DIGEST> \
  --output-dir /data/tmp/restoration-v2-substrate-screening
```

每个 state 的 attempt marker 在首次 policy call 前 exclusive-create。`--resume` 只允许复用已有 terminal
record；若 marker 存在但 terminal record 不存在，必须将该 run 视为 interrupted/invalid，禁止 hidden retry、
top-up 或替换 state。production security contract 以该 CLI 为准；测试中的 dependency-injection hooks 不能
作为正式入口。

### Restoration v2.1 full-45 执行顺序

v2.1 full-45 使用独立 contract、runner、root、ledger 和 HF repo，不能续写 fixed-15 或旧 v2 screening。
正式 Hyper00 顺序固定为：fresh immutable fixed-15/processor evidence byte validation → clean pushed `main` 与
formal source inventory validation → derived artifact/exact 45-state projection validation → live host/container/GPU
identity validation → durable ledger/root/run manifest claim → policy runtime import/model construction → 45-state
execution → deterministic USTAR → private HF immutable upload/fresh-download → Git manifest/committed-binding
validation。完整 argv 和 canonical paths 见 `code/README.md`，科学 gate 与 resume 语义见
`docs/restoration_v2_1_full_45.md`。

sibling ledger 是 root 外的 durable high-water journal，不只是一次性 claim marker。每次 state attempt 与 terminal
落盘都经过 atomic replace、file fsync 和 parent-directory fsync；resume 会把 journal high-water 与 state/attempt
inventory exact 对齐。两边任何缺失或超前都进入 canonical `INVALID`，不能通过删除成对文件绕开 no-retry。

该正式 job 只使用 container `cuda:0`，对应 Hyper00 host physical GPU 2。GPU job 前仍须重新执行
host/GPU/disk/container preflight 与至少 10 秒 idle sampling，并确认 canonical root、ledger、archive 均不存在；
长任务同时启动 utilization monitor。fixed batch-1 protocol 不能为了提高 utilization 改写 scientific schedule，
但低利用率窗口必须立即检查并记录原因。

2026-07-15 的 constructor preflight 已在 Aries 对 14/14 payload 通过，exact evidence 见
`data/results/restoration_v2_constructor_preflight/`。冻结 interface manifest 保留 run 前 `pending`，实际
状态由该 result summary 更新。

同日 executor dispatch formal attempt `rv2-20260715T101814Z-53016a40` 已在 Aries 对 14/14 cases 通过，
negative actuation control 为 HTTP 500，独立 reducer verdict 为 `PASSED_EXECUTOR_DISPATCH`。完整四件套见
`data/results/restoration_v2_executor_dispatch/`；该结果闭合第 4 项 action dependency，但不替代其余七项
pre-output dependencies。

exact-ID/exposure 是纯 CPU 数据步骤。Hyper00 canonical run 已从 pushed
`main@30879c09e896a61929c66935f98c21e6c3fc7ff5` 的 clean detached worktree 正式运行，未占用
GPU。正式命令显式传入 source root、parent manifest 与 Git revision，两个输出路径在执行前不存在：

```bash
python3 -m scripts.materialize_restoration_v2_selection \
  --v2-contract code/configs/causalcache_restoration_v2.json \
  --v1-config code/configs/independent_reference_gate_v1.json \
  --source-file-manifest data/manifests/independent_reference_gate_v1_source_files.json \
  --source-root /data/source/guiodyssey-independent-v1 \
  --parent-manifest /data/staging/causalcache-parent/manifest.json \
  --v1-summary data/results/independent_reference_gate_v1/summary.json \
  --git-revision <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --output-selection /data/tmp/restoration-v2-selection.json \
  --output-exposure /data/tmp/restoration-v2-exposure.json
```

runner 对 2.25 GB source files 先做 size/SHA 验证，再读取 raw rows 两遍以重建全 pool 和固定 65 个
state witnesses；parent tar 只含 23 条，没有被当作 confirm source。两个空目录的全量构建字节级一致，并分别通过
独立 validator。canonical selection/exposure SHA256 为 `292c7e52...` / `bc122482...`，证据见
`data/results/restoration_v2_selection/`；dependencies 2/3 已闭合。该步骤未启动 GUI-Owl，也未产生任何
policy/restoration output。

OCR/image backend 是纯 CPU 数据步骤，不触发 GPU cleanup 或 utilization monitor。implementation contract、
exact runtime lock、model SHA 和 synthetic fixture 见 `docs/restoration_v2_ocr.md`。Hyper00 staging 固定为
`/data/.venv/causalcache-ocr-v2` 与 `/data/artifacts/causalcache-ocr-ppocrv5-mobile-v1`；它们都是可重建的
local cache，不是 canonical artifact。三模型的 canonical artifact 已固定为 private HF model
`gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en@0dbc766a73ee88d10d52285d434dbfec58617835`，并已从该
revision fresh re-download 验证 6/6 files。正式 synthetic golden 必须从已 push 的 clean detached commit 在两个
独立 Python process 中各写入一个原先不存在的 output path，再比较 canonical JSON bytes。命令显式传入
backend config、model dir、fixture 和 output，不通过环境变量覆盖任何 OCR 参数：

```bash
cd /data/repo/code
/data/.venv/causalcache-ocr-v2/bin/python -m scripts.validate_restoration_v2_ocr_backend \
  inspect-golden \
  --backend-config configs/restoration_v2_ocr_backend.json \
  --model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 \
  --wheel-dir /data/tmp/causalcache-ocr-v2-wheels \
  --fixture ../data/fixtures/restoration_v2_ocr_golden.json \
  --output /data/tmp/causalcache-ocr-v2-golden/repeat-1.json
```

可从仓库根目录运行 `make validate-restoration-v2-ocr-artifact`，离线复核 Git source、HF
repo/revision/file inventory、fresh re-download 标记与 synthetic summary。模型上传和 synthetic golden 已
通过；label/development 中按冻结规则 policy-blind 选出的 real-screen behavioral golden 也已完成。confirm
screenshot 未参与挑选或 backend 调整，完成态 manifest 已回写，dependency 5 现为 passed。

real-screen source contract 已在 output 前冻结，source-only validation 不运行 OCR：

```bash
cd /data/repo/code
/data/.venv/causalcache-ocr-v2/bin/python -m scripts.validate_restoration_v2_real_screen \
  source \
  --source-contract ../data/manifests/restoration_v2_real_screen_source.json \
  --repository-root ..
```

正式 materialization 必须从新的 pushed `main` clean checkout 运行；`--output-dir` 必须原先不存在。以下命令
分别用 `repeat-1` 和 `repeat-2` 两个新目录启动两个独立 Python process，除 output path 外 argv 完全相同：

```bash
cd /data/repo/code
/data/.venv/causalcache-ocr-v2/bin/python -m scripts.materialize_restoration_v2_real_screen \
  --v2-contract configs/causalcache_restoration_v2.json \
  --v1-config configs/independent_reference_gate_v1.json \
  --source-file-manifest ../data/manifests/independent_reference_gate_v1_source_files.json \
  --source-root /data/source/guiodyssey-independent-v1 \
  --selection-manifest ../data/manifests/restoration_v2_selection.json \
  --exposure-manifest ../data/manifests/restoration_v2_exposure.json \
  --backend-config configs/restoration_v2_ocr_backend.json \
  --source-contract ../data/manifests/restoration_v2_real_screen_source.json \
  --model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 \
  --wheel-dir /data/tmp/causalcache-ocr-v2-wheels \
  --git-revision <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --output-dir /data/tmp/restoration-v2-real-screen-<SHORT_SHA>-repeat-1
```

每个输出都必须用 artifact mode 从原始 Parquet 独立复算并重放 6 次 OCR：

```bash
/data/.venv/causalcache-ocr-v2/bin/python -m scripts.validate_restoration_v2_real_screen \
  artifact \
  --source-contract ../data/manifests/restoration_v2_real_screen_source.json \
  --repository-root .. \
  --output-dir /data/tmp/restoration-v2-real-screen-<SHORT_SHA>-repeat-1 \
  --git-revision <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --source-root /data/source/guiodyssey-independent-v1 \
  --v2-contract configs/causalcache_restoration_v2.json \
  --v1-config configs/independent_reference_gate_v1.json \
  --source-file-manifest ../data/manifests/independent_reference_gate_v1_source_files.json \
  --selection-manifest ../data/manifests/restoration_v2_selection.json \
  --exposure-manifest ../data/manifests/restoration_v2_exposure.json \
  --backend-config configs/restoration_v2_ocr_backend.json \
  --model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 \
  --wheel-dir /data/tmp/causalcache-ocr-v2-wheels
```

两次 validator 的 5 个 file SHA 与 `artifact_tree_sha256` 必须一致。只上传其中一份 exact 5-file tree 到
private dataset `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`；`.gitattributes`/`README.md` 位于 repo
root，tar/JSONL/manifest 位于 `golden/real-screen-v1` prefix。tag 后按 full immutable revision fresh
snapshot-download，再在 snapshot root 上运行同一 artifact validator。HF
revision、tag、完整 file hashes、两次 UTC brackets、runtime/container identity 与 re-download evidence 回写
Git 完成态 manifest 后，才能把 dependency 5 标为 passed。本步骤 CPU-only，不触发 GPU cleanup/monitor。

上述 formal flow 已在 pushed `main@dcc6e217b4885cef5f745d987a1ec74e57109717` 完成：两次
materialization + replay 的 5-file tree 均为 `605d6396...7e25`；private HF tag
`ocr-real-screen-golden-v1.0.0` 解析到 `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`，fresh immutable
re-download 后第三次 replay 通过。完整证据见
`data/results/restoration_v2_ocr_backend/real_screen_summary.json`。这只闭合 dependency 5；baseline dependency
另由 `data/manifests/restoration_v2_baselines.json` 闭合。完整 derived artifact 现已闭合；在该里程碑当时，
execution config 仍阻止 GUI-Owl policy output。

### 完整 GUIOdyssey derived artifact

正式 derived builder 必须在 `HEAD == origin/main == --git-revision` 的 clean Hyper00 checkout 中运行。
这是 CPU-only 数据构建，不使用 GPU；每次先重验 2.25 GB 的 16 个 Parquet、selection/exposure 与 OCR
artifact source，再直接生成并 replay exact 210 条 OCR。不得传入外部 OCR JSONL。第一次 run 的显式命令为：

```bash
cd /data/CausalCache/code
/data/.venv/causalcache-ocr-v2/bin/python -m scripts.build_guiodyssey_restoration_v2 \
  --v2-contract configs/causalcache_restoration_v2.json \
  --v1-config configs/independent_reference_gate_v1.json \
  --source-file-manifest ../data/manifests/independent_reference_gate_v1_source_files.json \
  --source-root /data/source/guiodyssey-independent-v1 \
  --selection-manifest ../data/manifests/restoration_v2_selection.json \
  --exposure-manifest ../data/manifests/restoration_v2_exposure.json \
  --ocr-backend-config configs/restoration_v2_ocr_backend.json \
  --ocr-backend-manifest ../data/manifests/restoration_v2_ocr_backend.json \
  --model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 \
  --wheel-dir /data/tmp/causalcache-ocr-v2-wheels \
  --git-revision <FULL_PUSHED_MAIN_SHA> \
  --output-dir /data/tmp/causalcache-restoration-v2-derived-<SHORT_SHA>-repeat-1
```

repeat-2 必须使用新的空 output directory，两个 artifact tree、OCR aggregate 和逐文件 bytes 必须完全一致。
上传 private HF dataset `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile` 后创建稳定 tag
`restoration-v2-derived-v1.0.0`。由于同一 repo 保留既有 `golden/real-screen-v1` prefix，fresh immutable
download 必须只投影 `.gitattributes`、`README.md` 与 `derived/restoration-v2-v1/**` 到 clean root；不能把
HF cache metadata 或 golden prefix 混入 exact 6-file validator root。随后执行：

```bash
cd /data/CausalCache/code
/data/.venv/causalcache-ocr-v2/bin/python -m scripts.validate_guiodyssey_restoration_v2 \
  --output-dir /data/tmp/causalcache-restoration-v2-derived-<SHORT_SHA>-hf-redownload \
  --v2-contract configs/causalcache_restoration_v2.json \
  --v1-config configs/independent_reference_gate_v1.json \
  --source-file-manifest ../data/manifests/independent_reference_gate_v1_source_files.json \
  --selection-manifest ../data/manifests/restoration_v2_selection.json \
  --exposure-manifest ../data/manifests/restoration_v2_exposure.json \
  --ocr-backend-config configs/restoration_v2_ocr_backend.json \
  --ocr-backend-manifest ../data/manifests/restoration_v2_ocr_backend.json \
  --model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 \
  --wheel-dir /data/tmp/causalcache-ocr-v2-wheels \
  --git-revision <FULL_PUSHED_MAIN_SHA>
```

只有 tag resolution、immutable download file hashes、第三次 210-record OCR replay、UTC brackets、完整 argv、
host/container/runtime 与 negative declarations 全部写入 Git completion manifest 并 push 后，dependency 1
才可标为 passed。本步骤仍不生成 policy/restoration output。

2026-07-15 完成证据：builder commit
`1a01f2323647d092cab67f0531ecb877a4a255de` 在 Hyper00 CPU-only runtime 两次构建的 UTC brackets 为
`14:26:37.779--14:47:57.486Z` 和 `14:51:24.170--15:12:43.238Z`；两次均得到
35 trajectories / 175 events / 65 states / 210 images / 210 OCR records，exact 6-file bytes 相同。artifact
tree SHA256 为 `475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e`，OCR aggregate 为
`1e04ddbdd2fd6e5fc50206c436f512908b269fc95566476c799a075eb9af7010`。

private HF tag `restoration-v2-derived-v1.0.0` 解析到 immutable revision
`89f136abaff797e14fe758a198996e51032a10a6`；该里程碑当时 repo `main` 为 9 files，旧
`ocr-real-screen-golden-v1.0.0@9ebbbbbc4666e8a065f4ecb5240491c70f05e21b` 保持不变。第一次
`hf download` preflight 因同时传入 `--cache-dir` 与 `--local-dir` 被 CLI 在下载前拒绝，零文件落盘，属于
superseded preflight failure；正式重试必须使用新的空目录并只投影 exact 6 files。fresh immutable projection
随后在 Hyper00 于 UTC `15:20:56.587--15:31:30.534Z` 完成第三次 210-record replay，tree/OCR aggregate
一致。三次均未加载 policy 或生成 policy/restoration output；dependency 1 已 passed，轻量证据见
`data/results/restoration_v2_derived_artifact/`。该里程碑结束时第 8 项 execution config 仍 pending，policy
inference 当时继续 locked。

executor dispatch 必须按 `docs/restoration_v2_executor_dispatch.md` 先运行 host-side live Docker inspection，
再在 exact pushed `main` checkout 运行 14-case dispatch 与 negative actuation control，最后用独立 reducer 从
raw records 重算。不得把此前的手工 transport probe、constructor summary 或旧 environment smoke 冒充
本次 formal evidence。

每次 formal run 使用显式 unique attempt ID，所有已产生文件均 exclusive-create；失败 attempt 必须在新
attempt 前连同错误回写 Git，且不能伪造无法通过 reducer 的 canonical package。passing attempt 的
before/raw/after/canonical 四件套全部保留；post-inspection 必须确认 dispatch 后仍是同一
container/image/port/source，canonical package 内嵌两次 inspection raw JSON，并从 action/reset/health
等 compact HTTP raw body 重算 length、SHA256 与 decoded response。大体积 screenshot pixels 不嵌入 Git，
只保留 frozen runner 计算的 transport SHA256 与 1080x2400 shape，因此 shape check 依赖该 runner source。

正式 run 必须 checkout 已 push 的 exact commit 并保持 clean detached worktree；开发阶段的同一 topic
可以复用 `/data/repo` 后回到 `main` 执行 `git pull --ff-only origin main`，不要重复 clone 到多个散乱
目录。当前 validated AndroidWorld 路径是在 Aries 同一 host 上运行 emulator HTTP containers 与本地
policy runtime；尚未验证“Aries emulator + Hyper01 policy”的拆机拓扑。若以后实现 policy RPC 或
受控 tunnel，必须作为独立里程碑测试并记录，不得把它当成现有 runner 已支持。worker 数可以因机器
改变，但 validation plan、instance denominator 与失败处理不能改变。

62-instance validation 的命令必须显式传入
`--early-stop-when-success-is-mathematically-impossible`。每个 episode 作为原子 checkpoint 写盘后，runner
用固定 62 分母计算 `observed_success + unobserved_count`；只有该上界严格小于达到
50% 所需的 31 个 success 时才停止分配新 episode。已在途的 worker 必须完成 score 和
tear-down，所以最终 checkpoint 可比首次触发边界多最多 `worker_count-1` 条。非 resume 运行
必须使用空 output directory；resume 前必须校验 plan index、instance 与 filename。
每条 episode 还必须带 `run_contract`，固定 Git commit、model snapshot identity、runtime/processor、
visual preprocessing、generation、plan hash 和 server digest。resume 只能接受完整 contract 相同的
checkpoint，防止 Think 误复用 Instruct 的旧 episode。CLI 传入的 `--run-git-commit` 必须是
full SHA，且 runner 会校验它等于 clean checkout 的实际 HEAD。

## Aries/Taurus fallback adapter

Aries 是当前 AndroidWorld closed-loop MVP 的 validated host。每次运行仍需重新检查所有本地盘，不能
把上次使用的 `/mnt/data6/jiaxuanluo/causalcache` 当作永久最佳路径：

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no aries '
hostname
uname -m
test -e /dev/kvm && ls -l /dev/kvm
nvidia-smi
df -hT / /mnt/data /mnt/data2 /mnt/data3 /mnt/data4 /mnt/data5 /mnt/data6 /mnt/data7
docker ps -a --format "{{.Names}}" | sort
'
```

根据当次 `df -hT` 选择一个 local `/mnt/data*` personal directory，显式挂载为容器 `/data`；HF cache
也选择有空间的 local disk，不能写 Aries 根盘或 Taurus/Aries cross-mount 做重 I/O。Taurus 使用相同
原则，适合小模型 smoke、数据处理和 sample-level evaluation。GPU 数量不设固定默认：按 workload
实际并行度与当次空闲卡选择；非 Taurus/Aries host 单任务最多 4 卡（除非用户显式授权），
Taurus/Aries 可用尽可能多的有效空闲卡。GPU job 只在 warmup 和代表性 startup steady-state 窗口验证
每张已分配 GPU 持续利用率至少 80%；该窗口通过后不要求持续监控。

AndroidWorld emulator container 与 policy container 的权限需求不同：emulator 需要 KVM/privileged
设备访问，policy container 不应因此继承 `--privileged`。policy container 必须使用 Docker 的显式
`--gpus '"device=<physical-id>"'` 绑定，并在加载模型前同时确认 `nvidia-smi` 只列出一张卡、
`torch.cuda.device_count() == 1`，以及容器内 `cuda:0` UUID 对应选中的物理 GPU。2026-07-14 的 Aries
preflight 发现 image 自带 `NVIDIA_VISIBLE_DEVICES=all`；在 `--privileged` 下它会绕过预期隔离并看到
8 张卡，因此正式 Think policy container 去掉 `--privileged`，保留单卡 device request。科学参数仍由
CLI 显式传入，不能用 `CUDA_VISIBLE_DEVICES` 代替该设备契约。

同一个 attribution-label dataset 只能指定一种 canonical hardware/runtime；restoration v2 默认在 Hyper00
H200 生成，Aries A6000 只做固定 canary 或 fallback，不能把两种芯片生成的 labels 混进同一 train
split。跨芯片 replication 单独记录和报告。

## HF token 的安全传递

Mac 的 `~/hf_key.txt` 只作为 stdin 输入，不能出现在 argv、环境变量、远端文件或日志。若 remote
container 必须直接上传 private artifact，可使用单次 stdin API 调用；下面命令中的 repo/path 必须
显式替换，输出只保留 immutable HF commit OID：

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no hyper01 \
  "docker exec -i <CONTAINER_NAME> python3 -c 'import sys; from huggingface_hub import HfApi; api = HfApi(token=sys.stdin.readline().strip()); info = api.upload_folder(repo_id=\"gavinlaw/<HF_REPO_ID>\", repo_type=\"dataset\", folder_path=\"/data/experiments/<RUN_ID>\", path_in_repo=\"data\"); print(info.oid)'" \
  < ~/hf_key.txt
```

上传后从 Mac 用已认证的 HF client 验证 repo visibility、文件清单和 revision，再把 exact revision
写回 Git。不得把 token 复制到 `/data` 作为长期 secret。

## 跨芯片一致性 smoke

新 host 或新镜像进入正式实验前，在同一个 committed fixture 上比较：

1. model/data revisions 与 SHA256；
2. processor `image_grid_thw`、effective visual tokens、input token count；
3. finite logits、唯一 action parse、canonical executable action；
4. deterministic decoding 下的 raw output；若浮点 logits 或 natural-language description 有微小差异，
   记录差异，不放宽 tool-call grammar 或 executable equivalence；
5. AndroidWorld reset、execute、score、tear-down 与 server image digest。

只有 smoke 通过后才启动完整 rollout。H200 与 A6000 的 latency、peak memory 可以不同；action、grid、
token budget、task reward 与 gate decision 不应因芯片不同而改变。

GUI-Owl Think 在同一 H200 上的两次 `do_sample=false` 运行已观察到一个 Action description
句点的引号内/外位置差异，tool-call JSON 与 canonical action 不变。因此 executable gate 可按
canonical action 执行；但后续 teacher-forced token distance 不得假定 raw generation byte-identical，必须
固定 action token boundary 并报告 repeat-forward variance。

历史 v1 replacement teacher 已在 Hyper01 完成独立 1/5-image interface smoke。Aries 上不再单独执行并随后
重复某个冻结 validation instance；完整 runner 写出的第一个原子 episode checkpoint 同时作为
closed-loop infrastructure canary。其成功或失败都不能触发 prompt、parser、model、task plan 或 gate
threshold 修改，正式分母始终来自同一个冻结 62-instance plan。

## Run metadata 与回写

每个 material run 至少记录：Git commit、完整命令、host、GPU model/id、driver、CUDA、PyTorch、
Transformers、dtype、Docker image/digest、model/data HF revision、seed、worker 数、起止时间、结果与
failure classification。字段契约见 `code/configs/run_manifest.schema.json`。轻量 summary 写入
`data/results/<run>/`；raw traces/checkpoints 先上传 HF，再把 repo/tag/revision 写回 README、manifest
与 `docs/progress.md`。
restoration v2 还必须记录 scientific/execution config SHA、derived artifact immutable revision、exact
confirm IDs manifest 与 exposure-ledger hash、accessibility/OCR identity、action-interface source hashes、
每图 `image_grid_thw`/effective visual tokens、policy-visible text tokens、microbatch size，以及 batch-1
GPU KL 对 audited CPU KL 的 equivalence result。
没有 RNG 参数的 deterministic runner 必须记录 `seed=null`，不得为了满足 metadata 伪记一个
未实际设置的 seed。

AndroidWorld validation 的本地 `episodes/*.json` 是 resumable checkpoints，不是 HF 上传布局。run
完成或产生数学确定的 early-stop summary 后，使用
`scripts.package_androidworld_validation` 对 summary、plan、run contract 与 episode 集合做 fail-closed
复核，并生成单个 deterministic gzip JSONL shard。相同输入重复打包必须得到相同 SHA256；HF stable
repo 内按 `data/<policy-slug>/` 和 `runs/<policy-slug>/` 隔离 policy 版本，旧 tag 不覆盖。上传 payload
得到 immutable OID 后再创建 tag；HF payload manifest 不写自己的 OID，最终 OID 只回写 Git 中的
dataset/run manifests，避免 revision 自引用。

一个里程碑的完成顺序固定为：

```text
preflight → run/smoke → verify → upload reusable artifacts to HF
→ update README/docs/data summary → test → commit → push main
```

任何一步尚未上传或 push，都必须在 README/docs 标记为 local staging，不能口头视为完成。

## Spatial reference audit v1 执行边界

该审计计划在 Hyper01 单张 H200 上运行；只有 live preflight 确认资源后才能把 host 写成正式结果。GPU job 前必须
重跑 host/GPU/disk/container preflight 和至少 10 秒 continuous-zero cleanup；三个 profile 必须使用同一
container、同一 visible physical GPU 和同一 clean pushed source commit。运行中启动 utilization monitor；由于
protocol 固定 batch-1，低利用率窗口需要记录 model load、CPU processor 或 teacher/generation 的归因，不能为
提高利用率改变 scientific schedule。

父 v2.1 raw archive 与 derived artifact 必须先从其 immutable HF revision materialize 到 `/data/artifacts` 并逐
hash 验证。唯一 canonical output root 是 `/data/experiments/causalcache/spatial-reference-audit-v1`，sibling ledger
是 `/data/experiments/causalcache/.spatial-reference-audit-v1.attempt.json`。profile 和 state 都在首个 forward 前
durable claim；incomplete attempt、删除 root/ledger、换路径或重跑均不能用于继续同一 v1。

不可重试 ledger 创建前还必须通过 exact runtime preflight：image digest
`sha256:6a8f60af...d349acfa`、Python `3.12.3`、PyTorch `2.11.0+cu130`、CUDA `13.0`、cuDNN
`91900`、Transformers `5.6.0`、driver `570.172.08`。runner 记录完整 live 值；CLI image digest 只要与 config
不同就必须在 durable claim 前拒绝。固定 scientific environment audit 名单必须全部 absent，允许的 cache/device
routing variables 不受影响。auto attention 必须实际解析为 non-eager；eager profiles 所有 non-null observed
implementation 必须为 eager。

validation 完成后，唯一 raw archive 是
`/data/experiments/causalcache/spatial-reference-audit-v1.tar`。必须调用
`scripts.package_spatial_reference_audit_v1` 并显式传入 repo/config/source commit/root/ledger/output；packager
只接受 terminal ledger，以 exclusive-create 写 deterministic USTAR，拒绝 symlink/non-regular member，并在写前、
写后重建 archive bytes。已有 archive 不能覆盖或删除后重包。

本次唯一 attempt 的原 validator 已在 summary 前 fail closed，profile/state 不得重跑。纯离线 repair 必须先从
新的 clean pushed `main` 执行，且只能使用下面的 canonical config/path：

```bash
cd /data/CausalCache/code
python3 -m scripts.validate_spatial_reference_audit_v1_validation_repair_v1 \
  --repository-root /data/CausalCache \
  --config /data/CausalCache/code/configs/spatial_reference_audit_v1_validation_repair_v1.json
```

repair 会在任何写入前重新绑定 pre-repair root/ledger、三个 terminal、原 formal log/exit、旧 source/config 与
parent v2.1 USTAR；它不加载 processor/model，不调用 GPU。只有 dynamic grid accounting、parent shape witness、
teacher aligned-input inventory 和原 validator 的其余全链全部通过后，才 exclusive-create canonical
`summary.json`。已有 summary 或 archive 时必须拒绝，不复制或改写原 formal log/exit。随后仍使用上述旧
packager，`--source-git-commit` 必须传产生 raw 的 `c093bd8f92ab97427acb427bd2d66fb6b20b556a`，不能传
repair commit；repair commit 由 summary 内部单独记录。

2026-07-15 该流程已从 clean `main@a2528d7e95e73c25639568650f63abce58e4e491` 一次完成。repair
summary SHA256 为 `1d6dc90c602a3cb97ff58da8fafe03eefe31b19d1673084a132cc328c9229bf4`，scientific payload
SHA256 为 `2d73e395d5ee91349adf904b344b7ae25569e79ce8bfa35be9df7902ab8dc79e`。冻结 packager 产生
1,873,920-byte / 72-member USTAR，SHA256
`d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc`。private HF immutable revision
`d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1` 已从全新 cache 下载并完成 byte identity、canonical rebuild
与 exact file allowlist 检查；tag `spatial-reference-audit-v1` 解析到同一 revision。

严格 CUDA deterministic GEMM 需要 `CUBLAS_WORKSPACE_CONFIG`，与本项目“不用 environment variable 传
scientific 参数”的规则冲突。因此 v1 不启用 `torch.use_deterministic_algorithms(True)`，只比较 legacy auto 与
eager fixed-seed/TF32-off numerical control；文档和结果不得把后者写成数学 deterministic。confirm input/output、
restoration coalition、baseline、selector 和 gate training 都不属于本 job。

聚合判定不能只看 eager：eager 不稳定进入 semantic；eager 稳定而 auto 不稳定才是 eager-specific recovery；若
两者都稳定，只能报告本次两个 backend 都稳定且 numerical attribution inconclusive。FP32 与 margin 不改变分支。

## Restoration v2.2-eager fresh-45 执行边界

本阶段的 source freeze 与唯一正式执行均已闭合，完整 contract 见 `docs/restoration_v2_2_eager.md` 和
`code/configs/causalcache_restoration_v2_2_eager.json`。spatial audit 的正式父结论是
`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`，只授权冻结 eager runtime；v2.1 的
`NO_GO_V2_1_FULL_45_SUBSTRATE` 不变，confirm 仍 locked。formal source freeze 已在 clean pushed
`main@b3a6303d69b1145fbf195e0bd18b9b3065a6f213` 建立；config SHA256 为
`f473bb8a1657072235dd73bf78a93aff27b438d7baca65b7ef6096cf985effa7`，50-file inventory SHA256 为
`bb6351e4dd5b4470ed1add86dbcbaa714a56063ba8ecf07268b6f13f630f9e54`。

正式 run 前按本文件通用 GPU preflight 检查同一 Hyper host 的两张 H200、磁盘、container 与至少 10 秒 idle
window，并启动 utilization monitor。拓扑不能根据 live 速度动态改变：同一 container 中 logical `cuda:0` 的
`even` worker 固定 23 个 even-index states，logical `cuda:1` 的 `odd` worker 固定 22 个 odd-index states；
禁止 state stealing、跨 host 和跨 container。两个 worker 的 scientific runtime metadata 与 image digest 必须
一致，只允许 device/UUID/PCI/logical index 不同。

coordinator 在任一 worker import policy runtime 或构造 model 前，必须先 fresh-download 并验证 parent spatial、
v2.1 full-45、pilot、processor 与 derived evidence，再 exclusive-create global sibling ledger
`/data/experiments/causalcache/.restoration-v2-2-eager-full-45-substrate-v1.attempt.json`，统一 claim canonical root
`/data/experiments/causalcache/restoration-v2-2-eager-full-45-substrate-v1`。该 ledger/root 与 v2.1 完全独立；旧 raw、
state records、aggregate、terminal 或 ledger 不能复制、resume 或计入 fresh 45-state denominator。任一 worker
失败、parity inventory 重叠/缺失或 marker 无 terminal 都使整个 attempt `INVALID`。

v2.2 双 worker lifecycle 明确不支持 `--resume`。任一进程、coordinator、model construction 或主机中断后，
canonical attempt 只能封存为 `INVALID`；即使已有完整 terminal prefix，也不能由新 invocation 跳过后续补跑。
两个 worker 的独立 sibling high-water ledger 必须在 global claim 时预绑定，并在每个正式 state 前后 durable
更新，防止删除 worker root 或局部 ledger 后伪造“尚未尝试”。

scientific delta 只能是 BF16 eager fixed-seed/TF32-off numerical control 与双 worker transport，不声称 strict
CUDA determinism。official-tool interface、45-state projection、per-state 两次 generation/三次 teacher/two-KL
schedule、gate 和全 attempt 90/135/90 ceilings 均保持 v2.1 不变。confirm、expert、restoration、baseline 与
gate construction/training/selection 的计数必须为 0。

两个 worker 还必须同时 exact 匹配 spatial audit 的 container digest、Python `3.12.3`、PyTorch
`2.11.0+cu130`、CUDA `13.0`、cuDNN `91900`、Transformers `5.6.0`、driver `570.172.08` 和 H200 型号；
只允许 logical device、container-visible NVML index、UUID 与 PCI bus ID 不同。logical device 固定为 0/1；
NVML index 可以是 preflight 选中的任意两个不同非负编号。两份 runtime identity 必须在 coordinator barrier
中先证明 stack 相同且 UUID/PCI/NVML identity 不同，之后才允许首个 state marker 或 generation。任一 stack
或双卡 identity 漂移都必须在 model forward 前使 attempt `INVALID`。

terminal 后才允许从 global ledger、canonical root 与两个 disjoint worker inventories 构建 deterministic USTAR。
唯一 Hyper00 run 已于 `2026-07-16T06:05:09.464587Z`--`06:15:19.889274Z` 完成，45/45 states、
90/135/90 operations、zero retry/top-up，正式为 `PASS_V2_2_EAGER_FULL_45_SUBSTRATE`。102-file USTAR SHA256
为 `b22827e6e2d8d33b03686fc177dc8f9c55c5133470fbe40f9fb3e33cce809fb5`；private HF tag
`v2.2-eager-full-45-substrate-v1` 解析到 immutable revision
`3577099d505b8c652d764f41269df911128ec767`，fresh-download 与 source archive byte-identical。Git 轻量结果见
`data/results/restoration_v2_2_eager_full_45_substrate/`。source-only validator 通过本身不授权 GPU run；本次
execution 的独立 global claim 才授权并记录了正式 policy operations。

若 coordinator 在 aggregate 前硬中断，只能执行 policy-free `seal-interrupted` 将 existing claim、两个 sibling
high-water 与 forensic inventory 封存为 `INVALID`；不得重新调用 production runner。root ledger 缺失或落后
只在它是 sibling prefix 时合法封存，root 超前/non-prefix 仍 fail closed。HF manifest 必须比较 canonical source
archive 与独立 fresh-download path 的完整 bytes，并绑定 exact repo/path/immutable revision。

## Restoration v2.2 exact-label 执行边界

label source 已在 clean pushed `main@3942d687d03bf63ea683fe8ad906a161eb10dc27` 冻结；contract
`code/configs/causalcache_restoration_v2_2_labels.json` 的 SHA256 为
`56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87`，29-file source inventory
SHA256 为 `8d0ecf6b0df75f9fa7753631b8fca5efd98c71a7953007e684393e6e8fe52d9b`。完整回归 575 tests
通过（skipped 11），`make paper` 通过；这些只证明 source freeze，不代表 raw labels 已生成。

正式运行前先按本文件通用流程完成 Hyper host/GPU/disk/container 检查、至少 10 秒 idle cleanup，并只向新
container 暴露两张 preflight 选出的 H200；启动后必须核对 `torch.cuda.device_count()==2` 并运行 utilization
monitor。同机同容器内 `even` worker 固定处理 23 states，`odd` worker 固定处理 22 states，microbatch 固定为
1；禁止 state stealing、动态 batch、跨 host/container、resume、retry 与 top-up。parent v2.2 raw 和 derived
artifact 必须从各自 immutable revision fresh materialize 并逐 byte/hash 验证，不能直接复用旧共享盘 output。

以下 canonical 路径是已经封存的 v1 identity：

```text
output root: /data/experiments/causalcache/restoration-v2-2-eager-labels-v1
global ledger: /data/experiments/causalcache/.restoration-v2-2-eager-labels-v1.attempt.json
raw archive: /data/experiments/causalcache/restoration-v2-2-eager-labels-v1.raw.tar
```

唯一 scientific schedule 是 45 states、420 条 raw `D(S)`、45 个 primary exact-subset oracle、435 条
deployment conditional-marginal labels、465 teacher forwards、420 GPU KL measurements 与 0 generation。
confirm、gate training/selection、matched-NLL 与 closed-loop 的 operation count 必须全部为 0；任一混入、缺失/
重复 coalition、non-finite distance、worker/aggregate failure 或路径/identity drift 都将唯一 attempt 原地封存为
`LABEL_ATTEMPT_INVALID`，不得删除后重跑。

正式 argv 以 `docs/restoration_v2_2_labels.md` 为唯一说明入口。terminal 后才可用
`scripts.manage_restoration_v2_2_label_artifact` 将 root 与 sibling ledgers 打成 deterministic USTAR。planned
private HF target 是 `gavinlaw/causalcache-restoration-labels-mobile`。v1 原计划 tag 为
`v2.2-eager-train-dev-exact-v1`，但 zero-forward invalid attempt 没有创建 repo/tag/artifact；replacement v2 tag
已冻结为 `v2.2-eager-train-dev-exact-v2`。成功 attempt 必须 upload 后取得 immutable revision，再下载到不同路径，重跑 raw reducer并
验证 archive byte identity，最后才能回写 Git compact summary/artifact binding。

### v1 zero-forward failure 与 replacement 前置条件

2026-07-16 的 v1 formal attempt 已 durable claim，但两个 worker 在加载任何 model/runtime 或写 state marker 前发现
`--model-dir` 指向的 immutable snapshot directory 不存在；global ledger 因此以 0 attempted / 0 completed states
封存为 `LABEL_ATTEMPT_INVALID`。teacher forward、KL、raw label、oracle、conditional marginal 与所有 prohibited
work count 均为 0。原 output/root/三个 ledger 必须永久保留，禁止删除后重跑；轻量 binding 位于
`data/results/restoration_v2_2_eager_labels_v1_attempt/`。

replacement 必须使用全新的 attempt/output/ledger/archive/HF tag identity，并在任何 durable claim 前验证：model
root 是非 symlink 的 real directory、basename 精确等于 model repo slug、manifest 要求的 revision/file inventory/size/hash
完整、两 worker 能看到相同路径。shell smoke 必须用 fail-fast 或显式读取 exit code，禁止在失败的 `test` 后无条件
打印 success。只有新 source/config/tests commit 并 push clean `main` 后，replacement formal run 才获授权。

replacement v2 已冻结为：

```text
contract: code/configs/causalcache_restoration_v2_2_labels_v2_repair.json
config SHA256: 7fc96883b905a5f026c18e65c3c7e377972559c775ac7ec95030ea1f04c2f94c
attempt/revision: restoration-v2-2-eager-labels-v2 / v2_preclaim_repair
model projection: /data/artifacts/models/GUI-Owl-1.5-8B-Instruct
output root: /data/experiments/causalcache/restoration-v2-2-eager-labels-v2
global ledger: /data/experiments/causalcache/.restoration-v2-2-eager-labels-v2.attempt.json
raw archive: /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar
HF tag/path: v2.2-eager-train-dev-exact-v2 / raw/v2.2-eager-train-dev-exact-v2.tar
```

pre-claim validator 复用 production snapshot verifier，逐 hash 验证 14 files / 17,545,907,171 bytes、
`.snapshot.json` revision 和 Transformers source；它不构造 model、不初始化 CUDA，也不创建 ledger。v2 formal
run 必须显式传以上 model/output/ledger，不能再使用 HF cache snapshot path。

## Restoration v2.2 expansion exact-label A/B 与执行顺序

expanded substrate 已以 192/192 states、185 memory-sensitive 正式 PASS，raw artifact 固定到 private HF
immutable revision `25ac19cf6ef98adc243d421cd0039ac104ddb539`。后续 label run 使用独立 config
`code/configs/causalcache_restoration_v2_2_expansion_labels_v1.json`，SHA256 为
`65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5`。完整科学/失败边界与最终 CLI 见
[`restoration_v2_2_expansion_exact_labels.md`](restoration_v2_2_expansion_exact_labels.md)。当前新增的是 source
A；没有 expansion label output，source-only validator 也不授权 GPU。

GPU authorization 必须按以下不可合并的 Git 顺序完成：

1. contract、coalition input、parent crosswalk、runner、raw reducer/manager、tests 与 docs 作为 source A commit
   并 push canonical `main`；
2. 在 clean `main@A == origin/main` 上确定性物化
   `code/configs/causalcache_restoration_v2_2_expansion_labels_runner_v1.json`；
3. 只把 runner freeze 作为 execution B commit 并 push；
4. formal host 必须是 clean `main == origin/main == B`，committed validator 重建 A 的 source blobs、证明 A 是
   B 的 ancestor 并验证 freeze bytes 后，才可返回
   `VALID_COMMITTED_PUSHED_EXPANSION_EXACT_LABEL_RUNNER_FREEZE`；
5. A、pending-B materializer 或未 push B 的任何本地状态都不得启动 model/GPU。

唯一 attempt identity 固定为：

```text
attempt: restoration-v2-2-expansion-exact-labels-v1
output root: /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1
global ledger: /data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1.attempt.json
raw archive: /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1.tar
HF repo: gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile
HF tag/path: v2.2-expansion-exact-labels-v1 / raw/v2.2-expansion-exact-labels-v1.tar
```

formal run 前按本文件通用 preflight 检查 Hyper host、两张 H200、disk 与 containers，执行至少 10 秒
continuous-zero cleanup，只把选中 GPU 暴露给新容器并确认 `torch.cuda.device_count()==2`。source-locked monitor
sidecar 必须先 ready，外部 utilization monitor 从 preclaim 前开始并保留完整原始日志。even / odd workers 固定
处理 96 / 96 states；同一 state 的 fresh reference、repeat reference 与全部 coalitions 保持同卡，microbatch=1，
禁止 state stealing、resume、retry、top-up 或根据利用率改变 denominator。

preclaim 必须从不同 fresh paths 验证 substrate archive 与 policy-blind derived artifact；共享盘现有目录只能作为
下载 cache。label schedule 固定 64 trajectories / 192 states、$n=2/3/4$、$B=2$、1,792 raw $D(S)$ rows、
1,984 teacher forwards 与 1,792 KL。full logits、log-probabilities 和 KL intermediates 保持 GPU-only，每个 KL
只把最终 scalar 搬到 host；generation、expert、gate、matched-NLL、closed-loop、confirm、retry 与 top-up 均为
0。raw $D(S)$ 是唯一 canonical truth；1,856 deployment edges、3,072 full edges、1,984 interactions、576
attributions 与 192 oracles 必须在 terminal 后由 policy-free validator 独立重算，所有 negative marginals 保留。

PASS 后才允许 deterministic USTAR、private HF upload/tag、取得 immutable revision并下载到不同 fresh path。
fresh archive 必须与 source archive byte-identical，并再次从 raw table 重算全部 counts；随后 Git 只记录 compact
summary、hash 与 immutable HF binding。该 immutable artifact 已在 repaired-label private dataset 的 resolved
commit `7a6c254b8cec0dd3d8111dfc9c080de357e5cef3` 闭合，因此 formal-58 gate data blocker 已解除；它不自动
授权 matched-NLL、closed-loop 或 confirm，也不代表 gate 已训练。

原 GPU attempt 已永久 INVALID，后续 repaired-label publication 必须使用独立的 CPU/network-only 协议
[`restoration_v2_2_expansion_labels_scientific_repair_publication.md`](restoration_v2_2_expansion_labels_scientific_repair_publication.md)。
publisher 只接受 completed no-GPU child 的 frozen archive/claim/completion；从 clean pushed `main` 以
archive+sidecar exact-pair commit、annotated tag 和 immutable fresh replay 闭合。publication claim/sidecar 不能
解锁 gate。content mutation 前还必须以 mode-0600 remote-base receipt 绑定完整 reachable history 与 recursive
blob tree；pair 只能在该 base 上增加两个 target，不能改动任何既有 blob。fresh replay 后先保留 completion
staging，再以 same-inode hard-link 创建 final completion；该 link 是最后 namespace mutation。只有 final seal 才
清除 formal-58 label-data blocker。该 final seal、幂等 replay 与独立只读 postflight 现已完成，正式记录见
[`../data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/`](../data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/)。
publication 未申请 GPU、未重启 model，也不改变 matched-NLL、closed-loop 与 confirm 的独立锁。首次 formal-58
cache v1 run 已在 feature transport byte validation 处、任何 semantic decode 前 fail-closed：旧 config 的 expansion
trajectories SHA 与 Git-pinned immutable producer evidence 不一致。旧 mode-0600 claim 被保留，cache/HF output 均不
存在，详见 [`../data/results/gate_v1_formal58_cache_v1_attempt/`](../data/results/gate_v1_formal58_cache_v1_attempt/)。
独立 namespace 的 one-leaf transport-repair 已闭合：Source-A=
`4f8c01b026167d6e9429716a082f3abd7c0c1bc9` 的 config SHA256 为
`aaf82fd5e994588bc22f0f745139e349219863eee5fa8cb4cc93c4a9e48e123a`，唯一 direct-child Execution-B=
`f96c197c0fd31bfd299b5ab6e9e1416f6183bc6d` 的 runner SHA256 为
`14141d0d790c4cf4d6c12b3bc336e1e6d2ebd26fde5f4be40b58627bd86a2b34`。Hyper00 no-GPU `run` 返回
`VALID_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`，fresh immutable replay 后的只读 `validate` 返回
`REVALIDATED_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`，没有创建 claim/cache/state，remote mutation count 为 0。
canonical private HF dataset 为
[`gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile)，tag
`gate-v1-formal58-cache-transport-repair-v1` → immutable commit
`a61b31bf2e69be00f94469f4a2f2d6b336fcc386`，annotated-tag object
`c603397b9b1b1c3ad472f49125f827b8a096997d`。cache completion 只让 formal-58 成为单独 preregistered train-only
gate workflow 的 input；没有 gate training、OOF、checkpoint、fresh-16、旧 dev-5、matched-NLL、closed-loop 或
confirm run。详细 execution evidence 见
[`../data/results/gate_v1_formal58_cache_transport_repair_v1/`](../data/results/gate_v1_formal58_cache_transport_repair_v1/)
与 [`gate_v1_formal_cache_transport_repair.md`](gate_v1_formal_cache_transport_repair.md)。

formal-train Source-A 曾在不读 semantic cache、不加载 PyTorch、不创建 local state 与不访问 HF 的
source-only 边界下冻结，config 为
`code/configs/causalcache_gate_v1_formal_train_v1.json`，SHA256
`bff920266b3f691618005b239c8a0aaa99369b45c0612ae628eec1a8f7eebf2f`。它绑定上述 repair cache 的 feature/label/manifest SHA256、tag、
resolved commit、join audit，以及 gate preregistration SHA256
`37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b`。详细 contract 见
[`gate_v1_formal_train.md`](gate_v1_formal_train.md)。

Source-A validator 的固定输出仍是该历史时刻的 `training_executed=false` 与
`execution_authorized=false`。随后唯一 Execution-B=`bad28b74c421ccf6be1ab2f4407f7bad3414a2f2` 已在
CPU-only/no-GPU runtime 完成 formal-58 OOF 与 final fit；10 checkpoints、2 full OOF reports 与 4 manifests
已发布到 private HF model `gavinlaw/causalcache-gate-v1-formal58-selector-mobile`。tag
`gate-v1-formal58-train-v1` 解析到 manifest commit
`23f6786075c7bff91f93fd7e8a878e070efb72a9`，完成态只读 replay remote mutation count 为 0，详见
[`../data/results/gate_v1_formal58_train_v1/`](../data/results/gate_v1_formal58_train_v1/)。整个训练保持
fresh-16、legacy dev-5、confirm-20、matched-NLL 与 closed-loop access 为 0。后续 fresh-16 v1 A/B 已冻结，
但其唯一 formal attempt 在 derived remote-tree preflight 永久 pre-semantic fail closed；不能修改 formal
checkpoints、续跑 v1 或把 combined-21 混入同一 primary execution。

## fresh-16 full-inventory repair 执行边界

父 scientific protocol、v1 failure 与 versioned repair 的完整记录分别见
[`gate_v1_fresh16_evaluation.md`](gate_v1_fresh16_evaluation.md)、
[`../data/results/gate_v1_fresh16_evaluation_v1_attempt/`](../data/results/gate_v1_fresh16_evaluation_v1_attempt/) 和
[`gate_v1_fresh16_inventory_repair.md`](gate_v1_fresh16_inventory_repair.md)。repair config SHA256 为
`5ba1b2d433c01defa0faae92a163dc9a9a916d967218abdc7607bd3724829a44`。

repair 的 remote metadata preflight 必须对 derived revision `630363a6adb692d72774f16dd0653a50216313ff`
验证精确 15-path tree；下载与 byte/semantic consumption 仍只限原 4 个 label-expansion files。父协议的
16/48/144/448 denominator、formal model、10 checkpoints、labels、gate、thresholds、bootstrap、label firewall、
9+4 output target paths 与 49/97 双 H200 schedule全部不变。repair 只允许改变 source lineage、full-tree metadata、
local namespace/runtime receipt 与 planned HF destination。

旧 parent v1 evidence 必须继续保留。inventory-repair Source-A=
`6fb3e868e293bce191ce30a5c6a15ecc544c4591`、唯一 Execution-B=
`c22734ffc85935882f57ddb081c9194d6dae92d0` 已 commit/push 并被唯一 Hyper00 attempt 消费。该 attempt 已通过
retained-v1、full-15 non-label metadata、14-file/17,545,907,171-byte GUI-Owl projection 与两张 H200 UUID
preflight，随后完成 16 trajectory / 80 OCR semantic decode、80 selected images、48 feature states、144
candidates、10 checkpoint loads、2 policy workers 与 97 vision forwards。

执行在 `heuristic_local_seal` 后调用 `pretty_json_bytes(claim.claim)` 时因 `mappingproxy` 不可 JSON serialization
而 exit `1`，精确停在 ordinal `0..7` receipts 之后、`label-access-claim` 写入之前。fresh label semantic/access、
primary report、HF mutation、旧 dev-5、confirm、matched-NLL 与 closed-loop 均为 0，不能报告 GO/NO-GO。
repair v1 的 state/artifact/runtime-receipt/log bytes 必须保留；88-file / 51,508,707-byte artifact tree 的失败
执行器 canonical inventory SHA256 为
`5443db6df16234c29b32501bc9f766670d428141fb02c4a665be936e1ca2587c`。旧与 repair planned HF repositories
均不存在。完整 binding 见
[`../data/results/gate_v1_fresh16_inventory_repair_v1_attempt/`](../data/results/gate_v1_fresh16_inventory_repair_v1_attempt/)。

独立 versioned claim-serialization repair 已冻结 repair v1 receipts/seal/artifact/log/runtime、successor absence，
并由 Source-A=`f0dd53b…0ed1`、唯一 Execution-B=`ce523ff…a634` 在全新 state/artifact/runtime-receipt/HF
identities 上完成 formal run 与 immutable validate。旧 repair v1 的 `run`/`validate` 仍禁止再次调用，也禁止
删除旧 roots 后在同一 namespace 续跑。

## fresh-16 claim-serialization repair 完成态

Source-A protocol 见
[`gate_v1_fresh16_claim_serialization_repair.md`](gate_v1_fresh16_claim_serialization_repair.md)，config 为
`code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json`。本层只允许把
`LabelAccessClaim.claim` 改为 canonical-JSON deep snapshot；formal training 已完成，full-15 inventory、实际消费的
4 files、GUI-Owl projection、10 checkpoints、label bytes、16/48/144/448 geometry、evaluation、output relative
paths 与 GO thresholds 全部不变。

Source-A source-only milestone 保持 network/write/torch/fresh semantic/label/model/report/HF mutation 全 0，且
当时 `evaluation_executed=false`、`execution_authorized=false`、runner-freeze B absent。config SHA256 为
`3979573be235d630ee2f46dc23be8747a843190c9b57ee81e1b3a17b4416d8c7`，57-path source inventory SHA256 为
`572992cf5ac75142cdf8f0e82bdfc2caad43ba37632c741eae277285db54d689`；focused regression 为 99 passed +
7 subtests passed，全仓为 1263 passed、16 skipped、4 deselected、617 subtests passed。4 个 deselect 均为已完成
历史 A/B 的 B-absence lifecycle tests。后续 B 以单独 direct-child commit 生成，source-only 计数不被追改。

新 identity 固定为：

| 项目 | value |
| --- | --- |
| state root | `/data/experiments/causalcache/gate-v1-fresh16-evaluation-claim-serialization-repair-v1` |
| artifact root | `/data/artifacts/causalcache/gate-v1-fresh16-evaluation-claim-serialization-repair-v1` |
| Docker receipt | `/data/experiments/causalcache/.gate-v1-fresh16-evaluation-claim-serialization-repair-v1.docker-inspect.json` |
| private HF dataset | `gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile` |
| immutable tag | `gate-v1-fresh16-claim-serialization-repair-v1` → `3541fe1ea2c46e555c29cc53483e6f3b809f8f81` |

旧 parent/inventory-repair roots、mode-0600 receipts/seal/artifact/log/runtime bytes 必须原样保留；failure evidence
Git commit 固定为 `7fbfe1b8314ea61d7d646a47be902fdf1c6d4af9`。完成态严格执行了 Source-A push → clean A
validation → mechanical direct-child B separate push → Hyper preflight + exact 2 H200 → new runtime receipt → token
前验证 local evidence/new roots → owner/write-role 与三个 repo absence 验证 → unique completed run → immutable
validate。run/validate exit 0，ordinal `0..15`、9+4 publication 和 tag replay 均闭合；结果为 selector/set-conditioning
双 `NO-GO`。当前不得重跑完成态或打开 post-GO stages；下一步只允许新的 read-only failure-decomposition contract。
