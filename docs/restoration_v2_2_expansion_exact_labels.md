# Restoration v2.2 expansion exact labels

> 当前状态：192-state label-expansion substrate 已以
> `PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1` 闭合，private HF immutable revision 为
> `25ac19cf6ef98adc243d421cd0039ac104ddb539`。expanded exact-label contract、coalition input、
> raw-artifact reducer、双 H200 runner 与 artifact manager 已进入 source A；科学 config SHA256 为
> `65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5`。本阶段仍是
> source-only：runner source A commit/push 后，还必须单独物化、commit、push runner freeze B，B 的
> committed validator 通过前不得运行 GPU。当前没有 expansion label、gate checkpoint、matched-NLL、
> closed-loop 或 confirm output。

## 目标与 canonical truth

本阶段只为冻结的 64 条 expansion trajectories 生成完整 coalition distance table。每条 trajectory 固定
decision steps 4、5、6，候选事件数分别为 $n=2,3,4$，部署预算固定为 $B=2$。完整历史策略在 substrate
阶段生成的 canonical action 只作为 teacher-forced token span；label run 不再 generation，也不读取 expert
action。

每个 state 只把完整 power set 上的 raw $D(S)$ 写成科学真值。exact-subset oracle、deployment conditional
marginal、full-hypercube edge、pair interaction 与 exact-permutation attribution 都由 artifact validator 在
policy-free CPU reduction 中重新计算，不能信任 runner 写出的 derived headline。定义保持：

\[
\Delta_j(S)=D(S)-D(S\cup\{j\}),\qquad
I_{ij}(S)=\Delta_i(S\cup\{j\})-\Delta_i(S).
\]

负 marginal、负 utility recovery 与 non-monotonic coalition 必须原样保留，不做 clamp、删样本、top-up 或
replacement。

runtime、model identity 与 GPU operation accounting 属于 **source-locked producer evidence**：artifact validator
只验证其归档结构、source contract 和计数闭合，不声称在打包阶段独立重跑 model/runtime。artifact 阶段可独立
声称的范围只有两项：从 canonical raw $D(S)$ 到全部 derived labels 的 CPU reduction，以及利用 parent substrate
USTAR、derived artifact 与 OCR config 对 192 个 dynamic state/1,792 个 coalition input witness 的 external replay。

## 固定分母与 operation schedule

| 项目 | 固定总数 |
| --- | ---: |
| trajectories / states | 64 / 192 |
| even / odd worker states | 96 / 96 |
| raw $D(S)$ rows | 1,792 |
| deployment-reachable conditional edges，$B=2$ | 1,856 |
| full-hypercube edges | 3,072 |
| pair interactions | 1,984 |
| exact-permutation attribution rows | 576 |
| primary exact-subset oracles | 192 |
| teacher forwards / GPU KL measurements | 1,984 / 1,792 |

1,984 次 teacher forward 精确分解为 192 次 fresh full-history reference、192 次 repeat reference 和
1,600 次 non-full coalition forward。1,792 次 KL 精确分解为 192 次 repeat-reference KL 与 1,600 次
non-full coalition KL；full coalition 的 $D(S)=0$ 由 reference identity 物化，不额外调用 candidate teacher
或 KL。generation、confirm、expert read、gate training/forward/selection、matched-NLL、closed-loop、retry 与
top-up 的计数全部固定为 0。

reference 与 candidate full-vocabulary logits、log-probabilities及 KL intermediate 始终留在同一 CUDA device。
每次 measurement 只允许把最终 distance scalar 搬到 host；full tensor host transfer 固定为 0。每个 state 的
reference、repeat 与全部 coalitions 必须由同一 worker/device 完成，禁止 state stealing、跨卡 reference cache
或动态 microbatch；microbatch 固定为 1。

## 冻结输入与隔离边界

科学 config 是
[`../code/configs/causalcache_restoration_v2_2_expansion_labels_v1.json`](../code/configs/causalcache_restoration_v2_2_expansion_labels_v1.json)，
其 SHA256 为
`65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5`。它绑定：

- substrate raw archive：private HF
  `gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile`，tag
  `v2.2-label-expansion-substrate-v1`，immutable revision
  `25ac19cf6ef98adc243d421cd0039ac104ddb539`，archive SHA256
  `4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d`；
- policy-blind expansion derived artifact：private HF
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@630363a6adb692d72774f16dd0653a50216313ff`；
- frozen GUI-Owl eager runtime、official-tool prompt/parser、strong low-fidelity summary、post-state-image-only
  restoration 与 GPU full-vocabulary KL；
- 48 train + 16 fresh development trajectories 的 192-state projection、双 worker parity 与 exact counts；
- confirm/test prompt、image、action、decoder 和 teacher access 全部禁止。

formal preclaim 必须先 fresh materialize 并验证 substrate archive 与 derived artifact。共享机器上的旧目录只可
作为 cache，不能替代 immutable validation；任何 bytes、state order、history slice、candidate event IDs、model
snapshot、runtime stack 或 canonical path 漂移都必须在 global attempt claim / model import 前 fail closed。

## Source A / runner freeze B

source contract 本身明确返回
`policy_or_gpu_execution_authorized_by_this_validator=false`。执行授权分两次 Git milestone：

1. **source A**：contract、coalition input、parent crosswalk、runner、artifact reducer/manager、tests 与本文档
   commit 并 push canonical `main`；
2. 从 clean pushed `main@<SOURCE_A>` 确定性物化
   `code/configs/causalcache_restoration_v2_2_expansion_labels_runner_v1.json`，只把该 freeze 作为
   **execution B** commit 并 push；
3. 在 formal host 的 clean `main == origin/main == B` 上运行 committed validator。它必须证明 A 是 B 的
   ancestor、freeze bytes 已提交、当前六个 reserved execution sources 与 A 的 Git blobs 完全相同，才返回
   GPU authorization。

从 source A 物化 freeze 的命令为：

```bash
cd code
python3 -m scripts.manage_restoration_v2_2_expansion_labels_artifact \
  materialize-runner-freeze \
  --repository-root .. \
  --runner-source-git-commit <FULL_SOURCE_A_SHA> \
  --output ../code/configs/causalcache_restoration_v2_2_expansion_labels_runner_v1.json
```

该命令只允许在 clean pushed A 上运行；返回 pending-B 状态，不授权 GPU。freeze commit/push 后再执行：

```bash
cd code
python3 -m scripts.manage_restoration_v2_2_expansion_labels_artifact \
  validate-runner-freeze \
  --repository-root .. \
  --runner-freeze ../code/configs/causalcache_restoration_v2_2_expansion_labels_runner_v1.json
```

只有 `VALID_COMMITTED_PUSHED_EXPANSION_EXACT_LABEL_RUNNER_FREEZE` 才是正式运行的 Git 前置条件。

## 唯一 attempt 与 artifact identity

```text
attempt: restoration-v2-2-expansion-exact-labels-v1
output root: /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1
global ledger: /data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1.attempt.json
raw archive: /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1.tar
HF repo: gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile
HF tag: v2.2-expansion-exact-labels-v1
HF path: raw/v2.2-expansion-exact-labels-v1.tar
```

output root、global ledger、两个 sibling ledgers、publication staging 或同一 attempt 的任一残留存在时，禁止通过
删除、换路径、`--resume` 或 replacement 重新运行。中断/失败 attempt 必须原地封存为
`INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`；只有 192/192 states、全部 exact counts、finite distances、repeat KL
不超过 $10^{-4}$、zero prohibited operations 和 raw-table independent reduction 全部通过，才能返回
`PASS_V2_2_EXPANSION_EXACT_LABELS_V1`。

## Formal 双 H200 顺序

正式执行必须先遵循 [`execution.md`](execution.md) 的 host/GPU/disk/container preflight、至少 10 秒
continuous-zero cleanup，并只向容器暴露两张选定的 H200。容器内要求 `torch.cuda.device_count()==2`；even / odd
固定映射到 `cuda:0` / `cuda:1`，各 96 states。source-locked monitor sidecar 必须在 main runner 前 ready，外部
utilization monitor 原始日志也必须归档；低利用率只触发诊断，不能改变 state、batch 或 scientific denominator。

下面命令是最终 runner CLI 形状；只有 execution B 已 commit/push 且 validator 通过后才可填写实际路径执行：

```bash
cd /data/CausalCache/code

python3 -m scripts.run_restoration_v2_2_expansion_labels monitor-sidecar \
  --ready-file /data/tmp/expansion-exact-labels-monitor.ready.json \
  --log-file /data/tmp/expansion-exact-labels-monitor.jsonl \
  --summary-file /data/tmp/expansion-exact-labels-monitor.summary.json &

python3 -m scripts.run_restoration_v2_2_expansion_labels \
  --repository-root .. \
  --contract configs/causalcache_restoration_v2_2_expansion_labels_v1.json \
  --runner-freeze configs/causalcache_restoration_v2_2_expansion_labels_runner_v1.json \
  --parent-substrate-archive /data/artifacts/fresh/v2.2-label-expansion-substrate-v1.tar \
  --derived-artifact-root /data/artifacts/fresh/restoration-v2-label-expansion-v1 \
  --ocr-backend-config configs/restoration_v2_ocr_backend.json \
  --snapshot-manifest configs/gui_owl_1_5_8b_snapshot.json \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id <FULL_64_HEX_CONTAINER_ID> \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --output-dir /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1 \
  --global-ledger /data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1.attempt.json \
  --preflight-log /data/tmp/expansion-exact-labels-preflight.log \
  --utilization-monitor-log /data/tmp/expansion-exact-labels-monitor.jsonl \
  --monitor-ready-file /data/tmp/expansion-exact-labels-monitor.ready.json \
  --monitor-summary /data/tmp/expansion-exact-labels-monitor.summary.json
```

runner 对 argv 的 option order、canonical repository inputs、model path、host identity、image digest、attempt path
与 ledger path 都做 exact validation；不要重排参数或另加未冻结 option。

## 成功后的回写顺序

terminal PASS 后才允许：

1. 用 artifact manager 对 canonical root、global/sibling ledgers 和 monitor evidence 做 independent raw reduction；
2. 生成 deterministic USTAR 到 canonical archive path，并记录 archive/tree/payload hashes；
3. 上传 private HF canonical repo/path，创建冻结 tag 并取得 immutable revision；
4. 下载到不同 fresh path，验证 archive byte identity，再独立重算 1,792 raw rows 与全部 derived counts；
5. 只把 compact summary、artifact manifest、HF immutable revision 与下一步写入 Git；raw tensor/evidence 不进 Git。

immutable expansion labels 完成前，formal-58 gate training 继续 locked。完成后也只能按已冻结 gate contract 训练；
matched-NLL、closed-loop 与 confirm 仍各自需要后续独立 freeze，不能由本次 label PASS 自动解锁。

以下命令是完整的 formal package、upload、fresh-download 和 manifest 流程。所有命令中的
`--source-git-commit <SOURCE_A_SHA>` 都必须填写 runner freeze 所绑定的完整 40 位 **source A** SHA，不能填写
execution B SHA。parent archive、derived artifact 与 OCR config 在 package、fresh validation 和 manifest 三处都会
重新生成 dynamic input witnesses；只校验上传后 archive 自身一致、但没有 external input replay 的结果不能作为
formal manifest。

```bash
cd /data/CausalCache/code

SOURCE_A_SHA=<FULL_40_HEX_SOURCE_A_SHA>
CONFIG_SHA256=65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5
RAW_ARCHIVE=/data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1.tar
PARENT_ARCHIVE=/data/artifacts/fresh/v2.2-label-expansion-substrate-v1.tar
DERIVED_ROOT=/data/artifacts/fresh/restoration-v2-label-expansion-v1
OCR_CONFIG=configs/restoration_v2_ocr_backend.json
UPLOAD_ATTESTATION=/data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1.upload.json
ARTIFACT_MANIFEST=/data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1.manifest.json

python3 -m scripts.manage_restoration_v2_2_expansion_labels_artifact package \
  --raw-output-dir /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1 \
  --global-attempt-ledger /data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1.attempt.json \
  --output "${RAW_ARCHIVE}" \
  --source-git-commit "${SOURCE_A_SHA}" \
  --config-sha256 "${CONFIG_SHA256}" \
  --parent-substrate-archive "${PARENT_ARCHIVE}" \
  --derived-artifact-root "${DERIVED_ROOT}" \
  --ocr-backend-config "${OCR_CONFIG}"

python3 -m scripts.manage_restoration_v2_2_expansion_labels_artifact upload-and-tag \
  --source-archive "${RAW_ARCHIVE}" \
  --source-git-commit "${SOURCE_A_SHA}" \
  --config-sha256 "${CONFIG_SHA256}" \
  --hf-repo gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile \
  --hf-path raw/v2.2-expansion-exact-labels-v1.tar \
  --hf-tag v2.2-expansion-exact-labels-v1 \
  | tee "${UPLOAD_ATTESTATION}"

HF_REVISION=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["immutable_revision"])' \
  "${UPLOAD_ATTESTATION}")
test "${#HF_REVISION}" -eq 40

python3 -m scripts.manage_restoration_v2_2_expansion_labels_artifact validate-fresh-hf \
  --source-archive "${RAW_ARCHIVE}" \
  --download-dir /data/artifacts/fresh/expansion-exact-labels-hf-validation \
  --source-git-commit "${SOURCE_A_SHA}" \
  --config-sha256 "${CONFIG_SHA256}" \
  --hf-repo gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile \
  --hf-path raw/v2.2-expansion-exact-labels-v1.tar \
  --hf-tag v2.2-expansion-exact-labels-v1 \
  --hf-immutable-revision "${HF_REVISION}" \
  --parent-substrate-archive "${PARENT_ARCHIVE}" \
  --derived-artifact-root "${DERIVED_ROOT}" \
  --ocr-backend-config "${OCR_CONFIG}"

python3 -m scripts.manage_restoration_v2_2_expansion_labels_artifact create-manifest \
  --source-archive "${RAW_ARCHIVE}" \
  --download-dir /data/artifacts/fresh/expansion-exact-labels-hf-manifest \
  --source-git-commit "${SOURCE_A_SHA}" \
  --config-sha256 "${CONFIG_SHA256}" \
  --hf-repo gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile \
  --hf-path raw/v2.2-expansion-exact-labels-v1.tar \
  --hf-tag v2.2-expansion-exact-labels-v1 \
  --hf-immutable-revision "${HF_REVISION}" \
  --parent-substrate-archive "${PARENT_ARCHIVE}" \
  --derived-artifact-root "${DERIVED_ROOT}" \
  --ocr-backend-config "${OCR_CONFIG}" \
  --output "${ARTIFACT_MANIFEST}"
```

`upload-and-tag` 会幂等创建或复用 private dataset repo，但 canonical path 和 tag 必须同时不存在；upload 使用
pre-upload parent commit 防止并发覆盖，tag 使用 `exist_ok=False`。两个 fresh download 目录和 manifest output
必须不存在或为空，不能复用第一次验证产生的非空目录。manifest 必须记录
`external_input_replay_verified=true`，并把 runtime 独立复验标记为 false。
