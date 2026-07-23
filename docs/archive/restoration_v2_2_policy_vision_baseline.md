# Restoration v2.2 policy-vision feature-only baseline

> 当前状态：v1 source contract 已冻结；唯一 formal invocation 因 pinned PyTorch UUID object type mismatch 在
> 0 feature 时 `INVALID`，canonical output 未生成。confirm/test、gate、matched-NLL 与 closed-loop 仍锁定。

## 1. 这个 comparator 测什么

本阶段补齐 primary `n=4,B=2` selector matrix 中最后一个非学习视觉 baseline。它的准确名称是：

> **frozen-policy-backbone vision similarity**

它使用冻结 GUI-Owl checkpoint 的 vision tower 和 final main spatial merger，但不输入 task goal、文本 query、
history action、summary 或 OCR，也不调用 language model。因此它是 checkpoint-aware、image-conditioned 的视觉
相似度，不是 behavioral policy-aware selector，更不是 restoration attribution。

对每个 primary state，candidate 是 events 1--4 的 post-action screenshot，query 是 event 5 post-state，也就是
decision step 6 的 current observation。两侧完全对称地经过：

```text
raw screenshot
-> pinned AutoImageProcessor
-> all 27 GUI-Owl vision blocks
-> final main spatial merger pooler_output
-> FP32 mean over visual-token rows
-> L2 normalization
```

若第 (j) 个 event 与 current 的 normalized embedding 分别为 (z_j,z_t)，冻结分数为：

\[
s_{j,t}=z_j^\top z_t.
\]

四个分数按 descending 排序，`math.isclose(rel_tol=1e-9, abs_tol=1e-12)` 时较小 event step 优先；
选出的 top-2 coalition 最终按时间顺序序列化。负 restoration utility 或 recovery 不删除、不裁剪。

## 2. 输入与 source identity

machine-readable contract 是
[`../../code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json`](../../code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json)，
SHA256：

```text
a2319f8ea52d53fa01487cdbcbfef20b86ac81dcfab1c8a36ce4583b0b503023
```

它固定绑定：

- selector-geometry v2 repair exact-three artifact；
- restoration label raw archive 的 private HF immutable revision 与 SHA256；
- derived dataset exact-six projection；
- 已验证 OCR/RGB v2 result 的 15-state/75-image identity fields；
- `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；
- 14-file、17,545,907,171-byte model snapshot；
- Transformers 5.6.0 三份 vision/processor source SHA；
- 现有 `pooler_output` extractor 与 v2.2 eager runtime source identity。

OCR/RGB result 在 feature stage 只充当 state/image identity witness。feature scorer/worker 不读取 archived OCR
tokens、OCR score、resized 256×256 RGB、OCR/RGB selection 或 comparator result。正式 GPU 运行前只对 raw
label archive 做 path/size/SHA256 byte-identity verification；其中的 restoration rows 不在此时解析。完整
(D(S)) table 只在 feature selection 完成后由 CPU reducer parse/join；reducer 此时也读取已冻结 OCR/RGB
similarity、selection、recovery 与 geometry comparators，用于 paired reporting，不反向影响 vision selection。

固定 denominator 为 15 states：10 条 `v2_label_train`、5 条 `v2_development`，共 75 张唯一 screenshot 和
60 个 canonical cosine。confirm/test state、payload、processor input、feature 与 score 均为 0。

## 3. Feature tensor contract

唯一允许的预处理入口是：

```python
AutoImageProcessor.from_pretrained(
    model_dir,
    min_pixels=2_621_440,
    max_pixels=2_621_440,
    local_files_only=True,
)
image_processor(images=five_rgb_pil_images, return_tensors="pt")
```

禁止 `AutoProcessor`、tokenizer、chat template 或 prompt wrapper。processor 必须是
`Qwen2VLImageProcessor`，输出 exact-two keys：

- `pixel_values`: CPU contiguous FP32，shape `[sum(t*h*w),1536]`；
- `image_grid_thw`: CPU contiguous int64，shape `[5,3]`。

两者先在 CPU 完整校验，再显式搬到 worker 的单一 CUDA device。模型必须是 H200 上的 BF16、eager
attention、`eval()`、`requires_grad=False`；TF32 关闭。feature 只允许：

```python
model.get_image_features(
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    return_dict=True,
).pooler_output
```

每图输出必须是 BF16 `[t*h*w/4,4096]`。pre-merger 1152-d `last_hidden_state` 与 layer 8/16/24
DeepStack features 全部排除。mean/L2 在 GPU 上用 FP32 完成；4096-d embedding 不转 CPU、不保存到 Git/HF。
只允许 grid/token metadata、norm validation scalars 和最终 cosine scalars 回 host。

processor-only 预检在任何 model feature output 前固定了 canonical geometry：

| `image_grid_thw` | Images |
| --- | ---: |
| `[1,136,76]` | 15 |
| `[1,148,68]` | 15 |
| `[1,150,66]` | 5 |
| `[1,152,68]` | 20 |
| `[1,80,128]` | 20 |

15 个 state batches 的 pixel rows 为 `51680×7、50320×3、49500×1、51200×4`；总 raw patches
`767,020`、merged tokens `191,755`。这些是输入预处理 identity，不是 policy-vision scientific result。

## 4. 双 GPU 执行与 replay

正式 stage 使用 2 张 H200、两份独立 model replica、无 DDP。primary state 的原始 index 为
`2,5,...,44`，固定 parity shard：

- `even / cuda:0`：`[2,8,14,20,26,32,38,44]`，8 states；
- `odd / cuda:1`：`[5,11,17,23,29,35,41]`，7 states。

每个 worker 都独立执行 model snapshot full hash、runtime verification 和 model load；GPU UUID、PCI bus、
logical device 与 `nvidia-smi` index 必须交叉一致。每个 canonical state 只做一次 processor batch，随后在同一
CUDA tensors 上做两次 feature pass：第一次产生 canonical score，第二次是 same-device replay。

另外 odd worker 对 primary ordinal 0（state index 2，
`0131649930078879:decision_step:006`）做一次 cross-device sentinel。预注册稳定性条件为：

- 每个 cosine 的 absolute difference 不超过 `1e-6`；
- 四候选 ranking 完全相同；
- selected coalition 完全相同。

不声称 strict bitwise CUDA determinism。任何 replay、source、runtime、hash、shape、finite/norm、count 或 worker
failure 都使本 identity fail closed，canonical scientific aggregate 不发布；不得看结果后放宽 tolerance、换层、
改 pooling 或删 state。

冻结 operation schedule 为：

| Operation | Canonical | Verification | Total |
| --- | ---: | ---: | ---: |
| Image processor batches | 15 | 1 | 16 |
| Image assignments/extract/decode | 75 | 5 | 80 |
| Vision feature forwards | 15 | 16 | 31 |
| Cosine scalar transfers | 60 | 64 | 124 |
| Model / processor loads | - | - | 2 / 2 |

top-model forward、language-model forward、LM head、generation、teacher forward、KL、OCR inference、gate
training、matched-NLL、closed-loop、confirm 与 sealed test access 全部固定为 0。runtime 对 top model、language
model、LM head 和 `generate` 安装 fail-before-output guard，而不是只依靠调用约定。

## 5. 统计与结果边界

feature worker 输出只含 60 个 canonical cosine、selection、grid/token counts、replay 与 runtime provenance。
CPU reducer 之后才从 immutable raw (D(S)) table lookup selected coalition，计算：

\[
U(S)=D(\varnothing)-D(S),\qquad
R(S)=\frac{U(S)}{D(\varnothing)}.
\]

正式 summary 按 train/development/overall 报告：

- mean/median/min/max recovery、negative count、mean raw utility；
- mean (D(\varnothing)) 与 ratio-of-sums small-denominator sensitivity；
- exact at-most-2 / exact-cardinality-2 match、Jaccard 与 regret；
- 对 exact、true greedy、budget-independent、Shapley、recent、analytic random 和 OCR/RGB 的 paired
  trajectory bootstrap（10,000 resamples、seed 271828、90% percentile interval）；
- 五条 development paired deltas；
- cosine range、top2-top3 margin、near-tie、selection frequency、recent/OCR coalition overlap 和 grid geometry。

预注册异常点 `0141544666483837` 必须单独复核。若 vision 仍选择有害的 `[3,4]` 或出现负 recovery，样本仍
保留；若 cosine 高度饱和，则结论是 global-mean representation 退化，不是数据错误。

## 6. Source-only validation 与正式运行

在 formal output 不存在时验证 source contract：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_policy_vision_contract \
  --repository-root .. \
  --contract configs/causalcache_restoration_v2_2_policy_vision_baseline.json
```

正式 runner 只能从 clean、已 push 的 `main` 执行。GPU UUID、container/host identity 与所有路径必须显式传入；
preflight 后使用下面的完整参数面；`<GPU_UUID_0/1>`、container hostname/name 只能替换为本次实际值，不能改动
科学路径、model revision 或 image digest：

```bash
cd /data/CausalCache/code
python3 -m scripts.run_restoration_v2_2_policy_vision_baseline run \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json \
  --labels-archive /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar \
  --derived-root /data/tmp/causalcache-restoration-labels-v2-derived \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --snapshot-manifest /data/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json \
  --source-git-commit <SOURCE_COMMIT> \
  --expected-gpu-uuid <GPU_UUID_0> \
  --expected-gpu-uuid <GPU_UUID_1> \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-hostname "$(hostname)" \
  --container-name <SGLANG_OMNI_JAXAN_TIMESTAMP_NAME> \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --nvidia-driver-version 570.172.08 \
  --host-evidence /data/tmp/policy-vision-host-evidence.json \
  --output-dir /data/CausalCache/data/results/restoration_v2_2_policy_vision_baseline_v1
```

`--source-git-commit` 必须等于 clean checkout 的 `HEAD` 与 `origin/main`，两个 UUID 的顺序定义 container logical
`cuda:0/1`。四类 identity 的含义分别是：`--host-alias` 为本地 SSH alias 且 runner 只接受
`hyper00/hyper01`；`--host-hostname` 是宿主 preflight `hostname` 的原样记录，alias 与物理宿主的对应关系由
operator attestation 保证，runner 不在容器内反推；`--container-hostname` 必须 exact 等于 runner 进程看到的
`socket.gethostname()`；`--container-name` 是 Docker object name，二者通常不同。container name 与 image
digest 也来自宿主侧 `docker inspect` attestation；runner 检查命名格式及 digest 与冻结值相等，但容器内不调用
Docker daemon 自证。`--host-evidence` 是 GPU job 启动前由宿主 `docker inspect` 与 `nvidia-smi` 生成、复制进
容器且不可 group/other write 的 strict JSON record；它绑定完整 64-hex container ID、name、image digest、
driver 与按 logical `cuda:0/1` 排列的两个 GPU UUID。runner 记录该文件 SHA256，并要求默认 12-hex container
hostname 是完整 container ID 的前缀。

正式 preflight 依次闭合 clean pushed source、canonical output/hidden staging 均不存在、raw label archive
byte identity、derived exact-six projection、snapshot manifest/path 与 15-state identity witness。两个 worker
之间还会检查 model directory 存在；随后各自完成 model snapshot full hash 与 runtime/GPU verification，才允许
第一次 feature forward。没有 `--resume`；“一次 formal attempt、失败不得静默重试”是 preregistered operator
procedure，当前 runner 不持久化 failed-attempt ledger，因此失败时必须由执行者保留并回写证据。发布使用隐藏
staging 目录，三份文件写完后才 atomic rename；异常时 staging 会清除。canonical 与 hidden staging 为：

```text
data/results/restoration_v2_2_policy_vision_baseline_v1/
data/results/.restoration_v2_2_policy_vision_baseline_v1.staging/
```

只允许 exact-three lightweight files：`README.md`、`state_scores.jsonl`、`summary.json`。raw image、label、
4096-d features、model weights 或 checkpoint 不进入 Git。若以后需要可复用 embedding cache，必须另建 HF
dataset artifact，并绑定 image SHA、model revision、extractor source 和 tensor schema；本 baseline 默认不生成
该 cache。

result 产生后，从包含 exact-three artifact 的 clean descendant checkout 做 CPU-only reconstruction：

```bash
cd /data/CausalCache/code
python3 -m scripts.run_restoration_v2_2_policy_vision_baseline validate \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json \
  --labels-archive /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar \
  --source-git-commit <SOURCE_COMMIT> \
  --output-dir /data/CausalCache/data/results/restoration_v2_2_policy_vision_baseline_v1
```

这里的 `<SOURCE_COMMIT>` 必须仍是原 formal GPU run 的 source SHA，不是包含 result 的 descendant `HEAD`。

v1 formal invocation 从 `main@c0937056e94d110cd67e593288f9e0c3a3b24809` 启动，但
`torch.cuda.get_device_properties().uuid` 的真实类型是 `torch._C._CUuuid`，不是 source parser 允许的
`str/bytes`。它在 model snapshot full hash、processor/model load、feature forward 与 output staging 前 fail
closed；轻量证据见
[`../../data/results/archive/restoration_v2_2_policy_vision_baseline_v1_attempt/`](../../data/results/archive/restoration_v2_2_policy_vision_baseline_v1_attempt/)。
同一 v1 不重试，必须先冻结只扩展 pinned UUID object normalization 的 versioned replacement。当前不能把本
source freeze 写成 comparator 已通过，也不能据此开始 gate training。

## 7. GPU UUID type-only v2 repair

v2 overlay 位于
`code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json`，SHA256 为
`23733169ef5ba60a84f4447080ef12e1893858aa375ff50d8d4e3595699e774e`。它严格绑定 parent v1 contract、
`main@a809a908207768759f6a15874aeac552a7fd5e08` 的 exact-two failure files、zero-feature failure stage 与 v1
canonical/staging absent。v2 没有修改任何 comparator science，只新增显式 runtime profile：

```text
cuda_device_property_uuid_torch_c_cuuuid_v2
```

该 profile 必须从当前 loaded PyTorch 取得 `torch._C._CUuuid` type object，并要求
`type(observed_value) is torch._C._CUuuid`；通过后只做 `str(value)`，再进入原 canonical UUID parser。仅伪造
`__module__`/`__name__`、普通字符串、bytes、其他 object 或缺少 loaded type 都会 fail closed。UUID format、
expected UUID、`nvidia-smi` exact-one-row、PCI bus ID 与 logical-device binding 不变。v1 默认 profile 仍只接受
`str/bytes`，因此 repair 不是对 v1 source 的静默放宽。

source-only validator：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_policy_vision_v2_contract \
  --repository-root .. \
  --contract configs/causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json
```

正式执行必须重新做 GPU preflight，并创建全新 timestamp container；不得复用 v1 container 或 v1 output
identity。其余 immutable paths、两个 logical GPU UUID 的顺序和宿主 evidence 规则继承第 6 节：

```bash
cd /data/CausalCache/code
python3 -m scripts.run_restoration_v2_2_policy_vision_baseline_v2 run \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json \
  --labels-archive /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar \
  --derived-root /data/tmp/causalcache-restoration-labels-v2-derived \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --snapshot-manifest /data/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json \
  --source-git-commit <V2_SOURCE_COMMIT> \
  --expected-gpu-uuid <GPU_UUID_0> \
  --expected-gpu-uuid <GPU_UUID_1> \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-hostname "$(hostname)" \
  --container-name <NEW_SGLANG_OMNI_JAXAN_TIMESTAMP_NAME> \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --nvidia-driver-version 570.172.08 \
  --host-evidence /data/tmp/policy-vision-v2-host-evidence.json \
  --attempt-ledger /data/experiments/causalcache/restoration-v2-2-policy-vision-v2-gpu-uuid-repair-attempt.json \
  --output-dir /data/CausalCache/data/results/restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair
```

runner 在所有 path/source/host/label-byte/derived/snapshot preflight 通过后、任何 model hash/load 或 feature forward
之前，以 `O_CREAT|O_EXCL` 和 mode `0600` 创建上述固定 ledger，并 `fsync` 文件与父目录。ledger 一旦存在，
同一 v2 protocol 永久拒绝第二次 claim；异常不得删除或换路径。v1 旧入口也已 tombstone，`run` 会在读取任何
input 或进入 GPU path 前拒绝。

v2 仍是一次 attempt、无 resume、失败不得同 protocol 静默重试。result 未生成前，`VALID_*_SOURCE_CONTRACT`
只表示 repair 边界自洽；它不表示 feature replay、recovery、exact match 或任何 paper gate 已通过。result commit
后只能从 clean descendant checkout 用同一个 `<V2_SOURCE_COMMIT>` 执行 `validate`，逐 byte 重建 exact-three
artifact。这里的重建以 committed `state_scores.jsonl` 中记录的 feature rows、immutable labels 和 identity
witness 为输入；CPU `validate` 不重新加载模型或重算 vision features。随后再做不 import 项目 reducer 的独立
reducer/math 审计，二者都不能替代 formal GPU replay checks。

### v2 formal outcome

唯一 v2 attempt 从 `main@fe7640395d3b6aea2e5e3a8cc34a49efc5ba2d2f` 启动并永久 claim ledger。UUID v2
repair 通过；新的 fail-closed 点是 `image_processor.size` interface。pinned runtime 返回
`transformers.image_utils.SizeDict`，`dict(size)` 精确为冻结的
`{"longest_edge": 2621440, "shortest_edge": 2621440}`，但该类型不实现 `collections.abc.Mapping`。因此
v2 在 policy model load、processor batch、feature forward、cosine、selection 和 semantic label load 前退出，
canonical output/staging 均不存在。

该结果只否定 v2 的 interface predicate，不否定冻结 preprocessing 数值或 comparator。same-protocol retry 已由
ledger 禁止；若继续，必须先提交
`data/results/archive/restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/`，再冻结新的 versioned
SizeDict-interface repair。不能事后修改 v2，也不能据此开始 gate/confirm。

## 8. Exact SizeDict-interface v3 repair

v3 使用新 protocol
`causalcache_restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair`，并把 v2 failure commit
`main@b1d0755` 与 failure README/JSON exact bytes 作为 parent evidence。它继承 v2 的 UUID exact-type repair，
只替换 image-processor size interface profile。config SHA256 为
`794474d8bc60463ba10fdd772691461f5910ca5e5501f7cccff4c53542b84b7f`。合法 tuple 只有：

```text
(cuda_device_property_uuid_torch_c_cuuuid_v2,
 transformers_image_utils_size_dict_exact_edges_v3)
```

v3 必须从当前 loaded Transformers 取得 `transformers.image_utils.SizeDict` type，并要求
`type(value) is loaded SizeDict`、type module/name 精确、`isinstance(value, Mapping)` 为 false，且
`height/width/max_height/max_width` 四个 fields 都存在并为 `None`。随后 `dict(value)` 必须是 built-in `dict`，
key set 恰为 `shortest_edge/longest_edge`，两值都等于 `2621440`。同名伪造类、subclass、Mapping、generic
iterable、缺失/额外/错误 field 或 key 一律在 policy model load 前拒绝。v1/v2 profile、runtime ID 和 metadata
schema 不变。

v1/v2 runner 均已永久 tombstone。v3 canonical output 固定为
`data/results/archive/restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair`，唯一 attempt ledger 固定为
`/data/experiments/causalcache/restoration-v2-2-policy-vision-v3-size-dict-interface-repair-attempt.json`。ledger 在
任何 model/feature access 前以 `O_CREAT|O_EXCL`、mode `0600` durable claim；异常不得删 ledger、换路径、resume
或 same-protocol retry。

source-only validator：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_policy_vision_v3_contract \
  --repository-root .. \
  --contract configs/causalcache_restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair.json
```

正式入口沿用冻结的双 worker 参数，只替换 v3 identity：

```bash
cd /data/CausalCache/code
python3 -m scripts.run_restoration_v2_2_policy_vision_baseline_v3 run \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair.json \
  --labels-archive /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar \
  --derived-root /data/tmp/causalcache-restoration-labels-v2-derived \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --snapshot-manifest /data/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json \
  --source-git-commit <V3_SOURCE_COMMIT> \
  --expected-gpu-uuid <GPU_UUID_0> \
  --expected-gpu-uuid <GPU_UUID_1> \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-hostname "$(hostname)" \
  --container-name <NEW_SGLANG_OMNI_JAXAN_TIMESTAMP_NAME> \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --nvidia-driver-version 570.172.08 \
  --host-evidence /data/tmp/policy-vision-v3-host-evidence.json \
  --attempt-ledger /data/experiments/causalcache/restoration-v2-2-policy-vision-v3-size-dict-interface-repair-attempt.json \
  --output-dir /data/CausalCache/data/results/archive/restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair
```

该 PASS 只证明修复边界和 source inventory 自洽；在新的 clean pushed source 完成正式 run 之前，不报告
policy-vision feature、recovery 或 comparator 结论，也不运行 gate、matched-NLL、closed-loop 或 confirm/test。

### v3 formal outcome 与 CPU validation 边界

唯一 v3 GPU attempt 从 `main@a935a3cf5efb1fa7a952ca6f45a5994609b367e9` 完成全部冻结 schedule：15 个
canonical states、75 unique images、15 canonical + 15 same-device replay + 1 cross-device sentinel vision
forwards。两级 replay 的 maximum absolute score difference 都是 0；policy、language model、LM head、
generation、gate、matched-NLL、closed-loop 和 confirm/test operation 都是 0。exact-three artifact 已原子发布，
scientific payload SHA256 为 `811e59c780851c19b7fe21314be1e52c1dbfcfa002e63a9857fd34d90ba94f48`。
raw restoration-label archive bytes 在 GPU 前仅做 immutable SHA 验证；只有 selection 完成后才做 semantic
parse/join，feature workers 未接收 label semantics。

首次 CPU `validate` 随后在 row provenance 重建处 fail closed。formal feature record 的 state 是
`index/role/trajectory_id/state_id` 四键；evaluated output row 按 frozen output schema 保存同四键加
`decision_step_id/candidate_event_step_ids/budget_event_capacity`。reconstructor 使用 `dict(row["state"])`，没有
投影回 feature-state，导致严格相等检查报 `policy-vision row-to-worker provenance drifted`；worker/device/GPU
字段本身没有漂移。

同一 source 上的 bounded CPU diagnostic 只做四键 projection，复用原 rows、execution、immutable labels 与
witness，逐 byte 重建三份 artifact 均 exact match。在 versioned repair 执行前，该 artifact 记为
`COMPLETED_PENDING_VERSIONED_CPU_REPLAY_VALIDATION`，不是 scientific `INVALID`；随后只能冻结 reporting-only
CPU validation repair，不得修改 artifact bytes、重跑 GPU 或提前解锁 gate/confirm。下节记录该 repair 的正式闭合。

独立数值审计已复算所有 15-state selection、utility、comparator delta 与 hash，没有发现 scientific blocker。
overall mean normalized recovery 是 `0.404691`；由于小 summary-only distance 的负 outlier 对逐状态 ratio 的
影响较大，同时保留 ratio-of-sums `0.703501`。artifact 中 `exact_coalition_overlap_with_*` 的语义是
policy-vision selection 与该 comparator selection 完全相同，并非与 exact-subset oracle 相同。

### v3 validation-repair v1 source freeze 与 formal outcome

repair v1 使用独立 config/module/runner，不编辑旧 v3 reconstructor。唯一修复是先严格要求 recorded evaluated
state 恰好包含七键，再按固定顺序投影为 `index/role/trajectory_id/state_id` 四键；其余 feature record 字段仍由
producer `a935a3cf` 的原函数重建。config SHA256 为
`64f63ab7563c227423c15ef82d5fd74d8be11579248908c7ec2880137e5d6ddf`，artifact commit 固定为
`597f050297342d9f29eb383985c2014f5782b2bb`。

source-only validator：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_policy_vision_v3_validation_repair_v1_contract \
  --repository-root .. \
  --contract configs/causalcache_restoration_v2_2_policy_vision_v3_validation_repair_v1.json \
  --validation-source-git-commit <CLEAN_PUSHED_REPAIR_SOURCE_COMMIT>
```

formal CPU-only runner：

```bash
cd /data/CausalCache/code
python3 -m scripts.run_restoration_v2_2_policy_vision_v3_validation_repair_v1 run \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_2_policy_vision_v3_validation_repair_v1.json \
  --labels-archive /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar \
  --producer-attempt-ledger /data/experiments/causalcache/restoration-v2-2-policy-vision-v3-size-dict-interface-repair-attempt.json \
  --validation-source-git-commit <CLEAN_PUSHED_REPAIR_SOURCE_COMMIT> \
  --output-dir /data/CausalCache/data/results/archive/restoration_v2_2_policy_vision_baseline_v3_validation_repair_v1 \
  --attempt-ledger /data/experiments/causalcache/restoration-v2-2-policy-vision-v3-validation-repair-v1-attempt.json \
  --completion-seal /data/experiments/causalcache/restoration-v2-2-policy-vision-v3-validation-repair-v1-completion-seal.json
```

formal container 不传 `--gpus`，且 runner 必须观察到零 NVIDIA device node。它在 claim 新 CPU ledger 前固定
runtime identity；claim 后才做 label semantic parse/reconstruction。它在 atomic publish 前创建独立 `0600`
O_EXCL completion seal，把 claim SHA、finished time 与两份输出的 exact size/SHA 锁在 result 之外；seal 已存在但
exact output 未完整发布时，该次 protocol 永久 invalid。全部 producer files、原 GPU ledger、claim ledger、
completion seal 与 source 在 publication 前后重验。

唯一 formal audit 已从 `main@dbb45637cf79c3573bbbc6051b8b480e3f76d69d` 完成。validation Python closure 是
156 paths / `e8daf215ca51b4c4d9e5f8a82399ec70d7d68700351968cace3386e235160a1d`。运行环境为 Hyper00
`node-radixark-16-0000`、container `sglang-omni-jaxan-07170735` / ID
`02df32fa30f768217854fa12365ae81d64a7b6817f32ec36b0f9fe29bc439261`、image
`hongccc/sglang-omni:dev@sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
Docker DeviceRequests 为空。runtime 使用 Python 3.12.3，container hostname 是 `02df32fa30f7`，platform 为
`Linux-6.8.0-1043-aiext-x86_64-with-glibc2.39`，runner 记录 CPU、零 NVIDIA device node、零 CUDA runtime
import 与零 forbidden module import。

正式 sibling result
[`data/results/archive/restoration_v2_2_policy_vision_baseline_v3_validation_repair_v1/`](../../data/results/archive/restoration_v2_2_policy_vision_baseline_v3_validation_repair_v1/)
返回 `VALID_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`。README/summary exact-two 分别为
836 bytes / `3492dbae9a13d3d1d70e7aacf2cd7b405d1f9cc5956af980c519c8fb3ceee7e9` 与 10461 bytes /
`f7a5ff63a754d06d7b61dcc46516ee2ed22e0a6a3b92b8b0be8a68869cf362b1`；attempt ledger 为
2918 bytes / `6b65bef9d6edb73ee4275e89923bfd0e6123fb275fccb2b5dda9e57245f7b5d3`，completion seal 为
771 bytes / `837c52f403dbb9f21bf968ff5a0ba10199d06fe36ee49431787eff5154c6a52b`。三份 producer
artifact 全部逐 byte 相同，denominator 是 15 feature records / 60 candidate scores；所有 operation count 为 0，
唯一修复是 exact 7-key→4-key projection，其余 reconstruction fields 不变。

clean-descendant validation 已在独立 validation-repo 的 clean
`main@174801112c58d831249fd54f4f8bc9af01524b44` 完成。runner 的 validation source 仍为
`dbb45637cf79c3573bbbc6051b8b480e3f76d69d`，状态是
`REVALIDATED_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`。postflight exact-two README/summary
仍为 mode 0644，SHA256 仍为 `3492dbae9a13d3d1d70e7aacf2cd7b405d1f9cc5956af980c519c8fb3ceee7e9` /
`f7a5ff63a754d06d7b61dcc46516ee2ed22e0a6a3b92b8b0be8a68869cf362b1`；attempt ledger / completion seal
仍为 mode 0600，SHA256 仍为 `6b65bef9d6edb73ee4275e89923bfd0e6123fb275fccb2b5dda9e57245f7b5d3` /
`837c52f403dbb9f21bf968ff5a0ba10199d06fe36ee49431787eff5154c6a52b`。validate 没有 GPU operation，也没有
修改 artifact、config、code 或 test。执行容器已停止但保留用于审计，exit code 137。
