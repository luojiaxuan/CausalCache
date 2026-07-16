# Restoration v2.2 policy-vision feature-only baseline

> 当前状态：source-only contract 已实现，formal GPU output 尚未生成。confirm/test、gate、matched-NLL 与
> closed-loop 仍保持锁定。

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
[`../code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json`](../code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json)，
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

source SHA 由 formal argv/result provenance 记录；formal result commit、descendant CPU audit 与最终数值将在
执行后追加。当前不能把本 source freeze 写成 comparator 已通过，也不能据此开始 gate training。
