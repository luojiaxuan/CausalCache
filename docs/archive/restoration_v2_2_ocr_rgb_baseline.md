# Restoration v2.2 OCR/RGB baseline v1

## 当前状态

v1 已完成 **source-only freeze**，但第一次正式 attempt 已在零 feature-score 阶段 fail closed。machine-readable contract 是
[`../../code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json`](../../code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json)，
SHA256 固定为
`08f57505d71e603d81d6915ef27cf008d0fe097a56d70a05f8b3519211e2e6f9`；protocol ID 是
`causalcache_restoration_v2_2_ocr_rgb_baseline_v1`。正式结果目录
`data/results/restoration_v2_2_ocr_rgb_baseline_v1/` 与 staging 均不存在，任何 recovery、match rate 或 paired
delta 都还不能报告为结果。compact failure binding 见
[`../../data/results/archive/restoration_v2_2_ocr_rgb_baseline_v1_attempt/`](../../data/results/archive/restoration_v2_2_ocr_rgb_baseline_v1_attempt/)；
replacement 已使用新 protocol/output identity 完成 source freeze，见
[`restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md`](restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md)。

clean pushed `main@aa5898f25e2e7d647363fe701ac90134bf744a5c` 的 formal runner 在第 0 条 derived trajectory
identity scan 失败。每条 trajectory 的同一个 `source_id` 合法地出现在 top-level 与 nested selection metadata，
两个 raw value 完全相同；v1 byte lexer 却要求该字段整行只能出现一次。失败早于 selected trajectory semantic
parse、OCR record parse、image payload extract、decode-resize 和全部 OCR/RGB/combined score，所以这些 operation
全为 0；GPU、policy、gate、confirm/test 也全为 0。这不是 OCR/RGB baseline 的负结果。

这一步回答一个有限问题：在已经冻结的 primary `n=4,B=2` selector slice 上，只用 OCR 与低级 RGB
相似度选择两个历史 event，能够恢复多少 frozen-policy behavior。它不是 learned gate、policy-vision baseline、
matched-NLL 或 closed-loop evaluation。

## 冻结输入与 primary slice

contract 同时绑定以下 immutable inputs：

- selector geometry repaired result：artifact commit
  `d0f25d812869d5fc7b58284a25abe3aa8049b0aa`，scientific payload SHA256
  `cc505443a7efdc68c8eeca754f24c9143cabf72a090f024fde05011e783cbb21`；
- restoration labels：private HF dataset
  `gavinlaw/causalcache-restoration-labels-mobile@8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`，raw USTAR SHA256
  `99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e`；
- derived dataset：private HF dataset
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@89f136abaff797e14fe758a198996e51032a10a6`，
  exact-six tree SHA256 `475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e`；
- baseline implementation manifest、Pillow 12.2.0 runtime、OCR backend config 与所依赖 source hashes。

formal reducer 会先验证完整 label archive 的 45 states / 420 条 $D(S)$ rows，但只把其中 step 6、候选
event 1--4 的 15 个 primary states / 240 条 $2^4$ distance rows 送入本阶段计算。primary denominator 为：

| 项目 | 冻结数量 |
| --- | ---: |
| `v2_label_train` trajectories / states | 10 / 10 |
| `v2_development` trajectories / states | 5 / 5 |
| 合计 trajectories / states | 15 / 15 |
| 候选 event-image 与 current-image | 60 + 15 |
| unique images | 75 |
| OCR / RGB / combined candidate scores | 60 / 60 / 60 |

每个 state 的 query 是 decision step 6 的 current observation，也就是 event 5 的 post-state image；四个候选是
event 1--4 各自的 post-state image。高保真预算固定为两个 event。

## Feature 与选择规则

OCR 不重新推理，直接读取 derived artifact 中已经归档的 `full_spatial_tokens`。token 经过 Unicode NFKC、
whitespace collapse 与 strip 后按集合计算 Jaccard；两个空集合的 similarity 固定为 1。RGB 分支复用冻结的
`prepare_image_bytes`：Pillow bilinear resize 到 256×256 RGB，再计算每通道 16 bins、共 $16^3$ cells 的 joint
histogram cosine。候选分数固定为：

\[
s_j=0.5\,\mathrm{Jaccard}_{\mathrm{OCR}}(j,t)
  +0.5\,\mathrm{Cosine}_{\mathrm{RGBHist}}(j,t).
\]

按 combined score 取 top-2；相等按冻结的 `rel_tol=10^{-9}`、`abs_tol=10^{-12}` 判断，再优先较小的 event
step。输出 coalition 按时间顺序序列化。这个 selector 总是选满两个，不允许根据结果临时加 threshold。

## At-most oracle 与 exact-cardinality oracle

两种 oracle 必须同时报告，不能混称为“exact”：

- **Exact subset / at-most-$B$ oracle** 在 $|S|\le 2$ 的空集、singleton 与 pair 中直接最小化 $D(S)$；distance
  相同时依次优先更小 cardinality 与 lexicographically earlier coalition。它允许在恢复 event 会降低 utility 时
  少选或不选，是原 memory budget 问题的真正上界。
- **Exact-cardinality oracle** 只在 $|S|=2$ 的六个 pairs 中最小化 $D(S)$，distance 相同时按 coalition
  lexicographic order 打破平局。因为 OCR/RGB selector 固定选两个，它是 cardinality-matched 对照。

每个 state 同时报告 actual utility、normalized recovery、相对两种 oracle 的 regret、coalition exact match 与
Jaccard。还会带入 repaired geometry 中的 true conditional greedy、budget-conditioned independent、full-path
Shapley independent、recent 与 analytic exact-cardinality random，避免只和一个弱 baseline 比较。

## 统计契约

统计单位固定为 trajectory。先在 trajectory 内对 state 等权，再在每个 role 内对 trajectory 等权；本 primary
slice 每条 trajectory 恰好一个 state，但仍保留该一般定义。train 与 development 分开重采样，overall 使用
role-stratified trajectory bootstrap。每个 paired comparator 固定 10,000 次 bootstrap、seed 271828、90%
percentile interval、tie epsilon $10^{-12}$。development 的五条 paired deltas 也逐 trajectory 落盘，不能只报告
均值。

## Confirm-safe 与 operation ceiling

完整 derived exact-six projection 会逐文件 hash 和 inventory 校验。trajectory/OCR JSONL 的所有行只扫描 identity
字段，image tar 只扫描 member identity，随后仅对 15 个 allowlisted trajectories 和 75 张 allowlisted images
进行 semantic parse、payload extract 与 feature reduction。其余记录（包括 confirm）保持 opaque：不会解析
confirm prompt、image 或 OCR 内容，不会提取 confirm image payload，也不会进入 scorer。这里的“opaque scan”
不表示 confirm bytes 从未被读取做文件 hash；它严格表示 confirm semantic access 与 feature access 都为零。

冻结的负操作包括：

- GPU、OCR inference、OCR model load、policy model load：全部 0；
- policy/policy-vision forward、generation、teacher forward、KL：全部 0；
- gate training/forward、matched-NLL、closed-loop：全部 0；
- confirm state access、confirm feature score、sealed-test access：全部 0；
- non-primary geometry 与 nonselected derived record 的 semantic parse：全部 0。

因此正式工作是 CPU feature reduction，不需要申请 GPU，也不能借本阶段打开 confirm。

## Source validation 与 v1 invalid attempt

source-only validator 不生成 aggregate：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_ocr_rgb_contract \
  --repository-root .. \
  --contract configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json
```

source commit push 后，v1 已在 Hyper00 的 Python 3.12 / Pillow 12.2.0 CPU runtime 执行一次并永久
fail closed。下列命令只作为 v1 provenance，**不得再次运行**：

```bash
cd /absolute/path/to/CausalCache/code
/data02/jaxan/.venv/causalcache-ocr-v2/bin/python \
  -m scripts.run_restoration_v2_2_ocr_rgb_baseline run \
  --repository-root /absolute/path/to/CausalCache \
  --contract /absolute/path/to/CausalCache/code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json \
  --labels-archive /immutable/raw/v2.2-eager-train-dev-exact-v2.tar \
  --derived-root /immutable/restoration-v2-derived-exact-six \
  --source-git-commit <CLEAN_PUSHED_MAIN_SHA> \
  --output-dir /absolute/path/to/CausalCache/data/results/restoration_v2_2_ocr_rgb_baseline_v1
```

v1 canonical output 从未创建。replacement 只能修 identity lexer：trajectory 每行必须恰有两个相同的
unescaped `source_id`，OCR 每行必须恰有一个 `image_member_path`；同一行全部 occurrence 的 UTF-8 decoded
value 必须完全一致，零 occurrence、escaped identity、数量漂移或不一致值仍 fail closed。allowed identities、
opaque nonselected semantics、feature/selector/statistics、operation ceiling、
immutable input identity 与 source-closure 规则全部保持不变。replacement 必须使用新的
config/protocol/output directory 并绑定 v1 failure record。

## 下一步

1. v1 zero-score failure binding 已在 `main@2870d8ae26542a184647e8b6d97b8c79e4e12641` commit/push；
2. 新 identity 的 exact-occurrence/equal-value lexer repair source 已冻结；
3. replacement 已从 clean `main@a9bede85ab8bd10623c5755b944b3c26865c6485` 在 Hyper00 CPU-only
   runtime 完成 canonical aggregate、pre-commit byte replay 与独立数值审计；
4. 有效 v2 result 见
   [`restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md`](restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md)；
   轻量 result 已在 `main@7e59591573cb31f178dfd07422cc2e3c8aeff573` commit/push，并通过 clean
   descendant committed replay；
5. 单独冻结并执行 policy-vision feature-only baseline；
6. visual comparator matrix 完整后，才冻结 gate training、matched-NLL 与 closed-loop contract。

上述任何步骤都不授权读取 confirm/test 或训练 gate。
