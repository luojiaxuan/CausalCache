# Restoration v2.2 OCR/RGB baseline v2 identity repair

## 当前状态

本阶段已从 clean pushed `main@a9bede85ab8bd10623c5755b944b3c26865c6485` 在 Hyper00 完成 formal
aggregate、同源逐 byte replay 与独立复算。machine-readable contract 是
[`../code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json`](../code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json)，
SHA256 为 `d68cb032ef3c56c330d57329507d409b20f878b7510bd09d88e2eff1f3f898f3`；protocol ID 是
`causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair`。canonical result directory 固定为
[`../data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/`](../data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/)。
source freeze 时该目录不存在；正式 run 后只允许 exact-three 文件，result commit 后从 descendant clean
`main` 再做 committed replay。

v1 attempt 是 zero-score implementation invalid，而不是 negative scientific result。其 immutable failure binding 已在
`main@2870d8ae26542a184647e8b6d97b8c79e4e12641` 提交，见
[`../data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/`](../data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/)。
v2 必须保持 v1 canonical output 与 staging 永久不存在，不能覆盖或重写 v1 identity。

## 正式结果

固定 primary `n=4,B=2` slice 的结果如下。所有 recovery 都按 state 的
`(D(empty)-D(selected))/D(empty)` 计算；没有删除、clamp 或重加权状态。

| Split | States | Mean normalized recovery | Exact at-most-2 match | Exact-2 match |
| --- | ---: | ---: | ---: | ---: |
| Train | 10 | 0.771189 | 3/10 | 3/10 |
| Development | 5 | 0.019571 | 0/5 | 0/5 |
| Overall | 15 | 0.520650 | 3/15 | 3/15 |

主要配对结果：

- overall OCR/RGB minus budget-conditioned independent 为 `-0.292103`，90% trajectory bootstrap interval
  `[-0.625159,-0.046921]`，win/tie/loss=`0/8/7`；
- development 对应差值为 `-0.746732`，interval `[-1.785407,0]`，`0/3/2`；
- overall 相对 recent 为 `+0.144670`，但 interval `[-0.046489,0.451989]` 跨 0；
- overall 相对 analytic random expectation 为 `+0.187471`，interval `[-0.071090,0.569338]` 也跨 0；
- overall OCR/RGB/exact-subset mean recovery 为 `0.520650/0.893364`，ratio of means 为 `0.582796`。

因此这个结果只闭合一个弱的 non-learned similarity comparator。它在 train 上经常恢复较多 utility，但在固定
5 条 development trajectories 上不稳定；不能声称显著优于 recent/random，更不能把它写成 CausalCache
主方法的成功或失败。policy-vision、learned gate、matched-NLL 与 closed-loop 仍未运行。

### 保留的 non-monotone 异常点

development trajectory `0141544666483837` 是唯一负 recovery state，也是 15 个 states 中最小的
`D(empty)`：

- combined similarity 排名为 `4 > 3 > 2 > 1`，选择 `[3,4]`；
- `D(empty)=0.000616045843344`、`D([3,4])=0.002243123482913`；
- raw utility `-0.001627077639569`、normalized recovery `-2.641163246449`；
- exact/true-greedy 选择 `[1,2]`，`D=0.0000470415543532`、recovery `0.923639523160`。

四个 RGB cosine 都在 `0.995689--1.000000`，combined ranking 主要由 OCR overlap 决定；最接近 current 的
events 3/4 恰好增加 frozen-policy distance。raw label、图像/OCR path/SHA、coalition lookup 和算术均一致，
所以这是允许的 non-monotonicity 叠加小分母放大，不是符号错误或重复图。该 state 必须保留在五条 development
denominator 中。去掉它的 leave-one-out 数值只能作为 post-hoc sensitivity，不得替换正式 `0.019571`。

### Artifact identity

- scientific payload SHA256：`5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2`；
- `README.md`：1603 bytes，SHA256 `526347e1928510a3b5f3791631fd35d08e6c53420b9994fd37a4b89fc592b4bf`；
- `state_scores.jsonl`：57,465 bytes，SHA256 `07e29538232620bbea3bb12d1c0ce2649ee505285f53daf006dea3f670ba2d0a`；
- `summary.json`：38,130 bytes，SHA256 `3ae9c27c93c3813f3458d31e1740ff9a97d0c9af6beab9f926f627d8611c7b29`；
- formal Python closure：140 paths，inventory SHA256
  `3d60b7b1ddb60c7a1c377c90ac171304273ecd97fe14109532f3297fff474541`；
- raw labels 与 derived images/OCR 继续复用既有 private HF immutable revisions，没有新增 reusable artifact。

独立审计未 import `code/causalcache` 或 formal reducer：stdlib 读取 raw label USTAR、lexical scan 全部
35/210 JSONL lines，只 semantic-parse allowlisted 15 trajectories 与 75 OCR records；Pillow 12.2.0 从原始 PNG
独立 resize 并重建 16³ histogram。44/44 checks、0 mismatch；60 scores、15 complete rows、geometry
comparators、三 split aggregate、development deltas 与 7×10,000 bootstrap 的 max absolute diff 全为 0，并独立
复原同一 scientific payload SHA。audit projection SHA256 为
`9e8424940a357b63100d1c088b7c72a47e123556dfafccf51f198279830b12df`，remote feature projection SHA256 为
`3bf180327e8e9416553f4b80a50d713512b13cbb9d749465733b7c9235d107e2`，lexical audit SHA256 为
`70ba220dc3fcc3b9b3b0142670a73884f4d125f3f8d719b85ad6b994cb8bb2c9`。这些摘要与 artifact regression 是
Git source of truth；临时审计工作目录不是 canonical artifact。

## 唯一修复

v1 scanner 对每条 JSONL line 要求 identity field 恰好出现一次。immutable derived schema 中：

| 文件 | Lines | Identity field | 每行 raw occurrences | 行内值 |
| --- | ---: | --- | ---: | --- |
| trajectories | 35 | `source_id` | 2 | 35/35 完全相同 |
| OCR records | 210 | `image_member_path` | 1 | 单值 |

trajectory 的两个 `source_id` 分别来自 top-level 与 nested selection metadata。v2 scanner 固定
trajectory=`2`、OCR=`1`，并要求同一行所有 unescaped raw identity literal 都能解码为 UTF-8 且完全相同。字段
occurrence 为 0、数量漂移、escaped value 或行内不一致都会 fail closed。allowlist 之外的行仍只做 byte-level
identity scan，不做 JSON semantic parse；allowlist 内 record parse 后还会再次核对 top-level identity。

共享 reducer 的默认参数仍是 v1 的每行一次，因此 v1 行为没有被静默修复。只有 v2 runner 显式传入 `(2,1)` 与
新的 output protocol ID。unit test 同时证明：v1 default 对重复 identity 仍失败；v2 接受两个相同 occurrence，但
拒绝 1/3 次、不一致、escaped 和缺失字段。

## 保持不变的科学契约

v2 直接加载并 hash 验证 parent v1 contract：

- parent config SHA256：`08f57505d71e603d81d6915ef27cf008d0fe097a56d70a05f8b3519211e2e6f9`；
- parent source commit：`aa5898f25e2e7d647363fe701ac90134bf744a5c`；
- primary slice：15 个 `n=4,B=2` states、75 张 unique images、60 个 candidates；
- OCR：archived `full_spatial_tokens` set Jaccard；
- RGB：Pillow bilinear 256×256、16³ joint histogram cosine；
- combined score：`0.5 * OCR + 0.5 * RGB`，固定选满 top-2；
- at-most-2 exact 与 exact-cardinality-2 oracle、geometry comparators、10k trajectory bootstrap 全部不变；
- restoration label、derived dataset、selector geometry、baseline implementation/runtime identity 全部不变；
- GPU、OCR inference/model load、policy/policy-vision、teacher/KL、gate、matched-NLL、closed-loop、confirm/test
  operation ceiling 全部保持 0。

v1 没有产生 state row、aggregate 或 scientific payload，因此不能声称“v2 scientific values 与 v1 相同”。准确说法
是：parent scientific contract 未变，v1/v2 scientific-value comparison 不存在。

## Source 与 failure binding

v2 contract 会 fail closed 验证：

- `git show aa5898f:parent-config` 与当前 parent config 逐 byte 相同；
- failure commit 是 parent source 的 descendant；
- failure 目录 exact-two files 的 size/SHA 与 failure commit Git blobs 相同；
- failure summary 的 error class/message、零 score/output、负操作计数和 `(35×2, 210×1)` byte forensic；
- shared reducer 的 parent/repaired source SHA256，以及 parent→repair source commit 的 exact changed-path allowlist；
- failure commit 必须是 formal source commit 的 ancestor；
- v1 canonical result 与 `.tmp` 均不存在；
- formal source closure 包含 source commit 下 `code/causalcache/` 与 `code/scripts/` 的全部 tracked Python。

source validator（仅用于 canonical output 生成前的 source freeze）：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_ocr_rgb_contract_v2 \
  --repository-root .. \
  --contract configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json
```

source-only validator 还要求 v2 canonical output 与 `.tmp` 都不存在；result 生成后应使用 formal runner 的
`validate` mode，而不是重新声称 source-only pre-output 状态。

focused tests：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m unittest \
  tests.test_restoration_v2_2_ocr_rgb \
  tests.test_restoration_v2_2_ocr_rgb_v2 \
  tests.test_restoration_v2_2_ocr_rgb_v2_artifact -v
```

## Formal run 与验证

正式 run 已在 Hyper00 Python 3.12.3 / Pillow 12.2.0 CPU runtime 使用下列命令执行；其中 source commit 为
`a9bede85ab8bd10623c5755b944b3c26865c6485`：

```bash
cd /data/CausalCache/code
/data/.venv/causalcache-ocr-v2/bin/python \
  -m scripts.run_restoration_v2_2_ocr_rgb_baseline_v2 run \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json \
  --labels-archive /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar \
  --derived-root /data/tmp/causalcache-restoration-labels-v2-derived \
  --source-git-commit a9bede85ab8bd10623c5755b944b3c26865c6485 \
  --output-dir /data/CausalCache/data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair
```

runner 要求 `HEAD == origin/main == source commit`、branch `main`、worktree clean、新 output directory，且 v1
canonical output/staging absent。canonical v2 output exact-three files 为 `README.md`、`state_scores.jsonl`、
`summary.json`。run 完成后先用同一 source/argv 的 `validate` 做逐 byte replay；result commit/push 后，再从 clean
descendant `main` 重复 committed validation。

正式 acceptance 还要求独立 stdlib audit 从 state rows 重算 coalition、utility/recovery、oracle regret/match/Jaccard、
train/development/overall aggregate 与 paired deltas，并确认 15 states、60 scores、75 image identities、operation
ceilings 和 v1 absence。只有这些全部通过，OCR/RGB comparator 才算闭合。

## 下一步

1. source freeze 已 commit/push；
2. Hyper00 v2 CPU-only aggregate、pre-commit replay 与独立审计已完成；
3. commit/push exact-three lightweight result，再做 descendant validation；
4. 单独冻结并执行 policy-vision feature-only baseline；
5. visual comparator matrix 完整后，才冻结 gate training/evaluation contract。

本 repair 不授权 confirm/test、gate、matched-NLL 或 closed-loop。
