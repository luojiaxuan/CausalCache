# Restoration v2.2 OCR/RGB baseline v2 identity repair

## 当前状态

本阶段完成 **source-only freeze**，尚未生成 v2 aggregate。machine-readable contract 是
[`../code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json`](../code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json)，
SHA256 为 `d68cb032ef3c56c330d57329507d409b20f878b7510bd09d88e2eff1f3f898f3`；protocol ID 是
`causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair`。canonical result directory 固定为
`data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/`，source freeze 时必须不存在。

v1 attempt 是 zero-score implementation invalid，而不是 negative scientific result。其 immutable failure binding 已在
`main@2870d8ae26542a184647e8b6d97b8c79e4e12641` 提交，见
[`../data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/`](../data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/)。
v2 必须保持 v1 canonical output 与 staging 永久不存在，不能覆盖或重写 v1 identity。

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

source validator：

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
  tests.test_restoration_v2_2_ocr_rgb_v2 -v
```

## Formal run 与验证

source commit/push 后，在 Hyper00 已验证的 Python 3.12.3 / Pillow 12.2.0 CPU runtime 执行：

```bash
cd /data/CausalCache/code
/data/.venv/causalcache-ocr-v2/bin/python \
  -m scripts.run_restoration_v2_2_ocr_rgb_baseline_v2 run \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json \
  --labels-archive /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar \
  --derived-root /data/tmp/causalcache-restoration-labels-v2-derived \
  --source-git-commit <CLEAN_PUSHED_MAIN_SHA> \
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

1. commit/push source freeze；
2. 在 Hyper00 执行 v2 CPU-only aggregate、pre-commit replay 与独立审计；
3. commit/push exact-three lightweight result，再做 descendant validation；
4. 单独冻结并执行 policy-vision feature-only baseline；
5. visual comparator matrix 完整后，才冻结 gate training/evaluation contract。

本 repair 不授权 confirm/test、gate、matched-NLL 或 closed-loop。
