# Restoration v2.2 label-expansion derived artifact

本阶段把冻结的 48/16 structural split 还原成 policy-blind、可供后续 substrate/label runner 使用的 GUIOdyssey
多模态 artifact。它不运行或加载 policy，不产生 restoration value，也不包含任何 per-decision current expert
action。

## 固定 inventory

Formal artifact 固定为：

| 内容 | 数量 |
| --- | ---: |
| Trajectory | 64 |
| Shared history event | 320 |
| Decision view | 192 |
| Image member | 384 |
| OCR record | 384 |

每条 trajectory 保存 event 1--5 和 decision step 4/5/6。共享 event storage 中较后的 action 只对较后的
decision 合法；任何 consumer 都必须按该 decision 的 `history_event_step_ids` 切片，不能把全部五个 events
直接送入较早 state。每个 decision record 本身不保存当前 target action，validator 也拒绝
`validated_action/canonical_action/expert_action` 等字段。

Artifact 固定为六个文件：root `.gitattributes`、root `README.md`、一个 deterministic image tar、一个 OCR
JSONL、一个 trajectory JSONL 和一个 manifest。payload prefix 是
`derived/restoration-v2-label-expansion-v1`，canonical HF dataset repo 继续使用
`gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`；旧 immutable revision 不修改。

## 输入与 fail-closed 边界

Formal build 绑定：

- label-expansion config、parent selection 与已落盘 structural manifest；
- 原 independent v1 config 与 16-Parquet source manifest；
- frozen OCR backend config 与 artifact manifest；
- clean、已 push 的 canonical `main` full SHA；
- 原始 16 个 Parquet 的实时 size/SHA256；
- frozen 48/16 trajectory order、transport row、salted selection hash 与 step 4/5/6 geometry。

Builder 先完成 source rehash、64-row reload、图片路径/格式和 384-member inventory，再初始化 RapidOCR。输入、
source 或 image preflight 失败时不会创建 output。支持实际 `.png/.jpg/.webp`，但 suffix 必须与 magic bytes
一致，每条 trajectory 必须恰好有 observation 000--005。

写盘后立即执行 formal full OCR replay，并逐项验证 exact-six tree、payload member/record counts、OCR aggregate、
event reconstruction、content witnesses、role order、legacy/confirm overlap 和 per-decision slicing。独立 validator
还拒绝 symlink、special entry、额外文件和 generator/input Git-byte drift。

## 正式命令

只有 structural manifest 与 exposure ledger 都已 commit/push 后才运行。`SOURCE_ROOT` 必须指向经 source manifest
固定的 16 个 Parquet；OCR model/wheel 使用现有 frozen backend artifact。

```bash
cd code
python3 -m scripts.build_guiodyssey_restoration_v2_expansion \
  --label-expansion-config configs/causalcache_restoration_v2_2_label_expansion_v1.json \
  --parent-selection-manifest ../data/manifests/restoration_v2_selection.json \
  --expansion-selection-manifest ../data/manifests/restoration_v2_2_label_expansion_selection.json \
  --v1-config configs/independent_reference_gate_v1.json \
  --source-file-manifest ../data/manifests/independent_reference_gate_v1_source_files.json \
  --source-root <SOURCE_ROOT> \
  --ocr-backend-config configs/restoration_v2_ocr_backend.json \
  --ocr-backend-manifest ../data/manifests/restoration_v2_ocr_backend.json \
  --model-dir <FROZEN_OCR_MODEL_DIR> \
  --wheel-dir <FROZEN_OCR_WHEEL_DIR> \
  --git-revision <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --output-dir <NEW_EXCLUSIVE_OUTPUT_DIR>
```

Fresh checkout / fresh download 验证使用同一组输入：

```bash
python3 -m scripts.validate_guiodyssey_restoration_v2_expansion \
  --output-dir <ARTIFACT_ROOT> \
  --label-expansion-config configs/causalcache_restoration_v2_2_label_expansion_v1.json \
  --parent-selection-manifest ../data/manifests/restoration_v2_selection.json \
  --expansion-selection-manifest ../data/manifests/restoration_v2_2_label_expansion_selection.json \
  --v1-config configs/independent_reference_gate_v1.json \
  --source-file-manifest ../data/manifests/independent_reference_gate_v1_source_files.json \
  --ocr-backend-config configs/restoration_v2_ocr_backend.json \
  --ocr-backend-manifest ../data/manifests/restoration_v2_ocr_backend.json \
  --model-dir <FROZEN_OCR_MODEL_DIR> \
  --wheel-dir <FROZEN_OCR_WHEEL_DIR> \
  --git-revision <FULL_CLEAN_PUSHED_MAIN_SHA>
```

当前仅冻结 source；HF upload、immutable revision 与 fresh-download replay 尚未发生。
