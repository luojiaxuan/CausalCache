# Data 目录

本目录只保存适合 Git review 的小数据与轻量实验记录，不是大型 artifact 仓库。

```text
data/
├── fixtures/    # 单测和 contract 使用的最小确定性 fixture
├── manifests/   # 轻量 source/artifact index，不含 raw screenshots
└── results/     # README、summary.json、轻量 CSV/manifest
```

以下内容禁止作为唯一副本留在 Git 或共享机器本地盘：raw screenshots、完整 rollout traces、生成
dataset、model weights、checkpoint、adapter。它们以 Hugging Face dataset/model repo 为 canonical
source，并在顶层 `README.md` 和相关 result README 中记录 repo、revision/tag、schema 与生成命令。

`data/cache/`、`data/raw/`、`data/staging/`、`data/local/` 与 `data/tmp/` 已被 Git 忽略，只能用于
本地短期 staging。注意：仓库相对路径 `data/` 与容器持久挂载点绝对路径 `/data` 是两个不同概念。

当前 v2 interface fixtures：

- `fixtures/gui_owl_v2_action_roundtrip.json`：14 个合法、23 个非法 native action cases，覆盖全部
  canonical actions、aliases、system buttons 与 coordinate endpoints；
- `fixtures/restoration_v2_prompt_low_fidelity.json`：synthetic step-6 state，固定 events 1--5、四个
  candidates、八字段 summaries 与 current-equivalent event 5；同一 prefix fixture 还覆盖 development
  steps 4/5，validator 共穷举 28 个 coalitions，其中 step 6 为 16 个；
- `manifests/restoration_v2_interfaces.json`：上述 fixtures、v2 action/prompt/LF code 与接口说明的逐文件
  size/SHA256。它只证明本地 CPU interface validation；真实 AndroidWorld 运行状态由独立 result evidence
  给出，不事后改写 source manifest 的 pre-evidence sentinel。
- `fixtures/restoration_v2_ocr_golden.json`：内嵌 2x2 RGB 与两行 English OCR PNG bytes，source/prepared
  hashes 和两次 byte-identical Hyper00 inspection 的 expected fields 已冻结；最终 static fixture 已从新
  pushed commit 两次独立通过 `validate-golden`，canonical evidence 位于
  `results/restoration_v2_ocr_backend/`。

冻结后的 constructor run 证据位于 `results/restoration_v2_constructor_preflight/`：14/14 payload 已被
pinned `JSONAction` 接受。formal device-side result 位于 `results/restoration_v2_executor_dispatch/`：14/14
cases 与 negative control 已通过独立 reducer。interface manifest 的 `pending` 是运行前 source snapshot，
不被事后改写；实际 run status 以 result summary 为准。

restoration v2 exact selection/exposure 产物：

- `manifests/restoration_v2_selection.json`：111-trajectory eligible pool、8/15/20 role proof、exact
  30/15/20 states 与 65 个 content witnesses，SHA256 `292c7e52...`；
- `manifests/restoration_v2_exposure.json`：append-only pre-output exposure evidence，SHA256
  `bc122482...`；
- `manifests/restoration_v2_ocr_backend.json`：14 个 Git source、private HF OCR model revision、6 个 HF file
  hashes，以及 real-screen source/result、private HF dataset revision、5 个 dataset file hashes 与三次
  replay 的完成态 fail-closed index；当前 `dependency_5_closed=true`；
- `manifests/restoration_v2_real_screen_source.json`：在任何 real-screen OCR output 前冻结的 17-file source
  contract，记录 45 states / 180 occurrences / 75 unique images、55/20 orientation、confirm SHA overlap 0
  与 exact 3+3 screenshot keys；当前只是 source truth，不是 OCR result 或 derived artifact 完成证据；
- `results/restoration_v2_ocr_backend/real_screen_summary.json`：Hyper00 两次 materialization、两次独立 replay、
  private HF upload/tag、fresh immutable re-download 后第三次 replay 的完整 argv、UTC brackets、runtime、
  5-file hashes 与 negative declarations；
- `manifests/restoration_v2_baselines.json`：冻结五个 non-oracle baselines、final-main-merger vision extractor、
  GUI-Owl snapshot/Transformers source identity 与 11 个 Git source SHA；当前 `dependency_6_closed=true`；
- `manifests/restoration_v2_derived_artifact.json`：绑定 builder commit、两次 byte-identical Hyper00 build、
  private HF tag/immutable revision、exact 6-file hashes 与 fresh-download replay；当前
  `dependency_1_closed=true`；
- `results/restoration_v2_derived_artifact/`：完整 derived artifact 的轻量结果、复现参数、UTC brackets、
  superseded download preflight failure 与 negative declarations；
- `results/restoration_v2_selection/`：Hyper00 runtime、exact confirm IDs、失败 attempt 记录与两次
  byte-identical 全量构建结论。

当前 reusable artifacts：

- Independent GUIOdyssey gate：private HF dataset
  `gavinlaw/causalcache-guiodyssey-independent-mobile@v0.1.0`
  (`84c9f5a335e9612ccb4bd566f977574f359b2485`)；schema v0.4，reference 8 trajectories/75 decisions，
  oracle 15/132；
- Independent UI-TARS reference run：同一 private HF dataset
  `@reference-gate-v1` (`b3e1245c6c6a1723fe2ca3a861148008df39df46`)；raw per-decision summary 与
  GPU monitor 位于 `runs/independent-reference-gate-v1/`，immutable re-download verified；

- GUIOdyssey pilot：private HF dataset
  `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- GUI-Owl AndroidWorld validation traces：private HF dataset
  `gavinlaw/causalcache-androidworld-validation-mobile@v0.2.0`
  (`0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`)；旧 Instruct artifact 保留在 immutable
  `v0.1.0`。
- Restoration v2 OCR models：private HF model
  `gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en@v1.0.0`
  (`0dbc766a73ee88d10d52285d434dbfec58617835`)；三份 ONNX、model card 与 manifest 已 fresh
  re-download，6/6 file hashes verified。
- Restoration v2 real-screen OCR golden：private HF dataset
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@ocr-real-screen-golden-v1.0.0`
  (`9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`)；5/5 files fresh re-download verified，完整 tree
  SHA256 `605d6396...7e25`；该旧 tag 保持不变。
- Restoration v2 full derived artifact：同一 private HF dataset
  `@restoration-v2-derived-v1.0.0` (`89f136abaff797e14fe758a198996e51032a10a6`)；repo `main` 为 9 files，
  exact 6-file derived projection 已 fresh re-download，tree SHA256 `475e6cf2...a6e`，210 条 OCR replay
  通过。

最新轻量运行记录：

- `results/restoration_v2_ocr_backend/`：保留首次 package-source key 冲突与 mutable-status 两个 superseded
  attempt；最终 static fixture 已从 pushed commit 两次通过 `validate-golden` 且 byte-identical。synthetic
  golden、immutable HF model artifact、real-screen golden 与 HF dataset immutable re-download 均 passed；

- `results/restoration_v2_derived_artifact/`：builder commit `1a01f2323647d092cab67f0531ecb877a4a255de`
  在 Hyper00 CPU-only runtime 两次构建 exact 6-file bytes 相同；HF immutable re-download 后第三次
  210-record replay 通过。三次均未加载 policy 或产生 restoration output；dependency 1 已 passed；

- `results/restoration_v2_selection/`：Hyper00 formal selection/exposure 通过独立 validator；20 条
  confirm trajectories 已按 fixed first-20/no-top-up 规则冻结，未加载 policy 或使用 GPU；

- `results/restoration_v2_executor_dispatch/`：Aries formal attempt 通过 14/14 cases，negative actuation
  control 为 HTTP 500，canonical verdict 为 `PASSED_EXECUTOR_DISPATCH`；未加载 policy 或使用 GPU；

- `results/independent_reference_gate_v1/`：正式独立 reference 得到 69/75 parsed、27/75 match、swipe
  0/2，合法输出 `NO_GO_CURRENT_REFERENCE_STACK`；oracle split 未运行；

- `results/ui_tars_hyper00_hardware_anchor/`：Hyper00 H200 精确复现旧 A6000 UI-TARS 9/9 parsed、4/9
  executable-match vector，允许独立 reference gate 留在 Hyper00；

- `results/gui_owl_1_5_8b_think_smoke_strict/`：Hyper01 首次 1/5-image Think checkpoint smoke；
  finite logits 通过，strict parser 因闭合 thinking prefix 按设计拒绝。
- `results/gui_owl_1_5_8b_think_smoke/`：format-only parser 适配后的原参数重跑；finite logits
  与 parse 均为 2/2，interface smoke 通过。
- `results/gui_owl_1_5_8b_think_androidworld_validation/`：Aries 42-checkpoint 数学 early-stop；
  parse 512/513，但 official-success 上界 29/62，candidate 被有效拒绝。
