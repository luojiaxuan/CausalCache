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
- `fixtures/subset_search_scenarios_v1.json`：additive、redundant、complementary trap、mixed/non-monotone 与
  variable-cost 五个 deterministic set-function cases；只用于 CPU search implementation validation。

首次 subset-search CPU attempt 在 pre-commit strict replay 中因 tuple/list JSON-normalization mismatch fail closed；
对应 output 已删除且未进入 Git。serialization regression fix push 后，`results/subset_search_ablation_v1/`
已从 clean `main@7569ce2` canonical rerun，并通过 pre-commit full-payload/JSON/hash replay；scientific payload
SHA256 为 `26846d50...74edf`。目录只含轻量 `summary.json` 与 `README.md`，复用已有 phase-0 config 和旧 v1
`coalition_distances.csv`；不复制 raw policy trace，也不创建新的 HF repo。result push 后已从 clean
`main@45bcf7e` 通过 committed full-payload validation，状态为
`VALID_SUBSET_SEARCH_ABLATION_V1`。

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
- `manifests/restoration_v2_readiness.json`：绑定 execution config SHA、implementation commit 与同一 14-source
  inventory。GPU-0 的首次 `SCREENING_ALLOWED + CONFIRM_LOCKED` 已保留在 Git history/readiness result；
  GPU-2 manifest 已绑定 implementation commit `14faaa4...452aa`；formal clean-Git authorization 已返回
  `SCREENING_ALLOWED + CONFIRM_LOCKED`；
- `results/restoration_v2_derived_artifact/`：完整 derived artifact 的轻量结果、复现参数、UTC brackets、
  superseded download preflight failure 与 negative declarations；
- `results/restoration_v2_selection/`：Hyper00 runtime、exact confirm IDs、失败 attempt 记录与两次
  byte-identical 全量构建结论。
- `results/restoration_v2_gpu_compute_audit/`：Hyper00 单张 H200 policy-blind CUDA audit 与独立 offline
  validation；batch-1/CPU、batch-2/two-batch-1、logits/log-probs、zero-stride、invalid-to-NaN 和
  microbatch-2/no-OOM 全部通过，但明确不关闭 dependency 8。
- `results/restoration_v2_processor_audit/`：保存 Hyper00 上真实 pinned `AutoProcessor` 的轻量审计证据；
  第一次 clean-commit attempt 因真实 Transformers `SizeDict` pixel-limit 表示与 source 假设不符而在任何
  policy output 前 fail closed；GPU-2 re-anchor formal audit 已通过，canonical summary SHA256 为
  `69bb8ddb...d119`。正式审计只
  加载 processor、哈希 model snapshot 与
  Transformers source，不允许 materialize model weights、调用 forward/generate 或产生 policy/restoration
  output；结果必须先由干净 pushed commit 生成，再回写并冻结 exact image-grid/token-shape evidence。
- `results/restoration_v2_substrate_screening/`：第一次 fixed 45-state screening 的轻量 aggregate、完整 runtime/
  input provenance、raw artifact manifest 与 format-only inventory；正式 verdict 为 `NO_GO_V2_SUBSTRATE`，
  strict parse 0/45，confirm 未打开。45 条 native outputs、attempt markers、state records 与 log 只在 private
  HF tar shard，不进入 Git。
- `manifests/restoration_v2_parser_compatibility_golden.json` 与
  `results/restoration_v2_parser_compatibility/`：在 formal replay 前固定 HF/archive/run identity、0/43/40 totals、
  五个 rejection identities 与 45-record classification SHA；clean pushed replay 得到
  `NO_GO_ADAPTER_ONLY`。Git result 只含 output hashes，不含 native text。
- `results/restoration_v2_1_processor_preflight/`、`results/restoration_v2_1_interface_pilot/` 与
  `results/restoration_v2_1_full_45_substrate/`：依次保存 v2.1 official-tools processor PASS、fixed-15 interface
  PASS 和唯一 full-45 substrate NO-GO 的轻量 manifest/reduction。full-45 得到 45/45 parse、32/45 exact
  repeat agreement 与 32 个 memory-sensitive states；raw outputs 只在 private HF deterministic USTAR 中。

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
  `@restoration-v2-derived-v1.0.0` (`89f136abaff797e14fe758a198996e51032a10a6`)；
  exact 6-file derived projection 已 fresh re-download，tree SHA256 `475e6cf2...a6e`，210 条 OCR replay
  通过。
- Restoration v2 first substrate trace：同一 private HF dataset
  `@restoration-v2-substrate-screening-v1.0.0`
  (`c073e143b935a79befd8ab1fd7123796792efad8`)；prefix
  `runs/restoration-v2-substrate-screening-v1/`，deterministic tar SHA256 `c3619a17...a45e`，manifest 与
  archive 已从 immutable revision fresh re-download 验 hash。
- Restoration v2.1 full-45 substrate trace：private HF dataset
  `gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile@v2.1-full-45-substrate-v1`
  (`814506ef1450838d4bc6ed3d89fe53e0773d92fb`)；94-file deterministic USTAR SHA256
  `8cd53d6e...f4fa4`、962,560 bytes，fresh immutable download verified；verdict
  `NO_GO_V2_1_FULL_45_SUBSTRATE`；clean `main@554c51e` committed-binding validation passed。
- Spatial reference audit v1：canonical private HF dataset
  `gavinlaw/causalcache-spatial-reference-audit-mobile@spatial-reference-audit-v1`；Git-tracked source/config、
  `fixtures/spatial_reference_audit_v1_parent_mismatches.json` 和
  `manifests/spatial_reference_audit_v1_exposure.json`。旧 `restoration_v2_exposure.json` 是 immutable
  historical pre-output snapshot；新 child ledger 记录已经发生的 v2/v2.1 output 与本 audit 的零新-output freeze。
  唯一 GPU attempt 的三个 raw profile 为 auto 7/13、eager 13/13、FP32 4/4；原 validator 在 summary 前因
  realized-grid 与 aligned-input 两个读取契约错误 fail closed，profile 未重跑。纯离线 repair、72-member
  deterministic USTAR 与 private HF immutable fresh-download 已闭合，正式 decision 为
  `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`。Git 轻量结果位于
  `results/spatial_reference_audit_v1/`，raw canonical artifact 位于
  `gavinlaw/causalcache-spatial-reference-audit-mobile@d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`。

- Restoration v2.2-eager：唯一 fresh-45 run 已闭合，Git 轻量结果位于
  `results/restoration_v2_2_eager_full_45_substrate/`；raw canonical artifact 位于 private
  `gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile@3577099d505b8c652d764f41269df911128ec767`
  （tag `v2.2-eager-full-45-substrate-v1`）。两个 H200 worker 共享独立 global ledger，固定 even 23 / odd 22
  inventory；45/45 parse/repeat/finite logits，正式为 `PASS_V2_2_EAGER_FULL_45_SUBSTRATE`。102-file USTAR
  SHA256 为 `b22827e6...09fb5`，fresh immutable download 已逐 byte 复核。confirm、restoration 与 gate artifact
  count 当前仍均为 0。

- Restoration v2.2 exact labels：v1 source contract 已在
  `main@3942d687d03bf63ea683fe8ad906a161eb10dc27` 冻结；formal v1 attempt 因 model snapshot directory 不存在，
  在 0 state marker / 0 teacher forward / 0 KL 时永久封存为 `INVALID`。轻量 binding 位于
  `results/restoration_v2_2_eager_labels_v1_attempt/`，原 Hyper00 root/ledger 不删除、不重跑。replacement v2 已从
  `main@5ae40d4aed4eb20b931216776b379bc6ae55629d` 完成 45/45 states，生成 420 条 canonical raw $D(S)$、
  45 个 exact-subset oracle、435 条 deployment conditional-marginal labels 与 465 个 pair interactions；0 retry、
  0 top-up、0 generation。Git compact result 位于 `results/restoration_v2_2_eager_labels_v2/`；raw 101-file USTAR
  位于 private HF `gavinlaw/causalcache-restoration-labels-mobile@8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`
  （tag `v2.2-eager-train-dev-exact-v2`），fresh immutable download 已闭合。gate checkpoint、matched-NLL pairs、
  closed-loop episodes 与 confirm artifact 仍不存在。

- Restoration v2.2 OCR/RGB comparator：v1 formal attempt 在 0 score 时因 identity lexer contract 错误永久
  `INVALID`，轻量 failure binding 位于 `results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/`；新 identity 的
  v2 repair 已从 `main@a9bede85ab8bd10623c5755b944b3c26865c6485` 完成 15-state/60-score CPU-only
  aggregate 与 byte replay。canonical exact-three Git result 位于
  `results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/`，scientific payload SHA256 为
  `5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2`。本步骤复用上述 exact-label 与
  full-derived HF immutable revisions，不创建新的 raw dataset/model artifact；GPU、OCR inference、policy、gate、
  matched-NLL、closed-loop 与 confirm/test operation 均为 0。result commit
  `7e59591573cb31f178dfd07422cc2e3c8aeff573` 已通过 clean descendant Hyper00 committed replay。

- Restoration v2.2 policy-vision comparator：v1 formal attempt 从
  `main@c0937056e94d110cd67e593288f9e0c3a3b24809` 启动，但 pinned PyTorch 2.11 将 device UUID 暴露为
  `torch._C._CUuuid`，与只接受 `str/bytes` 的 v1 probe 不兼容；它在 0 feature、0 model/processor load、
  0 semantic label load 时永久 `INVALID`，canonical output/staging 均不存在。轻量 failure binding 位于
  `results/restoration_v2_2_policy_vision_baseline_v1_attempt/`。UUID type-only v2 contract 使用全新的 identity；
  唯一 v2 attempt 跨过 UUID check 后，又因 pinned `SizeDict` 可转成冻结数值但不是 `Mapping` 而在 0 feature
  fail closed。failure binding 位于
  `results/restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/`；canonical result 仍不存在，因而
  这里不登记 recovery、feature cache 或新 HF artifact。v3 exact-SizeDict repair 已使用新的
  `results/restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair/` canonical identity。唯一 v3 GPU
  run 已生成 exact-three bytes：README/state-scores/summary SHA256 分别为 `70ce9496...ed1f`、
  `8c597726...58b9`、`5ed21d6c...1c0e`，scientific payload SHA256 为 `811e59c7...f48`。首次 CPU validate 因
  recorded evaluated-state 未投影回四键 feature-state 而 fail closed；failure binding 位于
  `results/restoration_v2_2_policy_vision_baseline_v3_cpu_validation_attempt/`。versioned CPU repair v1 随后从
  `main@dbb45637cf79c3573bbbc6051b8b480e3f76d69d` 完成 formal audit，正式 sibling result 位于
  `results/restoration_v2_2_policy_vision_baseline_v3_validation_repair_v1/`，状态为
  `VALID_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`。该目录严格只有 `README.md/summary.json`：
  分别为 836 bytes / `3492dbae9a13d3d1d70e7aacf2cd7b405d1f9cc5956af980c519c8fb3ceee7e9` 与
  10461 bytes / `f7a5ff63a754d06d7b61dcc46516ee2ed22e0a6a3b92b8b0be8a68869cf362b1`。它逐 byte
  重建原 exact-three、记录 15 states / 60 candidates、全零 operation count 与唯一 7-key→4-key projection；
  不产生新 dataset/model，raw labels 继续引用既有 private HF immutable revision
  `8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`。外部 mode-0600 attempt ledger SHA256 为
  `6b65bef9d6edb73ee4275e89923bfd0e6123fb275fccb2b5dda9e57245f7b5d3`，completion seal SHA256 为
  `837c52f403dbb9f21bf968ff5a0ba10199d06fe36ee49431787eff5154c6a52b`；两者不进入 Git。只读 revalidation
  已在独立 validation-repo 的 clean `main@174801112c58d831249fd54f4f8bc9af01524b44` 完成，validation source
  仍是 `dbb45637cf79c3573bbbc6051b8b480e3f76d69d`，runner 返回 `REVALIDATED`。postflight exact-two mode 0644 /
  hashes 与外部 ledger/seal mode 0600 / hashes 均未变化；无 GPU，且没有修改任何 artifact、config、code 或 test。
  执行容器已停止并保留用于审计，exit code 137。

最新轻量运行记录：

- `results/spatial_reference_audit_v1/`：唯一 Hyper01 raw attempt 经零-forward validation repair 后正式得到
  auto 7/13、eager 13/13 与 `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；72-member USTAR 已在 private
  HF immutable revision `d6b2312e...45c1` fresh-download 并完成 canonical rebuild；v2.1 NO-GO 不变；

- `results/restoration_v2_2_eager_full_45_substrate/`：唯一 Hyper00 双 H200 fresh-45 attempt 为 45/45 exact
  repeat、45/45 finite logits、45 memory-sensitive states，正式 PASS；deterministic USTAR 与 private HF
  immutable revision `3577099d...c767` 已 fresh-download 闭合，restoration/gate/confirm count 均为 0；

- `results/restoration_v2_2_eager_labels_v2/`：replacement formal label attempt 45/45 PASS；保存 immutable HF/raw
  binding 和 policy-free scientific reduction。$B=2$ oracle normalized recovery 为 overall/train/dev
  0.8735/0.9028/0.8149；75/435 conditional marginals 为负且覆盖 22/45 states。该目录不包含 raw labels，也不
  声称 gate、matched-NLL 或 closed-loop 结果；

- `results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair/`：15 个 primary `n=4,B=2` states 的 frozen
  OCR-token/RGB-histogram comparator；train/development/overall recovery 为
  `0.771189/0.019571/0.520650`，exact match 为 `3/10、0/5、3/15`。唯一 development 负 recovery state 保留，
  不做 post-hoc 删除或 clamp；这是已闭合的弱 similarity baseline，不是 learned gate 或 closed-loop 结果；

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

- `results/restoration_v2_gpu_compute_audit/`：Hyper00 GPU-2 formal H200 compute audit 通过，canonical summary
  SHA256 `dce79769...9dac`；独立 validator 从 run commit Git blobs 复核通过，policy/restoration output 均为
  false；

- `results/restoration_v2_processor_audit/`：Hyper00 GPU-2 real `AutoProcessor` formal summary 已通过；exact
  grids/tokens/tensors、source hashes、零 model-tensor/forward/generate/output declarations 均已冻结。该子项
  已完成；GPU-2 execution config/readiness authorization 已重新通过；

- `results/restoration_v2_readiness/`：GPU-2 clean-Git formal authorization 返回 8/8 passed、
  `SCREENING_ALLOWED + CONFIRM_LOCKED`，canonical summary SHA256 `36bd183f...aa02`；dependency 8 已重新闭合，
  仍不授权 confirm；首次 GPU-0 summary 归档为 `gpu0-summary.json`；

- `results/restoration_v2_runtime_reanchor/`：screening preflight 发现旧 GPU 0 busy 后，在空闲 physical GPU 2
  的新单卡容器重新完成 GPU compute + independent validation + real processor audit；三份 evidence 均为
  policy-output-free。该里程碑结束时 GPU-2 config/readiness 尚未重签，screening 当时保持暂停；

- `results/restoration_v2_substrate_screening/`：Hyper00 physical GPU 2 完成固定 45-state run；90-prompt shape
  sweep 通过，45/45 在第一次 reference generation strict parse 失败，teacher forward/KL/restoration label
  均为 0，合法 verdict `NO_GO_V2_SUBSTRATE`；无 retry/top-up/confirm access；

- `results/restoration_v2_parser_compatibility/`：从 `main@fc3adf1` 正式 replay immutable raw archive；strict
  0/45、single-action canonical-first 43/45、conservative recovery 40/45，低于 required 45/45，输出
  `NO_GO_ADAPTER_ONLY`；25 个 clean EOF、15 个 suffix recovery、0 个 model-emitted canonical closer；

- `results/restoration_v2_1_full_45_substrate/`：唯一 Hyper00 full-45 attempt 得到 90/90 parseable
  generations、32/45 exact canonical repeat agreement、96 次 teacher forward、64 次 GPU KL 与 32 个
  memory-sensitive states；冻结 reducer 输出 `NO_GO_V2_1_FULL_45_SUBSTRATE`，restoration/confirm/gate work
  全为 0；

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
