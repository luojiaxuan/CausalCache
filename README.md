# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

> Target venue: AAAI
> Status: v2 substrate = `NO_GO_V2_SUBSTRATE`; adapter-only replay = `NO_GO_ADAPTER_ONLY` (40/45 < required 45/45) / v2.1 fixed-15 interface pilot = `PASS`、full-45 substrate = `NO_GO_V2_1_FULL_45_SUBSTRATE` (32/45 exact repeat agreement) / bounded spatial audit = `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY` (auto 7/13、eager 13/13；immutable HF closed) / v2.2-eager fresh-45 = `PASS` (45/45 exact repeat；immutable HF closed) / restoration label v1 = zero-forward `INVALID` / restoration label v2 = `PASS` (45/45；immutable HF closed) / confirm locked

## 团队交接入口

当前路线已经从 v1 expert-aligned admission 切换为 versioned v2 stable self-behavior estimand。科学配置
[`code/configs/causalcache_restoration_v2.json`](code/configs/causalcache_restoration_v2.json) 已在任何 v2
policy output 前冻结，SHA256 为
`9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc`。primary policy 固定为
`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faec`；reference 只要求 restricted action 可解析、logits finite、
两次 canonical action 一致，expert alignment 仅分层报告且不能过滤状态。完整定义、data exposure 和
go/no-go 阈值见 [`docs/restoration_v2.md`](docs/restoration_v2.md)。第一次固定 development screening 已产生
45 个 v2 native policy outputs，但 strict parse 为 0/45，因此没有 teacher forward、KL、restoration label、
gate checkpoint 或方法效果结果；confirm 仍 locked。

新的 v2.1 interface rescue 已在任何 v2.1 policy output 前冻结为独立协议，machine-readable contract 是
[`code/configs/causalcache_restoration_v2_1_pilot.json`](code/configs/causalcache_restoration_v2_1_pilot.json)，
SHA256 为 `9d51a2ed5d6cc382f297c1b8af3100d784090f72800d637136b88982763fdbf7`。它改用 checkpoint
official `tools=` chat template、要求模型原生 `</tool_call>`、删除 `Action:` carrier，并固定先做零 policy
output 的 90-prompt processor preflight，再对 exact 15 个 `v2_development` states 各生成一次。pilot 必须
15/15 strict whole-output parse、15/15 model-emitted closer、15/15 AndroidWorld bridge，任何一次失败即
`NO_GO_V2_1_INTERFACE_PILOT`，不得 retry/top-up/teacher/KL/restoration，也不得向 decoder/processor 暴露
任何 confirm state/prompt/image。完整边界见
[`docs/restoration_v2_1.md`](docs/restoration_v2_1.md)。90-prompt real `AutoProcessor` preflight 已在 Hyper00
CPU-only 正式通过：90/90 official-tools prompts 合法，context overflow 为 0，input token 长度为
3,759--15,013；policy model 未实例化、weights 未 materialize 为 tensors，且没有 forward、generation 或
GPU operation。raw JSON 已上传 private HF，并由 fresh immutable download 逐 byte 复核。随后唯一 fixed-15
attempt 在 clean `main@70724bd` 正式通过：15/15 strict parse、15/15 model-emitted closer、15/15
AndroidWorld bridge，retry/top-up/truncation/teacher/KL/restoration/confirm generation 全为 0。raw 34-file
USTAR 已上传 private HF，并按 immutable revision fresh-download 验证。loader
会验证并 hash 包含 confirm bytes 的完整 artifact，但提供给 decoder/processor 的 confirm state、prompt、image
均为 0，`confirm_processor_prompt_count=0`、confirm generation count 也为 0。
唯一正式 attempt 固定为 `restoration-v2-1-interface-pilot-v1`，持久路径固定为
`/data/experiments/causalcache/restoration-v2-1-interface-pilot-v1`；runner 拒绝 alternate output path，已有
incomplete marker 也不得通过换目录重跑。global ledger/root/run manifest 会在 policy runtime import/model
construction 前 durable claim；constructor/OOM 与中断残留只会落成可打包 `INVALID`，`--resume` 不会继续
generation。raw pilot evidence 使用 deterministic USTAR 上传 private Hugging Face，Git 只保存 immutable
revision、hash 与 compact reduction。

unchanged-interface full-45 已冻结为独立 child contract
[`code/configs/causalcache_restoration_v2_1_full_45.json`](code/configs/causalcache_restoration_v2_1_full_45.json)，
SHA256 为 `0924dd66fab9440bed66585765e9b1f5ab6fb80fbdf65a1492efba7ae81e116b`。它固定 30 个
`v2_label_train` + 15 个 `v2_development` states、每 state 两次 fresh full-history generation、三次
teacher forward 与两次 GPU KL；全 attempt 上限为 90/135/90。45/45 parse、finite logits 与 canonical
agreement 都是硬门槛，且至少 8 个 states 的 summary KL 必须超过 repeat-noise epsilon。唯一正式 run
得到 45/45 strict parse、90/90 closer/bridge 和 32 个 memory-sensitive states，但两次 exact canonical action
完全一致的只有 32/45，低于冻结的 45/45 gate，因此结果为 `NO_GO_V2_1_FULL_45_SUBSTRATE`。13 个 mismatch
均保持 action type，包含 12 个 `click→click` 与 1 个 `swipe→swipe` coordinate jitter；这只是描述性分析，
不能把当前结果事后改写为 PASS。raw evidence 已上传 private HF 并从 immutable revision fresh-download
逐 byte 验证；manifest push 后，clean `main@554c51e` 的 committed-binding validator 返回
`VALID_RESTORATION_V2_1_FULL_45_ARTIFACT` 与同一 NO-GO。轻量结论见
[`data/results/restoration_v2_1_full_45_substrate/`](data/results/restoration_v2_1_full_45_substrate/)，完整契约与
resume/INVALID/artifact 边界见 [`docs/restoration_v2_1_full_45.md`](docs/restoration_v2_1_full_45.md)。

针对 13 个已暴露 coordinate mismatch 的 bounded numerical audit 已完成 source-only 冻结，见
[`docs/spatial_reference_audit_v1.md`](docs/spatial_reference_audit_v1.md) 与
[`code/configs/spatial_reference_audit_v1.json`](code/configs/spatial_reference_audit_v1.json)。它只比较同一 H200 上的
BF16 auto、BF16 eager numerical control 和 4-state FP32 descriptive probe，总上限为 60 次 generation、120 次
teacher forward；confirm、restoration、gate training 与 coordinate radius tuning 均为 0。旧 raw archive、13-state
父投影、source inventory、唯一 no-retry attempt ledger 和 append-only exposure child ledger 都由 hash fail closed。
严格 CUDA determinism 因需要项目禁止的 `CUBLAS_WORKSPACE_CONFIG` 而不作声称；image digest、Python、
PyTorch/CUDA/cuDNN、Transformers、driver、scientific environment absence 和 observed attention backend 都在
durable claim 前 exact 核对。判定分为 eager unstable→semantic、eager stable/auto unstable→eager-specific
recovery、两者均 stable→本次 numerical audit inconclusive；FP32 与 teacher-forced margin 都不控制结论，后者也
不是旧 generation-time margin。唯一 Hyper01 attempt 已完成全部 60 次 generation 与 120 次 teacher forward：
BF16 auto 为 7/13 exact stable、BF16 eager-control 为 13/13、FP32 描述性 probe 为 4/4；confirm、restoration
和 gate 均为 0。原独立 validator 随后因把 processor target `2560` 误当成所有 realized grid 的固定 token 数而
fail closed，且还把 teacher aligned inventory 错写为 `attention_mask + input_ids`。raw profile 与 terminal
ledger 已冻结且未重跑；纯离线 validation repair 只修这两个读取契约，并要求所有 shape metadata 与固定 parent
raw witness exact，见
[`docs/spatial_reference_audit_v1_validation_repair.md`](docs/spatial_reference_audit_v1_validation_repair.md)。
repair、72-member deterministic USTAR 与 private HF immutable fresh-download 已全部闭合，正式归约为
`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；这只支持冻结新的 `v2.2-eager` runtime，不改写 v2.1 NO-GO，
也尚未授权 restoration 或 confirm。轻量结果见
[`data/results/spatial_reference_audit_v1/`](data/results/spatial_reference_audit_v1/)。

[`docs/restoration_v2_2_eager.md`](docs/restoration_v2_2_eager.md) 与
[`code/configs/causalcache_restoration_v2_2_eager.json`](code/configs/causalcache_restoration_v2_2_eager.json)
冻结了新的 eager child。它继承 v2.1 official-tool interface、45-state projection、gate 与
90/135/90 operation ceiling，唯一 scientific delta 是 BF16 eager fixed-seed/TF32-off numerical control；不声称
strict CUDA determinism。正式 attempt 必须 fresh 跑 45 states，并固定在同机同容器的两张 H200 上按偶数
index 23 states / 奇数 index 22 states 分工。global sibling ledger 必须在两个 worker import runtime 前统一
claim；旧 v2.1 raw/ledger/state output 不能复用。唯一 Hyper00 attempt 已正式通过：45/45 parse、45/45 exact
repeat、45/45 finite logits、45 个 memory-sensitive states，90/135/90 calls 精确命中，retry/top-up 为 0。
confirm、restoration 与 gate work 仍全为 0。

正式 source freeze 已在 clean pushed `main@b3a6303d69b1145fbf195e0bd18b9b3065a6f213`
建立：contract SHA256 为
`f473bb8a1657072235dd73bf78a93aff27b438d7baca65b7ef6096cf985effa7`，50-file formal inventory
SHA256 为 `bb6351e4dd5b4470ed1add86dbcbaa714a56063ba8ecf07268b6f13f630f9e54`。执行 source 为
`main@8ae07519f14ac3635f292ee93a7b6d624507427e`；102-file deterministic USTAR 已在 private HF revision
`3577099d505b8c652d764f41269df911128ec767` fresh-download 并逐 byte 复核。轻量结果见
[`data/results/restoration_v2_2_eager_full_45_substrate/`](data/results/restoration_v2_2_eager_full_45_substrate/)。

restoration v2.2 label v1 因错误 model path 在 zero-forward 阶段永久封存为 `INVALID`；replacement v2 已从
`main@5ae40d4aed4eb20b931216776b379bc6ae55629d` 完成唯一双 H200 formal run：45/45 states `PASS`，生成
420 条 canonical $D(S)$、435 条 deployment conditional-marginal labels、45 个 exact-subset oracle 与 465 个
pair interactions，retry、top-up、generation 均为 0。$B=2$ exact oracle 的 normalized recovery 为 overall
0.8735（train 0.9028 / dev 0.8149），44/45 states 获得正 oracle utility；75/435 marginals 为负且分布在
22/45 states，说明后续 gate 需要保留 set conditioning，而不是证明 learned gate 已有效。

raw label USTAR 已在 private HF
[`gavinlaw/causalcache-restoration-labels-mobile@8f6baae5`](https://huggingface.co/datasets/gavinlaw/causalcache-restoration-labels-mobile/tree/8f6baae5c0b23b08915fa1b0fb848dd519b4c8db)
fresh-download 闭合，tag 为 `v2.2-eager-train-dev-exact-v2`。Git compact result 见
[`data/results/restoration_v2_2_eager_labels_v2/`](data/results/restoration_v2_2_eager_labels_v2/)，完整协议与
artifact identity 见 [`docs/restoration_v2_2_labels.md`](docs/restoration_v2_2_labels.md)。这只是 offline
oracle/labels；gate training、matched-NLL、closed-loop 和 confirm 均未运行。

这条路线与评审建议的关键映射已经冻结：reference estimand 是 stable self-behavior，高保真干预只加入
post-state image，八字段 strong low-fidelity summary 已实现；v2.1 只修复 versioned policy interface，不改变
这些科学设计，也不覆盖旧 v2 negative results。

合作方提出的 subset optimality 问题已拆成独立 search ablation，见
[`ablations/subset_search.md`](ablations/subset_search.md)。v1 source/config/CPU runner 已冻结：小规模 exact
subset oracle、true conditional-marginal greedy、2x2 bounded exchange、true-utility beam-$2/4/8$ 分别报告
actual utility、greedy/exact ratio 与全部 coalition query cost；同时把既有 phase-0 的 averaged-score
objective-projection gap 与真正 search regret 分开。首次 CPU attempt 因 tuple/list JSON boundary 在 pre-commit
replay fail closed，output 未保留；修复 push 后，canonical rerun 与独立 pre-commit replay 已逐项通过。正式
结果中 phase-0 averaged-score / true-greedy ratio 为 0.858594/1.0；complementary trap 的 raw greedy/beam-2
为 0.5，exchange/beam-4 为 1.0；旧 v1 cached table 的四个 state/budget cases 中 greedy 全部等于 exact。
这支持保留 set-conditioned greedy 主线与 offline stronger-search diagnostics，但不是新 real-policy generalization
evidence。全程零新 policy/GPU/confirm access，v2.1 不重开。轻量结果见
[`data/results/subset_search_ablation_v1/`](data/results/subset_search_ablation_v1/)。

official Jinja/tojson compatibility 已在输出前冻结：teacher target 与 official assistant `tool_calls` JSON
逐字节一致，包括 canonical insertion order 与原始 Unicode UTF-8；解码后的 text argument 仍须严格满足 NFKC。
interface source SHA256 为 `90cbefc851bed105de6ea0c8f719aae6313a479ca4589e4160fc4ec3e8de3964`。

v2 CPU interface 已独立实现并 hash-pinned，见
[`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md) 与
[`data/manifests/restoration_v2_interfaces.json`](data/manifests/restoration_v2_interfaces.json)。14 个合法
action、23 个非法 action、6,000 个完整标量坐标检查和 decision steps 4/5/6 共 28 个 coalition（含
step-6 全部 16 个）已通过；pinned AndroidWorld `JSONAction` constructor 已在 Aries 对 14/14 payload
通过，证据见
[`data/results/restoration_v2_constructor_preflight/`](data/results/restoration_v2_constructor_preflight/)。
device-side executor dispatch 已在 Aries 正式通过 14/14 cases，negative actuation control 为 HTTP 500，
独立 reducer verdict 为 `PASSED_EXECUTOR_DISPATCH`；证据见
[`data/results/restoration_v2_executor_dispatch/`](data/results/restoration_v2_executor_dispatch/)。因此 action
dependency 已闭合。baseline 公式、
GUI-Owl final-main-merger extractor、完整 snapshot/runtime verifier 与 11-file source manifest 已通过，
dependency 6 已闭合。

OCR/image backend 的 implementation identity 已冻结为 CPU-only
`RapidOCR 3.8.4 + ONNX Runtime 1.24.4 + PP-OCRv5 mobile English`，两个 wheel、det/rec 与虽关闭但
constructor 仍加载的 classifier model 都有 exact SHA；256x256 Pillow bilinear 与 uncapped full-screen
OCR record schema 也已实现。synthetic golden 已从最终 static fixture 两次独立通过；private HF model
`v1.0.0@0dbc766a73ee88d10d52285d434dbfec58617835` 已上传、按 immutable revision fresh re-download 并逐文件
验 hash，Git source/artifact manifest 也已通过 fail-closed validator。6-image policy-blind real-screen
golden 现已完成：pre-output source contract、materializer 与独立 replay validator 先冻结，45 个 screening
states 展开为 180 个
occurrences、75 张 unique images（55 portrait / 20 landscape），confirm 有 97 张 unique images 且 SHA
交集为 0；exact 3+3 screenshots 在未读取 OCR/policy output 时固定。Hyper00 两次独立构建与 replay、HF
immutable re-download 后第三次 replay 全部通过，完整 5-file tree SHA256 为 `605d6396...7e25`。private HF
dataset tag `ocr-real-screen-golden-v1.0.0` 固定到 `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`；OCR
dependency 5 已闭合，该旧 tag 保持可复现。见
[`docs/restoration_v2_ocr.md`](docs/restoration_v2_ocr.md)。

完整 derived dataset 已从 builder commit
`1a01f2323647d092cab67f0531ecb877a4a255de` 在 Hyper00 CPU-only runtime 独立构建两次：35 条
trajectory、175 个 event、65 个 state、210 张图与 210 条 OCR record 的 exact 6-file projection
byte-identical，tree SHA256 为 `475e6cf2...a6e`，OCR aggregate 为 `1e04ddbd...010`。private HF tag
`restoration-v2-derived-v1.0.0` 固定到 immutable revision
`89f136abaff797e14fe758a198996e51032a10a6`；fresh immutable download 后第三次 210-record replay 通过。
dependency 1 已闭合，且三次均未加载 policy 或生成 restoration output。详细证据见
[`data/results/restoration_v2_derived_artifact/`](data/results/restoration_v2_derived_artifact/)。

dependency 8 的 source implementation 前置现已就绪：
[`code/causalcache/restoration_v2_gpu_kl.py`](code/causalcache/restoration_v2_gpu_kl.py) 将 BF16
candidate logits 的 FP32 `log_softmax`、full-vocabulary KL 与 action-token mean 留在同一 CUDA
device；finite/normalization/nonnegative/final-finite predicates 也全部留在 device，非法数值只映射为最终
`NaN` distance，primitive 记录 `validation_scalar_host_reads=0` 与
`full_tensor_host_transfers=0`；
[`code/causalcache/restoration_v2_batching.py`](code/causalcache/restoration_v2_batching.py) 固定
microbatch size 2，按 exact `(image_count, sequence_length)` 分组且禁止自动 OOM
fallback；[`code/causalcache/policy/gui_owl_v2_runtime.py`](code/causalcache/policy/gui_owl_v2_runtime.py)
实现 pinned snapshot/Transformers source 校验、单卡 BF16、每图 target 2560 effective visual tokens、native
batch-1 generation，以及只在 GPU 返回 tool-call distance span logits 的 batch-1/2 teacher forcing。
synthetic-only CUDA audit CLI 也已实现。Hyper00 单张 H200 formal run 已从 clean pushed commit 通过：
batch-1 对独立 float64 CPU oracle 的最大误差为 `3.05e-8`，batch-2 对两次 batch-1 完全一致，
zero-stride/NaN/no-host-intermediate-read 均通过；独立 validator 已从 run commit Git blobs 复核 source 与
全部关键字段。证据见
[`data/results/restoration_v2_gpu_compute_audit/`](data/results/restoration_v2_gpu_compute_audit/)。这仍只是
policy-blind compute audit。真实输入执行底座现已实现：confirm-safe loader 只暴露
label-train/development 的 45 states / 90 images；processor-only audit 会在不实例化 model weights、forward
或 generate 的前提下验证真实 `AutoProcessor` 的 1-image、5-image、nested batch-2、token boundary 与 exact
pixel target；readiness validator 必须同时复核 GPU audit、processor audit、全部 source Git blobs、clean
pushed `main` 和 `CONFIRM_LOCKED`，才返回 `SCREENING_ALLOWED`。production runner 随后先完成 90-prompt
shape sweep，再允许首个 policy output，并以 attempt marker 禁止崩溃后的 hidden retry。上述 source 已通过
本地 tests。正式 Hyper00 processor audit 也已通过：真实 grid 对齐后每图 2,584 effective visual tokens，
1/5-image sequence lengths 为 2,943/13,286，且所有 model/policy/restoration negative declarations 均为 false。
证据见 [`data/results/restoration_v2_processor_audit/`](data/results/restoration_v2_processor_audit/)。首次 GPU-0
execution config/readiness 已在 clean `main` 通过，历史授权见
[`data/results/restoration_v2_readiness/`](data/results/restoration_v2_readiness/)；正式 preflight 随后发现该
physical GPU 已被其他任务占用，因而没有启动 screening。
空闲 GPU 2 上的新单卡 runtime 已重新通过 GPU compute 与 processor audits，证据见
[`data/results/restoration_v2_runtime_reanchor/`](data/results/restoration_v2_runtime_reanchor/)；planned GPU-2
canonical execution config 已改绑新 container/GPU/evidence，SHA256 为 `819cb973...91ca0`。runner 还会在
artifact/model load 前实时核对 GPU UUID、单卡可见性、driver、compute capability、PyTorch/CUDA/cuDNN 与
Transformers。GPU-2 readiness manifest 已绑定 clean implementation commit `14faaa4...452aa`，formal
clean-Git authorization 在 `main@caa4f37` 返回 8/8、`SCREENING_ALLOWED + CONFIRM_LOCKED`；当前结果见
[`data/results/restoration_v2_readiness/`](data/results/restoration_v2_readiness/)。

第一次 fixed 45-state substrate screening 已在 `main@a0001cb`、Hyper00 单张 physical GPU 2 完整执行。
90-prompt shape sweep 通过，45 个 state 均生成一次；但冻结 parser 因 native envelope mismatch 在
`reference_generation_1` 得到 45/45 `PARSE_FAILURE`，正式输出 `NO_GO_V2_SUBSTRATE`。因此 teacher forward、
KL 与 restoration label 均为 0，这不是“logits 不 finite”或“memory 不敏感”的测量。原始 trace 已打成单个
deterministic tar shard 上传 private HF，轻量结论与 immutable revision 见
[`data/results/restoration_v2_substrate_screening/`](data/results/restoration_v2_substrate_screening/)。只读审计中
保守的单动作 envelope-only normalization 上界为 40/45，仍低于 0.99 gate；原 v2 结果不改写，confirm 保持
locked。用于复核该上界的 versioned compatibility parser 与 immutable archive replay 已完成 source
implementation；40 条中只有 25 条在首个 JSON 后 clean EOF，另外 15 条需要丢弃精确白名单中的残余 opener/
extra brace，且 0 条由模型生成 canonical closer，因此不能称为 native well-formed output。auditor 会逐项绑定
HF immutable revision、archive/run-contract/source commit 与 pre-registered per-state classification hash，重新执行
strict parser、保守 adapter、canonical round-trip 与 AndroidWorld bridge。该正式 replay 已在
`main@fc3adf1` 完成，得到 `NO_GO_ADAPTER_ONLY`；完整结果见
[`data/results/restoration_v2_parser_compatibility/`](data/results/restoration_v2_parser_compatibility/)。当时必须先
冻结 versioned interface/generation rescue；该步骤后来已由 v2.1 processor、pilot 与 full-45 milestones
完成并得到独立结果。

exact-ID/exposure materializer 已实现为 policy-blind CPU pipeline：它必须从 pinned 16 个 Parquet 重建
完整 111-trajectory eligible pool，并逐字节复现 frozen pool SHA，不能误从只含 8+15 条 trajectory 的
parent tar 继续抽样。pipeline 固定 `decision_count>=5` 后的首 20 条、step 6、8/15/20 disjoint proof，
同时为 45 个 screening states 和 20 个 confirm states 写 image/action content witnesses。Hyper00 已从
`main@30879c0` 完成两次 byte-identical 全量构建，两次均通过独立 validator；canonical
selection/exposure SHA 分别为 `292c7e52...` / `bc122482...`，见
[`data/results/restoration_v2_selection/`](data/results/restoration_v2_selection/)。confirm 固定 20 条、覆盖
29 个 normalized app labels，未 top-up；confirm 仍没有任何 policy/restoration output。

executor-dispatch 的 live-inspection、negative-control 与 offline-reduction 契约见
[`docs/restoration_v2_executor_dispatch.md`](docs/restoration_v2_executor_dispatch.md)。本次 run 绑定已推送
commit `b6e57c2619e88b8646657b3190bf45853a86c3d2`，未加载 policy 或使用 GPU。

v2 的干预已收窄：所有 memory 始终保留相同 strong low-fidelity summary；恢复 event 时只增加一张
post-action state image，不增加 before image 或额外 action text。confirm 固定每条 trajectory 的 decision
step 6；events 1--4 是四个 visual candidates，event 5 的 post-state 等于 current observation，因而只保留
summary、不重复计图且不能被选择。reference 使用四张历史 post-state 加 current，覆盖每个 non-current
historical event 且不重复 event 5 的 current-equivalent image；
主预算容量为四个候选中最多选两个；exact-two 作为 cardinality-matched ablation。

历史 v1 结果保持有效且不回改：independent UI-TARS reference gate 得到 69/75 parsed、27/75（36.0%）
expert executable match、swipe 0/2，输出 `NO_GO_CURRENT_REFERENCE_STACK`。这否定的是 v1 reference
stack，不是 restoration 假设。raw 结果位于 private HF
`reference-gate-v1@b3e1245c6c6a1723fe2ca3a861148008df39df46`，轻量结论见
[`data/results/independent_reference_gate_v1/`](data/results/independent_reference_gate_v1/)。旧 15-trajectory /
132-decision oracle raw trajectories 已被 builder 读取和打包，但从未产生本项目 policy/restoration output；
v2 只按预先冻结顺序把它们用于 label-train/development screening。

更早的两状态 real-policy diagnostic 为 `INCONCLUSIVE_POSITIVE`：step 8、1024-token cap 下 exhaustive oracle
恢复 84.2%，recent/similarity 为 48.7%，random expectation 为 58.1%。由于它来自已观察 matched subset，
它仍只是一条 selection-biased existence signal，不能成为 paper `GO` 或训练 label。历史 v1 配置、artifact、
H200 anchor 和执行记录全部保留，见 [`docs/go_no_go.md`](docs/go_no_go.md) 与
[`docs/independent_gate_execution.md`](docs/independent_gate_execution.md)。

新合作者按以下顺序阅读：

1. [`docs/restoration_v2.md`](docs/restoration_v2.md)：当前 scientific contract、data roles 与 gates；
2. [`docs/restoration_v2_1.md`](docs/restoration_v2_1.md)：official-tool interface rescue 与固定 15-state pilot；
3. [`docs/restoration_v2_1_full_45.md`](docs/restoration_v2_1_full_45.md)：full-45 stable-reference substrate、gate 与一次性执行边界；
4. [`docs/restoration_v2_2_eager.md`](docs/restoration_v2_2_eager.md)：只改 eager runtime 的 fresh-45 双 H200 source-only contract；
5. [`docs/restoration_v2_2_labels.md`](docs/restoration_v2_2_labels.md)：已闭合的 attribution run、exact subset oracle、conditional-marginal labels 与 artifact identity；
6. [`ablations/interaction_aware_gate.md`](ablations/interaction_aware_gate.md)：interaction-aware student 的设计、能力边界与 ablation matrix；
7. [`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md)：action、strong LF 与
   post-state-only prompt 的冻结实现；
8. [`docs/execution.md`](docs/execution.md)：跨芯片执行、HF/Git 回写与 Definition of Done；
9. [`docs/restoration_v2_ocr.md`](docs/restoration_v2_ocr.md)：OCR/image identity、schema 与 golden 状态；
10. [`docs/progress.md`](docs/progress.md)：已完成里程碑、negative results 与下一步；
11. [`docs/experiment_contract.md`](docs/experiment_contract.md)：历史 v0.3 与不变的系统边界；
12. [`docs/go_no_go.md`](docs/go_no_go.md)：历史 v1 和当前 v2 判据；
13. [`code/README.md`](code/README.md) 与 [`data/README.md`](data/README.md)：代码和数据边界；
14. [`paper/main.tex`](paper/main.tex)：AAAI 正文 source。

仓库结构：

```text
README.md          # 总索引与交接状态
AGENTS.md          # 每步 Git/HF/compute 规则
paper/             # AAAI LaTeX package
ablations/         # 未冻结的方法比较、诊断设计与执行前边界
code/              # package、scripts、tests、configs、requirements
data/              # 小 fixture 与轻量 result summaries
docs/              # contract、execution、progress、decisions
```

快速验证：

```bash
make test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces \
  validate-restoration-v2-executor-dispatch validate-restoration-v2-selection \
  validate-restoration-v2-ocr-config validate-restoration-v2-ocr-artifact \
  validate-restoration-v2-baselines paper

cd code && python3 -m scripts.validate_restoration_v2_1_full_45_contract \
  --repository-root .. \
  --config code/configs/causalcache_restoration_v2_1_full_45.json
```

计算 placement：v2 offline substrate/attribution 默认使用 Hyper00 H200，Aries A6000 为 fallback。八项
pre-output dependencies 已闭合并完成第一次正式 screening；v2 保持
`NO_GO_V2_SUBSTRATE + NO_GO_ADAPTER_ONLY`。v2.1 versioned source/contract 已冻结，90-prompt
processor-only preflight 与 fixed-15 interface pilot 均已正式通过；唯一 unchanged-interface full-45 attempt
已在 Hyper00 完成并按冻结 gate 判为 `NO_GO_V2_1_FULL_45_SUBSTRATE`。失败点是 32/45 exact canonical
repeat agreement，而不是 interface parse 或未观测到 summary/full-history behavior distance；按照该 contract，
restoration、confirm 与 gate training 均不得继续。完整 artifact 中的 confirm bytes 只接受 loader
validation/hash，不向 processor 或 decoder 暴露任何 confirm state/prompt/image，也没有生成 confirm output。
若探索 executable/UI-element equivalence，必须先冻结新的 versioned source-only protocol，保留本次 NO_GO，
不能在当前 45 states 上 retroactive relabel。AndroidWorld closed-loop MVP 继续使用已验证的 Aries stack；
Hyper01 当前不参与本轮执行。

## 一句话主张

现有 GUI memory 方法主要根据 recency、similarity、attention 或 learned salience 保存历史。CausalCache 直接测量：**在实际部署预算附近，只给某个 GUI 事件增加 archived post-state image，能在多大程度上恢复冻结策略 parseable、finite、repeat-stable 的 self-behavior**，并把这种昂贵的 restoration attribution 蒸馏成在线 memory selector。

## 核心观察

两个 memory controller 即使获得近似相同的 next-action NLL，也可能产生完全不同的长期任务成功率。局部 action fidelity 无法区分：

- 当前动作暂时用不到、未来却决定任务成败的事件；
- 视觉上相似但对后续控制无关的事件；
- 需要保留精确文本、坐标、开关状态的 dependency-critical event；
- 仅仅获得较高 attention、但删除后并不改变策略行为的事件。

因此，memory quality 不应只由局部 prediction quality 衡量，还应由历史事件对未来 policy behavior 的**可恢复贡献**衡量。

## 问题定义

一条 GUI trajectory 表示为：

$$
H_t=\{e_1,\ldots,e_{t-1}\},\qquad e_j=(o_{j-1},a_j,o_j),
$$

其中每个 event 包含：

- action 前截图；
- 执行的 action；
- action 后截图；
- 可选的结构化 UI delta。

给定任务指令 $g$、当前观测 $o_t$ 和高保真事件预算 $B$，目标是选择：

$$
M_t\subset H_t,\qquad \sum_{j\in M_t}c_j\le B,
$$

使冻结 GUI policy 的长期 executable task success 最大。这里的 $B$ 限制 policy-visible multimodal context，不限制 persistent storage；原始事件保存在 archive，未选事件在当前 query 只暴露 deterministic strong summary，并单独报告实际 text tokens。

## 方法

### 1. Restoration Attribution

首先在 action-contract parse、finite-logit 与 repeat-stability 检查通过的开发状态上运行冻结策略 $\pi_0$，得到 self-behavior reference；untouched confirm 使用固定分母，失败状态不能事后删除：

$$
q_t=\pi_0(\cdot\mid g,o_t,H_t).
$$

将所有历史事件替换为低保真版本，得到 baseline memory；随后按照共享随机排列逐个恢复事件的高保真内容。对于已恢复集合 $S$，定义与完整历史行为的距离：

$$
D_t(S)=\sum_l w_l\,
KL\!\left(
q_{t,l}(\cdot\mid a^*_{<l})
\parallel
q^S_{t,l}(\cdot\mid a^*_{<l})
\right),
$$

其中 $a^*$ 是完整历史策略生成的 canonical action，KL 在 teacher-forced action token 上计算。这样无需枚举包含 coordinate 和 text argument 的完整 sequence action space。

事件 $e_j$ 在部署预算 $B$ 附近的 restoration gain 为：

$$
G^{(B)}_{j,t}=\mathbb{E}_{S\sim\mathcal C_B(j)}\!\left[
D_t(S)-D_t(S\cup\{j\})
\right].
$$

其中 $\mathcal C_B(j)$ 只包含为 $e_j$ 留出容量的 maximal near-budget coalitions。该定义是 budget-conditioned Shapley-style attribution；新意不在通用 Shapley estimator，而在 GUI mixed-fidelity intervention、stable policy-behavior value 和固定预算蒸馏。实际使用共享 antithetic permutation，并报告 standard error、Spearman、top-budget Jaccard、oracle utility 与 coalition reconstruction error。

### 2. 在线 Memory Gate（当前 independent baseline）

离线 restoration attribution 计算昂贵，因此训练轻量 gate：

$$
s_{j,t}=f_\theta(g,o_t,\tilde e_j,z_j,\Delta t,B)
$$

预测 $G_{j,t}$。这里采用 **query-time scoring**：历史事件在每个决策时刻根据当前状态重新打分，解决 arrival-time label 随未来时刻变化的契约问题。

训练目标组合为：

$$
L=L_{\text{regression}}
+\lambda_1L_{\text{pairwise-ranking}}
+\lambda_2L_{\text{top-}B}.
$$

推理时使用 positive-value knapsack；分数不超过阈值的事件不会被强制加入，因此实际选择可以少于预算容量。冻结 action policy，只训练 memory gate。

该写法是 averaged-$G^{(B)}$ 的 independent student baseline，不表示 event utility 真正可加。prospective
set-conditioned iterative gate、额外 conditional-edge labels、greedy 的 pure-complementarity failure 与完整
ablation matrix 见 [`ablations/interaction_aware_gate.md`](ablations/interaction_aware_gate.md)；它尚未进入 frozen
v2/v2.1 contract。

### 3. Mixed-Fidelity Memory

v2 每个 low-fidelity event 固定包含：

```text
step_id
action_type
action_argument
foreground_app
screen_text_added
screen_text_removed
screen_change
executor_result
```

系统由 raw event archive、cheap summary/index（可含预计算视觉 embedding）和 policy-visible high-fidelity context 三层组成。所有 memory 中 summary 序列化 byte-identical；high fidelity 只增加一张 post-action state image。第一版不做跨层 KV surgery，而是通过 mixed-fidelity input 重新运行 policy，避免位置编码与上下文依赖导致不合法的 KV 拼接。

## 实验设计

### Benchmarks

- 主要 closed-loop benchmark：AndroidWorld；
- 离线 action prediction 与 memory attribution：GUI-Odyssey、AndroidControl 或同类数据。

### Baselines

- no history、full history、summary only；
- recent top-$B$、random top-$B$；
- visual/text similarity；
- attention-based retention；
- learned salience selector；
- MementoGUI-style memory control；
- AndroTMem/anchor-style memory；
- offline restoration oracle；
- distilled CausalCache gate。

### Metrics

- closed-loop task success；
- action type、target、text argument accuracy；
- successor-action NLL；
- retained restoration mass；
- visual-token budget；
- inference latency；
- success per unit memory/compute。

## 决定论文成败的实验

构造 successor-action NLL 相近的 memory pairs，比较 terminal success：

$$
\operatorname{NLL}(M_1)\approx\operatorname{NLL}(M_2).
$$

当：

$$
\operatorname{RestorationMass}(M_1)>
\operatorname{RestorationMass}(M_2),
$$

检验 $M_1$ 是否仍显著获得更高任务成功率。这是论文最关键的证据：**restoration relevance 捕获了 one-step fidelity 无法解释的长期控制信息。**

## Ablations

- permutation 数量与 attribution variance；
- KL、JS、action log-prob recovery；
- event、screenshot、UI element 三种粒度；
- 去掉 pairwise ranking 或 top-$B$ loss；
- query-time gate 对比 arrival-time gate；
- 不同 memory budget；
- success、NLL 与 restoration mass 的独立相关性；
- 跨 app、任务长度和 policy backbone 泛化。
- independent average-value gate 对比 set-conditioned iterative gate，并按 interaction mass 分层报告 oracle gap；
  设计与限制见 [`ablations/interaction_aware_gate.md`](ablations/interaction_aware_gate.md)。

## 预期贡献

1. 提出与部署预算对齐的 restoration-guided GUI memory attribution。
2. 提出将反事实 restoration gain 蒸馏成在线固定预算 memory gate 的方法。
3. 证明 matched next-action fidelity 下，不同 memory 仍产生不同长期成功率。
4. 在相同多模态 memory budget 下，提高 long-horizon GUI task success。

## Claim 边界

这里的 causal 含义限定为：

> 对冻结策略行为进行受控恢复干预得到的 counterfactual attribution。

本项目不声称识别真实环境结构因果，也不把该方法包装成 world model。

## 与最接近工作的差异

- [MementoGUI](https://arxiv.org/abs/2605.18652) 通过训练数据学习 memory selection、compression 和 retrieval；
- [AndroTMem](https://arxiv.org/abs/2603.18429) 使用 causally linked state anchors；
- CausalCache 的事件重要性来自：恢复该事件后，冻结策略行为的边际恢复量，而不是人工 salience、结构规则或相似度。

## Falsification Criteria

满足任一条件就应弱化主张或停止投稿：

- restoration score 不能比 recency、attention 或 similarity 更好地预测 terminal success；
- matched-NLL 后 restoration mass 与成功率不再相关；
- distilled gate 明显无法逼近 restoration oracle；
- 相同预算下 task success 没有稳定提升；
- 方法收益完全来自增加输入 token；
- attribution 成本无法通过少量 permutation 控制。

## Paper Story

> Long-horizon GUI memory selection lacks policy-grounded supervision. CausalCache values an event by how much adding only its archived post-state image to an unchanged strong-summary history restores a frozen policy's stable self-behavior, then distills this budget-conditioned teacher into a query-time gate.

## 初始路线图

- [x] 固化问题定义、核心机制、实验主线与 claim 边界；
- [x] 建立并验证 AAAI-27 官方 LaTeX anonymous submission 骨架；
- [x] 冻结 action serialization、validated-reference requirements 与 mixed-fidelity experiment contract；
- [x] 固定并评估首个 frozen policy candidate；因 full-history coverage 仅 2/9，拒绝作为主 teacher；
- [x] 冻结 restoration v2 primary policy 与 stable self-behavior reference；GUI-Owl Instruct 仅作为 v2 substrate，不回改其 v1 AndroidWorld rejection；
- [x] 冻结 v2 restricted action、strong LF、post-state-only prompt 与 interface source hashes；
- [x] 在 pinned AndroidWorld 对 14/14 payload 闭合 `JSONAction` constructor；
- [x] 在 Aries 正式闭合 14-case executor dispatch 与 negative actuation control；
- [x] 实现并测试完整 111-pool reconstruction、exact-ID selection 与 append-only exposure materializer；
- [x] 从 pinned source 正式冻结 exact 20-state confirm IDs、65 个 state witnesses 与 exposure ledger；
- [x] 完成 OCR identity、synthetic/real-screen golden 与 immutable HF model/dataset artifact；
- [x] 完成 baseline formulas、policy-vision extractor 与 source hashes；
- [x] 完成完整 derived artifact、immutable HF revision 与 fresh-download replay；
- [x] 完成真实 processor audit，冻结 exact grid/token/tensor evidence；
- [x] 冻结 execution config；
- [x] 物化并正式验证 readiness manifest；
- [x] 实现 trajectory/event schema 与 deterministic low-fidelity summarizer；
- [x] 实现并测试 budget-conditioned restoration attribution 核心；
- [x] 在 synthetic frozen behavior 上验证方差、ranking stability、负 gain 和 interaction error；
- [x] 在已接入的真实轨迹上实现 teacher-forced policy distance 与固定 45-state substrate runner；
- [x] 完成第一次固定 45-state substrate screening；strict parse 0/45，合法输出 `NO_GO_V2_SUBSTRATE`，confirm 未打开；
- [x] 完成 immutable raw-trace parser compatibility replay；40/45 低于 required 45/45，正式为 `NO_GO_ADAPTER_ONLY`；
- [x] 冻结 versioned native-output/generation rescue source，并完成 v2.1 90-prompt processor-only preflight；
- [x] 完成 v2.1 fixed-15 native-output interface pilot；15/15 parse/closer/bridge，保留原 v2 negative result；
- [x] 冻结 unchanged-interface full-45 v2.1 child contract、runner 与 raw artifact chain；
- [x] 执行唯一 full-45 v2.1 substrate attempt；32/45 exact repeat agreement，正式为 `NO_GO_V2_1_FULL_45_SUBSTRATE`；
- [x] 记录 prospective interaction-aware gate ablation；仅 source-only proposal，不重开 v2.1；
- [x] 冻结 subset-search v1 source/config/CPU runner，并完成 formal synthetic + cached-table replay；
- [x] 首次 subset-search CPU attempt 在 pre-commit JSON round-trip validation fail closed；output 未保留；
- [x] push serialization fix 后从新 clean source canonical rerun；pre-commit scientific replay 通过；
- [x] 提交/push subset-search result，并从 clean descendant main 通过 committed full-payload validator；
- [x] 完成 bounded spatial audit、离线 validator repair、deterministic USTAR 与 private HF immutable binding；
- [x] 冻结新的 `v2.2-eager` runtime/source 与双 H200 fresh-45 contract；
- [x] 运行唯一 `v2.2-eager` fresh 45-state substrate并闭合 private HF immutable artifact；45/45 exact repeat，正式 PASS；
- [x] v2.2 已稳定，因此不进入 executable/UI-element equivalence 补救分支；
- [x] 冻结 restoration v2.2 attribution source、exact subset oracle 与 conditional-marginal label contract；source 已 push；
- [x] formal v1 attempt 按 contract 永久封存为 zero-forward `INVALID`；根因是 model snapshot existence 校验晚于 durable claim，而非 restoration estimand 结果；
- [x] 冻结 replacement v2 attempt identity、独立 outcome/HF path 与 pre-claim model snapshot validator；
- [x] 在固定双 H200 23/22 parity、microbatch 1 下完成 v2 formal labels，并闭合 private HF immutable artifact；
- [ ] 冻结 set-conditioned gate-training/evaluation contract；在新 contract 前不训练 gate、不构造 matched-NLL、不运行 closed-loop 或 confirm；
- [ ] 按冻结 contract 训练 gate、构造 matched-NLL memory pairs 并运行 closed-loop；
- [ ] 整理论文与复现实验配置。

## Source of Truth

### Code and Documentation

- GitHub: <https://github.com/luojiaxuan/CausalCache>
- Canonical branch: `main`
- Paper source: [`paper/main.tex`](paper/main.tex)
- Code layout and commands: [`code/README.md`](code/README.md)
- Small-data policy: [`data/README.md`](data/README.md)
- Cross-chip execution and handoff: [`docs/execution.md`](docs/execution.md)
- Ablation index: [`ablations/README.md`](ablations/README.md)
- Interaction-aware gate proposal: [`ablations/interaction_aware_gate.md`](ablations/interaction_aware_gate.md)
- Subset-search contract and interpretation: [`ablations/subset_search.md`](ablations/subset_search.md)
- Frozen subset-search config: [`code/configs/subset_search_ablation_v1.json`](code/configs/subset_search_ablation_v1.json)
- Subset-search canonical CPU result: [`data/results/subset_search_ablation_v1/`](data/results/subset_search_ablation_v1/)
- Spatial reference audit protocol: [`docs/spatial_reference_audit_v1.md`](docs/spatial_reference_audit_v1.md)
- Spatial validator-repair contract: [`docs/spatial_reference_audit_v1_validation_repair.md`](docs/spatial_reference_audit_v1_validation_repair.md)
- Spatial validator-repair config: [`code/configs/spatial_reference_audit_v1_validation_repair_v1.json`](code/configs/spatial_reference_audit_v1_validation_repair_v1.json)
- Spatial reference audit canonical result: [`data/results/spatial_reference_audit_v1/`](data/results/spatial_reference_audit_v1/)
- Restoration v2.2-eager source-only contract: [`docs/restoration_v2_2_eager.md`](docs/restoration_v2_2_eager.md)
- Restoration v2.2-eager machine-readable config: [`code/configs/causalcache_restoration_v2_2_eager.json`](code/configs/causalcache_restoration_v2_2_eager.json)
- Restoration v2.2-eager canonical result: [`data/results/restoration_v2_2_eager_full_45_substrate/`](data/results/restoration_v2_2_eager_full_45_substrate/)
- Restoration v2.2 label protocol: [`docs/restoration_v2_2_labels.md`](docs/restoration_v2_2_labels.md)
- Restoration v2.2 label machine-readable contract: [`code/configs/causalcache_restoration_v2_2_labels.json`](code/configs/causalcache_restoration_v2_2_labels.json)
- Restoration v2.2 label v2 repair contract: [`code/configs/causalcache_restoration_v2_2_labels_v2_repair.json`](code/configs/causalcache_restoration_v2_2_labels_v2_repair.json)
- Restoration v2.2 label source validator: [`code/scripts/validate_restoration_v2_2_label_contract.py`](code/scripts/validate_restoration_v2_2_label_contract.py)
- Restoration v2.2 formal label runner: [`code/scripts/run_restoration_v2_2_labels.py`](code/scripts/run_restoration_v2_2_labels.py)
- Restoration v2.2 label artifact manager: [`code/scripts/manage_restoration_v2_2_label_artifact.py`](code/scripts/manage_restoration_v2_2_label_artifact.py)
- Restoration v2.2 label v1 zero-forward failure: [`data/results/restoration_v2_2_eager_labels_v1_attempt/`](data/results/restoration_v2_2_eager_labels_v1_attempt/)
- Restoration v2.2 label v2 canonical result: [`data/results/restoration_v2_2_eager_labels_v2/`](data/results/restoration_v2_2_eager_labels_v2/)
- Material-run metadata schema: [`code/configs/run_manifest.schema.json`](code/configs/run_manifest.schema.json)
- Current restoration v2 contract: [`docs/restoration_v2.md`](docs/restoration_v2.md)
- Machine-readable v2 config: [`code/configs/causalcache_restoration_v2.json`](code/configs/causalcache_restoration_v2.json)
- Frozen v2 interface semantics: [`docs/restoration_v2_interfaces.md`](docs/restoration_v2_interfaces.md)
- Frozen v2 interface hashes: [`data/manifests/restoration_v2_interfaces.json`](data/manifests/restoration_v2_interfaces.json)
- Pinned AndroidWorld constructor preflight: [`data/results/restoration_v2_constructor_preflight/`](data/results/restoration_v2_constructor_preflight/)
- Executor-dispatch contract: [`docs/restoration_v2_executor_dispatch.md`](docs/restoration_v2_executor_dispatch.md)
- Pinned AndroidWorld executor-dispatch result: [`data/results/restoration_v2_executor_dispatch/`](data/results/restoration_v2_executor_dispatch/)
- Restoration-v2 selection materializer: [`code/causalcache/data/restoration_v2_selection.py`](code/causalcache/data/restoration_v2_selection.py)
- Frozen restoration-v2 exact selection: [`data/manifests/restoration_v2_selection.json`](data/manifests/restoration_v2_selection.json)
- Frozen restoration-v2 exposure ledger: [`data/manifests/restoration_v2_exposure.json`](data/manifests/restoration_v2_exposure.json)
- Restoration-v2 selection result: [`data/results/restoration_v2_selection/`](data/results/restoration_v2_selection/)
- Restoration-v2 OCR/image contract: [`docs/restoration_v2_ocr.md`](docs/restoration_v2_ocr.md)
- Restoration-v2 OCR backend config: [`code/configs/restoration_v2_ocr_backend.json`](code/configs/restoration_v2_ocr_backend.json)
- Restoration-v2 OCR implementation and validator: [`code/causalcache/restoration_v2_text_backend.py`](code/causalcache/restoration_v2_text_backend.py), [`code/scripts/validate_restoration_v2_ocr_backend.py`](code/scripts/validate_restoration_v2_ocr_backend.py)
- Restoration-v2 OCR runtime lock and synthetic fixture: [`code/requirements/restoration_v2_ocr_lock.txt`](code/requirements/restoration_v2_ocr_lock.txt), [`data/fixtures/restoration_v2_ocr_golden.json`](data/fixtures/restoration_v2_ocr_golden.json)
- Restoration-v2 OCR source/artifact manifest: [`data/manifests/restoration_v2_ocr_backend.json`](data/manifests/restoration_v2_ocr_backend.json)
- Restoration-v2 OCR evidence: [`data/results/restoration_v2_ocr_backend/`](data/results/restoration_v2_ocr_backend/)
- Restoration-v2 real-screen materializer: [`code/causalcache/data/restoration_v2_real_screen.py`](code/causalcache/data/restoration_v2_real_screen.py), [`code/scripts/materialize_restoration_v2_real_screen.py`](code/scripts/materialize_restoration_v2_real_screen.py)
- Restoration-v2 real-screen source/artifact validator: [`code/scripts/validate_restoration_v2_real_screen.py`](code/scripts/validate_restoration_v2_real_screen.py)
- Frozen real-screen pre-output source contract: [`data/manifests/restoration_v2_real_screen_source.json`](data/manifests/restoration_v2_real_screen_source.json)
- Passed real-screen run/HF summary: [`data/results/restoration_v2_ocr_backend/real_screen_summary.json`](data/results/restoration_v2_ocr_backend/real_screen_summary.json)
- Restoration-v2 deterministic baseline formulas: [`code/causalcache/restoration_v2_baselines.py`](code/causalcache/restoration_v2_baselines.py)
- Restoration-v2 policy-vision extractor: [`code/causalcache/policy/gui_owl_v2_vision.py`](code/causalcache/policy/gui_owl_v2_vision.py)
- Restoration-v2 baseline source manifest: [`data/manifests/restoration_v2_baselines.json`](data/manifests/restoration_v2_baselines.json)
- Restoration-v2 derived dataset builder: [`code/scripts/build_guiodyssey_restoration_v2.py`](code/scripts/build_guiodyssey_restoration_v2.py)
- Restoration-v2 derived dataset validator: [`code/scripts/validate_guiodyssey_restoration_v2.py`](code/scripts/validate_guiodyssey_restoration_v2.py)
- Restoration-v2 derived artifact completion index: [`data/manifests/restoration_v2_derived_artifact.json`](data/manifests/restoration_v2_derived_artifact.json)
- Restoration-v2 derived artifact evidence: [`data/results/restoration_v2_derived_artifact/`](data/results/restoration_v2_derived_artifact/)
- Historical experiment contract v0.3: [`docs/experiment_contract.md`](docs/experiment_contract.md)
- Frozen policy selection: [`docs/policy_selection.md`](docs/policy_selection.md)
- AndroidWorld benchmark-native stack: [`docs/androidworld_stack.md`](docs/androidworld_stack.md)
- AndroidWorld frozen task partition: [`docs/androidworld_task_partition.md`](docs/androidworld_task_partition.md)
- Progress record: [`docs/progress.md`](docs/progress.md)
- Synthetic estimator validation: [`data/results/synthetic_phase0/README.md`](data/results/synthetic_phase0/README.md)
- Qwen3-VL real-policy smoke test: [`data/results/qwen_policy_smoke/README.md`](data/results/qwen_policy_smoke/README.md)
- Qwen3-VL full-history coverage: [`data/results/qwen_policy_coverage/README.md`](data/results/qwen_policy_coverage/README.md)
- UI-TARS full-history coverage: [`data/results/ui_tars_policy_coverage/README.md`](data/results/ui_tars_policy_coverage/README.md)
- Independent gate artifact index: [`data/manifests/independent_reference_gate_v1_artifact.json`](data/manifests/independent_reference_gate_v1_artifact.json)
- Independent UI-TARS reference rejection: [`data/results/independent_reference_gate_v1/README.md`](data/results/independent_reference_gate_v1/README.md)
- OpenCUA-7B pinned snapshot manifest: [`code/configs/open_cua_7b_snapshot.json`](code/configs/open_cua_7b_snapshot.json)
- OpenCUA-7B pinned runtime dependency: [`code/requirements/opencua.txt`](code/requirements/opencua.txt)
- OpenCUA-7B logits and mixed-fidelity smoke: [`data/results/open_cua_policy_smoke/README.md`](data/results/open_cua_policy_smoke/README.md)
- OpenCUA-7B full-history coverage: [`data/results/open_cua_policy_coverage/README.md`](data/results/open_cua_policy_coverage/README.md)
- ShowUI-2B pinned snapshot manifest: [`code/configs/showui_2b_snapshot.json`](code/configs/showui_2b_snapshot.json)
- ShowUI-2B logits and mixed-fidelity smoke: [`data/results/showui_policy_smoke/README.md`](data/results/showui_policy_smoke/README.md)
- ShowUI-2B full-history coverage: [`data/results/showui_policy_coverage/README.md`](data/results/showui_policy_coverage/README.md)
- GUI-Owl-1.5-8B pinned snapshot manifest: [`code/configs/gui_owl_1_5_8b_snapshot.json`](code/configs/gui_owl_1_5_8b_snapshot.json)
- GUI-Owl-1.5-8B-Think pinned snapshot manifest: [`code/configs/gui_owl_1_5_8b_think_snapshot.json`](code/configs/gui_owl_1_5_8b_think_snapshot.json)
- Replacement teacher preregistration: [`code/configs/androidworld_replacement_teacher_v1.json`](code/configs/androidworld_replacement_teacher_v1.json)
- AndroidWorld stack preregistration: [`code/configs/androidworld_stack.json`](code/configs/androidworld_stack.json)
- AndroidWorld task partition manifest: [`code/configs/androidworld_task_partition.json`](code/configs/androidworld_task_partition.json)
- AndroidWorld validation execution plan: [`code/configs/androidworld_validation_plan.json`](code/configs/androidworld_validation_plan.json)
- GUI-Owl native logits/history smoke: [`data/results/gui_owl_native_smoke/README.md`](data/results/gui_owl_native_smoke/README.md)
- GUI-Owl model-default native-resolution smoke: [`data/results/gui_owl_native_resolution_smoke/README.md`](data/results/gui_owl_native_resolution_smoke/README.md)
- AndroidWorld environment/reward smoke: [`data/results/androidworld_environment_smoke/README.md`](data/results/androidworld_environment_smoke/README.md)
- GUI-Owl AndroidWorld validation smoke: [`data/results/gui_owl_androidworld_validation_smoke/README.md`](data/results/gui_owl_androidworld_validation_smoke/README.md)
- GUI-Owl configuration-invalid validation audit: [`data/results/gui_owl_androidworld_validation_attempt2/README.md`](data/results/gui_owl_androidworld_validation_attempt2/README.md)
- GUI-Owl native-resolution validation rejection: [`data/results/gui_owl_androidworld_validation/README.md`](data/results/gui_owl_androidworld_validation/README.md)
- GUI-Owl Think strict-parser smoke: [`data/results/gui_owl_1_5_8b_think_smoke_strict/README.md`](data/results/gui_owl_1_5_8b_think_smoke_strict/README.md)
- GUI-Owl Think passing native smoke: [`data/results/gui_owl_1_5_8b_think_smoke/README.md`](data/results/gui_owl_1_5_8b_think_smoke/README.md)
- GUI-Owl Think AndroidWorld validation rejection: [`data/results/gui_owl_1_5_8b_think_androidworld_validation/README.md`](data/results/gui_owl_1_5_8b_think_androidworld_validation/README.md)
- Restoration v2 real processor audit: [`data/results/restoration_v2_processor_audit/README.md`](data/results/restoration_v2_processor_audit/README.md)
- Restoration v2 execution config: [`code/configs/restoration_v2_execution_hyper00_v1.json`](code/configs/restoration_v2_execution_hyper00_v1.json)
- Restoration v2 readiness authorization: [`data/results/restoration_v2_readiness/README.md`](data/results/restoration_v2_readiness/README.md)
- Restoration v2 GPU-2 runtime re-anchor: [`data/results/restoration_v2_runtime_reanchor/README.md`](data/results/restoration_v2_runtime_reanchor/README.md)
- Restoration v2 fixed 45-state substrate result: [`data/results/restoration_v2_substrate_screening/README.md`](data/results/restoration_v2_substrate_screening/README.md)
- Restoration v2 parser replay golden contract: [`data/manifests/restoration_v2_parser_compatibility_golden.json`](data/manifests/restoration_v2_parser_compatibility_golden.json)
- Restoration v2 formal parser compatibility replay: [`data/results/restoration_v2_parser_compatibility/README.md`](data/results/restoration_v2_parser_compatibility/README.md)
- Restoration v2.1 frozen pilot contract: [`code/configs/causalcache_restoration_v2_1_pilot.json`](code/configs/causalcache_restoration_v2_1_pilot.json)
- Restoration v2.1 interface and execution boundary: [`docs/restoration_v2_1.md`](docs/restoration_v2_1.md)
- Restoration v2.1 passed processor preflight: [`data/results/restoration_v2_1_processor_preflight/README.md`](data/results/restoration_v2_1_processor_preflight/README.md)
- Restoration v2.1 passed fixed-15 interface pilot: [`data/results/restoration_v2_1_interface_pilot/README.md`](data/results/restoration_v2_1_interface_pilot/README.md)
- Restoration v2.1 frozen full-45 contract: [`code/configs/causalcache_restoration_v2_1_full_45.json`](code/configs/causalcache_restoration_v2_1_full_45.json)
- Restoration v2.1 full-45 execution boundary: [`docs/restoration_v2_1_full_45.md`](docs/restoration_v2_1_full_45.md)
- Restoration v2.1 full-45 NO-GO result: [`data/results/restoration_v2_1_full_45_substrate/README.md`](data/results/restoration_v2_1_full_45_substrate/README.md)
- Restoration v2.1 processor-only audit CLI: `cd code && python3 -m scripts.audit_gui_owl_v2_1_processor --help`
- Restoration v2.1 fixed-15 no-retry runner: `cd code && python3 -m scripts.run_restoration_v2_1_interface_pilot --help`
- Restoration v2.1 raw evidence manager: `cd code && python3 -m scripts.manage_restoration_v2_1_pilot_artifact --help`
- Restoration v2.1 full-45 runner: `cd code && python3 -m scripts.run_restoration_v2_1_full_45_substrate --help`
- Restoration v2.1 full-45 artifact manager: `cd code && python3 -m scripts.manage_restoration_v2_1_full_45_artifact --help`
- Build command: `make paper`
- Test command: `make test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces validate-restoration-v2-executor-dispatch validate-restoration-v2-selection validate-restoration-v2-ocr-config validate-restoration-v2-ocr-artifact validate-restoration-v2-baselines`；v2.1 pilot/full-45 contracts 分别用 `cd code && python3 -m scripts.validate_restoration_v2_1_contract --repository-root .. --config code/configs/causalcache_restoration_v2_1_pilot.json` 与 `cd code && python3 -m scripts.validate_restoration_v2_1_full_45_contract --repository-root .. --config code/configs/causalcache_restoration_v2_1_full_45.json`
- 当前状态：v1 UI-TARS reference 以 27/75、swipe 0/2 判负；v2 第一次固定 45-state screening 在
  clean `main` 完成，strict parse 0/45，正式为 `NO_GO_V2_SUBSTRATE + CONFIRM_LOCKED`；事后 immutable
  replay 的保守上界也仅 40/45，正式为 `NO_GO_ADAPTER_ONLY`。已有 45 个 native policy outputs，但没有
  teacher forward、KL、restoration label 或 CausalCache 方法效果结果。v2.1 official-tool contract、
  90-prompt processor preflight 与唯一 fixed-15 pilot 均已正式通过；后者 15/15 parse/closer/bridge，raw
  artifact 已绑定 private HF immutable revision。唯一 full-45 formal attempt 得到 45/45 parse、32/45 exact
  canonical repeat agreement、32/45 finite-logit coverage 与 32 个 memory-sensitive states，正式为
  `NO_GO_V2_1_FULL_45_SUBSTRATE`。bounded spatial audit 随后得到 auto 7/13、eager 13/13，并在不重跑
  profile 的离线 repair 后正式归约为 `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；private HF immutable
  artifact 已 fresh-download 复核。v2.2-eager 随后在两张 H200 上按 23/22 parity 完成唯一 fresh-45 attempt：
  45/45 parse、45/45 exact repeat、45/45 finite logits、45 个 memory-sensitive states，正式为
  `PASS_V2_2_EAGER_FULL_45_SUBSTRATE`；raw artifact 已绑定 private HF immutable revision。restoration v2.2
  label v1 source 已在 clean pushed `main@3942d687d03bf63ea683fe8ad906a161eb10dc27` 冻结，contract SHA256
  为 `56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87`，29-file inventory SHA256
  为 `8d0ecf6b0df75f9fa7753631b8fca5efd98c71a7953007e684393e6e8fe52d9b`。formal v1 attempt 因指定 model
  snapshot directory 不存在而在 0/45 attempted states、0 teacher forward、0 KL 时 fail closed；原 ledger/root
  保留且不可重跑，没有 raw labels 或 HF artifact。replacement v2 identity 与 pre-claim full snapshot validator 已
  冻结；下一步只运行固定双 H200 schedule。gate、matched-NLL、closed-loop 与 confirm 仍未运行且不属于本步。

### Data and Models

| Artifact | Canonical location | Revision/status | Notes |
| --- | --- | --- | --- |
| GUIOdyssey pilot trajectory | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-pilot-mobile> | `1de9c34ff029d4c01665cdaca74436ae24bff276`，private | schema v0.3；10 screenshots、9 events、9 decisions |
| Independent GUIOdyssey gate artifact | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-independent-mobile> | `v0.1.0` / `84c9f5a335e9612ccb4bd566f977574f359b2485`，private | schema v0.4；reference 8 trajectories/75 decisions；oracle 15/132；immutable re-download verified |
| Independent UI-TARS reference run | 同一 private independent dataset repo | `reference-gate-v1` / `b3e1245c6c6a1723fe2ca3a861148008df39df46` | 69/75 parsed、27/75 match、swipe 0/2；`NO_GO_CURRENT_REFERENCE_STACK`；oracle 未运行 |
| Rejected policy candidate | <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct> | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | full-history executable-match 2/9；不作为主 teacher |
| Rejected GUI-tuned candidate | <https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B> | `683d002dd99d8f95104d31e70391a39348857f4e` | parsed 9/9、executable-match 4/9；未通过预注册 50% gate |
| Rejected computer-use candidate | <https://huggingface.co/xlangai/OpenCUA-7B> | `a2efb7d2b104d477a4a2666a357e79550a28aafc` | parsed 7/9、executable-match 1/9；未通过预注册 gate |
| Rejected GUI navigation candidate | <https://huggingface.co/showlab/ShowUI-2B> | `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60` | parsed 9/9、executable-match 2/9；未通过预注册 gate |
| GUI-Owl-1.5-8B-Instruct | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct> | `06d5faecff74840bab2be2425e9c42667a5d04fc` | v1 AndroidWorld success teacher 被拒；v2 首次 fixed screen 因 strict envelope parse 0/45 判 `NO_GO_V2_SUBSTRATE` |
| Rejected replacement candidate | <https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Think> | `afe3707fc84caebc4d7046118b34493ecf8bb060` | 512/513 parsed；official-success 上界 29/62，未通过 50% gate |
| AndroidWorld native validation traces | <https://huggingface.co/datasets/gavinlaw/causalcache-androidworld-validation-mobile> | `v0.2.0` / `0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`，private | 42 Think traces；deterministic gzip JSONL；`v0.1.0` Instruct artifact 保持不变 |
| Restoration v2 OCR models | <https://huggingface.co/gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en> | `v1.0.0` / `0dbc766a73ee88d10d52285d434dbfec58617835`，private | 三份 ONNX、model card 与 manifest；fresh immutable re-download 后 6/6 file hashes verified |
| Restoration v2 derived dataset | <https://huggingface.co/datasets/gavinlaw/causalcache-guiodyssey-restoration-v2-mobile> | `restoration-v2-derived-v1.0.0` / `89f136abaff797e14fe758a198996e51032a10a6`，private | exact 6-file derived projection 已 fresh re-download 并第三次 replay；旧 OCR golden tag 仍固定到 `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b` |
| Restoration v2 first substrate trace | 同一 private restoration-v2 dataset repo | `restoration-v2-substrate-screening-v1.0.0` / `c073e143b935a79befd8ab1fd7123796792efad8` | fixed 45 states；strict 0/45、conservative recovery 40/45；raw shard + manifest fresh-download verified；`NO_GO_V2_SUBSTRATE` / `NO_GO_ADAPTER_ONLY` |
| Restoration v2.1 processor preflight | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile> | `v2.1-processor-preflight-v1` / `85576161b7cb8bbae14e46a482c42b5be5bf1d7e`，private | 90-prompt CPU-only PASS；raw SHA256 `5349ffc6...499191`、7,609,803 bytes；fresh immutable download verified |
| Restoration v2.1 interface pilot trace | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-1-interface-pilot-mobile> | `v2.1-interface-pilot-v1` / `bdff8ca71f150afd80d6291b4ecec76cbf9e7432`，private | fixed-15 PASS；raw USTAR SHA256 `f71d5fd5...32064`、133,120 bytes；fresh immutable download verified |
| Restoration v2.1 full-45 substrate trace | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile> | `v2.1-full-45-substrate-v1` / `814506ef1450838d4bc6ed3d89fe53e0773d92fb`，private | 45/45 parse、32/45 exact repeat agreement、32 memory-sensitive；`NO_GO_V2_1_FULL_45_SUBSTRATE`；raw USTAR SHA256 `8cd53d6e...f4fa4`、962,560 bytes；fresh immutable download verified |
| Spatial reference audit trace | <https://huggingface.co/datasets/gavinlaw/causalcache-spatial-reference-audit-mobile> | `spatial-reference-audit-v1` / `d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`，private | auto 7/13、eager 13/13、FP32 4/4 descriptive；`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；72-member USTAR SHA256 `d62ad05f...e5ecc`；fresh immutable canonical rebuild verified |
| Restoration v2.2-eager fresh-45 trace | <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile> | `v2.2-eager-full-45-substrate-v1` / `3577099d505b8c652d764f41269df911128ec767`，private | 45/45 parse/repeat/finite logits、45 memory-sensitive；`PASS_V2_2_EAGER_FULL_45_SUBSTRATE`；raw USTAR SHA256 `b22827e6...09fb5`、fresh immutable download verified |
| Restoration v2.2 exact labels（v2 planned） | `gavinlaw/causalcache-restoration-labels-mobile` | tag `v2.2-eager-train-dev-exact-v2`；repo/artifact not created | v1 zero-forward `INVALID`；v2 pre-claim repair source frozen，预期 45 states、420 raw `D(S)` rows、45 exact-subset oracle、435 deployment conditional marginals 尚未生成 |
| Gate checkpoints/adapters | Hugging Face model repo（待创建） | not created | 记录 policy backbone、训练配置与评测 provenance |

Pilot 的生成配置见 [`code/configs/guiodyssey_pilot.json`](code/configs/guiodyssey_pilot.json)，independent
artifact 见 [`code/configs/independent_reference_gate_v1.json`](code/configs/independent_reference_gate_v1.json)。
v2 完整 derived dataset、OCR 三模型、6-image real-screen golden 与第一次 screening raw trace 已成为 private
HF dataset/model canonical artifacts。Hyper00 model cache 仍可重建；native outputs 不留在 Git，而由完成态
Git manifest/result 绑定 HF immutable revision、逐文件 hashes 与 fresh-download evidence。

## Citation

项目仍处于研究与实验阶段，正式 citation 将在论文公开后补充。
