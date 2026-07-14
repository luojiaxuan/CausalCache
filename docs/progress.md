# 项目进展

## 当前目标

在 AAAI-27 截止前完成一个最小但可证伪的 CausalCache 实验链路：validated offline attribution、multi-budget gate、AndroidWorld closed-loop frontier 与 matched-NLL mechanism test。

## 已完成里程碑

### 2026-07-14：论文骨架

- 使用 AAAI-27 官方 author kit 建立 anonymous submission LaTeX；
- 编译并逐页检查 3 页论文骨架；
- 明确 Shapley-style attribution 不是 estimator novelty；
- 将 claim 收窄为 decision-time behavioral restoration，并由 closed-loop 验证长期意义；
- 引入 budget-conditioned value、validated teacher、policy-visible context budget 和可为空的选择。

### 2026-07-14：实验契约 v0.1

- 冻结 archive / low-fidelity index / policy-visible context 三层接口；
- 冻结五字段低保真事件 schema 与 executable action canonicalization；
- 冻结 reference validation、near-budget coalition、稳定性指标和 matched-NLL 协议；
- 增加机器可读配置、fixture、验证 CLI 与单元测试。

### 2026-07-14：Synthetic estimator validation

- 实现 shared antithetic permutation 与 variable-cost maximal near-budget coalition；
- 实现 exact permutation enumeration、standard error、positive-value knapsack、Spearman 与 top-budget Jaccard；
- 在 8-event 非线性 synthetic frozen behavior 上验证两个负 restoration-gain event 不会被强制选择；
- 5 seeds 下，$K=4$ 的 mean top-budget Jaccard 为 0.90，$K\ge 8$ 为 1.00；mean standard error 从 2.99（$K=4$）下降到 1.19（$K=32$）；
- exact marginal-score selection 的实际 utility 仅为 global subset optimum 的 85.9%，确认 interaction 会破坏 attribution 的可加性，后续真实实验必须报告 reconstruction error 并保留 budget-aware set loss。

以上结果只验证实现与指标链路，不构成论文效果证据。

### 2026-07-14：GUIOdyssey deterministic pilot artifact

- 从 `cua-lite/GUIOdyssey` 固定 revision 的单个 Parquet shard 提取 upstream GUIOdyssey 成功轨迹 `0054832199799795`；
- 实现 tap、long press、type、swipe、system button 的 executable canonicalization，以及 action 与 terminal signal 分离；
- 生成 10 screenshots、9 events、9 decisions 的 deterministic tar shard，连续两次构建 SHA256 一致；
- 明确拒绝失败轨迹，且 policy-visible manifest 不包含 expert inline reasoning；
- 上传到私有 Hugging Face dataset；当前 canonical revision 为 `1de9c34ff029d4c01665cdaca74436ae24bff276`；
- 该 artifact 只验证真实数据接口，不构成 restoration 或任务成功率证据。

### 2026-07-14：Qwen3-VL real-policy forward smoke

- 固定 `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` 的 14 个加载文件，并校验四个 weight shard 的 LFS SHA256；
- 实现统一的 summary-only、mixed-fidelity、full-history prompt contract 与 JSON executable action parser；
- 在 GUIOdyssey trajectory `0054832199799795` 的 decision step 4 上，三种 fidelity 输入都生成与 recorded action 完全匹配的 `type_text("Cryptocurrency Market")`；
- 输入长度分别为 501、865、1,593 tokens，单张 A6000 peak allocated memory 为 17.71、17.85、18.10 GB；
- 结果见 `results/qwen_policy_smoke/`。这只验证 policy forward 链路，不能说明 restoration 有收益。

### 2026-07-14：Qwen3-VL full-history coverage（negative result）

- 在同一成功 trajectory 的全部 9 个 decisions 上运行 full-history deterministic generation；
- schema v0.3 输入下，9 个输出中 8 个满足 executable JSON schema；decision step 4 的 `type_text` 和 step 8 的 `swipe` 与 recorded action 匹配；
- 在 schema v0.2 executable equivalence 下，tap 为 0/7、swipe 为 1/1、type_text 为 1/1，总 coverage 为 2/9（22.2%）；
- 1 个状态输出 unsupported `click` alias，另外 6 个 tap 的 coordinate bin 不匹配；
- 按预注册 validation contract，拒绝 Qwen3-VL-8B-Instruct 作为主 frozen teacher，不放宽标准；
- 结果见 `results/qwen_policy_coverage/`。这不是对 CausalCache 机制的 falsification，而是 backbone selection 的负结果。

### 2026-07-14：Executable equivalence contract v0.2

- 将不同 policy grammar 的 swipe/start-end 与 scroll/direction 统一为 viewport content direction；
- 例如手指从屏幕底部向顶部滑动统一为 `scroll:down`；
- 重建 deterministic pilot、更新私有 HF revision，并在相同规则下重跑 Qwen3-VL coverage；
- 该修改在 UI-TARS coverage 前完成，避免为新 candidate 事后放宽验证。

### 2026-07-14：High-fidelity action contract v0.3

- 发现 archived raw tool call 已保存，但 high-fidelity policy prompt 错误地只暴露 coarse canonical target；
- high-fidelity event 现显式包含并暴露原始 action arguments，low-fidelity event 仍只暴露 coordinate bin 或 scroll direction；
- 重建 deterministic pilot、更新私有 HF revision，并最后一次重跑 Qwen3-VL consistency coverage；
- Qwen3-VL coverage 仍为 2/9，因此 rejection 决策不变。

### 2026-07-14：UI-TARS-1.5-7B coverage（negative result）

- 固定并逐文件校验 `ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`；
- 实现 native mobile prompt、Qwen2.5-VL resized-coordinate normalization 和统一 executable parser；
- 在相同 9-decision pilot、相同 full-history validation contract 下得到 9/9 parsed、4/9 executable match；
- tap 为 2/7、swipe 为 1/1、type_text 为 1/1，action-type gate 通过，但 44.4% overall coverage 低于预注册的 50%；
- 不因一个相邻 coordinate-bin miss 事后放宽 equivalence，拒绝该 candidate 进入 attribution pilot；
- 结果见 `results/ui_tars_policy_coverage/`。这仍是 backbone selection 的负结果，不是 CausalCache 方法效果。

### 2026-07-14：OpenCUA-7B 接口审计

- 固定 `xlangai/OpenCUA-7B@a2efb7d2b104d477a4a2666a357e79550a28aafc` 与 38 个必要文件；
- 确认 remote `forward()` 返回 token logits，满足后续 teacher-forced component distance 接口；
- 确认官方 PyAutoGUI grammar 可映射到统一 `ExecutableAction`，absolute coordinate 需按 smart-resized image 归一化；
- 记录自定义 1D RoPE、tokenizer/chat template、remote code 与长于官方默认 image history 的复现风险；
- 实现共享 resized-coordinate conversion、OpenCUA prompt、PyAutoGUI parser、pinned runtime 与显式 logits probe，22 个单元测试通过；
- 接口审计通过，只允许在相同预注册 gate 下继续，不据此认定它适合作为主 teacher。
- 第一次 load smoke 在容器预装的 Transformers 5.6.0 上因 remote `tie_weights()` 签名不兼容而失败；依据上游源码固定独立 `transformers==4.53.0` runtime，不对模型代码做运行时补丁。
- 第二次 import smoke 发现 system-site `kernels 0.14.1` 与 Transformers 4.53.0 所需的 Hugging Face Hub 冲突；在同一 venv 内固定兼容的 `kernels==0.11.7`，不修改全局容器包。
- 第三次 load smoke 发现 adapter 错用 Transformers 5.x 的 `dtype=` 参数；固定 4.53.0 runtime 已改为对应的 `torch_dtype=`，其余实验接口不变。
- 第四次 smoke 成功加载 28/28 权重 shard，但 processor 要求 system message 使用 typed content list；消息容器已修正且新增回归测试，prompt 文本不变。
- 第五次 smoke 成功得到 finite logits，并验证 1/3/7-image generation；发现 summary/full 各输出两个 PyAutoGUI call，parser 已收紧为每个 decision 必须恰好一个 executable call，拒绝静默截取。
- canonical smoke 的 summary-only logits shape 为 `[1, 526, 152064]` 且全部 finite；只恢复 event 2 时唯一 action 与 recorded `type_text("Cryptocurrency Market")` 匹配；结果见 `results/open_cua_policy_smoke/`，不作为 attribution 效果证据。
- 实现 OpenCUA 9-decision coverage runner，逐步记录 image count、input tokens、raw output、single-action parse、显存和 latency，并直接读取预注册 gate 配置。

### 2026-07-14：OpenCUA-7B coverage（negative result）

- 在相同 9-decision full-history gate 下得到 7/9 parsed、1/9 executable match；
- tap 为 1/7、swipe 为 0/1、type_text 为 0/1，overall gate 与 action-type gate 均失败；
- 3--19 image full-history 输入全部运行完成，最长 5,172 tokens、峰值 allocated GPU memory 18.71 GB，排除 OOM 或短 context 作为主要失败原因；
- 失败集中在 desktop-oriented grounding、multi-action output 和 mobile trajectory action mismatch；
- 拒绝该 candidate 进入 attribution pilot，结果见 `results/open_cua_policy_coverage/`。

### 2026-07-14：ShowUI-2B 接口审计与 revision 冻结

- 官方 model card 的 phone navigation action space 明确定义 `INPUT`、`SWIPE`、`TAP`、`ANSWER` 与 `ENTER`，因此不是只能输出点击坐标的 grounding-only 接口；
- 固定 `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`，并记录 11 个 runtime 文件的 size 与 SHA256；
- 固定使用原生 phone prompt、单 dictionary output 与 `[0, 1]` 相对坐标；
- 下一步先完成 parser/prompt 单测与单 GPU smoke，再在完全相同的预注册 gate 上评估 9 个 full-history decisions。

## 当前 artifact 状态

- Git 代码、配置、论文与轻量测试 fixture：本仓库 `main`；
- GUIOdyssey pilot：私有 Hugging Face dataset `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- Rejected policy candidate：上游 Hugging Face model `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；共享机器副本只是可重建 cache；
- Rejected GUI-tuned policy candidate：上游 Hugging Face model `ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`；Aries 副本只是可重建 cache；
- Rejected computer-use policy candidate：上游 Hugging Face model `xlangai/OpenCUA-7B@a2efb7d2b104d477a4a2666a357e79550a28aafc`；Aries snapshot 与 venv 只是可重建 cache；
- Pending GUI navigation policy candidate：上游 Hugging Face model `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`；尚未完成 smoke 与 coverage gate；
- 完整 attribution dataset：尚未生成，pilot 扩展后仍需单独登记 revision；
- gate checkpoint：尚未生成，目标 Hugging Face model repo 待 owner 确认；
- 当前没有仅存在共享机器或本地磁盘上的正式实验 artifact；Taurus/Aries 目录只作为 HF artifact 的 staging/cache。

## 未决策项

- transfer backbone；
- canonical action path 上 teacher-forced component distance 的具体 token boundary；
- pilot 应扩展到多少 app、trajectory 和 horizon 才足以进入 attribution 主表。

这些项目必须经过可获得 logits、许可证、磁盘和算力检查后再冻结，不能为了填配置而猜测。

## 下一步

实现 ShowUI-2B 原生 phone navigation adapter，依次完成单测、单 GPU smoke 与预注册 coverage gate。若仍未通过，则停止在当前 pilot 上继续枚举 backbone，转向选择一个与 benchmark 原生适配的 policy/evaluation stack。
