# 项目进展

## 当前目标

AAAI-27 的论文目标仍是 offline restoration attribution、multi-budget gate、AndroidWorld closed-loop frontier
与 matched-NLL mechanism test，但当前 v2.1 实验路线已按 substrate gate 停止。八项 pre-output dependencies、
official-tools processor preflight 与 fixed-15 interface pilot 均已闭合；唯一 full-45 run 虽得到 45/45 parse 和
32 个 memory-sensitive states，但 exact canonical repeat agreement 只有 32/45，因此正式为
`NO_GO_V2_1_FULL_45_SUBSTRATE`。没有生成 restoration label、baseline selection、gate checkpoint、confirm
policy output 或 CausalCache 方法效果结果。该 NO-GO 的 Git/HF committed binding 已在 clean
`main@554c51e` 闭合；bounded spatial audit 随后正式得到 eager-specific exact-stability recovery，新的
v2.2-eager 已完成 source-only contract freeze，但 fresh-45 尚未运行。在该 substrate gate 明确 PASS 前，
不运行 restoration、confirm 或 AndroidWorld test split。

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
- 结果见 `data/results/qwen_policy_smoke/`。这只验证 policy forward 链路，不能说明 restoration 有收益。

### 2026-07-14：Qwen3-VL full-history coverage（negative result）

- 在同一成功 trajectory 的全部 9 个 decisions 上运行 full-history deterministic generation；
- schema v0.3 输入下，9 个输出中 8 个满足 executable JSON schema；decision step 4 的 `type_text` 和 step 8 的 `swipe` 与 recorded action 匹配；
- 在 schema v0.2 executable equivalence 下，tap 为 0/7、swipe 为 1/1、type_text 为 1/1，总 coverage 为 2/9（22.2%）；
- 1 个状态输出 unsupported `click` alias，另外 6 个 tap 的 coordinate bin 不匹配；
- 按预注册 validation contract，拒绝 Qwen3-VL-8B-Instruct 作为主 frozen teacher，不放宽标准；
- 结果见 `data/results/qwen_policy_coverage/`。这不是对 CausalCache 机制的 falsification，而是 backbone selection 的负结果。

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
- 结果见 `data/results/ui_tars_policy_coverage/`。这仍是 backbone selection 的负结果，不是 CausalCache 方法效果。

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
- canonical smoke 的 summary-only logits shape 为 `[1, 526, 152064]` 且全部 finite；只恢复 event 2 时唯一 action 与 recorded `type_text("Cryptocurrency Market")` 匹配；结果见 `data/results/open_cua_policy_smoke/`，不作为 attribution 效果证据。
- 实现 OpenCUA 9-decision coverage runner，逐步记录 image count、input tokens、raw output、single-action parse、显存和 latency，并直接读取预注册 gate 配置。

### 2026-07-14：OpenCUA-7B coverage（negative result）

- 在相同 9-decision full-history gate 下得到 7/9 parsed、1/9 executable match；
- tap 为 1/7、swipe 为 0/1、type_text 为 0/1，overall gate 与 action-type gate 均失败；
- 3--19 image full-history 输入全部运行完成，最长 5,172 tokens、峰值 allocated GPU memory 18.71 GB，排除 OOM 或短 context 作为主要失败原因；
- 失败集中在 desktop-oriented grounding、multi-action output 和 mobile trajectory action mismatch；
- 拒绝该 candidate 进入 attribution pilot，结果见 `data/results/open_cua_policy_coverage/`。

### 2026-07-14：ShowUI-2B 接口审计与 revision 冻结

- 官方 model card 的 phone navigation action space 明确定义 `INPUT`、`SWIPE`、`TAP`、`ANSWER` 与 `ENTER`，因此不是只能输出点击坐标的 grounding-only 接口；
- 固定 `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`，并记录 11 个 runtime 文件的 size 与 SHA256；
- 固定使用原生 phone prompt、单 dictionary output 与 `[0, 1]` 相对坐标；
- 第一次 smoke 的三种 fidelity 都生成正确 `INPUT` 文本但返回 `position=None`；在 gate 前明确按统一 executor 只消费文本参数，位置非空时才额外校验；
- 修正后的 smoke 返回 `[1, 695, 151936]` finite logits，且 1/3/7-image 三种输入均生成唯一、正确的 `type_text('cryptocurrency market')`；
- 峰值 allocated GPU memory 不超过 4.45 GiB，确认单张 A6000 可运行；
- 下一步在完全相同的预注册 gate 上评估 9 个 full-history decisions。

### 2026-07-14：ShowUI-2B coverage（negative result）

- 在相同 9-decision full-history gate 下得到 9/9 parsed、2/9 executable match；
- tap 为 1/7、swipe 为 0/1、type_text 为 1/1，overall gate 与 action-type gate 均失败；
- 3--19 image 输入全部运行完成，最长 5,305 tokens、峰值 allocated GPU memory 5.01 GiB；
- 后半段多次过早生成 `ANSWER('task complete')`，说明失败来自 behavior coverage 而非 runtime；
- 拒绝该 candidate 进入 attribution pilot，结果见 `data/results/showui_policy_coverage/`；
- 四个预登记 candidate 全部未过门槛，停止在当前 pilot 上继续枚举 backbone，转向 benchmark-native policy/evaluation stack 设计。

### 2026-07-14：AndroidWorld benchmark-native stack 冻结

- 固定 `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc` 作为新的 primary policy candidate；
- 官方 AndroidWorld adapter 报告 69.0% success，并原生使用最近 5 张截图与更早 action text，接口与 mixed-fidelity memory 问题直接对齐；
- 固定 canonical AndroidWorld 与 MobileAgent adapter revisions、模型 14 个 runtime files、SHA256、task hash partition 和 validation gate；
- Aries 已确认 x86_64、Docker 27.2.1 与 `/dev/kvm` 可用；
- 详细执行顺序见 `docs/androidworld_stack.md`。当前只完成 stack preregistration，尚未把 candidate 标记为 accepted。

### 2026-07-14：GUI-Owl native logits/history smoke

- 14 个 pinned snapshot files 已在 Aries 持久盘完成 SHA256 校验；
- 补齐 AndroidWorld `open`、`answer`、`key`、`recents` executable action types，并实现 pinned `mobile_use` prompt/parser；
- 单图与 native 5-image history 均返回 finite logits，并各生成唯一合法 click；
- 5-image 输入为 2,365 tokens，峰值 allocated GPU memory 17.07 GiB；
- 两条输出都与 fixture tap 匹配只视为 incidental，不计入 coverage；
- 结果见 `data/results/gui_owl_native_smoke/`。下一步启动 AndroidWorld emulator/reward smoke。

### 2026-07-14：AndroidWorld container compatibility fix

- pinned MobileAgent Dockerfile 因 `openjdk:18-jdk-slim` 不再可解析而无法原样构建；
- 新增严格、可测试的构建前处理，仅替换为 `eclipse-temurin:17-jdk-jammy`；
- upstream `setup.py` 的未声明 `pkg_resources` 在 uv 隔离构建中引发第二次失败，
  按构建器建议严格切换为 `--no-build-isolation`；
- 源码编译的 Python 3.11 环境不能复用 Ubuntu Python 3.10 的 wheel，因此固定 uv
  `0.11.28`，并预装 `wheel==0.45.1` 与 upstream 已锁定的 `grpcio-tools==1.71.0`；
- 保留 upstream checkout 与 pinned revision 不变，未修改 task、reward、agent 或 prompt。

### 2026-07-14：AndroidWorld environment/reward smoke

- 在 Aries KVM 上构建并启动 Pixel 6 / API 33 / Google APIs x86_64 emulator，读取到 116 个
  AndroidWorld task types；
- 使用 validation seed `271828` 和单组合 suite 初始化 `SystemWifiTurnOn[0]`；
- 4 个状态变更全部通过 `/execute_action` 完成，截图 payload SHA256 发生变化；
- `/task/score` 从 0.0 变为 1.0，随后 `/task/tear_down` 成功，正式 smoke 耗时 44.189 秒；
- Contacts 首次 setup 有权限文案 mismatch warning，后续 validation 必须单独标记 setup failure；
- 结果见 `data/results/androidworld_environment_smoke/`。下一步先提交冻结 task partition manifest，
  之后才启动 GUI-Owl validation rollout。

### 2026-07-14：AndroidWorld task partition 冻结

- 从 live pinned registry 读取并排序 116 个 task types，registry SHA256 为
  `185ae2019706693bd32ecc25ffd0c8f87be87331cae6f7d7e31c91674c962b89`；
- 按预注册 SHA256 bucket rule 得到 train 60 templates / 180 instances、validation 31 / 62、
  test 25 / 75；
- 不对不均匀 split 做事后 rebalance，final test 在 validation gate 通过前保持 sealed；
- manifest 与重建测试见 `code/configs/androidworld_task_partition.json` 和
  `docs/androidworld_task_partition.md`。

### 2026-07-14：AndroidWorld validation execution plan

- 只实例化 validation split，未生成 final-test 动态参数；
- 固定 62 个 goal、template、complexity、home reset flag 与官方 dynamic step budget，instance
  records SHA256 为 `8204e7f832f1d70becd51f299977f0e4a322a080bbfab64010345c48f901b0e8`；
- 62 个实例全部从 home screen 开始，step budget 最小 10、最大 60；
- 发现 upstream wheel 漏装 `task_evals` 子包，按官方 server 相同条件从 Docker `/` 源码根运行，
  未修改 benchmark package；
- 计划见 `code/configs/androidworld_validation_plan.json`，下一步只跑单个 validation instance 的
  GUI-Owl closed-loop smoke。

### 2026-07-14：GUI-Owl 单实例 AndroidWorld closed-loop smoke

- 首次 `SystemWifiTurnOn[0]` 尝试在 policy generation 前暴露 upstream a11y reset failure；保留
  官方 `initialize_task → agent.reset` 顺序，没有为通过 smoke 跳过 reset；
- `ClockStopWatchPausedVerify[0]` 首条模型动作暴露 MobileAgent HTTP server 的 JSONAction schema
  mismatch：旧 schema 拒绝四坐标 swipe；新增 fail-closed build-context patch，与 pinned GUI-Owl
  converter 的 `new_json_action` 对齐；
- 修复 image `sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486`
  已通过独立四坐标 swipe transport smoke；
- 最终 frozen-policy episode 从 reward 0 开始，4/4 outputs parsed，3 个 click 与 1 个
  `status/task_complete` 全部执行成功，policy 明确 done，最终 reward 与官方 success 均为 1；
- 单实例耗时 78.675 秒，最多 4 张可见截图、2,040 input tokens、16.90 GiB peak allocated GPU
  memory；完整 trace 见 `data/results/gui_owl_androidworld_validation_smoke/`；
- 该结果只允许进入完整 62-instance validation rollout，不接受 GUI-Owl 为主 teacher，也不构成
  CausalCache 方法效果证据。

### 2026-07-14：完整 validation 首次启动与 action-alias 修复

- 首次完整 rollout 在 4 个独立 emulator worker、单 A6000、单模型串行 generation 下启动；
- 前 16 个已 checkpoint episode 中，2 个 Clock 实例 official success，另外 14 个在首步被本项目
  bridge 错误拒绝；这些首步都是结构合法的 `action=open_app`，错误不是 policy parse failure；
- pinned MobileAgent converter 同时接受 `open` 与 `open_app` 并映射到 AndroidWorld
  `action_type=open_app`，HTTP executor 也已在环境 smoke 中执行同一 action type；
- rollout 被立即中止，16 个 checkpoint 标记为 implementation-invalid 并从正式 gate 排除；诊断摘要见
  `data/results/gui_owl_androidworld_validation_attempt1/`；
- bridge 已补齐 `open_app` alias 与双 alias regression test。正式 rollout 必须从空 checkpoint 目录
  重启，不能 `--resume` 这次无效尝试。

### 2026-07-14：完整 validation 第二次启动与 native-resolution 审计

- 修复 `open_app` 后从空目录启动 4-worker validation，在 45 个 checkpoint 时得到 12 个
  official success、7 个 infrastructure failure、1 个 parse failure；即使余下 17 个全部成功，
  上界也只有 29/62；
- 在按协议 early-stop 后复核 pinned adapter，发现 runner 强制 256 visual tokens/图，而上游先做
  1080×2400→1092×2408 resize，再由 GUI-Owl processor 产生 grid `[1,150,68]`；merge size 2
  对应 2,550 effective visual tokens/图；
- 同一 pinned processor 对原图和上游 resize 图均返回 `[1,150,68]`，因此 model-default resolution
  是可机器复现的唯一修正，不是基于 success 调参；
- 第二次运行整体标记为 configuration-invalid，不用于拒绝 GUI-Owl，也不改变 validation plan、
  gate threshold、prompt、action equivalence 或模型权重；诊断见
  `data/results/gui_owl_androidworld_validation_attempt2/`；
- runtime 现在显式区分 model-default 与 fixed-token preprocessing，并记录实际 `image_grid_thw`
  和 effective visual token count；
- model-default 1/5-image smoke 随后通过：GUIOdyssey fixture 每图 grid `[1,116,138]`、4,002
  visual tokens，5 图共 20,010 visual tokens / 21,185 input tokens；last-token logits finite，
  generation 是唯一合法 click，5-image generation 峰值 22.77 GiB；
- 该 fixture 的横向截图 token 数高于 AndroidWorld；1080×2400 AndroidWorld processor replay
  仍固定为 `[1,150,68]` / 2,550 tokens。结果见 `data/results/gui_owl_native_resolution_smoke/`。

### 2026-07-14：GUI-Owl 正式 native-resolution validation 判负

- 从空结果目录启动 4-worker、单 A6000、model-default resolution 的 frozen validation；
- 在 47/62 个原子 checkpoint 时得到 15 个 official success、23 个正常终止失败、9 个
  infrastructure exception；496/496 actions parsed；
- 所有 generation 均记录 `[1,150,68]` grid 与每图 2,550 effective visual tokens，关闭了前次
  256-token configuration-invalid 问题；
- 固定 62 条分母下至少需要 31 个成功；剩余 15 条全成功的上界也只有 30/62，因此按预注册规则
  early-stop，并停止 GPU runner 与 utilization monitor；
- 不报告受 worker 完成顺序影响的 15/47 为 benchmark success，只报告固定分母下界 15/62 与
  上界 30/62；test partition 保持 sealed；
- GUI-Owl 正式拒绝为 validated teacher，不升级 AndroidWorld attribution contract，不在该 stack
  上生成 restoration labels 或训练 gate；结果见 `data/results/gui_owl_androidworld_validation/`；
- 47 条 reusable traces 已聚合为单个 gzip JSONL shard 并上传到 private Hugging Face dataset
  `gavinlaw/causalcache-androidworld-validation-mobile@v0.1.0`
  (`3fcca45fffe9842c9fcebbf5c6c27c9540bb1515`)；Git 只保留 README、summary 和 manifest，不重复
  提交逐 episode 小文件。

### 2026-07-14：仓库 SoT 与跨芯片执行结构

- 顶层固定为 `README.md` 总索引、`paper/` LaTeX、`code/` 可执行逻辑、`data/` 小数据、`docs/`
  交接记录；原 package/scripts/tests/configs/requirements 与轻量 results 已机械迁移并保留 Git history；
- `pyproject.toml` 从 `code/` 发现 `causalcache` 与 `scripts` packages；Makefile、tests、active configs、
  README/docs 链接同步更新，并新增 layout regression test；
- 新增 `AGENTS.md`、`code/README.md`、`data/README.md` 与 `docs/execution.md`，固定每步
  verify→HF→docs→commit→push main 的完成条件；
- 实测 Hyper01 为 x86_64、8×H200 且 KVM 可用，但 Docker root `/var/lib/docker` 位于只剩约
  7.7G 的根盘，尚无 AndroidWorld image；因此 H200 先承担 policy/offline 工作，已验证的 Aries
  stack 继续承担 closed-loop MVP；
- Mac `~/hf_key.txt` 只允许通过 stdin 用于单次 HF API 调用，不进入 argv、环境变量、远端持久盘、
  Git 或日志；
- 迁移后 51 个单测、contract validation、synthetic deterministic replay、fresh editable install、30 个
  JSON、全部相对 Markdown links 与 AAAI LaTeX 构建均通过；synthetic summary 的陈旧
  `contract_version` metadata 从 `0.1.0` 对齐到实际 config `0.3.0`，其数值结果不变；
- 本次只改变 repository/execution contract，不改任何历史 experiment semantics 或结果数字。

### 2026-07-14：Replacement teacher v1 预注册

- 本轮唯一 candidate 冻结为
  `mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060`；官方报告
  AndroidWorld 71.6%，但该数字不作为本项目 gate 结果；
- 新 checkpoint 与已接入的 Instruct 版本共享全部 10 个非权重 runtime files，四个 BF16 weight
  shard 的 SHA256 单独固定在 `code/configs/gui_owl_1_5_8b_think_snapshot.json`；
- prompt、native 5-image history、action grammar、model-default visual preprocessing、deterministic
  generation、62-instance validation plan、95% parse gate 与 50% success gate 全部保持不变；
- Hyper01 只承担 standalone logits/parser smoke；Aries 继续承担已验证的 closed-loop stack，未预注册
  跨主机 policy/environment topology；
- 在任何 candidate inference 前发现并修正 smoke 协议错误：原 decision step 4 只会产生
  4-image history，现固定为 step 6 以真正覆盖 5-image 上限；同时显式固定
  `max_new_tokens=256`；
- `UI-Voyager` 未选为主 teacher，因为官方推理只暴露当前截图，加入历史截图会改变其已报告策略接口；
- 首次 smoke 后的当时状态为 `smoke_format_adaptation_pending`，不是 accepted teacher，也不是
  CausalCache 效果证据。

### 2026-07-14：GUI-Owl Think 首次 strict-parser smoke

- Hyper01 单 H200 在 Git `e40a0c780c96dda1d43ac5ae1469ccca86d9887d` 运行，model/data revision、
  Docker digest、完整 argv 与 runtime 已写入
  `data/results/gui_owl_1_5_8b_think_smoke_strict/run_manifest.json`；
- single-image 与 5-image history 的 last-token logits 均 finite，实际 image count 为 1/5，峰值
  显存约 19.15/24.47 GB；
- 两个 raw output 都是一个闭合 `<think>...</think>` prefix，后接一个原生
  `Action + mobile_use <tool_call>`；旧 strict parser 按设计拒绝，parse coverage 0/2；
- 该次运行状态为 `invalid`，不是 candidate rejection；预注册只允许在任何 validation 前
  做一次与 action correctness 无关的 fail-closed format adaptation，并先提交测试。

### 2026-07-14：GUI-Owl Think format-only parser 适配

- 只允许 raw output 开头最多一个小写、闭合的 `<think>...</think>` block；剔除后仍
  full-match 单行 `Action:` 和唯一 `mobile_use` tool call；
- 未闭合、多 block、非前缀位置、suffix 或额外文本全部 fail closed；历史 action description
  与当前 executable parser 复用同一 boundary helper；
- 两个已记录 smoke raw outputs 与人工 malformed cases 均有回归测试；未查看或使用
  AndroidWorld success，prompt、action mapping、equivalence 和 threshold 均未修改；
- 该里程碑结束时状态为 `smoke_rerun_pending`，必须先 push 该 parser commit 才能原参数重跑。

### 2026-07-14：GUI-Owl Think interface smoke 通过

- Hyper01 单 H200 在 Git `72c59c4d5f2f9c4f7eb42758d6606b1736e6121d` 以原 model/data/fixture/
  generation 参数重跑，完整 provenance 见
  `data/results/gui_owl_1_5_8b_think_smoke/run_manifest.json`；
- single-image 和真实 5-image history 的 finite logits 与 parse 均为 2/2，通过预注册
  interface gate；`executable_match` 0/2 仍只作 diagnostic；
- 与首次运行比较，single-image raw output 逐字相同；5-image 只有 Action description 中
  一个句点的引号内/外位置不同，thinking、tool call 与 canonical action 相同；
- 该里程碑结束时状态为 `androidworld_validation_pending`，不是 accepted teacher；只允许进入冻结的
  62-instance validation plan。

### 2026-07-14：AndroidWorld automatic early-stop orchestration

- full runner 新增显式 `--early-stop-when-success-is-mathematically-impossible`，不再依赖
  外部人工观察后终止 GPU process；
- 只在 episode JSON 原子写盘后计算固定分母 success 上界；上界严格低于 gate 时设置
  stop event，已在途 worker 仍完成 score/tear-down；
- early-stopped summary、CLI 输出与 full-run summary 均可正常写入；`--resume` 增加 plan index、
  instance、filename 与完整 run contract 校验，非 resume 运行拒绝非空 output directory；
- 每条 episode 新增 Git/model/runtime/processor/generation/plan/server identity；CLI 的 full Git SHA 必须
  等于 clean checkout HEAD，防止 Think run 静默复用 Instruct checkpoint；
- 62 条、50% gate 的边界测试固定：47 checkpoints/16 successes 不能停，47/15 必须停。

### 2026-07-14：GUI-Owl Think Aries 正式运行 preflight

- 在 Aries `/mnt/data6/jiaxuanluo/causalcache` 准备独立 clean checkout、venv、model cache 与空实验目录；
  preflight checkout 为 Git `aaa19a9efc72b3a48cb05375f19649a8f9e3f210`，正式 runner 启动前会同步
  本里程碑对应的最新 pushed `main` 并把 full SHA 写入每条 episode contract；
- model snapshot 为 `mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060`，
  14/14 pinned files 已核验；runtime 为 Python 3.12.3、PyTorch 2.11.0+cu130、CUDA 13.0、
  Transformers 5.6.0；
- policy container `sglang-omni-jaxan-07141905` 使用 image
  `sha256:81b5df11b32ad8460be270a67066196cb7c6d4fb92cb5d05a44fb06d1ec88d21`，只暴露 Aries
  physical GPU 1（A6000 UUID `GPU-7bf06053-364d-92a4-fdd0-b04b2dd66291`，容器内 `cuda:0`）；
- 首次带 `--privileged` 的空 policy container 会看到全部 8 张 GPU，未加载模型即删除重建；正式
  container 去掉该权限后 `nvidia-smi` 与 PyTorch 均只看到一张卡；
- 四个 pinned AndroidWorld executors 的 `5000–5003/health` 全部通过，server image digest 保持
  `sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486`；Think validation
  output directory 在启动前不存在；
- 为避免提前观察后重复 validation 样本，不再另跑 `ClockStopWatchPausedVerify[0]`。完整 runner 的首个
  原子 checkpoint 同时作为 infrastructure canary，结果不得用于修改冻结协议。

### 2026-07-14：AndroidWorld validation deterministic packaging

- 新增 `scripts.package_androidworld_validation`，对完整或数学确定 early-stop summary、冻结 plan、
  episode instance/index/filename/run contract 与聚合计数做 fail-closed 复核；
- raw episode 按 `plan_index` 写成一个 canonical UTF-8、`mtime=0` 的 deterministic gzip JSONL shard，
  避免把 resumable checkpoints 作为大量小文件上传；
- HF stable repo 内按 `data/<policy-slug>/` 与 `runs/<policy-slug>/` 隔离 policy artifact；payload manifest
  记录目标 repo/tag 和内容 SHA256，但不记录尚未产生的 HF OID，避免 revision 自引用；
- 相同输入双重打包 byte-identical、early-stop 决定性与错误 contract 拒绝均已覆盖；当前全量 65 个
  tests 与 contract validation 通过。该工具在正式 run Git commit 之后实现，只用于事后 artifact
  packaging，不改变已运行的 policy、plan 或 gate。

### 2026-07-14：GUI-Owl Think AndroidWorld validation 判负

- 在 clean Git `36526c58997be5409f65c5976fb35e17aa007ad7`、Aries physical GPU 1、四个 pinned
  AndroidWorld executors 上运行冻结的 62-instance plan；Python 3.12.3、PyTorch 2.11.0+cu130、
  CUDA 13.0、Transformers 5.6.0、BF16、model-default visual resolution、最多 5 images、
  `do_sample=false`、`max_new_tokens=256`；
- 40 个原子 checkpoint 时已有 8 个 success，剩余 22 条即使全部成功也只有 30/62，触发自动停止；
  两个在途 worker 完成 score/tear-down 后，最终 summary 为 42 checkpoints、9 successes、20
  unobserved，上界 29/62（46.77%），低于 31/62 gate；
- 513 model steps 中 512 parsed（99.81%），parse gate 通过；outcomes 为 9 official success、23
  terminal failure、1 parse failure、9 infrastructure failure；9 exceptions 包括 6 个 HTTP 500、2 个
  live-instance mismatch、1 个 nonzero initial reward；
- wall time 6384.83 s；GPU monitor 的 86 个 10 秒 window-average utilization mean/median 为
  84.7%/87%，30 个窗口至少 90%，forward peak 反复为 100%；较低窗口来自 emulator round-trip，未改变
  单 GPU 与冻结 worker topology；
- 相同输入独立打包两次 byte-identical；42 条 raw traces 上传 private HF dataset
  `gavinlaw/causalcache-androidworld-validation-mobile@v0.2.0`
  (`0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`)，上传后 5 个文件 SHA、解压记录数与 plan index
  均重新核验；旧 `v0.1.0` tag 未改写；
- candidate 状态改为 `rejected_by_androidworld_validation_gate`。按预注册规则停止 replacement round，
  test split 保持 sealed，不升级 teacher contract、不生成 restoration labels、不训练 gate。

## 当前 artifact 状态

### 2026-07-14：go/no-go existence diagnostic 预注册

- 在任何 restoration forward 前冻结 `exploratory_oracle_diagnostic_v1`：Qwen exact revision、旧
  GUIOdyssey artifact、decision step 4/8、预算 512/1024、full-vocabulary
  teacher-forced action-path mean KL、全部可行 coalition、固定 baselines 与四分支判据；
- 明确这两个 states 是在 2/9 coverage 后观察到的 matched subset，只能产生当前 representation/policy
  stack 的工程结论或硬负信号，禁止训练 gate、替换 primary teacher gate 或输出论文级 `GO`；
- paper-level go/no-go 仍要求未观察 restoration 的独立多轨迹 manifest，且必须先通过未修改的 50%
  full-history executable-match reference gate；旧 single-trajectory rejection 保留披露；
- compute 改用 Hyper00（Hyper01 有用户任务）：只读审计时 Hyper00 8×H200 均为空、`/data01` 与
  `/data02` 空间充足；正式 GPU forward 前仍须重新运行 10 秒 idle-cleanup preflight，并显式绑定至多
  一张即时空闲 GPU。

### 2026-07-14：pre-forward visual-cost correction

- artifact 的 10 张截图均为 2208×1840；在未运行任何 restoration forward 的前提下，用 pinned
  Transformers 5.6.0 `smart_resize` 静态核验 256-token resize target，实际尺寸为 476×392；
- 按 patch 14、merge 2 核算，每张图是 238 effective visual tokens，每个 restored event 的前后两张图
  成本为 476。原预注册中把 processor target 直接写成 effective cost 512，现已显式纠正；
- 预算 cap 保持 512/1024 不变，因此可行 coalition 仍分别为最多 1/2 个事件，states、distance、baselines、
  threshold 与 forward 数量均未改变。正式 runner 还会逐 coalition 对 `image_grid_thw` 做 fail-closed 复核。

### 2026-07-14：teacher-forced action-path KL runtime

- Qwen runtime 新增 canonical action 无 special-token 编码、严格 causal shift 的
  `prompt + action[:-1]` teacher forcing，以及 full-vocabulary float32 log-prob 输出；
- 每次 forward fail-closed 检查单 batch、logits shape、vocabulary boundary 与 finite 值，并记录 prompt/
  forced token 数、image grid、effective visual tokens、latency 和 peak GPU memory；
- KL helper 同时返回 per-token、sum 和 mean，拒绝 materially negative 或非 finite 输入；纯 CPU 测试覆盖
  shift、special-token rejection、self KL 与已知 Bernoulli KL，尚未产生 GPU 实验结果。

### 2026-07-14：go/no-go deterministic reducer

- 新增纯 CPU reducer：首个 action JSON 的 sorted compact canonicalization、16×16×16 joint RGB
  histogram cosine，以及完整 coalition table 上的 recent、similarity、uniform maximal-random 与 global
  oracle；
- oracle ties 固定按 distance、cost、event ids 排序；global oracle 可选择空 memory，负 restoration 不会
  被强制计为收益；normalized recovery 使用预注册的 `max(D(empty), epsilon)` 分母；
- outcome reducer 要求输入 states 与 config 完全一致，并保证 positive evidence 来自同一个
  memory-sensitive state；非法、重复或非 finite 输入统一输出 `INVALID`。

### 2026-07-14：go/no-go fail-closed runner

- runner 已闭合 full-history executable validation、canonical action tokenization、full/repeat reference、
  36 个 exhaustive mixed-fidelity coalition forwards、实际 visual-token 复核、deterministic baselines、exact
  与 sampled restoration selectors、selected-coalition generation 和最终 outcome reducer；
- attribution 内部把 manifest 的 1-based step ids 显式映射到 estimator 的 contiguous 0-based ids，再映射
  回 result；$K\in\{4,8,16\}$、5 seeds 只读取同一 KL cache，报告 standard error、Spearman、Jaccard 与
  exact-selector utility ratio；
- CLI 要求 clean full Git SHA、dataset SHA、model snapshot repo/revision 与 container image digest，记录完整
  argv、host/GPU、Python/PyTorch/Transformers、时间、latency 与 peak memory；成功不落 full logits，失败原子
  写入一个 `failure.json`；90 项 CPU tests 与 contract validation 通过，尚未启动正式 GPU forward。

### 2026-07-14：go/no-go attempt 1 implementation-invalid

- Hyper00 preflight 重新确认 GPU 0/1 空闲，formal run 只绑定 physical GPU 0；clean Git `527711b`、
  pinned dataset SHA、model snapshot 与 container image digest 全部通过 runner contract；
- decision step 4 full-history generation 通过 executable match，但首个 action-path reference forward
  在 Qwen3-VL 3D RoPE 前 fail closed：追加 12 个 action prefix tokens 后 attention mask 为 2023，未同步
  扩展的 `mm_token_type_ids` 仍为 2011；
- 本次生成 0 个 restoration distances，`valid_for_diagnostic=false`，不解释为方法结果；轻量失败摘要见
  `data/results/go_no_go_diagnostic_v1_attempt1/`；
- 修复同步扩展 `attention_mask=1` 与新文本的 `mm_token_type_ids=0`，未知 aligned tensor 继续拒绝；
  states、预算、distance、baselines 与 threshold 不变，更新 clean main 后从新目录重跑。

### 2026-07-14：go/no-go diagnostic v1 `INCONCLUSIVE_POSITIVE`

- 修复后的 clean Git `2715f31` 在 Hyper00 physical GPU 0 完成 38 个 reference/coalition forwards、2 个
  repeat probes 和 selected-memory generations；两状态 repeat KL=0，全部实际 visual-token accounting 与
  full-history executable validation 通过；
- step 8 summary-only action-path KL 为 0.109727；512-token cap 下 oracle `[1]` recovery 0.526，对比
  recent/similarity `[7]` 0.332、random 0.368；1024 cap 下 oracle/restoration `[1,7]` recovery 0.842，
  对比 recent/similarity `[6,7]` 0.487、random 0.581；
- step 8 的非 recent event 1 exact gain 为正且远高于 epsilon；$K=16$ 五 seeds 全部选择 `[1,7]`，
  min Spearman 0.964、Jaccard/utility ratio 1.0，满足 selection-biased diagnostic 的 positive rule；
- step 4 也 memory-sensitive，但 absolute KL 仅 0.000349；1024 oracle 相对 recent 的 normalized gain 只有
  0.007。全部 selected memories 生成的 action 仍 executable-match，因此没有 action recovery 或 success
  结论；
- canonical compact result 与全部 coalition distances 已写入
  `data/results/go_no_go_diagnostic_v1/`。这两个 states 来自已观察的 matched subset，明确禁止升级为 paper
  `GO`、训练 gate 或改写 rejected teacher decision；
- wall 170.30 s 中记录的 teacher-forced GPU forward 仅 4.09 s；monitor 三个 active 10 秒窗口均为 0%，
  瓶颈是 CPU full-vocabulary KL。独立扩展前必须完成 GPU-side KL/batching，否则 attribution 成本 gate
  不通过。

### 2026-07-14：独立 UI-TARS reference gate 预注册

- 在读取新的 GUIOdyssey source rows 或运行新 policy output 前，冻结
  `code/configs/independent_reference_gate_v1.json`；source pool 为 exact transport revision 的前 16 个
  mobile/use train shards，旧 trajectory `0054832199799795` 显式排除；
- eligibility 只读取 source structure，限制 successful mobile、最后一步 terminal、单 executable action、
  4--12 decisions、安全唯一 source ID、合法图片与原 parser 可表示 action domain；不允许开放式
  `except ValueError: skip`；
- trajectory 用 fixed salt + NUL + source ID 的 SHA256 全序排列；reference/oracle 分别取满足
  8 trajectories/48 decisions/3 apps 与 15 trajectories/60 decisions/3 apps 的最短不重叠前缀，policy
  output 后禁止 top-up；
- UI-TARS revision、256 visual-token target、256 generation cap、10x10 equivalence 与原 interface file
  SHA 全部锁定。reference 仍用原 50% overall + tap/swipe/type_text 每类至少一 match，不新增结果驱动
  threshold；
- Hyper00 正式运行前必须先在旧 9-decision artifact 复现 A6000 的 9/9 parsed、4/9 match boolean vector；
  anchor 不一致则不查看 Hyper00 独立 outputs，转 Aries 执行未修改协议。
- Hyper00 已从 immutable transport revision 下载冻结的 16 个 source shards，共 2,252,923,738 bytes；
  在任何 row decoding 前逐文件计算 size/SHA256 并写入
  `data/manifests/independent_reference_gate_v1_source_files.json`。后续 builder 必须逐项验证，不能接受
  revision 相同但 local bytes 不一致的输入。
- Hyper00 随后用 exact UI-TARS snapshot 在旧 9-decision artifact 运行 behavioral anchor；9/9 parsed、4/9
  match 及逐 step vector 与 Aries A6000 完全相同，`HARDWARE_ANCHOR_PASSED`，因此无需因芯片差异转回
  Aries。compact result 位于 `data/results/ui_tars_hyper00_hardware_anchor/`；
- anchor peak 17.83 GB，generation latency 合计 30.77 s。monitor 两个 active windows 平均只有 21%/18%，
  原因是单样本 variable-history preprocessing + 短 generation；reference gate 保持单卡监督执行，正式
  coalition-scale oracle 前必须先做 batching/concurrency 与 GPU-side KL。
- 新增 v0.4 multi-trajectory builder：逐文件 hash、命名 eligibility exclusion、NFKC app normalization、
  salted shortest-prefix split、image collision/path 检查和 byte-deterministic tar；旧 v0.3 API 不变；
- 新增 formal reference runner：模型加载前验证 clean Git、frozen interface、base 50% gate、HF artifact
  revision/tar/manifest SHA、完整 split denominator、UI-TARS snapshot 实际文件 SHA 与 H200 anchor；合法
  scientific failure 输出 `NO_GO_CURRENT_REFERENCE_STACK`，实现/契约异常原子写 `failure.json`；
- full suite 从 91 增至 106 tests，另通过 contract validation、py_compile 与 whitespace check；
- 在 clean Git `7bd1851` 上从 212 rows 得到 111 eligible trajectories；命名 exclusions 为 length-above 81、
  length-below 1、旧 source 1、invalid executable action 6、terminal failure 12；
- 两个独立 output dirs 均生成同一 manifest SHA
  `3870900442dd0c8f037c9127e0c61c53c9d08c3735164c3c57c857f2c65eb949` 与 163,061,760-byte tar SHA
  `b16bd9c631c3e0ad2576715db6aad72be82ca9dda71612576c5f8b8ab52b0fe2`；230 images、tar 231 members；
- reference 最短前缀为 8 trajectories/75 decisions/14 app labels（tap 58、swipe 2、type_text 9、home 6）；
  disjoint oracle 为 15/132/23（tap 93、swipe 6、type_text 20、home 13）；
- artifact 已上传 private HF
  `gavinlaw/causalcache-guiodyssey-independent-mobile@v0.1.0`
  (`84c9f5a335e9612ccb4bd566f977574f359b2485`)；从 immutable OID force-download manifest/tar 后 size/SHA
  与 pre-upload bytes 完全一致。此时尚未运行任何 independent policy output；下一步为 formal reference
  gate。

### 2026-07-14：独立 UI-TARS reference gate 判负

- 在 clean Git `585fd2aa8061f069008d985552ddaec4bbfd5246`、完整 artifact/model/interface/hardware-anchor
  fail-closed validation 后，于 Hyper00 physical GPU 1 运行冻结的 8 trajectories / 75 decisions reference；
- 69/75 outputs parsed，27/75 executable match（36.0%）；预注册 threshold 至少需要 38/75；tap 21/58、
  swipe 0/2、type_text 5/9，因此 overall 与 required-action 两个 gate 均失败；
- 6 个 parse failure 都是 prompt 允许但 parser 未实现的 `open_app`。这是 v1 action-contract mismatch；即使
  事后把 6 个全部乐观计为正确也只有 33/75（44.0%），且 swipe 仍为 0/2，所以 no-go 对它稳健；
- 逐 decision 复核确认 75 个 `(source_id, decision_step)` 唯一、source list 等于冻结 reference split、旧
  anchor source 已排除、75/75 match flags 可从 raw record 重算；正式输出为 `summary.json`，没有
  `failure.json` 或 incomplete denominator；
- 按 protocol 输出 `NO_GO_CURRENT_REFERENCE_STACK` 并停止；disjoint 15-trajectory / 132-decision oracle
  split 未运行，不生成 restoration labels、不训练 gate、不改 v1 prompt/parser/equivalence/threshold；
- raw per-decision summary 与 monitor 上传 private HF
  `gavinlaw/causalcache-guiodyssey-independent-mobile@reference-gate-v1`
  (`b3e1245c6c6a1723fe2ca3a861148008df39df46`)；四个文件从 immutable revision 强制重下载并核验
  SHA256，轻量结论见 `data/results/independent_reference_gate_v1/`；
- 单 H200 wall 296.75 s、generation latency 合计 185.26 s、peak allocated 18.20 GB；10 个 active monitor
  windows 的平均 window utilization 20.4%、sample max 83%。逐 decision preprocessing/短 generation
  仍需 batching，但科学 no-go 已使本路线不进入 coalition-scale oracle；
- GUIOdyssey source 是 train shards，不能仅凭当前 provenance 排除 UI-TARS 训练数据重叠。任何 v2 必须
  在新 untouched split 前先预注册一致的 `open_app` action contract；这不会回改 v1 结论。

### 2026-07-15：Restoration v2 scientific contract 冻结

- 保留 v1 `NO_GO_CURRENT_REFERENCE_STACK`，不修改原 prompt、parser、threshold 或 negative result；v2
  是 stable self-behavior 的新 estimand，不是降低 expert top-1 gate；
- primary substrate 固定为
  `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`，reference admission
  只检查 restricted action parse、finite logits 和两次 canonical action 一致；expert alignment 只报告，
  不得筛选状态；
- strong low-fidelity event 固定八字段且出现在每个 memory；high fidelity 只增加一张 post-action image，
  不增加 before image、raw action arguments 或额外 action text；
- confirm 固定 decision step 6：events 1--4 为 visual candidates，event 5 的 post-state 等于 current，因而
  summary-only 且不能选择；reference 为四张历史 post-state 加 current，primary capacity `B=2`，允许
  positive-value selector abstain，另做 exact-two ablation；
- teacher forcing 固定 native assistant prefix + nonsemantic Action carrier，并只在 deterministic
  `<tool_call>...</tool_call>` token span 上计算 full-vocabulary mean KL；不复用 full-history 自由文本
  description，避免向 coalition 泄漏 target；
- 旧 15 条 oracle trajectories 的 raw artifact 已读取/打包但无 policy/restoration output，按原顺序拆为
  10 条 label-train、5 条 development；confirm 继续 v1 frozen hash order，从 exact 8+15 后取 20 条
  structurally eligible trajectory，每条一个 step-6 state，禁止 output-driven top-up/filter；
- substrate gate 固定至少 20 states、parse 0.99、finite 1.0、repeat agreement 1.0、memory-sensitive 至少 8；
  confirm 固定 20 states、`K=16`、oracle recovery 0.30、strongest-baseline gain 0.10 + paired 90% bootstrap、
  Spearman 0.80、Jaccard 0.75、utility ratio 0.90；
- v2 action inventory 删除 `key`/`Menu`，接受 `tap`/`open_app` alias，坐标固定 `[0,999]` 且映射进有效像素；
  exhaustive round-trip fixture 100% 通过前禁止 policy output；
- recent/random/OCR+RGB/policy-vision baseline 公式、tie-break 与 source hashes 必须在 confirm 前冻结；
  Spearman/Jaccard/utility ratio 只在 memory-sensitive states 上取 deterministic median，并锁定 ties 和
  degenerate denominator；
- scientific config SHA256 为
  `9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc`；config validator 与回归测试已加入，
  完整说明见 `docs/restoration_v2.md`。这一步没有运行 GPU 或产生新 artifact。

### 2026-07-15：Restoration v2 CPU interface 冻结

- 新增独立 `gui_owl_v2` action/prompt adapter，未修改历史 v1：prompt inventory 与 parser/bridge 精确闭合
  九个 canonical actions，只接受 `tap/open_app` 两个 alias，拒绝 `key/Menu/time/terminate:failure`；
- strict parser 拒绝 missing/extra/wrong-type 参数、duplicate JSON keys、非 finite JSON、thinking 与额外
  prose；canonical target 固定 nonsemantic Action carrier 和 NFKC compact tool call；
- `[0,999]` 到 pixel 使用 integer half-up 等价公式，6 个 extents 的全部 6,000 个 scalar checks 均满足
  endpoints、monotonic 与严格不越界；
- strong LF 八字段 serializer 固定 key order、compact UTF-8 newline、ordered screen-text arrays、32-token
  cap、foreground/executor provenance、screen-change bins，并补齐 swipe direction/displacement 等实现细节；
- post-state-only builder 在 image load 前校验 trajectory/event/decision identity、summary bytes/SHA 与
  current-equivalent event，并对实际 image bytes 重算 SHA 后才 decode；steps 4/5/6 共穷举 28 个
  coalitions（step 6 为 16 个），每个 state 内所有 text blocks byte-identical，image count 始终为
  `1+|S|`，before image、raw source tool call 与额外 action description 从未进入 prompt；
- action fixture 为 14 valid / 23 invalid，CPU parse/canonicalize/reparse/AndroidWorld-payload 全部通过；
  source/fixture/spec 由 `data/manifests/restoration_v2_interfaces.json` 逐文件 hash；
- 真实 pinned AndroidWorld `new_json_action.JSONAction` constructor 和 device-side executor dispatch 尚未
  执行，manifest 分别标为 `pending`；这一步未加载 GUI-Owl、未生成 policy output，也没有新的 HF
  artifact。完整语义见
  `docs/restoration_v2_interfaces.md`。

### 2026-07-15：Pinned AndroidWorld constructor preflight

- 在 Aries 创建 exact CausalCache commit `a9e2afa` 与 MobileAgent revision `11cea575...` 的两个 clean
  detached checkout，没有复用历史 dirty 开发目录；
- 14 个冻结合法 payload 全部通过真实 `android_world.agents.new_json_action.JSONAction(**payload)`；
  module SHA256 为 `14ca00cabf3d5b83e4d55cb683a09a4beccbbc658e21039ca5cf8cef3f543e3f`；
- 结果绑定 interface manifest SHA256 `02744d82...`、runtime image repo digest `6a8f60af...`、完整 argv、
  host/container 与起止时间，见 `data/results/restoration_v2_constructor_preflight/`；
- 该结果不证明 device-side executor dispatch；未加载 policy、未使用 GPU、未生成任何 v2 policy output。

### 2026-07-15：Executor-dispatch formal runner 冻结

- 新增 Aries host-side Docker inspection，绑定 runtime/server container ID、actual image ID、5003→5000
  port mapping、persistent mounts、live health 与四个 executor source hashes；
- formal runner 从每个 native output 重新执行 v2 parser 与 bridge，不直接信 fixture payload；14 个 case
  保持单 worker、固定顺序、每 case 独立 reset、每 case 一次请求且零 retry；
- valid response 必须 exact HTTP 200 JSON echo；另对缺坐标 click 先从 pinned source 记录 constructor-acceptance
  witness，再要求相同 payload 在 live actuation 必须 HTTP 500，并要求失败后 health 与 cleanup reset 正常；
- 独立 reducer 从 raw records 重算 14-case denominator、action-type counts、response binding 和 verdict；
- pre/post inspection 对同一 unique attempt 做时间夹持和 container/source identity 比较；compact
  action/reset/health response 保留 raw UTF-8 body，reducer 重算 bytes/SHA/JSON；大体积 screenshot 只记录
  frozen runner 计算的 digest 与 shape；所有 attempt files exclusive-create，失败不能被同路径成功重跑覆盖；
- 当前仅完成代码与 tests，必须先 commit/push，再在 Aries 正式运行。本步骤没有 policy/GPU output。

### 2026-07-15：Executor-dispatch formal run 通过

- 从已推送 commit `b6e57c2` 的新 clean detached checkout 运行 formal attempt
  `rv2-20260715T101814Z-53016a40`，未复用旧开发目录；
- pre/post inspection 锁定相同 Aries runtime/server container、image、5003→5000 port、mount 与四个 live
  source hashes；dispatch 总耗时 53.39 秒，首帧为 1080x2400；
- frozen parser/bridge 重新生成的 14 个 cases 全部返回 exact HTTP 200 success echo；11 个 AndroidWorld
  action types 的 raw denominator 与 counts 由 reducer 重算；
- 缺坐标 click 的 pinned constructor-acceptance witness 通过，相同 payload 在 live actuation 返回 HTTP 500，
  随后 health 与 cleanup reset 正常；
- canonical offline verdict 为 `PASSED_EXECUTOR_DISPATCH`，summary SHA256
  `61956a457fb15a8fdfd35baf1a9f07c48ab45910a94829d537c9a591b2ffcc39`；四件套见
  `data/results/restoration_v2_executor_dispatch/`；
- 第 4 项 action dependency 已闭合。本次未加载 policy、未使用 GPU、未生成 v2 policy/restoration output。

### 2026-07-15：Restoration-v2 selection/exposure materializer

- 确认 parent HF manifest 虽记录完整 111-pool hash，但 `trajectories` 只含原 8+15 条，不能从 parent tar
  猜出 confirm；正式选择必须重扫 pinned 16 个 Parquet；
- 新增 fail-closed CPU pipeline：先验证 2.25 GB source files，再重建 212 rows / 111 eligible pool，要求
  count、canonical pool SHA、exclusion counts 与原 reference/oracle 顺序全部精确复现；
- confirm 算法固定为排除 exact 8+15、保留 `decision_count>=5`、取首 20 后才检查 app diversity，禁止
  为 diversity top-up；`decision_count>=5` 恰好保证 decision step 6 存在；
- selection manifest 保存完整 pool records、8/15/20 disjoint proof、train/dev/confirm 的 30/15/20 state
  IDs，以及每个 state 的 current、全部已存在 candidate events、紧邻 current-equivalence event 与 action hashes；
- exposure 使用 append-only events 与 reducer，术语明确为 confirm `policy-output untouched`，而不是
  `raw unseen`；已有 v1 UI-TARS output 只覆盖 reference 8 条；
- synthetic/mutation tests 覆盖 step-6 语义、fixed-prefix/no-top-up、pool mutation、overlap 和 exposure
  digest。当前只完成实现；必须从已推送 commit 在 Hyper00 正式生成两个 manifest 后，dependencies 2/3
  才能标为 passed。本步骤未生成 policy/restoration output。

### 2026-07-15：Selection formal attempt 1 被 validator 拒绝

- 从已推送 `main@826b45f` 的 clean detached Hyper00 worktree 重建出 212 rows / 111 eligible pool，builder
  产出 fixed-prefix 20 IDs、30/15/20 state counts；未加载 policy、未使用 GPU；
- 独立 validator 随后拒绝 exposure：canonical JSON 使用 `sort_keys=True`，落盘再读回后 role-object key
  order 改变，而 validator 错把 dict insertion order 当作 expert-access event 的语义；
- 这是 serialization/validator contract bug，不是 selection scientific failure；attempt 路径
  `/data/tmp/restoration-v2-selection-826b45f` 保留且不会覆盖，但不能作为 canonical artifact；
- 修复将 role inventory 改为显式 frozen tuple，并新增“serialize→parse→validate”回归测试。必须先
  commit/push 修复，再从新 commit/new output path 重跑 formal materialization。

### 2026-07-15：Selection formal attempt 2 通过，后因 metadata 不全被 supersede

- 从 pushed `main@ed0706ecb72d9a828f452f7308859f95f9554c55` 的同一 clean detached Hyper00
  worktree，在全新 output path 重跑；source 212 rows、111 eligible count、pool SHA `84d685...`、五类
  exclusion counts 与 frozen 8/15 role order 全部复现；
- 排除 exact 23 条后剩 88 条，其中 84 条满足 `decision_count>=5`；confirm 是 structural order 首 20，
  对应 global indices 23--29、31--43，index 30 因结构条件被跳过，不存在 result-dependent filter；
- confirm 覆盖 29 个 normalized app labels，远高于 frozen minimum 3；8/15/20 union 为 43 且两两不交；
- exact state denominator 为 label-train 30、development 15、confirm 20；65 states 的 current/candidate/
  current-equivalence image 与 validated-action hashes 均已冻结；
- selection SHA256 `13197eedc413717a3f190aa53453c6f34b1db82c57f0b47945d552f38bec74f3`，
  exposure SHA256 `0b1a4dfdb9f23be1a7456f78801f26e0b0800b7f32f011af913257b6919535ae`；
- 独立 validator 通过；第二个空目录的全量重建与第一次 passing build 两文件 byte-identical；canonical
  ID 与科学选择结果有效；但轻量 summary 未保存完整 argv、起止时间与 dtype 声明，因此不作为
  最终 canonical runtime record，也不事后伪造缺失 metadata。

### 2026-07-15：Selection canonical formal run 通过

- 从 pushed `main@30879c09e896a61929c66935f98c21e6c3fc7ff5` 的全新 clean detached Hyper00
  worktree 重跑，formal run 为 `2026-07-15T11:11:29.634935126Z`--`11:11:36.857590859Z`；
- 完整 argv、validator argv、所有 input/config/source hashes、container digest、Python/pyarrow、
  `dtype=not_applicable`、null seed 与 negative output declarations 已记录在 result summary；
- canonical selection SHA256 `292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b`，
  exposure SHA256 `bc1224826e0642c2374f6cfbebd676389d8c17ce6bf8a789f08903740cf97a95`；
- 第二次 rebuild 为 `11:11:52.565730649Z`--`11:12:00.086714732Z`，字节级一致，且两次均通过
  独立 validator；dependencies 2/3 正式闭合。本步骤未加载 policy、未使用 GPU、未生成任何
  v2 policy/restoration output。

### 2026-07-15：OCR/image implementation identity 冻结

- 确认 Hyper00/Aries 都没有现成 OCR runtime 或 weights；选择 CPU-only `RapidOCR==3.8.4` +
  `onnxruntime==1.24.4`，显式 PP-OCRv5 mobile detector + English recognizer，关闭 orientation cls，
  intra/inter-op threads 均为 1；
- Hyper00 persistent venv 已安装 full exact lock 并 `pip check` 通过；det/rec 从 RapidOCR pinned
  ModelScope v3.8.0 URLs 下载，inactive classifier 从 hash-pinned wheel 提取，三个 SHA 与 upstream
  manifest 一致；
- 新增 fail-closed backend config、exact runtime lock、lazy optional-dependency adapter、canonical node/record
  schema、256x256 Pillow bilinear implementation、config/golden validator 与 mutation tests；
- Git golden 已冻结 2x2 RGB 和两行 English text PNG bytes 及 prepared hashes；OCR expected output
  仍必须从已 push commit 在 Hyper00 两个独立进程生成；
- 本里程碑未使用 GPU，未加载 GUI-Owl，未产生 policy/restoration output。HF model
  upload、immutable re-download、source manifest 和 real-screen golden 尚 pending，因此 dependency 5 未闭合。
- 从 pushed `main@0e1dfa1` 的首次 Hyper00 synthetic inspection 完成真实 OCR forward，识别得到
  `Causal Cache / Step 42`，但 serialized package-source evidence 以 basename 为 key，两个不同目录的
  `main.py` 发生覆盖；该 run 已登记为 `INVALID_EVIDENCE_SCHEMA_PACKAGE_PATH_COLLISION`，不回写 expected，
  修复后必须从新的 pushed commit 重跑两个独立进程。
- 修复后从 pushed `main@09e4f6d` 在 Hyper00 两个独立进程运行 `inspect-golden`，UTC brackets 为
  `11:47:47.140669493Z--11:47:48.656088201Z` 与
  `11:47:57.685531727Z--11:47:59.202436027Z`；两份 canonical JSON byte-identical，SHA256 均为
  `6cda74bb4f33708795842c947dac53e380e9f0d4e3debcbf9bb38334a645af44`。expected fields 已回写 fixture，
  但仍必须从包含 expected 的新 pushed commit 跑 `validate-golden` 才能标记 synthetic golden passed。
- pushed `main@b82c3d8` 的两次 `validate-golden` 均通过且输出 byte-identical（SHA256 `e1fee087...5245`），
  但审计发现 fixture 顶层 status 仍含 mutable pending lifecycle。该 status 已改为永久内容描述
  `synthetic_expected_inspection_frozen`；因此 b82c3d8 validation 作为 superseded passing attempt 保留，
  必须再从最终 static-status fixture commit 重跑后才给 synthetic golden 最终 verdict。
- 从 pushed `main@820fa54` 对最终 static fixture 两次运行 `validate-golden`，UTC brackets 为
  `11:53:34.926118232Z--11:53:36.505118659Z` 与
  `11:53:53.276034846Z--11:53:54.879217749Z`；两次均为 `PASSED_OCR_GOLDEN_VALIDATION`，validation JSON
  byte-identical，SHA256 `3f4fde7c...49a6`，最终 fixture SHA256 `8c81feb3...bca6`。synthetic golden passed；
  在该 run 时 HF immutable revision、6-image real-screen golden 和 final manifest 仍 pending。

### 2026-07-15：OCR model artifact immutable-verified

- 已创建 private HF model `gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en`，上传 model card、artifact
  manifest 与 det/rec/inactive-cls 三份 ONNX；tag `v1.0.0` 解析到 full immutable revision
  `0dbc766a73ee88d10d52285d434dbfec58617835`；
- 从该 full revision fresh re-download 6 个文件并逐文件复算 size/SHA256，6/6 与上传前 manifest 一致；
  verified at `2026-07-15T12:01:12Z`，HF CLI `1.23.0`；
- 新增 `data/manifests/restoration_v2_ocr_backend.json` 与离线 `artifact-source` validator，绑定 8 个 Git
  source、HF repo/revision/file inventory、fresh re-download 与 synthetic summary；正式 outcome 为
  `PASSED_OCR_ARTIFACT_SOURCE_VALIDATION`；
- 本步骤不把 model cache、token 或 ONNX 放进 Git；Hyper00 staging 现在只是可重建 cache。未加载 GUI-Owl、
  未生成 policy/restoration output；dependency 5 仍只因 6-image real-screen golden 与完成态 manifest pending。

### 2026-07-15：Real-screen golden source/materializer 冻结

- 在读取任何 real-screen OCR/policy/restoration output 前冻结 17-file source contract；当前 SHA256 为
  `374a38c997a1ee9a715a8cf6ce9b7ca26edc1cf56f503c2d42a97436afac16c5`，source-only validator outcome 为
  `PASSED_REAL_SCREEN_SOURCE_VALIDATION`；
- 从已冻结 selection witnesses 复算 45 screening states / 180 occurrences，其中 candidate post-state 135、
  current 45；按 SHA 去重得到 75 unique images（55 portrait / 20 landscape / 0 square）；confirm 97 unique
  images 与 eligible pool 的 SHA overlap 为 0；
- policy-blind 排序已固定 portrait/landscape 各 3 张 exact SHA/path/dimensions；同 SHA 的全部 occurrence paths
  会逐路径验 bytes，canonical representative 是最小 member path，不因当前数据碰巧一 SHA 一 path 而改变契约；
- 新增 raw Parquet reload、deterministic 5-file HF payload materializer 与独立 validator。validator 重建完整
  75-image pool、重放 6 次 OCR、逐字节比较 canonical USTAR/JSONL/manifest，并输出覆盖 `.gitattributes`、
  `README.md` 和三个 payload 文件的 `artifact_tree_sha256`；canonical USTAR trailer 与 HF extra-file
  mutation 均有回归覆盖；
- 本里程碑只冻结 source/runner，不是 real-screen OCR 结果。下一步先 push clean `main`，再在 Hyper00
  CPU runtime 独立构建两次、验证 5-file tree byte-identical、上传 private HF dataset、按 immutable revision
  fresh re-download 并重放 validator。完成前 dependency 5 仍 pending，且没有 v2 policy/restoration output。

### 2026-07-15：Real-screen OCR golden 与 dependency 5 通过

- 从 pushed `main@dcc6e217b4885cef5f745d987a1ec74e57109717` 的 clean Hyper00 checkout，用 CPU-only
  RapidOCR runtime 两次独立 materialize exact 6-image payload；UTC brackets 为
  `12:56:51.799980862Z--12:57:21.514769385Z` 与 `12:58:23.052897194Z--12:58:53.066616371Z`；
- 两个输出目录的 `.gitattributes`、README、raw-image USTAR、OCR JSONL 与 manifest 5/5 byte-identical，
  complete tree SHA256 `605d6396b0cde84697ff3f2407a3630d7ccdb822f55c2bdae56be82e942a7e25`；
  两次独立 validator 都从 2.25 GB raw Parquet 重建 75-image pool 并重放 6 次 OCR，结果为
  `PASSED_REAL_SCREEN_ARTIFACT_VALIDATION`；
- exact tree 已上传 private HF dataset
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@ocr-real-screen-golden-v1.0.0`，tag 解析到 immutable
  revision `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`；从全新本机 cache 下载后 5/5 hashes 一致，送回
  Hyper00 第三次 raw-source + OCR replay 仍通过；
- 完成态 `data/manifests/restoration_v2_ocr_backend.json` 绑定 14 个 Git source、HF model/dataset 两个
  immutable revisions、11 个远端文件 identity 与
  `data/results/restoration_v2_ocr_backend/real_screen_summary.json`。离线 outcome 仍为
  `PASSED_OCR_ARTIFACT_SOURCE_VALIDATION`，但现报告 `dependency_5_closed=true`；
- confirm images、GUI-Owl policy 与 restoration output 均未使用或生成。OCR dependency 5 已 passed；完整
  derived artifact、baseline implementation/source hashes 与 execution config 仍 pending。

### 2026-07-15：Deterministic baseline 纯公式实现

- 新增 `code/causalcache/restoration_v2_baselines.py`，实现 summary-only、recent、uniform-random exact
  expectation、OCR+RGB similarity 与 frozen-policy vision similarity；所有输入均按四候选 exact coverage
  fail closed；
- random 不采样，也不使用 confirmatory attribution seed；它按 lexicographic 顺序枚举六个 2-of-4 subsets，
  对 runner 提供的六个 normalized recovery 求解析均值；
- OCR+RGB 使用 frozen OCR normalization 后的 token set Jaccard 与 256x256 RGB 16x16x16 joint histogram
  cosine 各 0.5；policy-vision 对 spatial-merger 后的 visual token rows 做 mean-pool、L2-normalize 与 cosine；
- 11 个定向 tests 与 16 个 scientific-contract tests 通过。本步骤未加载 policy、未生成 policy/restoration
  output；policy-vision extractor identity 和完成态 source-hash manifest 仍 pending，因此 dependency 6 尚未
  标为 passed。

### 2026-07-15：Policy-vision extractor 与 dependency 6 通过

- 新增唯一 policy-vision baseline 入口 `code/causalcache/policy/gui_owl_v2_vision.py`；旧 generic
  Python-double vision entry 已删除，避免 runner 在两套数值语义间误选；
- extractor 只调用 Qwen3-VL final main merger `pooler_output`，按逐图 `t*h*w/4` 边界在 accelerator 上做
  BF16→FP32 mean/L2/cosine；明确排除 pre-merger `last_hidden_state` 和 layer 8/16/24 DeepStack outputs；
- 正式调用前必须逐文件验证 GUI-Owl 14-file snapshot（17,545,907,171 bytes）、repo/revision、exact local
  inventory、Transformers 5.6.0 与三份 source SHA。Aries verifier 对实际 snapshot 全部通过；Hyper00/Aries
  source SHA 一致；
- Aries 真实 PyTorch runtime 的 5 个 fake-model Torch tests 全部通过；测试不加载 GUI-Owl，不调用 language
  model、LM head 或 generation，也未读取 confirm score；
- `data/manifests/restoration_v2_baselines.json` 绑定 11 个 Git source 与全部公式/identity，离线 validator
  outcome 为 `PASSED_BASELINE_SOURCE_VALIDATION`；dependency 6 已 passed。

### 2026-07-15：完整 derived builder/validator source 就绪

- 新增 policy-blind full builder、independent validator 与 15 项定向 schema/artifact tests；正式 inventory 固定为
  35 trajectories、175 events、65 states、210 张 `observation-000..005` 与 210 条 uncapped OCR records；
- builder 不再接受裸外部 OCR JSONL，而是验证 exact raw source 后，用 pinned RapidOCR/ONNX Runtime/wheel/model
  直接生成记录；builder post-write 和 standalone validator 都重跑 210 OCR 并逐条 exact compare；
- raw GUIOdyssey tool call 必须与 archived canonical executed action 的 type/target/text/case 全字段一致；
  low-fidelity 保存 OCR multiset delta、screen-change、discard counts、exact serialization/SHA，high-fidelity 只保存
  post-state image pointer；
- formal 路径同时绑定 clean `origin/main` Git revision、v2 contract HF repo、v1 source dataset、selection/exposure、
  OCR model/wheel/package-source/runtime/recognizer identity，并对 nonformal counts、dirty checkout、identity drift、
  action drift、OCR replay drift与 extra files fail closed；
- 本机完整 212 tests 通过（9 项因可选 Pillow/PyTorch runtime 跳过）；Aries pinned Pillow 相关回归 66/66
  通过。独立审计发现的裸 OCR、counts、action、provenance 与 replay blockers 均已修复；尚未运行正式全量
  build，因此 dependency 1 仍 pending，且仍无 policy/restoration output。

### 2026-07-15：完整 derived artifact 与 dependency 1 通过

- exact pushed builder commit `1a01f2323647d092cab67f0531ecb877a4a255de` 在 Hyper00 CPU-only runtime
  完成两次正式 build：UTC `14:26:37.779--14:47:57.486Z` 与
  `14:51:24.170--15:12:43.238Z`；两次均为 35 trajectories / 175 events / 65 states / 210 images /
  210 full OCR records；
- 两次 exact 6-file artifact bytes 完全一致，tree SHA256 为
  `475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e`，OCR aggregate 为
  `1e04ddbdd2fd6e5fc50206c436f512908b269fc95566476c799a075eb9af7010`；
- artifact 已上传 private HF dataset
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@restoration-v2-derived-v1.0.0`，tag 解析到 immutable
  `89f136abaff797e14fe758a198996e51032a10a6`。repo `main` 当前为 9 files；旧
  `ocr-real-screen-golden-v1.0.0` 仍固定到 `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`；
- 第一次 `hf download` CLI preflight 同时传入 `--cache-dir` 与 `--local-dir`，在下载前被参数检查拒绝，
  零文件落盘，记为 superseded preflight failure。随后使用新的空目录只投影 exact 6 files，fresh immutable
  download 成功；
- Hyper00 在 UTC `15:20:56.587--15:31:30.534Z` 对该 immutable projection 完成第三次 210-record OCR
  replay，tree/OCR aggregate 与两次 build 完全一致。三次均为 `policy_loaded=false`、
  `policy_output_generated=false`、`restoration_output_generated=false`；dependency 1 已闭合，详细轻量证据见
  `data/results/restoration_v2_derived_artifact/`。

### 2026-07-15：GPU-resident KL 与 deterministic microbatch 源码前置

- 新增 `causalcache.restoration_v2_gpu_kl`：正式 API 只接受同一 CUDA device 上的
  `[B,T,V]` tensor，reference 固定为 normalized FP32 log-probabilities，candidate 为 BF16
  logits 或 normalized FP32 log-probabilities；FP32 `log_softmax`、full-vocabulary KL 和 token mean
  都不离开 GPU；
- cached `[1,T,V]` reference 通过 zero-stride `.expand(B,-1,-1)` 复用，正式返回值是
  GPU 上的 FP32 `[B]`；独立 float64 CPU oracle 只用于审计，等价容差为
  `atol=1e-6, rtol=1e-5`；
- 新增 `causalcache.restoration_v2_batching`：只允许显式 microbatch size 2，按 exact
  `(image_count, sequence_length)` 升序分组，组内按唯一 `input_index` 排序，并固定
  `automatic_oom_fallback=false`；
- Mac 全量 233 tests passed，10 个 optional PyTorch/Pillow runtime tests skipped。本步没有加载
  policy、不使用 GPU，也没有生成 v2 policy/restoration output。Hyper00 CUDA audit、实际
  runtime 与 execution config 仍 pending，因此 dependency 8 未通过。

### 2026-07-15：GUI-Owl v2 runtime 与 CUDA audit runner 源码前置

- 新增 `causalcache.policy.gui_owl_v2_runtime`：完整验证 pinned model snapshot 与 Transformers source，
  固定单卡 BF16、每图 target 2560 effective visual tokens 并记录 actual grid/tokens、native assistant
  prefix/fixed carrier/tool-call action span；batch-1/2 teacher forcing 必须共享同一 decision state 的同一个
  canonical reference action，并只返回 GPU BF16 logits；native generation 固定 deterministic kwargs 并
  strict parse；
- GPU KL 删除 validation path 的四次 `.item()`：finite/normalization/nonnegative/final-finite predicates
  留在 device，invalid example 映射为最终 `NaN` distance；primitive audit 固定
  `validation_scalar_host_reads=0`、`full_tensor_host_transfers=0`；
- 新增 `scripts.audit_restoration_v2_gpu_compute`：从 clean exact Git checkout 绑定显式
  host/GPU/container/runtime identity，并审计 batch-1/float64 CPU、batch-2/two-batch-1、
  logits/log-probs、zero-stride shared reference、invalid-to-NaN 和 single-state microbatch 2/no-OOM；
- Mac 全量 256 tests passed，10 个 optional runtime tests skipped。本步骤未使用 GPU、未实例化或加载
  GUI-Owl，也未生成 v2 policy/restoration output。Hyper00 formal CUDA audit、execution config 与
  readiness validator 仍 pending，因此 dependency 8 未通过。

### 2026-07-15：Hyper00 formal GPU compute audit

- 从已推送 clean detached `main@a0f06f495a78bee43112651040fca6c3d31701ac` 在 Hyper00 单张 H200
  执行 policy-blind synthetic audit；preflight 复核 host/GPU/disk/container，两轮 10 秒采样均为 0%，
  physical GPU 0 被显式选中，container 内只可见 `cuda:0`；
- batch-1 GPU 对两个独立 float64 CPU oracle 的绝对误差为 `6.48e-9`、`3.05e-8`；batch-2 对两次
  batch-1 与 logits 对 pre-normalized log-probs 的误差均为 `0.0`；zero-stride reference storage 共享、
  invalid-to-NaN、validation scalar host reads 0、full-tensor host transfers 0 均通过；
- formal summary SHA256 为 `0b0adbd0...8134`。新增完全独立的 offline validator：不 import audit runner、
  GPU KL 或 batching，而从 run commit Git blobs 重新计算 source hashes，并复核 standard JSON、argv、
  UTC、host/container/GPU identity、数值 equivalence、NaN 与 planner；validation outcome passed；
- 加入 validator 与 negative tests 后，Mac 全量 262 tests passed，10 个 optional runtime tests skipped；
- 本 run `policy_loaded=false`、`policy_output_generated=false`、`restoration_output_generated=false`。
  它只闭合 compute primitive audit，不是 model runtime pass；dependency 8 与 screening 仍 locked。

## Artifact 状态

- Git 代码、配置、论文与轻量测试 fixture：本仓库 `main`；
- GUIOdyssey pilot：私有 Hugging Face dataset `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- GUIOdyssey independent gate：私有 Hugging Face dataset `gavinlaw/causalcache-guiodyssey-independent-mobile@v0.1.0` (`84c9f5a335e9612ccb4bd566f977574f359b2485`)；schema v0.4，immutable re-download verified；
- Rejected policy candidate：上游 Hugging Face model `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；共享机器副本只是可重建 cache；
- Rejected GUI-tuned policy candidate：上游 Hugging Face model `ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`；Aries 副本只是可重建 cache；
- Rejected computer-use policy candidate：上游 Hugging Face model `xlangai/OpenCUA-7B@a2efb7d2b104d477a4a2666a357e79550a28aafc`；Aries snapshot 与 venv 只是可重建 cache；
- Rejected GUI navigation policy candidate：上游 Hugging Face model `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`；Aries snapshot 只是可重建 cache；
- GUI-Owl-1.5-8B-Instruct：上游 Hugging Face model `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；v1 native validation 上界 30/62、未通过 50% success gate；v2 只将同一 checkpoint 用作 stable self-behavior substrate；
- Rejected replacement policy：上游 Hugging Face model `mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060`；native validation 上界 29/62，未通过 50% gate；
- AndroidWorld native validation traces：私有 Hugging Face dataset `gavinlaw/causalcache-androidworld-validation-mobile@v0.2.0` (`0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`)；旧 Instruct artifact 保持在 `v0.1.0`；
- restoration v2 executor evidence：Git `data/results/restoration_v2_executor_dispatch/`，formal verdict
  `PASSED_EXECUTOR_DISPATCH`，14/14 cases，summary SHA256 `61956a45...`；
- restoration v2 exact selection/exposure：Git `data/manifests/restoration_v2_selection.json` 与
  `data/manifests/restoration_v2_exposure.json`，SHA256 分别为 `292c7e52...` / `bc122482...`；20 confirm
  trajectories、45 screening states，formal verdict `PASSED_PREOUTPUT_SELECTION_VALIDATION`；
- independent candidate dataset 已冻结；reference raw record 位于 private HF
  `@reference-gate-v1` (`b3e1245c6c6a1723fe2ca3a861148008df39df46`)；reference 判负，oracle records
  按协议未生成。旧 oracle raw trajectories/images/expert actions 已被 builder 读取和打包，不是 raw unseen；
- restoration v2 derived dataset 的 canonical destination 是 private HF
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`；完整 artifact 已固定在
  `restoration-v2-derived-v1.0.0@89f136abaff797e14fe758a198996e51032a10a6`，旧 real-screen OCR golden
  保持在 `ocr-real-screen-golden-v1.0.0@9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`；
- restoration v2 OCR 三模型的 canonical artifact 是 private HF model
  `gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en@v1.0.0`
  (`0dbc766a73ee88d10d52285d434dbfec58617835`)；6/6 files 已从 immutable revision fresh re-download
  验 hash，Hyper00 `/data/artifacts/causalcache-ocr-ppocrv5-mobile-v1` 只是可重建 cache；
- selection-biased go/no-go compact result：Git `data/results/go_no_go_diagnostic_v1/`；raw 209 KiB debug
  summary 只含可丢弃的 per-token/runtime 展开，canonical distances 与结论已压缩进 Git；
- gate checkpoint：尚未生成，目标 Hugging Face model repo 待 owner 确认；
- 当前没有仅存在共享机器或本地磁盘上的正式实验 output；Taurus/Aries/Hyper 目录只作为 Git/HF artifact
  的 staging/cache。

## 八项 pre-output dependencies 状态

1. derived artifact immutable HF revision/file hashes：passed，
   `restoration-v2-derived-v1.0.0@89f136abaff797e14fe758a198996e51032a10a6`，tree SHA256
   `475e6cf2...a6e`；
2. exact confirm trajectory/state IDs：passed，selection SHA256 `292c7e52...`；
3. exposure ledger：passed，ledger SHA256 `bc122482...`；
4. restricted prompt/parser/bridge/executor fixture：passed；CPU prompt/parser/bridge、真实 pinned `JSONAction`
   constructor 与 device-side executor dispatch 均有独立 evidence；
5. pinned accessibility/OCR identity：implementation/config/weights SHA、synthetic golden 与 HF immutable
   model revision、6-image real-screen golden、HF dataset immutable re-download 与完成态 manifest passed；
6. baseline specification/source hashes：passed，见 `data/manifests/restoration_v2_baselines.json`；
7. v2 interface source hashes：passed，见 `data/manifests/restoration_v2_interfaces.json`；
8. 引用 scientific-config SHA 的 execution config：passed。GPU KL/microbatch/runtime/
   audit source、Hyper00 formal compute audit 与 real processor audit 均已通过；confirm-safe screening loader、
   readiness validator 与固定 45-state production runner source 也已实现；完成态 config/source-hash
   inventory、readiness manifest 与 clean-Git authorization 全部通过。

GPU-side scalar KL、GUI-Owl runtime 与 coalition microbatch 已 implementation-ready，formal CUDA audit 已
通过。新增 runner 在 readiness 8/8 前不 import policy runtime；readiness 后也先对固定 45 states 的
reference/summary 共 90 个 prompt 做 processor-only shape sweep，任何 context overflow 在首个 forward 前
输出 `INVALID_BEFORE_POLICY_FORWARD`。每个 state 在首次 policy call 前写 exclusive attempt marker；崩溃后
存在 marker 而无 terminal record 时禁止 resume 重跑。合法 scientific failures 只包括 parse、repeat-action
mismatch 与 non-finite distance；contract/runtime error 不得伪装成 `NO_GO_V2_SUBSTRATE`。

### 2026-07-15：Restoration v2 screening execution source 完成

- 新增只读 derived loader，先运行 public artifact validator，再逐 byte 复核 canonical JSONL/USTAR、selection
  witnesses 与 90 个 screening image SHA；API 只暴露 10 条 label-train + 5 条 development、45 states，confirm
  state/image 无法寻址；
- 新增真实 `AutoProcessor` audit：完整 pinned model snapshot/Transformers source hash、deterministic portrait/
  landscape PNG、assistant-prefix/carrier/tool boundary、1-image、5-image 与 nested batch-2 均 fail closed；它只
  调用 `AutoProcessor.from_pretrained`，权重文件只做 SHA 读取，不实例化为 tensors，不 forward/generate；
- 新增 readiness validator：dependency 8 必须同时包含历史 GPU audit/独立 validation、真实 processor summary、
  14 个固定 source roles、clean pushed `main` 与 ancestor Git blobs；公开状态只可能是
  `SCREENING_ALLOWED + CONFIRM_LOCKED`，不能借 dummy `real_processor_*` 文件绕过；
- 新增 45-state runner：每 state 两次 deterministic full-history generation、两次独立 reference teacher
  forward、一次 summary teacher forward；reference FP32 log-probs 与 full-vocabulary KL 留在 GPU，只读取两个
  final distance scalars；global `max(1e-4,10*mean_repeat_kl)` 决定 memory sensitivity；
- parse failure 保留 raw generated text/metadata；OOM、prompt/shape/teacher/kernel error 是 fatal invalid，不能
  自动 fallback 或污染科学 gate；完整本地验证为 309 tests passed；当前最小本机环境另有 10 个
  optional-dependency skips（8 个 Pillow、2 个 PyTorch/GPU），所有既有
  contract/interface/executor/selection/OCR/baseline validators 通过；
- 本里程碑仍是 source-only：没有运行 GUI-Owl policy、没有生成 v2 policy/restoration output、没有创建新的
  HF artifact，dependency 8 仍 pending。

### 2026-07-15：第一次 real processor audit 按设计 fail closed

- clean pushed commit `6535e71d07b841e381f131f471d74467d6bef649` 在 Hyper00 容器
  `6263d8cd...c3ac21` 运行 processor-only audit；完整 17 GB model snapshot SHA 已先通过，随后在任何
  processor case、model weight materialization、forward/generate 或 policy/restoration output 前，以
  `actual processor min/max target or merge size drifted` 终止；
- 根因不是 scientific config 或 pixel target 改变：Transformers `5.6.0` 的 pinned
  `Qwen2VLImageProcessor` 将显式 `min_pixels=max_pixels=2621440` 归一化到
  `image_processor.size.shortest_edge/longest_edge`，并不保留 direct `min_pixels/max_pixels` attributes；
- 审计器现改为严格验证该真实 `SizeDict` 表示、四个 non-edge fields 为 `None`、merge size 2，并把 class/
  module/representation 全部写入 evidence；readiness validator 同步要求这些 exact fields。该失败是
  pre-output superseded attempt，dependency 8 仍 pending，可在新 pushed commit 上安全重跑 processor audit。

### 2026-07-15：real processor audit 正式通过

- clean pushed commit `82442da8193063b59e7b538d321406e26401d393` 在同一 Hyper00 container 完成 formal
  audit；summary SHA256 `7d5ac1bd13ba5def46dfb2ca419d59bb0da1ff970f186e9fd092d4006d8b43b8`，UTC
  bracket `17:38:30.280432Z--17:38:51.584295Z`；
- 完整 14-file / 17,545,907,171-byte model snapshot 与 pinned Transformers source 复核通过；唯一 pretrained
  loader 是 `AutoProcessor.from_pretrained`，model weights 未 materialize 为 tensors，forward/generate、
  policy output、restoration output 均为 false；
- 真实 processor 为 `Qwen3VLProcessor + Qwen2Tokenizer + Qwen2VLImageProcessor`；portrait/landscape grids
  为 `[1,152,68]` / `[1,68,152]`，对齐后每图 2,584 effective visual tokens；1-image/5-image sequence 为
  2,943/13,286，nested batch-2 sequence 为 2,946，无 padding；
- readiness validator 现对实际 tokenizer 3/8/15 token boundaries、全部 grid/sample/tensor inventory、PyTorch/
  Pillow runtime 与 class/module 做 exact equality，不再只检查“内部自洽”；processor 子项已闭合，但
  dependency 8 仍等待 execution config、完整 source inventory 与 readiness manifest。

### 2026-07-15：execution config 完成

- 新增 `code/configs/restoration_v2_execution_hyper00_v1.json`，SHA256
  `f2b6521ed8b1d65d4b6170c94c5c4cf8e46d135e352bdb52b21bf4e8f64173a5`；
- exact 8-dependency evidence inventory、14 个 source roles/current hashes、scientific config SHA、derived HF
  identity、GUI-Owl snapshot、Hyper00/H200/container runtime、GPU KL dtype/host-transfer contract 与
  microbatch=2/no-fallback 全部通过 `_validate_execution_config`；
- canonical policy 额外绑定 processor summary SHA 与真实 2,584-token geometry、1/5-image/nested batch-2
  sequence lengths 和 3/8/15 teacher boundaries，防止把 2,560 构造 target 误报成实测 accounting；
- config 仍声明 `CONFIRM_LOCKED` 且不授权 inference；dependency 8 的最终公开状态仍等待后续 clean pushed
  commit 中的 readiness manifest 与 Git ancestry/source-blob validation。

### 2026-07-15：readiness manifest 物化，等待 clean authorization

- 新增 `data/manifests/restoration_v2_readiness.json`，绑定 config SHA `f2b6521e...73a5`、implementation
  commit `a2aeb7f1930d40cb569c8e2adfc0dc39e950d131` 与同一 14-source inventory；
- manifest 结构、8/8 count、config/source hashes、canonical remote/main、pre-output declarations 与
  `SCREENING_ALLOWED + CONFIRM_LOCKED` 均已离线通过；confirm role 仍明确 forbidden；
- 当前工作树包含未提交 manifest，因此正式 validator 按设计尚不能通过 clean-Git gate。下一步先
  commit/push，再运行 CLI；在该 CLI 返回前仍不允许 policy import/output。

### 2026-07-15：dependency 8 正式闭合

- manifest commit `429c4584ba7c18eee0b96741b6c1514bd4d4d7ec` push 后，正式
  `scripts.validate_restoration_v2_readiness` 在 clean `HEAD == origin/main` 上返回
  `SCREENING_ALLOWED + CONFIRM_LOCKED`，8/8 dependencies passed；
- summary SHA256 `20b9e81050e56622cfe9bfe4dd1343f33513ae7f37fe25716d13b86ea8115964`；Git binding
  同时记录 implementation commit `a2aeb7f...d131` 与 current/origin commit `429c458...d7ec`；
- validator 未 import policy，policy/restoration output 均为 false。dependency 8 正式闭合，但授权范围仅为
  10 条 label-train + 5 条 development 的固定 45-state screening；`v2_confirm_primary` 继续 locked。

### 2026-07-15：screening preflight 触发 GPU-2 runtime re-anchor

- 10 秒 preflight 发现旧 config 锁定的 physical GPU 0 正被其他任务占用 112+ GiB 显存、86%--87%
  utilization；未共享该 GPU，未启动 screening；
- preflight 选择空闲 physical GPU 2。首次新容器因 `--privileged` 使 PyTorch 看见 8 卡，在任何审计/policy
  load 前删除；完成态 non-privileged 容器 `69f2b174...194df` 严格只见 `cuda:0`；
- 新 H200 UUID `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81` 上 formal GPU audit、独立 validator 与 real
  processor audit 全部通过，SHA256 分别为 `dce79769...9dac`、`ad53e186...2a6`、`69bb8ddb...d119`；
- 全程没有 policy/restoration output。旧 GPU-0 readiness 保留为历史 pass；planned GPU-2 screening 暂停，
  直到 canonical evidence、execution config 与 readiness manifest 重新绑定并正式授权。

### 2026-07-15：GPU-2 canonical transition 已锁定，等待 re-sign

- 将 re-anchor 的 GPU summary、independent validation、real processor summary 升级为 canonical evidence；
  SHA256 分别为 `dce79769...9dac`、`ad53e186...2a6`、`69bb8ddb...d119`，三份均与保留在
  `restoration_v2_runtime_reanchor/` 的原始副本 byte-identical；
- execution config 改绑新 container `69f2b174...194df`、GPU UUID
  `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81` 与新 evidence，14-source inventory 同步更新；完成态 config
  SHA256 为 `819cb973...91ca0`；
- production runner 新增 pre-import live runtime guard：Torch 与 `nvidia-smi` UUID 必须直接一致，并逐字段核对
  visible GPU count、driver、compute capability、SM count、Python、PyTorch/CUDA/cuDNN、Transformers；GPU 或
  software drift 在 artifact/model load 前 fail closed；
- readiness manifest 暂时设为 `SCREENING_LOCKED_RUNTIME_REANCHOR_PENDING_RESIGN` 且 implementation commit 为
  `null`，避免 GPU-0 历史 authorization 被误用于 GPU-2；focused GPU/readiness/runner tests 41/41、全量
  tests 314/314 通过，另有 10 个 optional-dependency skips；formal readiness CLI 按预期以 locked-state error
  非零退出；
- 本里程碑仍未 import policy runtime、未加载 model、未生成 policy/restoration output。下一 commit 只负责把
  manifest 绑定本 transition 的 clean pushed commit，不再修改 execution config 或 14-source files。

### 2026-07-15：GPU-2 readiness manifest 已 re-sign，等待 formal CLI

- transition commit `14faaa44cf1b2044b1f1bcb3c9dcfce36eb452aa` 已 push 到 canonical `main`；该 commit
  包含 exact GPU-2 config、14-source inventory、canonical evidence 与 live runtime guard；
- readiness manifest 恢复 `SCREENING_ALLOWED` schema，`implementation_git_commit` 精确绑定 `14faaa4...452aa`，
  config SHA 仍为 `819cb973...91ca0`，confirm 仍为 `CONFIRM_LOCKED`；
- 本步骤只修改 manifest、tests 与 docs，没有修改 execution config 或 14-source files，也没有 policy/model
  import/output。manifest commit/push 后必须在 clean `HEAD == origin/main` 运行 formal CLI；在此之前操作上仍
  locked。

### 2026-07-15：GPU-2 dependency 8 正式重新闭合

- re-sign commit `caa4f376026d13acd21db1f88b893bc92dd482b1` push 后，formal readiness CLI 在 clean
  `HEAD == origin/main` 返回 8/8、`SCREENING_ALLOWED + CONFIRM_LOCKED`；
- Git binding 记录 implementation commit `14faaa4...452aa` 与 validation/current/origin commit
  `caa4f37...82b1`，execution config SHA 仍为 `819cb973...91ca0`；
- GPU-2 canonical readiness summary SHA256 为 `36bd183f...aa02`；GPU-0 首次 summary 以
  `gpu0-summary.json`/`20b9e810...5964` 归档，不再授权当前 runtime；
- validator 再次明确 `policy_imported_by_validator=false`、policy/restoration output 均为 false。现在只允许
  固定 45-state development screening，confirm 继续 locked。

### 2026-07-15：第一次 fixed 45-state substrate screening 合法判负

- 从 clean pushed `main@a0001cbc4d4c3e2584ae3bbcff217ce664e5e064` 在 Hyper00 non-privileged container
  `69f2b174...194df`、单张 physical GPU 2（H200，`GPU-e19275bf...66b81`）运行；live runtime guard、derived
  artifact validation 与全部 90 个 processor-only prompt shapes 先通过，sequence length 范围
  3,245--14,499，未超过固定 context；
- 45 个固定 state（30 label-train + 15 development）各生成一次，共 1,943 tokens；45/45 都在
  `reference_generation_1` 触发 `GUIOwlV2GenerationParseError`，因此正式 aggregate 为 strict parse 0/45、
  `NO_GO_V2_SUBSTRATE`。没有 OOM、resume、state retry、top-up、sample mutation 或 confirm access；
- parse 之后未进入任何 teacher forward，故 finite-logit coverage、repeat agreement 与 memory-sensitive count
  的 0 值都是未测量下游量，不是 logits/non-sensitivity 证据；KL measurement 与 restoration label 均为 0；
- raw run 的 45 native outputs、45 state records、45 attempt markers、run manifest 与 log 已打成单个
  deterministic tar，上传 private HF
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@restoration-v2-substrate-screening-v1.0.0`
  (`c073e143b935a79befd8ab1fd7123796792efad8`) 并 fresh re-download 验 hash；Git 只保存轻量 summary 与
  artifact manifest；
- post-hoc format inventory 发现 45/45 都有 exact `Action:` + `<tool_call>` prefix，43/45 条 output 的
  单动作 first balanced JSON 可以 canonicalize，但这只是宽松 diagnostic；在拒绝多 action、截断、第二 JSON 与额外 observation 后，保守的 envelope-only
  normalization 上界只有 40/45，低于 frozen 0.99 gate（45 states 必须 45/45）。它不 retroactively 改写
  本次 0/45；confirm 继续 locked。

### 2026-07-15：immutable parser compatibility replay source 已实现

- 新增 versioned、format-only compatibility parser；原 strict parser 总是先运行，仅允许历史 trace 中明确定义的
  `Action:` envelope、四种 wrapper 与五种安全 suffix。它不补 `}`、不取多动作的第一个、不接受第二 JSON 或
  observation，并继续复用 frozen action canonicalizer 与 AndroidWorld bridge；
- 新增 immutable tar replay 与 clean-main CLI：绑定 archive SHA、exact 96-member order、run contract、source
  run commit、HF immutable revision、原 aggregate 和 45 个 no-retry failure records；archive 从已哈希 bytes
  解析，避免 path reopen TOCTOU；输出只保留 raw-output SHA，不复制 raw text；
- 独立 synthetic 45-state archive 覆盖预期 `0 strict / 43 single-action canonical-first / 40 conservative`
  计数、gate-pass outcome derivation，以及 SHA、member-order、official-result drift、dirty/remote Git、committed
  blob、exclusive write 和 non-finite JSON 的 fail-closed 路径；
- pre-registered golden manifest 进一步固定 45-record canonical classification SHA 和五个 rejection 的
  state/output hashes。40 个 accepted output 中仅 25 个是 clean EOF；其余 15 个分别丢弃重复 opener 8、extra
  brace 4、extra brace + opener 3，model-emitted canonical closer 为 0，因而必须称为 suffix recovery 而不是
  native well-formed parse；
- 本里程碑只冻结 auditor source，不把本地直接调用得到的计数当作正式结果。必须先 commit/push，再从 clean
  `main` 对 HF immutable archive 运行 formal CLI，随后才允许把 replay summary 写入 Git；全量 tests 为
  327 passed、10 optional-dependency skips。

### 2026-07-15：immutable parser compatibility replay 正式判负

- source commit `fc3adf13d48bb016014f7efa62bd27c8a4d12f49` push 后，从 clean `main` 运行 formal
  CLI；审计前后均确认 `HEAD == origin/main == GitHub remote main`，实际 import 的四个 source 与 committed
  blobs 完全一致；
- 已提交 golden contract、HF immutable revision、archive SHA、run contract/source commit、96-member tar
  inventory、原 0/45 aggregate、45-record classification SHA 与五个 rejection identities 全部通过；
- 正式结果为 strict 0/45、single-action canonical-first 43/45、conservative recovery 40/45；后者低于
  frozen 0.99 gate 对应的 required 45/45，因此输出 `NO_GO_ADAPTER_ONLY`；
- 40 个 accepted 中 clean EOF 25、suffix recovery 15，model-emitted canonical closer 0；该证据排除
  “只换 parser 就能继续 v2”的路线，但没有否定 restoration hypothesis；
- replay 没有 raw-text Git exposure、model import/load、forward/generation、retry、top-up 或 confirm access。
  完整 formal result SHA256 为 `78272ee5...a6e0`，见
  `data/results/restoration_v2_parser_compatibility/`；结果回归加入 committed-result/golden binding 后，全量
  tests 为 328 passed、10 optional-dependency skips。

### 2026-07-15：Restoration v2.1 official-tool interface 与 pilot contract 已冻结

- v2 的 `NO_GO_V2_SUBSTRATE` 和 parser replay 的 `NO_GO_ADAPTER_ONLY` 保持不变；v2.1 明确是新的
  policy-interface reset，不是对旧 raw output 的 relabel 或 compatibility adapter；
- 新增独立 official `tools=` schema、two-message prompt、single-line whole-output parser 和 AndroidWorld
  bridge round-trip。assistant 不再输出 `Action:` carrier；alias、CRLF、多行 JSON、duplicate key、非 finite
  number、非 NFKC text、truncated JSON、第二 call/JSON、observation 与任何前后文本均 fail closed；
- 新 runtime 固定 pinned chat-template file/text SHA、assistant/opener/closer/EOS/pad token IDs；generation
  使用 closer token `151658` 作为唯一 EOS并 suppression 标准 EOS，greedy batch-1、256-token cap、
  `skip_special_tokens=false`。没有 model closer 就按 truncation 失败，不做 host completion；
- teacher distance span 从 opener 到 model closer inclusive，且 reference/candidate teacher logits 对 generation
  forbidden EOS columns 应用相同的 GPU-resident、BF16 finite-min mask，避免 generation 与 KL 的策略支持
  不一致；
- machine-readable contract
  `code/configs/causalcache_restoration_v2_1_pilot.json` SHA256 为
  `9d51a2ed5d6cc382f297c1b8af3100d784090f72800d637136b88982763fdbf7`。它固定 processor-only
  45-state × 2-fidelity 共 90 prompts，随后只对 15 个 `v2_development` states 各生成一次；
- pilot 通过条件是 15/15 exact parse、15/15 model-emitted closer、15/15 AndroidWorld bridge，且所有
  truncation/extra-output/retry/top-up 计数为 0。pilot 明确禁止 teacher、KL、restoration、expert、label-train
  policy output，也不会向 decoder 暴露任何 confirm state/prompt/image；pass 也只授权 unchanged-interface
  full-45 substrate，不授权 confirm generation；
- 本里程碑是 source-only：validator 返回 processor preflight pending 且不授权 generation；尚未加载 v2.1
  model、forward、generate 或产生 policy/restoration output。下一步必须先从本 source commit push 后运行
  Hyper00 processor-only preflight。

### 2026-07-15：Restoration v2.1 formal audit/pilot runner source 就绪

- 90-prompt real `AutoProcessor` audit 会从 45 个 screening states 构造 full-history/summary-only 两种输入，
  检查 official tools 注入、tensor/shape、image-count distribution、token boundary 与 `seq_len + 256 <= 32768`；
  独立 reducer 从逐 prompt records 重算结论，不信任 runner 自报 aggregate；正式 evidence 还记录完整 argv、
  Hyper00/container/image、Python/platform/package、start/end/duration，并明确 CPU-only、无 GPU operation、seed
  not applicable；
- official Jinja/tojson correction 已在任何新 output 前冻结；teacher targets 与 official assistant
  `tool_calls` JSON 逐字节一致（canonical insertion order、原始 Unicode UTF-8），解码后的 text 仍做 strict NFKC；
  interface source SHA256 为 `90cbefc851bed105de6ea0c8f719aae6313a479ca4589e4160fc4ec3e8de3964`；
- fixed-15 runner 只按冻结顺序处理 `v2_development` states；每个 state 在 generation 前写 attempt marker，
  只允许一次 full-history generation，parse failure 是科学 `NO_GO`，OOM/runtime/contract failure 是 invalid，
  interrupted attempt 不得 hidden retry；formal attempt ID 与 persistent output path 已在 pre-output contract
  固定，alternate output path 不能绕过 marker；
- formal runner 在任何 policy runtime import/model construction 前先 durable claim global ledger、root 与
  run manifest。constructor/OOM 写 0/0 canonical `INVALID`；ledger-only、root-partial 或 marker-only crash 在
  `--resume` 时只能封存为 `INVALID`，不能继续 generation。成功构造后 exclusive 写完整 35-key
  `runtime_identity.json`，并在每个 attempt/generation 前复核；
- raw evidence packager 已在 output 前冻结：normalized deterministic USTAR 同时包含 canonical output 与 sibling
  ledger；PASS/NO_GO 重算 exact fixed-15 gate，INVALID 复核 strict partial inventory，并从 source commit 重放
  config、selection、policy/source blobs 与 processor artifact binding。raw archive 进入 private HF，Git 只保存
  immutable revision/hash/compact reduction；
- confirm 语义已精确化：loader 必须验证并 hash 包含 confirm bytes 的完整 artifact，但 decoder/processor
  暴露的 confirm state、prompt、image 都为 0；`confirm_processor_prompt_count=0`，confirm generation
  count=0；
- 该里程碑仍是 source-only。正式 Hyper00 processor audit 与 fixed-15 pilot 均未运行，不报告 parse、closer、
  bridge 或 generation 结果，也不运行 teacher/KL/restoration；旧 `NO_GO_V2_SUBSTRATE` 与
  `NO_GO_ADAPTER_ONLY` 不变；
- 与路线评审的映射未改变：stable self-behavior reference、post-state-only intervention 和 strong
  low-fidelity summary 均已在 v2 冻结，v2.1 只关闭 official-tool interface 风险；
- 输出前验证为 v2.1 63/63、全仓 391 passed（10 个 optional-dependency skips）；全部既有 contract/interface/
  executor/selection/OCR/baseline validators 与 AAAI LaTeX build 通过。processor audit 与 pilot 尚未运行。

### 2026-07-15：v2.1 processor preflight 首次启动在 preprocessing 前 fail closed

- source `main@f689025929e608dcdd10d35b910cef0b2f2424e1` 已 clean push，并在 Hyper00/container
  `69f2b174...94df` 完成 host/disk/path preflight；
- 第一次 CPU-only CLI 把 `--derived-artifact-root` 错传为 repo 内层 `derived/restoration-v2-v1`。artifact loader
  在调用 `AutoProcessor` 前发现缺少根层 `.gitattributes`/`README.md`，因此 fail closed；
- `formal-result.json` 不存在，model weights、policy forward/generation、restoration 与 confirm processor input 均未
  发生。这不是 scientific result，也不消耗 fixed-15 policy attempt；
- 正确 root 是 `/data/tmp/causalcache-restoration-v2-derived-1a01f23-hf-redownload`。先修正并 push 正式 argv，
  再从新的 clean `main` 重启 processor-only audit。

### 2026-07-15：v2.1 90-prompt processor preflight 正式通过

- 修正文档 argv 后，从 clean pushed `main@d0205afd789cda602fbf8964505d9de9a1b1fe53` 在 Hyper00/container
  `69f2b174...94df` 运行真实 `AutoProcessor`；45 states × full-history/summary-only 共 90 prompts 全部完成，
  official tools 注入 90/90，image-count distribution 为 1-image 45、3/4/5-image 各 15；
- input token 长度 3,759--15,013，context overflow 为 0，最长输入加固定 256 generation budget 后为 15,269，
  低于 32,768。processor class、prompt、shape 与 teacher-golden records 均有独立 SHA256 reduction；
- 运行耗时 59.589 秒，明确记录 `gpu_operations_executed=false`、policy model 未实例化、weights 未
  materialize 为 tensors、forward/generate 未执行、confirm processor prompt 为 0。因此本步骤没有产生任何
  v2.1 policy output，也没有运行 teacher、KL、
  restoration 或访问 confirm input；
- 7,609,803-byte raw JSON（SHA256 `5349ffc6b91bf93ed26d25104fe6c907f31ccc497007a5c0ae6ed5ddf5c84991`）
  已上传 private HF dataset `gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile`，tag
  `v2.1-processor-preflight-v1` 固定到 immutable revision
  `85576161b7cb8bbae14e46a482c42b5be5bf1d7e`；fresh immutable download 的 byte hash/size 与 API tag binding
  均已复核；
- Git 只保存 `data/results/restoration_v2_1_processor_preflight/` 的轻量 manifest/说明。该 PASS 只关闭
  processor gate；fixed-15 pilot 尚未运行，parse/closer/bridge 结果仍未知。

### 2026-07-15：v2.1 fixed-15 interface pilot 正式通过

- processor artifact 的 reuse-mode validator 先从 fresh immutable HF download 在 clean
  `main@70724bdb82484cd5645d174765724bd1f83b5ace` 重新得到 90 prompts PASS；随后 GPU preflight 确认
  Hyper00 全部 8 张 H200 无 compute process，canonical root/ledger 不存在；
- 唯一 formal attempt 使用 container-local `cuda:0`（映射 host physical GPU 2，UUID
  `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81`）、BF16、greedy batch-1、256-token cap；15 个冻结
  `v2_development` states 各 generation 一次，84.279 秒完成；
- 正式 outcome 为 `PASS_V2_1_INTERFACE_PILOT`：strict whole-output parse 15/15、model-emitted closer
  15/15、AndroidWorld bridge 15/15；generation 15，retry/top-up/truncation/extra output 均为 0；
- teacher forward、KL measurement、restoration label、expert action read、label-train/confirm generation 均为 0，
  没有 sample mutation。该 PASS 只证明 native official-tool interface 可用于下一阶段，不证明 stable
  reference、memory sensitivity 或 CausalCache 方法有效；
- utilization monitor 在 model load/processor 与 batch-1 generation 交错时观测到低于 90% 的窗口；已立即核查
  为单卡、固定 batch-1 protocol，不能再减少 GPU 数且不得为 pilot 改 batching。run 在 84.279 秒内完成；这不是
  throughput efficiency 结果，full-45 source 需继续记录 latency/utilization；
- canonical output 与 sibling ledger 已封装为 34-file deterministic USTAR，133,120 bytes，SHA256
  `f71d5fd575dde48ae8b3e50a19dd2fbecfa02d7d5ae6087f909a47dfd7032064`。private HF tag
  `v2.1-interface-pilot-v1` 解析到 immutable revision
  `bdff8ca71f150afd80d6291b4ecec76cbf9e7432`；fresh immutable download 的 byte hash/size 已复核；
- Git 只保存 `data/results/restoration_v2_1_interface_pilot/` 的 34-file inventory/binding 与说明。manifest
  push 后，artifact validator 已在 clean `main@64581118eb3375a03234ef49e5ce1f7bc41d354e` 从 fresh archive
  返回 `VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT`、`archive_hash_verified=true` 与同一 PASS；
  committed binding 已闭合。

### 2026-07-15：v2.1 full-45 substrate source contract 已冻结

- 新增独立 child protocol `causalcache_restoration_v2_1_full_45_substrate`；machine-readable contract
  `code/configs/causalcache_restoration_v2_1_full_45.json` 的 SHA256 为
  `0924dd66fab9440bed66585765e9b1f5ab6fb80fbdf65a1492efba7ae81e116b`，并绑定 fixed-15 PASS、
  90-prompt processor PASS、parent v2 scientific config 与 exact 30+15 state projection；
- exact projection SHA256 为 `65e7085f01bc26e425bab7f5148c6729bc8c6d0a62802ebc5e8dfeaed6439249`。
  每 state 固定两次 full-history generation；仅当两次均 strict parse、model-emitted closer、bridge 成功且
  canonical action 一致时，才执行两次 reference 和一次 summary-only teacher forward，再计算 repeat 与
  summary-reference GPU KL；
- 全 attempt 的 planned/maximum schedule 同为 generation 90、teacher 135、KL 90。parse/mismatch/non-finite
  仍留在 45 分母；runtime/contract/OOM/bridge invariant 进入 `INVALID`。gate 要求 45/45 parse、45/45 finite、
  45/45 repeat agreement，并有至少 8 个 states 满足 summary KL 大于
  `max(1e-4, 10 * mean(repeat KL))`；
- canonical attempt/root/sibling ledger/archive 与 private HF destination 均和 fixed-15 完全独立。resume 只跳过
  已有 terminal prefix；root 外 ledger 以 atomic replace + file/directory fsync 维护 attempted/completed high-water
  journal。marker 无 terminal、删除 marker+terminal pair 或删除整个目录都会使 attempt `INVALID`，不能重生成或
  换目录重跑；teacher suppression 也精确绑定 BF16 minimum，不接受任意负 finite 近似；
- source inventory 覆盖 full-45 source 及其递归本地 import closure；正式 runner 在 policy runtime import 前
  验证 clean pushed `main`、Git blobs、derived artifact，并从 private HF immutable revisions 的 fresh download
  复核 fixed-15 archive 与 processor JSON；
- raw evidence manager 将 canonical root 与 sibling ledger 打成 deterministic USTAR，并从逐 state records
  重算 denominator、operation counts、gate 与 outcome。正式 raw archive 只进入 private HF，Git 只保存轻量
  manifest 与 immutable binding；archive reader 会按同一 USTAR serializer 重建并要求原始 bytes 完全一致，
  因而拒绝 GNU tar、trailing bytes 与非 canonical header；
- 本里程碑是 source-only，validator 明确返回 policy/GPU 未授权；尚未运行 full-45 generation、teacher、KL、
  restoration 或 confirm。full-45 focused tests 33/33、全仓 424 tests（10 skips）、全部既有 validators 与
  AAAI LaTeX build 已通过。source commit/push 与 clean descendant validation 完成后，才允许唯一 formal
  attempt。

### 2026-07-15：v2.1 full-45 substrate 正式判为 NO-GO

- 唯一 `restoration-v2-1-full-45-substrate-v1` attempt 从 clean pushed source
  `7a5b6d5710fe4d054936b5aa474648149f725edb` 在 Hyper00 单张 H200 上完成；运行时间为
  `2026-07-15T22:59:20.590509Z`--`23:04:48.502942Z`，327.912 秒；
- 90/90 generations 均 strict parse、有 model-emitted closer 并通过 AndroidWorld bridge；但只有 32/45
  states 的两次 exact canonical action 完全一致，低于冻结的 45/45 gate。13 个 mismatch 全部保持 action
  type：12 个 `click→click`、1 个 `swipe→swipe`；label-train/development 分布为 10/3，step 4/5/6
  分布为 3/5/5；
- 32 个 agreement states 产生 96 次 teacher forward 与 64 次 GPU KL。repeat KL 全为 0，epsilon 固定为
  `1e-4`；32/32 summary-reference KL 超过 epsilon，min/median/mean/max 为
  0.000783/0.031472/0.044022/0.196197，因此 interface 和 history sensitivity signal 都存在；
- frozen reducer 仍必须输出 `NO_GO_V2_1_FULL_45_SUBSTRATE`，因为 repeat agreement 与相应 finite-logit
  coverage 均只有 32/45。retry、top-up、sample mutation、expert read、restoration coalition、baseline、gate
  training 和 confirm work 全为 0；不能把 coordinate jitter 事后 relabel 成 PASS；
- utilization monitor 观测到 model load/CPU processor 与 batch-1 generation/teacher 交错导致平均窗口低于
  90%，但 active GPU 峰值为 99--100%，仅使用一张 GPU，且 frozen batch-1 protocol 不允许为本次结果修改
  batching；
- canonical 94-file deterministic USTAR 为 962,560 bytes，SHA256
  `8cd53d6e56d5ad509da2af91d73d9e83b4db989ffc26aa18e1bca84e4c4f4fa4`。private HF tag
  `v2.1-full-45-substrate-v1` 固定到 immutable revision
  `814506ef1450838d4bc6ed3d89fe53e0773d92fb`；fresh immutable download 已逐 byte 复核，Git compact result
  位于 `data/results/restoration_v2_1_full_45_substrate/`；
- manifest push 后，Hyper00 clean `main@554c51e417d702a6bc759b1f592979a4c38c5283` 从 fresh immutable archive
  运行 committed-binding validator，返回 `VALID_RESTORATION_V2_1_FULL_45_ARTIFACT`、
  `archive_hash_verified=true`、94 files 与 tree SHA256 `6c5e515a...19281`。source commit、当前 descendant、
  archive、manifest、reducer outcome 与 HF revision 的完整 SoT chain 已闭合。

### 2026-07-15：Interaction-aware gate ablation 设计记录

- 合作方指出 synthetic 已观测到 non-additive event interaction，而当前独立 event score 会把
  coalition-conditioned marginal 压缩成 context-independent average value；该问题被确认为真实的
  subset-objective information bottleneck，不是 restoration attribution 本身假设 event 独立，也不是当前
  independent student 违反已经冻结的 averaged-$G$ teacher contract；
- 新增 `ablations/interaction_aware_gate.md`，记录 set-conditioned iterative gate：每选入一个 event 后，基于
  更新后的 coalition embedding 与剩余 budget 给候选重新打分；同时保留 independent gate 与 exact subset
  oracle 作为 ablation 两端；
- 文档明确三项限制：当前 near-budget label distribution 不覆盖 iterative gate 的全部前缀 cardinalities；标准
  permutation edge 不自动提供同一 coalition 下的 conditional-ranking pairs；pure complementarity 也可能让
  greedy 从空集停止。因此 set conditioning 只能声称缩小 interaction-induced oracle gap，不能声称解决任意
  interaction；
- 该里程碑仅为 Git source-only proposal，没有修改 frozen v2/v2.1 contract、代码或论文，没有运行 policy/GPU，
  没有读取 confirm，也没有新增 HF artifact。真实 ablation 仍被 v2.1 substrate NO-GO 阻断。

### 2026-07-15：Subset-search v1 source/config/runner 冻结

- 合作方进一步指出“conditional marginal 包含 interaction”并不保证最终 greedy subset 全局最优；该问题被拆成
  value-source 与 search-algorithm 两个轴，避免把既有 phase-0 的 0.859 误称为 greedy gap。前者是 exact
  averaged marginal 经 exact additive knapsack 后相对真实 subset objective 的 projection gap；只有 true
  conditional greedy 相对 exact subset 的差才是 search regret；
- 新增 deterministic black-box search module：exact subset、raw/density conditional greedy、从 raw-greedy
  seed 出发的 2x2 bounded exchange、允许 non-positive prefix 的 true-utility beam-$2/4/8$。所有方法允许空集，
  统一 utility/cost/cardinality/lexicographic tie-break，并记录全部 unique set evaluations、evaluated candidates、
  sequential rounds；exchange 成本包含 seed 与未采用 neighbors；
- 冻结 `code/configs/subset_search_ablation_v1.json`，SHA256
  `161518c3e951829528b63b9146878236d918d4d620b4a6acbecc53f7c4db921f`。输入包含五类 controlled
  functions、既有 phase-0 mixed synthetic、$n=8/12/16/24,b=3$ scale sweep，以及旧 v1 两个
  selection-biased development states 的完整 singleton/pair table；
- 旧 real table replay 明确使用 `distance_mean`、每 event 476 visual tokens、budget 512/1024；它不运行新
  policy forward，只验证 exact/greedy/exchange/beam 的 cached-table consistency。state sensitivity threshold
  `1e-4` 仅单列为 abstention sensitivity，canonical search stopping threshold 固定为 0，不能混成 interaction
  search gap；
- true-$U$ greedy/exchange/beam 在真实部署都需要 frozen-policy rerun，因此只称 offline optimizer diagnostic；
  learned marginal head 没有 direct set-utility/path contract 时不能直接用于 beam/removal。exact 只保证冻结
  single-step restoration utility 最优，不代表 terminal success 最优；
- focused tests 16/16、全仓 440 tests（10 个 optional-dependency skips）、既有 contract/interface/executor/
  selection/OCR/baseline validators 与 AAAI LaTeX build 全部通过。该里程碑只冻结 source/config/runner，尚未
  生成 formal result，没有运行 GPU、policy、gate、closed-loop 或 confirm；v2.1 继续保持
  `NO_GO_V2_1_FULL_45_SUBSTRATE`。

### 2026-07-15：Subset-search v1 首次 CPU attempt 在 artifact validation 前 fail closed

- source milestone `d5cd389f6a7b2720cc1daaa3373cefa52480cf7b` push 后，从 clean canonical main 运行
  frozen CLI；14 个 scenarios 在 CPU 上 51.668 秒完成，外部/untouched data、policy、GPU、confirm access 与
  learned-gate/closed-loop evaluation 均为 0；
- phase-0 exact averaged-marginal knapsack / exact subset 为 0.858594，而 true conditional greedy 为 1.0，
  正式确认前者应命名 objective-projection gap，不是 greedy search gap；
- controlled complement trap 中 raw greedy/beam-2 ratio 0.5，2x2 exchange/beam-4/8 为 1.0；mixed
  non-monotone 中 raw greedy 0.666667、exchange/beam-4/8 为 1.0；variable-cost 中 raw 0.833333、density
  1.0；
- $n=8/12/16/24,b=3$ exact queries 为 93/299/697/2325；raw greedy queries 22/34/46/70，ratio
  1.0/0.95098/0.95098/0.970588；beam-4 queries 51/88/125/197，在该 deterministic generator 上均 exact；
  2x2 exchange 虽均 exact，但 $n=24$ 使用 1,185 queries，不能简单视为便宜修复；
- 旧 v1 selection-biased cached table 的四个 state/budget cases 中，true greedy、exchange、beam 与 exact
  coalition 全部相同，greedy ratio 均为 1.0。step 8 / 1024 有 2 个 positive、19 个 negative pair
  interactions 但没有 greedy trap；真实 table 只验证 implementation consistency，不支持 stronger search 的
  paper-level收益；
- 单列 threshold sensitivity 发现把旧 `1e-4` state floor 当 search threshold 会令 step 4 / 1024 从 exact
  `[1,3]` 提前停在 `[3]`，ratio 0.944896；该差距明确归入 abstention，不归入 search regret；
- runner 输出后、任何 result commit 前执行独立 scientific replay；所有数值相同，但 in-memory trace coalition
  是 tuple，JSON load 后为 list，严格 payload equality 因类型不一致触发 `AssertionError`。这是 artifact
  serialization contract bug，不是搜索数值不稳定；按 fail-closed 边界，首次 result 目录已删除，未 commit/push；
- source 已在 `_trace` boundary 显式输出 list，并加入完整 JSON round-trip equality regression。下一步先
  commit/push 该 fix，再从新的 clean source canonical rerun；不能沿用首次 output 或放宽 validator。上述数值在
  canonical rerun 前只算 provisional diagnostics。

### 2026-07-15：Subset-search v1 canonical CPU rerun 与 pre-commit replay 通过

- JSON serialization fix 已作为 `main@7569ce2ac1e63f565be4e0d4dcc9626285aa355c` push；全仓 440 tests
  （10 skips）通过后，从该 clean canonical source 重新运行 frozen CLI；
- canonical run 于 `2026-07-15T23:58:44.535445Z`--`23:59:38.142234Z` 在 CPU 上完成，wall time
  53.606 秒，14 个 scenarios 的 scientific payload SHA256 为
  `26846d509d421dcb49f2d1554598893a65829c6a25610ef399ce69b383274edf`；policy/GPU operation 均为 0，
  未访问外部/untouched data 或 confirm；
- 写出后、任何 result commit 前再次从 config/input 构造完整 payload；`json.loads(json.dumps(payload)) ==
  payload`、与 summary scientific payload 的完整 equality 和 canonical SHA256 三项全部通过，输出
  `PRECOMMIT_SCIENTIFIC_REPLAY_MATCH`；
- canonical result 逐项复现首次 provisional 数值：phase-0 projection/true-greedy ratio 0.858594/1.0；
  complementary trap raw/beam-2 0.5、exchange/beam-4 1.0；mixed raw 0.666667、exchange/beam-4 1.0；
  variable-cost raw/density 0.833333/1.0；
- scale sweep exact queries 93/299/697/2325；raw greedy queries 22/34/46/70、ratio
  1.0/0.95098/0.95098/0.970588；beam-4 queries 51/88/125/197 且在本 generator 全部 exact；exchange
  到 $n=24$ 需要 1,185 queries；
- 旧 cached real table 的四个 cases 仍全部 greedy=exact；step 8 / 1024 的 2 positive + 19 negative pair
  interactions 没有形成 greedy trap。正式结论仍是保留 set-conditioned greedy 主线，stronger search 只作
  offline diagnostic；
- 轻量 summary/report 位于 `data/results/subset_search_ablation_v1/`。下一步 commit/push result，再从 clean
  descendant main 运行正式 committed validator。

### 2026-07-15：Subset-search v1 committed result validation 闭合

- canonical summary/report 已作为 `main@45bcf7e3ab984e2f4c977af46a27312b1ed8ba0a` commit/push；该
  result commit 只包含轻量 Git artifacts，没有新增 reusable dataset/model，因此无需创建 HF repo；
- 从 clean `main@45bcf7e` 调用独立 validator。它重新构造 14-scenario 完整 scientific payload，并同时校验
  source commit blobs、当前 source/config/input hashes、JSON round-trip identity 与 committed summary blob；
- validator 返回 `VALID_SUBSET_SEARCH_ABLATION_V1`，source commit 为
  `7569ce2ac1e63f565be4e0d4dcc9626285aa355c`，scientific payload SHA256 为
  `26846d509d421dcb49f2d1554598893a65829c6a25610ef399ce69b383274edf`；
- formal result 因而从 source、运行、pre-commit replay、result commit 到 clean-descendant committed replay
  全链路闭合。v2.1 outcome 仍为 `NO_GO_V2_1_FULL_45_SUBSTRATE`，没有 policy/GPU/confirm operation；
- 方法决策保持：set-conditioned greedy 是线上主线，exact 是小规模离线 ceiling，beam/local 只作可替换的
  true-utility offline diagnostics。旧 real table 中 greedy/exact 比为 1，尚无证据为线上 stronger search
  增加复杂度。

### 2026-07-15：Spatial reference audit v1 source-only 合约冻结中

- subset-search 支线保持闭合，主线回到 reference substrate；本 audit 只使用 v2.1 已暴露的 13 个
  `CANONICAL_ACTION_MISMATCH`（10 label-train、3 development；12 click、1 swipe），v2.1 的
  `NO_GO_V2_1_FULL_45_SUBSTRATE` 不重算、不改写；
- parent raw archive、13-state canonical projection、actual action/token hashes、derived artifact、selection、OCR、
  model snapshot、source inventory 和 append-only child exposure ledger 均进入 fail-closed config；旧
  `restoration_v2_exposure.json@bc122482...` 保持 byte-immutable，confirm 20 仍为 policy-output untouched；
- 严格 CUDA deterministic GEMM 需要项目规则禁止的 `CUBLAS_WORKSPACE_CONFIG`，因此 profile 诚实冻结为 BF16
  auto、BF16 eager-control 与 4-state FP32 eager-control，不声称数学 deterministic。eager-control 固定 seed、
  eager attention、cuDNN deterministic/benchmark off、TF32 off 和最高 float32 matmul precision；
- formal runtime 进一步在 durable attempt claim 前 exact 锁定 image digest
  `sha256:6a8f60af...d349acfa`、Python `3.12.3`、PyTorch `2.11.0+cu130`、CUDA `13.0`、cuDNN
  `91900`、Transformers `5.6.0`、driver `570.172.08`，并要求固定 behavior-changing environment-variable
  名单全部 absent；auto observed attention 必须 non-eager，eager profiles 的 non-null observation 必须全为 eager；
- operation ceiling 仍为 generation 60、teacher forward 120。每 state 的 teacher diagnostic 改为 2 次相同
  shape shared-prefix forward，从同一 vector 比较两个竞争 token；另对两条父 action 各做 1 次 full-branch
  diagnostic。margin 明确不是旧 generation-time score，也不进入 pass/fail；
- canonical root 与 root 外 sibling ledger 已冻结；profile/state 都必须在 forward 前 durable claim，alternate
  output、删除、retry、top-up 和跳过 profile 顺序全部禁止。raw root 计划上传 private HF dataset
  `gavinlaw/causalcache-spatial-reference-audit-mobile@spatial-reference-audit-v1`，当前 revision 仍 pending；
- raw packager 已在任何 audit forward 前冻结：它把 canonical root 与 sibling ledger 封装为单个按 path 排序、
  metadata 归一化的 deterministic USTAR，拒绝 symlink/non-regular member，exclusive-create 后重新读回并要求
  canonical-byte identity；本地 archive 固定为
  `/data/experiments/causalcache/spatial-reference-audit-v1.tar`，当前尚未生成；
- 未来 exact restoration 的 subset 计数已纠正：step-4/5/6 为 4/7/11，label-train 220 rows、development
  110 rows，总计 330；它只是新 substrate gate 通过后的 prospective table，不是当前 policy-forward 数，也未运行；
- 当前仍是 source-only：没有新 audit GPU profile、semantic result、restoration、gate training 或 confirm policy
  input/output。完成 source tests、commit/push 和 clean descendant validation 后，才允许 Hyper01 单张 H200
  preflight 与唯一 audit attempt。
- 聚合判定修正为三分支：eager unstable 才直接进入 semantic；eager 13/13 且 auto 非 13/13 才能称
  eager-specific recovery；若两个 BF16 profile 都是 13/13，只报告本次 numerical audit inconclusive，不归因于
  eager。FP32 与 margin 始终不决定分支。

### 2026-07-15：Spatial reference audit v1 formal raw 与 validation-repair 前置

- 从 clean pushed `main@c093bd8f92ab97427acb427bd2d66fb6b20b556a` 在 Hyper01 non-privileged
  container `7ca0845bbaa5...ff4b`、单张 `NVIDIA H200` 上完成唯一 attempt；三个 profile 与 sibling ledger
  都已 durable terminal，未发生 retry、resume、alternate path 或 output deletion；
- raw metrics 为 BF16 auto 7/13 exact generated-token stable、BF16 eager-control 13/13、FP32 eager-control
  4/4。operation counts 精确为 generation 60、teacher forward 120，confirm/restoration/gate 均为 0；FP32
  仍只作描述性 probe；
- 原 formal wrapper 在三个 profile 之后、summary 写入之前由 independent validator fail closed，exception 为
  `ValueError: generation per-image effective visual token count drifted`，exit code 为 1；raw root、ledger、formal
  log/exit 与三个 terminal 的 pre-repair SHA 已冻结，不允许重跑 profile；
- 只读诊断确认旧 validator 有两个纯离线 contract bug：它把 target `2560` 错当成所有 realized grid 的固定
  token 数，而实际合法 realized set 为 2516/2560/2584；它还把 teacher aligned inventory 错写为
  `attention_mask + input_ids`，frozen runtime 的实际定义是 `attention_mask + mm_token_type_ids`；
- 180/180 shape nodes 的 grid accounting 正确；60/60 generation 的 grid、effective visual tokens、text tokens 与
  prompt tokens 和已验证 v2.1 parent raw 的同 state witness exact。仅在内存中窄修上述两项、且不传
  `--summary-output` 的 dry-run 已让旧 validator 其余全链通过，没有第三个 blocker，candidate decision 为
  `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；
- 新增 source-only validation-repair child，旧 33-file audit source/config/validator/exposure ledger 全部
  byte-immutable。repair 先绑定 pre-repair tree、ledger、terminal、formal log/exit 和 parent archive，再用 dynamic
  grid accounting + parent-shape witness 运行其余旧 validator；全部读取验证完成后才 exclusive-create 一个
  provenance-complete `summary.json`，不加载 processor/model，不调用 GPU/policy，也不修改已有 raw；
- 完整说明见 `docs/spatial_reference_audit_v1_validation_repair.md`。repair source commit/push、远端 clean-main
  execution、旧 deterministic USTAR、HF immutable fresh-download 与 Git result 尚未闭合，因此 candidate decision
  暂不提升为正式 artifact verdict。

### 2026-07-15：Spatial reference audit v1 artifact 闭合

- validation-repair source 在 clean pushed `main@a2528d7e95e73c25639568650f63abce58e4e491` 冻结，full
  CPU suite 为 482 passed / 11 optional-dependency skips；唯一 Hyper01 offline repair exit 0，只新增 canonical
  `summary.json`，未加载 processor/model，也没有新增 generation、teacher、confirm、restoration 或 gate operation；
- repaired validator 复用旧 validator 全链，并把旧 config + 33-file source inventory 逐 blob 绑定回 raw commit
  `c093bd8f...`。60/60 generation shape、120/120 teacher shape 与 parent witness 通过；observed per-image effective
  visual tokens 为 2516/2560/2584，但只作 evidence inventory，不作 acceptance whitelist；
- 正式 decision 为 `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`：BF16 eager-control 13/13 exact stable，BF16
  auto 7/13；FP32 4/4 仍是 descriptive probe，不控制结论。原 `NO_GO_V2_1_FULL_45_SUBSTRATE` 永久不变；
- deterministic USTAR 含 72 members，1,873,920 bytes，SHA256
  `d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc`，tree inventory SHA256
  `6423c13f4b7067f5a2139442bab0be11a1ef5da9f880f51a1131768afa963b82`；
- private HF dataset `gavinlaw/causalcache-spatial-reference-audit-mobile` 的 canonical immutable revision 为
  `d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`，tag `spatial-reference-audit-v1` exact 解析到该 revision。
  从全新 cache 强制下载后，README/manifest/raw 三文件 allowlist、archive hash/size、canonical USTAR rebuild、
  72-member tree、summary SHA 与 scientific payload hash 均 exact；
- Git 只回写 `data/results/spatial_reference_audit_v1/` 的 compact summary/artifact binding。下一步不是
  restoration，而是先冻结一个全新 `v2.2-eager` source 与 fresh 45-state attempt；任何失败都不回写本 audit。

### 2026-07-15：Restoration v2.2-eager fresh-45 source-only freeze

- parent spatial artifact 固定为 `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`（BF16 auto 7/13、BF16
  eager-control 13/13）；它只授权新的 eager substrate，不改写
  `NO_GO_V2_1_FULL_45_SUBSTRATE`；
- machine-readable contract 为 `code/configs/causalcache_restoration_v2_2_eager.json`，完整说明为
  `docs/restoration_v2_2_eager.md`。config SHA256 与 clean pushed source commit 留待最终 freeze 后记录真实值，
  本 source-only milestone 不填预测值；
- scientific contract 只改变 runtime：BF16 eager、seed 0、TF32 off、cuDNN deterministic on/benchmark off、
  float32 matmul `highest`；明确不声称 strict CUDA determinism。official-tool interface、prompt/parser、teacher
  target、45-state projection、gate 与 per-state schedule 均继承 v2.1；container/Python/PyTorch/CUDA/cuDNN/
  Transformers/driver/H200 stack 则 exact 绑定回 spatial audit，避免引入第二个 runtime 变量；
- formal attempt 必须 fresh 运行全部 45 states，旧 v2.1 raw、state records、aggregate、terminal 与 ledger
  均不可复用或计数。全 attempt generation/teacher/KL 计划与硬上限仍为 90/135/90；
- worker topology 固定为同机同容器两张 H200：even worker 23 states，odd worker 22 states，不允许 state
  stealing。coordinator 必须在任一 runtime import/model construction 前，以独立于 v2.1 的 global sibling
  ledger 对整个 attempt 做 exclusive durable claim；任一 worker failure 或 parity inventory drift 都使 attempt
  `INVALID`；
- 两个 runtime identity 在首个 marker/generation 前经过 coordinator barrier；logical device 固定为 0/1，
  preflight 选择的 physical/NVML index 可为任意两个不同非负编号，并由 distinct UUID/PCI 交叉证明；
- v2.2 lifecycle 不支持 resume；任一 worker/coordinator/主机中断都会把唯一 attempt 终止为 `INVALID`，已有
  terminal prefix 也不能由新 invocation 跳过并补齐。global claim 会预绑定两个 worker 的 root 外 sibling
  high-water ledger，每个 state 前后都必须 durable 更新；stale/missing root mirror、orphan/missing state files
  只按 sibling truth 形成 forensic inventory，policy-free `seal-interrupted` 可封存但绝不 retry；
- confirm state/prompt/image/decoder/generation/teacher、restoration coalition/label、baseline 与 gate
  construction/training/selection 的计数全部固定为 0。planned private HF dataset 为
  `gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile`，当前没有 raw archive、immutable revision
  或 result verdict；
- 本里程碑没有调用 GPU、没有加载 processor/model、没有产生 v2.2 policy output。source-only validator 通过
  也不自动授权正式 run；执行仍需 clean pushed main、双 GPU preflight、fresh parent evidence 和独立 global
  claim。

## 下一步

Subset-search v1 与 spatial audit artifact 均已闭合，不继续为 optimizer 本身追加算法，也不得重跑任何 audit
profile。v2.2-eager source-only freeze 已完成；下一步只运行唯一 fresh 45-state 双 H200 substrate attempt，并
在 terminal 后闭合 deterministic USTAR、private HF immutable fresh-download 与 Git result。若新 substrate 仍
不稳定，再进入独立 source-only semantic-key / canonical-representative protocol。任何 semantic protocol 都不能
用当前 13 个 delta 调 radius，且必须在 synthetic
邻接控件 negative fixture 与独立 AndroidWorld UI-node calibration artifact 上先验证。之后只有新的 versioned
45-state substrate gate 通过，才允许 exact restoration；全部选择和阈值冻结后，最后才能打开 confirm。
