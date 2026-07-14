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
- 当前状态仍是 preregistered-not-run，不是 accepted teacher，也不是 CausalCache 效果证据。

## 当前 artifact 状态

- Git 代码、配置、论文与轻量测试 fixture：本仓库 `main`；
- GUIOdyssey pilot：私有 Hugging Face dataset `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- Rejected policy candidate：上游 Hugging Face model `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；共享机器副本只是可重建 cache；
- Rejected GUI-tuned policy candidate：上游 Hugging Face model `ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`；Aries 副本只是可重建 cache；
- Rejected computer-use policy candidate：上游 Hugging Face model `xlangai/OpenCUA-7B@a2efb7d2b104d477a4a2666a357e79550a28aafc`；Aries snapshot 与 venv 只是可重建 cache；
- Rejected GUI navigation policy candidate：上游 Hugging Face model `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`；Aries snapshot 只是可重建 cache；
- Rejected AndroidWorld-native policy candidate：上游 Hugging Face model `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；native validation 上界 30/62，未通过 50% gate；
- Preregistered replacement policy：上游 Hugging Face model `mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060`；尚未运行 Hyper01 smoke 或 AndroidWorld gate；
- AndroidWorld native validation traces：私有 Hugging Face dataset `gavinlaw/causalcache-androidworld-validation-mobile@v0.1.0` (`3fcca45fffe9842c9fcebbf5c6c27c9540bb1515`)；
- 完整 attribution dataset：尚未生成，pilot 扩展后仍需单独登记 revision；
- gate checkpoint：尚未生成，目标 Hugging Face model repo 待 owner 确认；
- 当前没有仅存在共享机器或本地磁盘上的正式实验 artifact；Taurus/Aries 目录只作为 HF artifact 的 staging/cache。

## 未决策项

- transfer backbone；
- canonical action path 上 teacher-forced component distance 的具体 token boundary；
- pilot 应扩展到多少 app、trajectory 和 horizon 才足以进入 attribution 主表。

这些项目必须经过可获得 logits、许可证、磁盘和算力检查后再冻结，不能为了填配置而猜测。

## 下一步

从 Git `main` 的精确提交在 Hyper01 单 H200 运行 GUI-Owl Think model-default 1/5-image
finite-logit/parser smoke。smoke 结果必须先回写并 push；通过后才能在 Aries 复用冻结的 62-instance
AndroidWorld validation gate。test partition 保持 sealed，不修改 prompt、action equivalence 或 threshold。
