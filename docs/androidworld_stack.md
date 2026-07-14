# AndroidWorld benchmark-native stack

## 为什么重建 stack

固定 GUIOdyssey trajectory 上的 Qwen3-VL、UI-TARS、OpenCUA 与 ShowUI full-history
coverage 分别为 2/9、4/9、1/9 与 2/9。继续针对该 trajectory 调 prompt 或枚举相似
backbone 会引入明显的 model-selection bias，也不能解决 AndroidWorld closed-loop 主实验
与离线数据 action interface 不一致的问题。

新 stack 不复用这四次 negative gate 做 candidate tuning。它直接采用有官方 AndroidWorld
agent adapter、公开权重与 benchmark 报告的原生 mobile policy，并在动态任务的独立
validation partition 上重新验证。

## Primary policy

- Model：`mPLUG/GUI-Owl-1.5-8B-Instruct`
- Revision：`06d5faecff74840bab2be2425e9c42667a5d04fc`
- Architecture：标准 `Qwen3VLForConditionalGeneration`
- License：MIT
- 官方报告 AndroidWorld success：69.0%
- Snapshot manifest：`configs/gui_owl_1_5_8b_snapshot.json`

选择它不是因为单纯的排行榜分数，而是因为官方 AndroidWorld agent 已经闭合本项目需要
的三个接口：mobile action tool call、multi-image history、较早 action 的文本退化表示。
标准 Transformers forward 预期可返回 token logits，但仍必须通过本项目 smoke 才能更新
为 accepted policy。

最初的 256-visual-token smoke 已证明 logits/parser 接口可运行，但后来核对 pinned adapter
发现它不复现上游视觉分辨率，因此不能作为 native-resolution smoke。结果及边界见
`results/gui_owl_native_smoke/`。emulator/reward smoke 已通过：
`SystemWifiTurnOn` 经 HTTP executor 后 score 从 0.0 变为 1.0，结果见
`results/androidworld_environment_smoke/`。candidate 仍需通过 task partition 与 validation
task-success gates，尚未写入主 experiment contract。

冻结 validation plan 上的 `ClockStopWatchPausedVerify[0]` 单实例 closed-loop smoke 已通过：
4/4 outputs parsed，policy 明确 terminate，environment reward 与 AndroidWorld 官方 success 均为 1。
结果见 `results/gui_owl_androidworld_validation_smoke/`。该 run 同样使用 256 visual tokens/图，
只关闭执行链路，不构成 native policy reproduction，也不替代 62-instance validation gate。

## Pinned benchmark code

- Canonical AndroidWorld：`google-research/android_world@3e50888527ef9f29b9157ecd537e408008bb1c85`
- GUI-Owl adapter：`X-PLUG/MobileAgent@11cea575561fb7800b5fb6b6cafa56f7a91de11f`
- Adapter path：`Mobile-Agent-v3.5/android_world_v3.5/android_world/agents/gui_owl.py`

环境与 reward 以 canonical AndroidWorld 为准；GUI-Owl prompt、history conversion 与 action
parser 以 pinned MobileAgent adapter 为准。若 fork 与 canonical environment 有行为差异，
必须先记录 diff，不能静默采用有利版本。

## Native context contract

官方 agent 每次保留最近 5 张截图，将更早历史退化为 executed-action text。其
`mobile_use` grammar 包含 click、long press、swipe、type、system button、open、wait、
answer 与 terminate，坐标范围为 `[0, 1000]`。

视觉预处理必须使用 model-default resolution。pinned adapter 先把 1080×2400 screenshot
resize 为 1092×2408；GUI-Owl 的 Qwen3-VL processor 随后得到 grid `[1,150,68]`，merge
size 为 2，因此有效视觉 token 数是 `150×68/4=2550`/图。项目已用相同模型 processor
分别重放原图和上游 resize 图，两者 grid 完全一致。此前强制 256 tokens/图的运行只保留约
1/10 的视觉 token，全部标记为 configuration-invalid，不能用于 policy gate。

第一步完全复现该 native context，不立即把现有“两张图一个 event block”的 GUIOdyssey
prompt 套上去。native smoke 通过后，再把 CausalCache high-fidelity event exposure 映射到
同一 5-image 上限，并按实际 visual tokens 计费。任何 event packing 变化都要在 attribution
前冻结并增加回归测试。

## 无泄漏 task partition

AndroidWorld 的 116 个 task templates 会动态生成大量参数实例。项目在看到 rollout 结果
前按 task name 的 SHA256 bucket 固定 template split：

- buckets 0--5：gate/attribution training；suite seed 314159，每个 template 3 个实例；
- buckets 6--7：model selection/validation；suite seed 271828，每个 template 2 个实例；
- buckets 8--9：final test；suite seed 161803，每个 template 3 个实例。

完整 task manifest 必须从 pinned registry 生成并提交到 Git；partition 生成后不得按成功率
移动 template。上游 policy 权重保持冻结，本项目只在 train partition 上训练 memory gate。

## Validated teacher 边界

GUIOdyssey offline analysis 继续要求 full-history action 与 recorded successful action
executable-match。AndroidWorld 没有逐状态 expert demonstration，因此 closed-loop 标签只从
terminal-success rollout 的决策产生，并单独报告成功任务覆盖率、保留决策数和未过滤
ablation。两个 validation source 不混成一个 coverage 数字。

这需要在真正生成 AndroidWorld attribution labels 前将 machine-checked contract 升级为
benchmark-specific validation；当前 v0.3 contract 仍保持不变，避免用尚未验证的 stack
提前修改方法结果。

## 预注册 gates 与执行顺序

1. 下载并校验 14 个 pinned model files；
2. [待重跑] model-default resolution 下的 native prompt/parser smoke：finite logits、单 action
   parse、1/5-image history；
3. [已通过] 启动 pinned AndroidWorld emulator，完成环境与 reward smoke；
4. [已完成] 从 pinned registry 生成并提交 task partition manifest；
5. 在 validation partition 复现 frozen policy，要求 parse coverage 至少 95%、task success
   至少 50%；
6. 只有通过后才升级 experiment contract、生成 restoration labels 和训练 gate。

validation 开始后不再调 prompt、executable equivalence 或 threshold。将 256-token 输入修正为
pinned model-default resolution 不是调参，而是恢复漏掉的上游预处理契约；修正前的结果全部作废，
阈值、task plan 与 policy 权重保持不变。若修正后的 gate 失败，停止该 stack 并报告
reproduction failure，不用 test partition 继续选模型。

## Compute placement

Aries 已确认 x86_64、Docker 27.2.1、`/dev/kvm` 可用，适合 Android emulator 与单卡
A6000 policy smoke。每个 GPU job 仍需 10 秒 idle preflight，并显式使用至多一张 GPU。

## Container compatibility

Pinned MobileAgent Dockerfile 的 `openjdk:18-jdk-slim` 在 2026-07-14 已无法从 Docker Hub
解析。为保持 upstream checkout 与 revision 不变，构建前使用
`scripts/prepare_androidworld_dockerfile.py` 严格将这一行替换为可解析的
`eclipse-temurin:17-jdk-jammy`。脚本要求原始行恰好出现一次，upstream 变化时会拒绝
静默打补丁。这是环境可用性修复，不修改 AndroidWorld task、reward、agent 或 prompt。

同一 Dockerfile 的 project install 会因 upstream `setup.py` 直接使用但未声明
`pkg_resources` 而在新版 uv 的隔离构建中失败。构建前处理因此还会严格将
`uv pip install . --system` 替换为 uv 建议的
`uv pip install . --system --no-build-isolation`，使用镜像中 Python 3.11 已安装的
setuptools。因 Ubuntu 的 `python3-wheel` 只安装给系统 Python 3.10，overlay 还会在
Python 3.11 环境预装 `wheel==0.45.1` 与 upstream `setup.py` 已声明的
`grpcio-tools==1.71.0`。构建工具 uv 固定为首次构建观测到的 `0.11.28`。这些修复
不改动 AndroidWorld Python 源码或 runtime dependency constraints。

MobileAgent HTTP server 另有一个 fork 内部 schema mismatch：原始 server 导入旧
`android_world.env.json_action`，但同 revision 的 GUI-Owl converter 产生四坐标 swipe，只被
`android_world.agents.new_json_action` 接受。构建 context 通过
`scripts/prepare_androidworld_server.py` 做单行 fail-closed import replacement；修复 image ID 为
`sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486`，正式 episode 前的
四坐标 swipe transport smoke 已通过。该修复保持 action coordinates 不变，只关闭 transport
schema 差异。

upstream wheel 只安装顶层 `android_world`，漏掉 `task_evals` 等子包；官方 server 能运行是因为
Docker entrypoint 的工作目录 `/` 包含完整 COPY 源码。validation plan builder 因此也必须在该
源码根目录解析 AndroidWorld，不能从不完整的 site-package wheel 单独运行。
