# Frozen policy 选择记录

## 选择标准

主 frozen policy 必须同时满足：

1. 权重公开且许可证允许研究复现；
2. 能在本地 forward 中访问 token logits；
3. 支持截图输入和 mobile GUI action grounding；
4. 能在一张 A6000 上完成 pilot forward；
5. 在固定 GUIOdyssey pilot 上通过 `code/configs/policy_coverage_gate.json`。

Pilot gate 在运行 GUI-tuned candidate 前冻结为：full-history executable-match coverage 至少 50%，并且当前 trajectory 出现的 tap、swipe、type_text 三类 action 各至少匹配一次。该 gate 只决定是否继续做 attribution pilot；正式实验仍需报告未过滤 coverage，并逐状态执行 executable-match validation。

## 候选

| Candidate | License / architecture | 优点 | 当前决定 |
| --- | --- | --- | --- |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Apache-2.0 / Qwen3-VL | 标准 Transformers、multi-image、logits 可用 | 拒绝：pilot full-history match 2/9 |
| [UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) | Apache-2.0 / Qwen2.5-VL | 官方 mobile action grammar、GUI agent tuning、标准 Transformers | 拒绝：pilot full-history match 4/9，低于预注册 50% gate |
| [ShowUI-2B](https://huggingface.co/showlab/ShowUI-2B) | MIT / Qwen2-VL | 仅 2B、原生 phone navigation grammar、logits 可用 | 拒绝：pilot full-history match 2/9 |
| [OpenCUA-7B](https://huggingface.co/xlangai/OpenCUA-7B) | MIT / custom code | computer-use tuning、公开权重 | 拒绝：pilot full-history match 1/9 |
| [GUI-Owl-1.5-8B-Instruct](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct) | MIT / Qwen3-VL | 官方 AndroidWorld adapter、原生 5-image history、logits 可用 | 拒绝：固定 62 分母 success 上界 30/62 |
| [GUI-Owl-1.5-8B-Think](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Think) | MIT / Qwen3-VL | 同一原生接口、官方报告 AndroidWorld 71.6% | 拒绝：固定 62 分母 success 上界 29/62 |

## UI-TARS 冻结配置

- Model：`ByteDance-Seed/UI-TARS-1.5-7B`
- Revision：`683d002dd99d8f95104d31e70391a39348857f4e`
- Architecture：`Qwen2_5_VLForConditionalGeneration`
- Native mobile grammar source：[UI-TARS `prompt.py`](https://github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/prompt.py)
- Snapshot manifest：`code/configs/ui_tars_1_5_7b_snapshot.json`

UI-TARS 评测使用其原生 mobile action grammar，再映射到项目统一的 `ExecutableAction`。mixed-fidelity event 内容、visual-token budget 和 validation contract 没有随 backbone 改变。

## UI-TARS gate 结果

- 结果：4/9（44.4%），低于冻结的 50% overall threshold；
- Parser coverage：9/9；
- Action-type gate：通过，tap 2/7、swipe 1/1、type_text 1/1；
- 决策：不进入 attribution pilot，不因 decision step 9 的相邻-bin miss 事后修改 equivalence；
- 完整记录：`data/results/ui_tars_policy_coverage/`。

下一候选优先评估 OpenCUA-7B。只有在确认其 custom code 可复现、logits 可访问、
mobile action grammar 可映射到当前 `ExecutableAction` 后，才固定 revision 并运行
同一 gate。ShowUI-2B 保留为 grounding-oriented 后备，不默认假定其具备完整规划能力。

## OpenCUA-7B 接口审计与冻结配置

- Model：`xlangai/OpenCUA-7B`
- Revision：`a2efb7d2b104d477a4a2666a357e79550a28aafc`
- Upstream code revision：`xlang-ai/OpenCUA@dfc91ba89f700d10f26ec50362d308571482ab8b`
- Architecture：remote `OpenCUAForConditionalGeneration`，基于 Qwen2.5-VL，但将 M-RoPE 改为 1D RoPE，并使用自定义 tokenizer/chat template；
- Logit access：remote `forward()` 返回带 `logits` 的 `LlavaCausalLMOutputWithPast`，满足 teacher-forced distance 的必要接口；
- Action grammar：官方 evaluator 输出 `pyautogui.*` 或 `computer.terminate(...)`，可映射到现有 tap、type_text、swipe、home、back、wait、stop；
- Coordinate：模型输出 smart-resized image 上的绝对坐标，必须用实际 `image_grid_thw` 归一化，不能直接当原图坐标；
- Multi-image 风险：官方 model card 强调 3-screenshot history，官方 evaluator 另提供 1/3/5-image 设置；本项目仍按原 contract 构造 full history，并原样报告实际 image count、input tokens、显存和 coverage；
- Snapshot manifest：`code/configs/open_cua_7b_snapshot.json`。
- Runtime dependency：`code/requirements/opencua.txt`，固定 `transformers==4.53.0` 与兼容的 `kernels==0.11.7`。

审计结论为可进入实测。运行使用 pinned remote code 和模型自带 processor，且不采用
上游示例中与当前 remote forward signature 不一致的 `grid_thws` 参数名。coverage gate
与 UI-TARS 完全相同，不因 OpenCUA 的 desktop-oriented grammar 调整判定标准。

第一次 load smoke 在容器预装的 Transformers 5.6.0 上失败：新版
`PreTrainedModel.post_init()` 会向 remote `tie_weights()` 传入 `recompute_mapping`，而
OpenCUA 的实现没有该参数。由于上游 `modeling_opencua.py` 明确基于 Transformers
4.53.0，本项目使用独立持久化 venv 固定该版本，不修改或 monkey-patch remote model。
独立 venv 使用 system-site PyTorch，因此也会看到容器的 `kernels 0.14.1`；该版本要求
新版 Hugging Face Hub。第二次 import smoke 据此失败后，项目在 venv 内显式安装兼容
Hub 0.36 的 `kernels 0.11.7`，覆盖系统可见版本。
第三次 load smoke 到达 `from_pretrained` 后发现 adapter 使用了 Transformers 5.x 的
`dtype=` 参数；固定 runtime 4.53.0 对应改用 `torch_dtype=`。该修改只修正加载 API，
不改变权重、prompt、processor 或 validation contract。
第四次 smoke 已成功加载全部 28 个权重 shard，随后发现 Transformers 4.53 的
multimodal processor 要求 system message 也使用 typed content list。adapter 已将
相同 system prompt 包装为 `[{"type": "text", "text": ...}]`，不改变 prompt 内容。
第五次 smoke 完成 load、logits forward 和 1/3/7-image generation；其中 summary-only
与 full-history 各输出两个 PyAutoGUI call。单 decision contract 不允许 parser 静默取
最后一个动作，因此 parser 现要求恰好一个 executable code line，多动作输出记为
parse failure。mixed-fidelity 输出单个正确 `write` action，仍按原 contract 匹配。

## OpenCUA-7B gate 结果

- 结果：1/9（11.1%），远低于冻结的 50% overall threshold；
- Parser coverage：7/9；
- Action-type gate：失败，tap 1/7、swipe 0/1、type_text 0/1；
- 计算可行性：3--19 images、最长 5,172 input tokens 均完成，峰值显存 18.71 GB；
- 决策：拒绝该 candidate 进入 attribution pilot，结果见 `data/results/open_cua_policy_coverage/`。

## ShowUI-2B 接口审计与冻结配置

- Model：`showlab/ShowUI-2B`
- Revision：`cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`
- Architecture：`Qwen2VLForConditionalGeneration`，标准 Transformers forward 可返回 token logits；
- License：MIT；
- Snapshot manifest：`code/configs/showui_2b_snapshot.json`；
- Coordinate：官方 navigation output 使用相对坐标 `[0, 1]`；进入统一 executable schema 前映射到 `[0, 1000]`；
- Native phone grammar：`INPUT`、`SWIPE`、`TAP`、`ANSWER`、`ENTER`；覆盖当前 pilot 的 tap、type_text 与 swipe；
- Output format：单个 Python dictionary，字段为 `action`、`value`、`position`。

接口审计结论为可进入实测。ShowUI 同时支持 point grounding 与完整 UI navigation；本项目
只使用官方 phone navigation prompt，不把 point-grounding 接口混入 action coverage。其
`INPUT` 虽含点击位置，但统一 schema 仍按当前 validation contract 映射为一次
`type_text`；不会为了该 candidate 修改 recorded action、gate threshold 或 equivalence。

第一次 smoke 中三种 fidelity 均输出正确的 `INPUT` 文本但将 `position` 设为 `None`。
当前 executor 的 `type_text` 与其他 candidate 一样只要求已聚焦文本框和文本参数，不执行
ShowUI 可选的组合点击；因此 adapter 在 coverage gate 前固定为：`position` 非空时校验其
坐标，空值时仍映射为单次 `type_text`。该修改不改变文本 equivalence、validated action
或 gate，只移除统一 executor 不消费的字段要求。

修正后的 canonical smoke 已完成：summary-only、恢复 event 2 与 full-history 分别使用
1、3、7 张图，均生成唯一且正确的 `INPUT('cryptocurrency market')`；summary-only
forward 返回 `[1, 695, 151936]` finite logits，峰值显存不超过 4.45 GiB。结果见
`data/results/showui_policy_smoke/`。这只证明候选可以进入预注册 coverage gate。

## ShowUI-2B gate 结果

- 结果：2/9（22.2%），低于冻结的 50% overall threshold；
- Parser coverage：9/9；
- Action-type gate：失败，tap 1/7、swipe 0/1、type_text 1/1；
- 计算可行性：3--19 images、最长 5,305 input tokens 均完成，峰值显存 5.01 GiB；
- 主要失败：后半段多次过早输出 `ANSWER('task complete')`；
- 决策：拒绝该 candidate 进入 attribution pilot，结果见 `data/results/showui_policy_coverage/`。

四个已登记 candidate 均未通过 gate。按预注册停止规则，不继续为当前单轨迹枚举相似
backbone，也不事后调整 prompt、坐标容差或 threshold；下一步重新选择 benchmark-native
policy/evaluation stack，并要求独立 held-out validation 与合法 token-logit access。

## AndroidWorld-native successor

新的 stack 固定 `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`。
它不参加已结束的 GUIOdyssey gate，而是在 pinned AndroidWorld registry 的独立 validation
partition 上按原生 adapter 复现。官方 adapter 保留最近 5 张截图并把更早 action 转为
文本，直接提供 mixed-fidelity history 接口；模型卡报告 AndroidWorld success 69.0%。

模型只有在 finite-logit/native parser smoke、emulator/reward smoke，以及预注册的至少
95% parse coverage 和 50% validation task success 都通过后，才会写入主 experiment
contract。完整 provenance、task split 与停止规则见 `docs/androidworld_stack.md`。

Native smoke 已得到单图与 5 图 finite logits、严格单 `mobile_use` parse，并确认一张
A6000 的峰值显存为 17.07 GiB。该 candidate 当前状态为
`rejected_by_native_validation_gate`，不作为 accepted teacher。model-default 正式 validation
在 47/62 checkpoint 时得到 15 个 official success；即使余下 15 条全成功也只有 30/62，低于
31/62 gate。496/496 action 均可解析，因此失败不是 serialization coverage 导致。结果见
`data/results/gui_owl_androidworld_validation/`。

## AndroidWorld replacement teacher v1

下一轮只预注册 `mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060`，
不并行枚举更多 backbone。模型卡报告 AndroidWorld success 71.6%，权重为 MIT licensed BF16
8.77B `Qwen3VLForConditionalGeneration`。它与已拒绝的 Instruct checkpoint 共享 chat template、
processor、tokenizer、model config、native 5-image history 与 `mobile_use` grammar；14 个 runtime
文件中只有四个 weight shards 不同。因此此次替换不引入新的 prompt 或 action adapter。

选择该模型只用于决定下一次预注册实验，不等价于本地 gate 已通过。固定执行顺序为：

1. Hyper01 单 H200 做 model-default 1/5-image finite-logit 与 strict parser smoke；
2. smoke 通过后，在完全相同的 62-instance AndroidWorld validation plan 上运行；
3. parse coverage 仍须至少 95%，task success 仍须至少 50%；
4. failure 时拒绝并停止本轮，success 时才升级 benchmark-specific teacher contract。

deterministic generation 继续使用 `do_sample=false`，因为 restoration contract 需要固定 canonical
action，且沿用既有栈的 `max_new_tokens=256`；官方 71.6% 只作为选型 prior，不声称可与
本项目 deterministic gate 直接比较。smoke fixture 固定在 decision step 6，使 history branch
真实经过 5-image 上限；早期配置中的 step 4 在任何 candidate inference 前经协议审计更正。首次 smoke
若只暴露与 action correctness 无关的稳定 Thinking prefix，可在任何 validation episode 开始前做
一次 fail-closed parser 适配并增加测试；不得根据动作正确性、success 或 validation state 调整格式。
完整冻结配置见 `code/configs/androidworld_replacement_teacher_v1.json`。

首次 smoke 的两个输出均只额外包含一个开头且闭合的 `<think>` block。因此在任何
AndroidWorld validation 前执行了上述一次性适配：共享 helper 只允许并剔除最多一个
闭合 prefix，随后仍执行原有的单 action/单 tool-call full match。记录输出与 malformed boundary
回归测试都已通过；该改动不使用 action correctness 或 benchmark reward。
适配后在 Hyper01 以原 model/data/fixture/generation 参数重跑，单图与 5 图的 finite logits
与 parse 均为 2/2，因此 interface smoke 通过。`executable_match` 0/2 按预注册只作
diagnostic；candidate 只被推进到 AndroidWorld validation，未被接受为 teacher。

Aries 正式 validation 使用完全相同的 62-instance plan、model-default preprocessing、5-image 上限、
deterministic generation 和 50% gate。40 个 checkpoint 时 success 上界已经严格低于 31/62；两个在途
worker 完成后得到 42 records、9 official successes、20 unobserved，因此完整成功率下界为 9/62、上界
为 29/62。512/513 actions parsed（99.81%），parse gate 通过，但 task-success gate 失败。该差异不能
归因于主要的 output grammar coverage。

9 个 exception 与 Instruct run 一样按固定分母保留：6 个 HTTP 500、2 个 live instance 与 frozen plan
不一致、1 个任务初始 reward 已为 1.0；没有 retry 或删除。完整结果见
`data/results/gui_owl_1_5_8b_think_androidworld_validation/`，raw traces 位于 private HF dataset
`gavinlaw/causalcache-androidworld-validation-mobile@v0.2.0`
(`0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`)。

最终决定：`rejected_by_androidworld_validation_gate`。官方 71.6% 只作为选型 prior，不能替代本项目
冻结 deterministic contract 的复现结果。按预注册停止本轮，不继续 test split，不升级 teacher contract，
不生成 restoration labels。任何新的 policy 或 validated-reference 来源必须作为下一轮独立预注册。

`MarsXL/UI-Voyager@c262b85` 没有选为主 teacher：其官方 inference 始终只传当前截图，所谓
`n_history_image` 只影响 SFT artifact 保存。为它加入历史截图会形成新的 OOD policy interface，
其报告的 81.0% AndroidWorld success 不能支持该修改后的接口。
