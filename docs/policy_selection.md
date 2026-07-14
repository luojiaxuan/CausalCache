# Frozen policy 选择记录

## 选择标准

主 frozen policy 必须同时满足：

1. 权重公开且许可证允许研究复现；
2. 能在本地 forward 中访问 token logits；
3. 支持截图输入和 mobile GUI action grounding；
4. 能在一张 A6000 上完成 pilot forward；
5. 在固定 GUIOdyssey pilot 上通过 `configs/policy_coverage_gate.json`。

Pilot gate 在运行 GUI-tuned candidate 前冻结为：full-history executable-match coverage 至少 50%，并且当前 trajectory 出现的 tap、swipe、type_text 三类 action 各至少匹配一次。该 gate 只决定是否继续做 attribution pilot；正式实验仍需报告未过滤 coverage，并逐状态执行 executable-match validation。

## 候选

| Candidate | License / architecture | 优点 | 当前决定 |
| --- | --- | --- | --- |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Apache-2.0 / Qwen3-VL | 标准 Transformers、multi-image、logits 可用 | 拒绝：pilot full-history match 1/9 |
| [UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) | Apache-2.0 / Qwen2.5-VL | 官方 mobile action grammar、GUI agent tuning、标准 Transformers | 第一 GUI-tuned candidate |
| [ShowUI-2B](https://huggingface.co/showlab/ShowUI-2B) | MIT / Qwen2-VL | 仅 2B、GUI grounding 专用、logits 可用 | UI-TARS 失败时的 grounding 后备 |
| [OpenCUA-7B](https://huggingface.co/xlangai/OpenCUA-7B) | MIT / custom code | computer-use tuning、公开权重 | 后备；custom code 增加复现风险 |

## UI-TARS 冻结配置

- Model：`ByteDance-Seed/UI-TARS-1.5-7B`
- Revision：`683d002dd99d8f95104d31e70391a39348857f4e`
- Architecture：`Qwen2_5_VLForConditionalGeneration`
- Native mobile grammar source：[UI-TARS `prompt.py`](https://github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/prompt.py)
- Snapshot manifest：`configs/ui_tars_1_5_7b_snapshot.json`

UI-TARS 将使用其原生 mobile action grammar，再映射到项目统一的 `ExecutableAction`。mixed-fidelity event 内容、visual-token budget 和 validation contract 不随 backbone 改变。
