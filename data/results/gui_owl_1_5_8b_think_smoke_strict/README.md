# GUI-Owl Think strict-parser smoke

## 结论

`mPLUG/GUI-Owl-1.5-8B-Think@afe3707` 的首次 Hyper01 smoke 关闭了视觉接口，但未通过
预注册的 strict-parser gate：

- single-image 与真实 5-image history 的 last-token logits 均 finite；
- 实际 image count 为 1/5，effective visual tokens 为 4,002/20,010；
- 生成峰值显存为 19.15/24.47 GB，单 H200 可运行；
- 两个输出都以一个闭合 `<think>...</think>` 开头，后面各含一个原生
  `Action + mobile_use <tool_call>`；旧 parser 因此按设计拒绝，parse coverage 为 0/2。

该结果的 scope 是 `interface_only_not_policy_coverage`，`executable_match` 不参与 gate，不是 teacher
acceptance 或 CausalCache 方法效果证据。预注册允许在任何 AndroidWorld validation 之前做一次
与 action correctness 无关的 format-only 适配；因此下一步只允许一个闭合 thinking prefix，
不改 `Action` 或 tool-call grammar。

## 可复现性

- 完整生成、logits、latency 与显存：[`summary.json`](summary.json)
- 主机、GPU、container digest、Git commit、HF revisions 与完整 argv：
  [`run_manifest.json`](run_manifest.json)
- `seed=null` 表示本 runner 没有 RNG 参数；实际 generation 固定为 `do_sample=false`，不把
  未设置的 seed 伪记为已执行参数。
