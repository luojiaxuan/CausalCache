# GUI-Owl Think native smoke

## 结论

`mPLUG/GUI-Owl-1.5-8B-Think@afe3707` 在 format-only parser 适配后通过预注册的 Hyper01
interface smoke：

- single-image 与真实 5-image history 的 last-token logits 均 finite；
- 实际 image count 为 1/5，effective visual tokens 为 4,002/20,010；
- 两个 raw output 均只含一个闭合 thinking prefix、一个 `Action` 和一个
  `mobile_use` tool call，parse coverage 为 2/2；
- 生成峰值显存为 19.15/24.47 GB，单 H200 可运行。

`executable_match` 为 0/2，但它在预注册中只是 diagnostic，不属于 smoke acceptance gate。
这个结果只把 candidate 推进到冻结的 62-instance AndroidWorld validation，不是 accepted
teacher，也不是 CausalCache 方法效果证据。

## 重复生成检查

与适配前的 [`strict-parser smoke`](../gui_owl_1_5_8b_think_smoke_strict/README.md) 比较：

- single-image raw output 逐字相同；
- 5-image raw output 只在 `Action` 描述中的一个句点放在引号内还是引号外上不同；
  thinking 内容、tool-call JSON、coordinate 和 canonical executable action 相同。

因此 `do_sample=false` 在该栈上不应被描述为 byte-identical natural-language generation。后续
restoration distance 必须显式固定 teacher-forced action token boundary，并单独测量重复 forward 方差。

## 可复现性

- 完整生成、logits、latency 与显存：[`summary.json`](summary.json)
- 主机、GPU、container digest、Git commit、HF revisions 与完整 argv：
  [`run_manifest.json`](run_manifest.json)
- `seed=null` 表示 runner 没有 RNG 参数；generation 实际使用 `do_sample=false`。
