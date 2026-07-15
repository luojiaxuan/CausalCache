# Restoration v2 real processor audit

本目录保存 dependency 8 的 policy-output-free processor evidence。它是 Git 中的轻量运行记录，不包含
model weights、screenshots、policy logits 或 restoration outputs。

## 完成态结果

- outcome：`PASSED_GUI_OWL_V2_PROCESSOR_AUDIT`；
- summary：`summary.json`；
- summary SHA256：`7d5ac1bd13ba5def46dfb2ca419d59bb0da1ff970f186e9fd092d4006d8b43b8`；
- run commit：`82442da8193063b59e7b538d321406e26401d393`，clean detached worktree；
- host/container：Hyper00 `node-radixark-16-0000`，
  `6263d8cd69fbe30a1313982126c493d08ed3997c1f58e1080c95481d03c3ac21`；
- container image digest：
  `sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- UTC bracket：`2026-07-15T17:38:30.280432Z` -- `2026-07-15T17:38:51.584295Z`；
- runtime：Python `3.12.3`、Transformers `5.6.0`、PyTorch distribution `2.11.0+cu130`、
  Pillow `12.2.0`，processor device `cpu`。

审计逐 byte 复核 14-file / 17,545,907,171-byte pinned model snapshot 与三份 Transformers source，随后只
调用一次 `AutoProcessor.from_pretrained`。model weight files 被读取用于 SHA256，但没有 materialize 为
tensors；没有调用 model loader、forward 或 generate。

## 冻结的真实 processor evidence

- processor/tokenizer/image processor：`Qwen3VLProcessor` / `Qwen2Tokenizer` /
  `Qwen2VLImageProcessor`；
- pixel target：构造参数 `min_pixels=max_pixels=2,621,440`，真实运行时表示严格为
  `image_processor.size.shortest_edge/longest_edge`；direct `min_pixels/max_pixels` attributes 不存在；
- portrait grid：`[1, 152, 68]`；landscape grid：`[1, 68, 152]`；每张实际 2,584 effective visual tokens；
- 1-image：sequence 2,943，text 359，`pixel_values=[10,336,1536]`；
- 5-image：sequence 13,286，text 366，`pixel_values=[51,680,1536]`；
- nested batch-2：sequence `[2,2946]`、两个 1-image samples 均无 padding，
  `pixel_values=[20,672,1536]`；
- assistant/carrier/distance token counts：3 / 8 / 15，joint boundary exact；
- processor tensors 全在 CPU，`pixel_values=float32`，其余记录 tensors 为 `int64`，均不需要 gradient。

顶层和 nested audit flags 均为：`model_weights_loaded=false`、
`model_weights_materialized_as_tensors=false`、`policy_forward_executed=false`、
`policy_generate_executed=false`、`policy_output_generated=false`、
`restoration_output_generated=false`。

## Superseded pre-output attempt

commit `6535e71d07b841e381f131f471d74467d6bef649` 的首次 formal attempt 在读取并验证 model snapshot 后，因
审计器误要求 direct `image_processor.min_pixels/max_pixels` 而 fail closed。真实 Transformers `5.6.0`
把这两个构造参数存入 `SizeDict.shortest_edge/longest_edge`。失败发生在任何 processor case、model tensor、
forward/generate 或 policy/restoration output 前；修复与根因记录在 commit `82442da` 和
`docs/progress.md`。

## 边界

该结果关闭 dependency 8 的 real-processor 子项，但不单独关闭 dependency 8。execution config、exact source
inventory 与 readiness manifest 尚需从包含本 summary 的后续 clean pushed commit 生成；在 readiness validator
返回 `SCREENING_ALLOWED + CONFIRM_LOCKED` 前，v2 policy inference 仍保持锁定。
