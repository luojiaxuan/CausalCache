# Policy-vision v2 GPU UUID repair attempt

状态：`INVALID_POLICY_VISION_V2_ZERO_FEATURE_SIZE_DICT_INTERFACE`。

唯一 v2 formal attempt 从 clean pushed
`main@fe7640395d3b6aea2e5e3a8cc34a49efc5ba2d2f` 在 Hyper00 两张 H200 上启动。v2 的核心修复已通过：
`torch.cuda.get_device_properties().uuid` 被验证为当前 loaded `torch._C._CUuuid` exact type，随后原
expected UUID、`nvidia-smi` 与 PCI binding 全部通过。

新的失败发生在 policy model load 和任何 feature forward 之前。pinned Transformers 返回的
`image_processor.size` 类型是 `transformers.image_utils.SizeDict`；它可以精确转换为冻结的
`{"longest_edge": 2621440, "shortest_edge": 2621440}`，但不实现 `collections.abc.Mapping`。v2 source 把
`Mapping` membership 与数值 identity 绑在同一个检查中，因此以
`policy-vision image processor size drifted` fail closed。这是 runtime interface mismatch，不是视觉几何或
restoration science 的负结果。

- policy model load：0；
- image processor batch：0；
- vision feature forward / cosine / selection：0；
- semantic label load、gate、matched-NLL、closed-loop、confirm/test：0；
- canonical output 与 hidden staging：均不存在；
- durable attempt ledger SHA256：
  `492e7c7aea0539fc5bc65c0d0be41b5659563a140d8e411a3b347b4c18584502`。

同一 v2 protocol 不重跑，ledger 与停止的容器 `sglang-omni-jaxan-07170602` 保留。下一步只能先提交本失败
证据，再冻结新的 versioned SizeDict-interface repair；不能把本 attempt 写成 comparator result，也不能开始
gate/confirm。
