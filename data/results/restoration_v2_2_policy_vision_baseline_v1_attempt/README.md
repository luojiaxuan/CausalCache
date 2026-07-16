# Policy-vision baseline v1 formal attempt

状态：`INVALID_POLICY_VISION_V1_ZERO_FEATURE_GPU_UUID_TYPE`。

唯一 v1 formal invocation 从 clean pushed
`main@c0937056e94d110cd67e593288f9e0c3a3b24809` 在 Hyper00 两张 H200 上启动。两个 worker 都必须先将
`torch.cuda.get_device_properties(device).uuid` 交给冻结的 canonical UUID parser；当前 PyTorch 2.11 返回
`torch._C._CUuuid`，而 v1 parser 只接受 `str/bytes`，因此第一次 model snapshot full hash、processor/model
load 或 feature forward 之前 fail closed。

本 attempt 产生 0 processor batch、0 vision forward、0 cosine、0 selection、0 semantic label load 和 0 result
file。canonical output 与 hidden staging 在失败后均不存在；gate、matched-NLL、closed-loop、confirm/test 也
保持 0。它不提供 policy-vision comparator 的科学结果。

同一 v1 identity 不重试。下一步只能先冻结 versioned replacement，且 repair 范围限定为识别 pinned
`torch._C._CUuuid` 后继续复用原 canonical UUID、expected UUID、PCI 与 `nvidia-smi` cross-check；不得修改图像、
feature、pooling、ranking、budget、labels、statistics 或 denominator。

完整 execution identity、zero-operation boundary、post-failure checks 与 traceback tail 见 `failure.json`；其
SHA256 为 `1cea24506b14b9dcd4bdfff77c82024417f65e1416a418caeec0d1192c2b0311`。
