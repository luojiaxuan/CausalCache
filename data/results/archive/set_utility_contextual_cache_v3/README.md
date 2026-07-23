# Contextual hidden cache v3

状态：`COMPLETED_TRAIN_TUNE_ONLY`。

25% nested train trajectories + 完整 tune 的 contextual cache 已完成并通过 finalizer：

- 350 trajectories / 3,703 states：train 2,640、tune 1,063；
- 8,813 unique contexts：query 3,703、event 5,110；
- 1,174 个原子 safetensors chunks，0 tensor-only、0 receipt-only；
- final GUI-Owl LM hidden width 4,096，text 最多 64 tokens；
- visual token 长度分布：448→4,073、459→1,332、464→1,676、480→1,732；
- 总文件大小 37,771,051,973 bytes；
- cache content SHA256 `af2b090d5b0c8daefe2586b3ebdad39aee7b7594f2847d237b25429bac791eb0`；
- manifest file SHA256 `c8a523c1b1febfea12322dd9e3ec50c15d601a1657401fb02c8d2bf8131b71b2`；
- extraction source revision `e413e5b32e9fb4230e2de5eb74dc56ffe7c1b2fc`。

执行使用 Hyper00/Hyper01 共 12×H200。跨机复制出现孤儿 tar 进程后，没有接受部分文件；清除 14 个无
receipt 半成品，并在 Hyper00 仅恢复缺失分区。最终 finalizer 对每个 chunk 的 byte count、SHA256、tensor
inventory、BF16 dtype 和 geometry 逐项验证。

Source of Truth：

- train/tune input：Hyper00
  `/data02/jaxan/artifacts/causalcache-contextual-inputs-v3-304631e`；
- complete cache：Hyper00
  `/data02/jaxan/runs/causalcache-contextual-hidden-v3-e413e5b`；
- independent original partition mirror：Hyper01 同名 run path；
- intended HF dataset：`gavinlaw/causalcache-set-utility-variable-history-mobile`，artifact path
  `artifacts/set-utility-contextual-cache-v3-af2b090d`；
- upload status：`PENDING_HF_UPLOAD`。

生成入口：

```bash
python code/scripts/materialize_set_utility_contextual_inputs.py ...
python code/scripts/extract_set_utility_contextual_hidden.py ...
python code/scripts/finalize_set_utility_contextual_hidden_cache.py ...
```

下一步：使用同一 cache 并行训练 multi-latent DeepSets 与 Set Transformer；在 tune 上生成 deployment-search
实际访问的 coalition truth，冻结后才允许 untouched evaluation。
