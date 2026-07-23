# go/no-go diagnostic attempt 1（implementation-invalid）

## 结论

本次运行不是 CausalCache outcome，禁止作为正负结果、attribution label 或 policy evidence。runner 在
decision step 4 的 full-history generation 通过 executable match 后，第一次 teacher-forced reference
forward 即 fail-closed；没有产生任何 coalition distance。

## 诊断

- clean Git：`527711bcf1b71ab84f7bdf68d0ddafe0c2444240`；
- host/GPU：Hyper00 `node-radixark-16-0000`，physical GPU 0，container `cuda:0`，NVIDIA H200；
- model：`Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；
- dataset：`gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`，
  shard SHA256 `54399e28df381c2114c1422dedc1aed17f1bd15b320e636155a070e181c33e72`；
- image：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- failure：Qwen3-VL processor 的 `mm_token_type_ids` 与 prompt `input_ids` 同为 2011 tokens；runtime
  追加 12 个 canonical action prefix tokens 时只扩展了 `attention_mask`，3D RoPE 因 2023 对 2011
  长度不一致而抛出 `IndexError`。

## 处理

teacher forcing 现在识别所有与 prompt 等长的 sequence-aligned tensors：`attention_mask` 对 action
prefix 填 1，`mm_token_type_ids` 对新文本 token 填 0；未知 aligned key fail closed。新增回归测试后从
新的 output directory、更新后的 clean pushed `main` 重跑。states、prompt、canonical action、预算、
distance、baseline 与 outcome threshold 均不变。

remote 原始 `failure.json` 仅为临时 staging；本目录的轻量摘要是该失败的 Git source of truth。
