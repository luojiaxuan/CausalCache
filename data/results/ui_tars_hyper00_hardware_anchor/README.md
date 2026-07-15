# UI-TARS Hyper00 hardware anchor

## 结论

`HARDWARE_ANCHOR_PASSED`。在查看任何 independent split policy output 前，Hyper00 H200 用冻结的
UI-TARS snapshot、旧 9-decision GUIOdyssey artifact、原 prompt/parser、256 visual-token target 与
deterministic decoding 完整重跑：9/9 parsed、4/9 executable-match，逐 decision boolean vector 与 Aries
A6000 历史结果完全一致。

这只证明当前 H200 runtime 没有改变 frozen gate 的 behavioral decision，不是新的 policy coverage 证据，
也不重新打开旧 single-trajectory rejection。Formal independent runner 可以把本目录的 compact
`summary.json` 作为强制 anchor 输入。

## 资源与吞吐

- 单张 H200，peak allocated 17.83 GB；9 次 generation latency 合计 30.77 s；
- monitor 的两个 active 10 秒窗口平均利用率为 21% 与 18%，max 为 50% 与 59%；
- 瓶颈是逐 decision 的 variable-history image preprocessing 与短 generation，已降到单 GPU 且全程监督。
  该 anchor 只有 9 个 decisions；独立 reference gate 仍应完整记录 monitor，进入 coalition-scale oracle 前
  必须先完成 batching/concurrency 与 GPU-side KL，不能用本次低利用率为扩展任务免责。

## Provenance

- Git：`89f4aa618ff1e74c57bef44dd69903e1c5134f7f`；
- model：`ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`；
- dataset：`gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- dataset shard SHA256：`54399e28df381c2114c1422dedc1aed17f1bd15b320e636155a070e181c33e72`；
- raw remote summary SHA256：`696d2d49efbfff5c2602091399fa0cb7c68f09cef6828e8aa7f10eff497bbe77`；
- monitor log SHA256：`9a09e5dc747df9c366fa5801a4bdbadaa60939336945e1d65e7cd8fde7465fcb`；
- Hyper00 physical GPU 0 / container `cuda:0`，NVIDIA H200，driver 570.172.08；
- container image id：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- Python 3.12.3、PyTorch 2.11.0+cu130、Transformers 5.6.0、BF16。

Raw output text 与重复的 snapshot file table 不作为 reusable artifact；compact summary 已保留 formal
validator 所需的 dataset/model identity、完整 match vector 与运行 provenance。
