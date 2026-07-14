# UI-TARS-1.5-7B frozen-policy coverage

## 结论

UI-TARS-1.5-7B 没有通过预注册的 frozen-policy coverage gate，因此不进入
restoration attribution pilot：

| Action type | Match | Coverage |
| --- | ---: | ---: |
| tap | 2/7 | 28.6% |
| swipe | 1/1 | 100% |
| type_text | 1/1 | 100% |
| **overall** | **4/9** | **44.4%** |

所有 9 个输出均能解析为统一 `ExecutableAction`，且每个 present action type
至少命中一次；但 overall coverage 低于预注册的 50% 阈值。该阈值不因结果接近
而事后放宽。

## 失败分析

- decision step 9 预测 `coordinate_bin:x8_y8`，recorded action 为相邻的
  `coordinate_bin:x7_y8`；固定的 10x10 bin equivalence 仍将其判为不匹配；
- 其余 tap 失败包含早期 search flow grounding 错误和中途 article target 错误；
- decision step 10 提前输出 `finished`，而 recorded action 仍需点击 Post；
- 因此这不是 parser failure，而是当前 trajectory 上可执行行为覆盖不足。

## 可复现配置

- Model：`ByteDance-Seed/UI-TARS-1.5-7B`
- Revision：`683d002dd99d8f95104d31e70391a39348857f4e`
- Dataset：`gavinlaw/causalcache-guiodyssey-pilot-mobile`
- Revision：`1de9c34ff029d4c01665cdaca74436ae24bff276`
- Validation：full-history executable match
- Visual budget：每张截图 256 visual tokens
- Runtime：单张 A6000，最大输入 5,178 tokens，峰值 allocated GPU memory
  17.81 GB

```bash
python3 -m scripts.run_ui_tars_policy_coverage \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile-v03/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/UI-TARS-1.5-7B \
  --coverage-gate configs/policy_coverage_gate.json \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --max-new-tokens 256 \
  --output-dir /data/experiments/ui_tars_policy_coverage_pilot
```

逐 decision 输出、解析结果、延迟、显存和 snapshot hash 见
[`summary.json`](summary.json)。该结果只用于 frozen-policy candidate selection，
不构成对 CausalCache 方法的验证或证伪。
