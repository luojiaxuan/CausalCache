# ShowUI-2B real-policy smoke

## 结论

固定的 `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`
已在单张 A6000 上闭合标准 Transformers、phone navigation prompt、multi-image
mixed-fidelity input、strict dictionary parser 与 token-logit 接口：

- summary-only forward 返回 finite token logits，shape 为 `[1, 695, 151936]`；
- summary-only、恢复 event 2、full-history 分别使用 1、3、7 张图；
- 三种输入都输出 `INPUT('cryptocurrency market')`，经 case-insensitive executable
  normalization 后与 decision step 4 的 recorded `type_text('Cryptocurrency Market')`
  匹配；
- 所有输出都只有一个 action dictionary，未触发 parser rejection。

该结果只验证模型、processor、prompt、mixed-fidelity input、parser 和 logits 链路，不能
说明 restoration attribution 有效，也不能替代 9-decision full-history coverage gate。

## 运行摘要

| Variant | Images | Input tokens | Latency after warmup | Peak allocated GPU memory | Match |
| --- | ---: | ---: | ---: | ---: | --- |
| summary only | 1 | 695 | 0.562 s | 4.19 GiB | yes |
| restore event 2 | 3 | 1,179 | 0.600 s | 4.27 GiB | yes |
| full history | 7 | 2,147 | 0.726 s | 4.45 GiB | yes |

Logits probe latency 为 0.731 秒，峰值 allocated GPU memory 为 4.33 GiB。Runtime 为
PyTorch 2.11.0+cu130、Transformers 5.6.0、bfloat16、单张 NVIDIA RTX A6000。
这些单次 latency 只用于发现数量级问题，不作为论文性能比较。

## 可复现命令

```bash
python3 -m scripts.run_showui_policy_smoke \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile-v03/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/ShowUI-2B \
  --decision-step-id 4 \
  --mixed-restored-step-id 2 \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --max-new-tokens 128 \
  --output-dir /data/experiments/showui_policy_smoke_step4
```

完整输出、snapshot hash、latency 与显存见 [`summary.json`](summary.json)。
