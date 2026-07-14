# Qwen3-VL policy smoke test

## 结论

固定的 `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` 已能在一张 A6000 上完成 summary-only、mixed-fidelity 与 full-history 三种真实 GUIOdyssey 输入的 deterministic generation。三种输入在 decision step 4 都生成：

```json
{"action_type":"type_text","text":"Cryptocurrency Market"}
```

规范化后均与 recorded validated action executable-match。该结果只证明真实模型链路闭合，不能说明 restoration attribution 有效，也不能估计完整 validated-reference coverage。

## 配置

- Dataset：`gavinlaw/causalcache-guiodyssey-pilot-mobile@6c840b1be9d96d23425c51ba7c02b35063cfa731`
- Trajectory：`0054832199799795`
- Decision step：4
- Mixed restored event：3
- Visual budget：每张图 256 tokens
- Runtime：PyTorch 2.11.0+cu130、Transformers 5.6.0、bfloat16、单张 NVIDIA RTX A6000
- 结果明细：[`summary.json`](summary.json)

| Variant | Input tokens | Executable match | Latency after warmup | Peak allocated GPU memory |
| --- | ---: | --- | ---: | ---: |
| summary only | 501 | yes | 0.578 s | 17.71 GB |
| mixed fidelity | 865 | yes | 0.643 s | 17.85 GB |
| full history | 1,593 | yes | 0.819 s | 18.10 GB |

这些 latency 来自单次 smoke forward，只用于发现数量级问题，不能作为论文性能比较。

## 复现命令

```bash
python3 -m scripts.run_qwen_policy_smoke \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/Qwen3-VL-8B-Instruct \
  --decision-step-id 4 \
  --mixed-restored-step-id 3 \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --max-new-tokens 96 \
  --output-dir /data/experiments/qwen_policy_smoke_step4
```
