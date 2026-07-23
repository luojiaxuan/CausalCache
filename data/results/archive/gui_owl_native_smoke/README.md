# GUI-Owl-1.5-8B 256-token interface smoke（非 native resolution）

## 结论

固定的 `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`
已在单张 A6000 上闭合 logits/parser 接口：

- 标准 `Qwen3VLForConditionalGeneration` 与 `Qwen3VLProcessor` 本地加载；
- 单图与 5-image history 均返回 finite token logits；
- 两种输入均生成且只生成一个符合 pinned `mobile_use` grammar 的 click；
- strict parser 将两条输出都规范化为 `tap / coordinate_bin:x3_y2`；
- 5-image 输入使用 2,365 tokens，峰值 allocated GPU memory 为 17.07 GiB。

本次人为固定为 256 visual tokens/图。后续对 pinned adapter 的端到端 processor replay 表明，
1080×2400 AndroidWorld screenshot 在 model-default resolution 下产生 grid `[1,150,68]`，即
2,550 visual tokens/图。本次结果因此只证明接口可运行，不能称为 native-resolution smoke，
也不能用于 AndroidWorld policy validation。

两种输出恰好都与 GUIOdyssey fixture 的 recorded tap executable-match，但这是 interface
smoke 的 incidental observation，不作为 GUIOdyssey policy coverage，也不重新打开已经结束
的 candidate gate。该结果只允许项目进入 AndroidWorld emulator/reward smoke。

## 运行摘要

| Variant | Images | Input tokens | Logits finite | Generation latency | Peak allocated GPU memory | Parse |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| single image | 1 | 1,212 | yes | 2.072 s | 16.71 GiB | one valid click |
| native history | 5 | 2,365 | yes | 2.133 s | 17.07 GiB | one valid click |

Runtime 为 PyTorch 2.11.0+cu130、Transformers 5.6.0、bfloat16；Qwen3-VL 的 visual
patch factor 按 model preprocessor 固定为 32。单次 latency 只用于发现数量级问题，不作为
论文性能比较。

## 可复现命令

```bash
python3 -m scripts.run_gui_owl_native_smoke \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile-v03/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --decision-step-id 6 \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --max-new-tokens 256 \
  --output-dir /data/experiments/gui_owl_native_smoke
```

完整输出、snapshot hash、logits shape、latency 与显存见 [`summary.json`](summary.json)。
