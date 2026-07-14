# GUI-Owl model-default native-resolution smoke

## 结论

固定的 `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`
已在 model-default visual preprocessing 下通过 1/5-image memory smoke：

- single-image 与 5-image history 的 last-token logits 均 finite；
- 两种输入都生成且只生成一个符合 pinned `mobile_use` grammar 的 click；
- 5-image generation 峰值 22.77 GiB，单张 A6000 可运行；
- runtime 记录实际 `image_grid_thw` 与 effective visual token 数，不再用人为 256-token 输入冒充
  native policy reproduction。

## 运行摘要

| Variant | Images | Grid/image | Visual tokens | Input tokens | Generation latency | Peak generation memory | Parse |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| single image | 1 | `[1,116,138]` | 4,002 | 4,976 | 3.299 s | 17.81 GiB | one valid click |
| 5-image history | 5 | `[1,116,138]` | 20,010 | 21,185 | 9.669 s | 22.77 GiB | one valid click |

5-image last-token logits probe 只物化 1 个 token 的 logits，峰值 20.83 GiB；这避免把无用的完整
`[sequence,vocab]` tensor 算入接口 smoke。GUIOdyssey fixture 是横向截图，因此每图 4,002
tokens；AndroidWorld 1080×2400 screenshot 的独立 processor replay 为 `[1,150,68]`，即每图
2,550 tokens。

## 可复现命令

```bash
python3 -m scripts.run_gui_owl_native_smoke \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile-v03/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --decision-step-id 6 \
  --device cuda:0 \
  --use-model-default-visual-resolution \
  --max-new-tokens 256 \
  --output-dir /data/experiments/gui_owl_native_resolution_smoke
```

完整 grid、token、latency、显存、输出与 snapshot hash 见 `summary.json`。该结果只关闭
native-resolution logits/parser/memory 接口，不是 AndroidWorld task-success 结果。
