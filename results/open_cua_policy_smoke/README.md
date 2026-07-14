# OpenCUA-7B real-policy smoke

## 结论

Pinned OpenCUA-7B runtime 已在单张 A6000 上闭合：

- 28/28 weight shards 加载成功；
- summary-only forward 返回 finite token logits，shape 为
  `[1, 526, 152064]`；
- summary-only、mixed-fidelity 和 full-history 分别使用 1、3、7 张图，均完成
  deterministic generation；
- 只恢复 event 2 时生成唯一的
  `pyautogui.write(message='Cryptocurrency Market')`，与 decision step 4 的
  recorded action 完全匹配；
- summary-only 与 full-history 各生成两个 executable code lines，因此按单 decision
  contract 记为 parse failure，不能静默截取其中一个动作。

这只验证 pinned remote model、processor、logits、multi-image input 和 parser 接口，
不构成 restoration 效果证据。尤其不能从单个 mixed-fidelity match 推断该事件具有稳定
causal value；下一步仍由预注册的 full-history coverage gate 决定 candidate 是否进入
attribution pilot。

## 运行摘要

| Variant | Images | Input tokens | Peak allocated GPU memory | Executable result |
| --- | ---: | ---: | ---: | --- |
| summary only | 1 | 526 | 16.85 GB | rejected：2 code lines |
| restore event 2 | 3 | 1,014 | 17.05 GB | match：`type_text` |
| full history | 7 | 1,990 | 17.46 GB | rejected：2 code lines |

Logits probe latency 为 0.67 秒，peak allocated GPU memory 为 16.82 GB。Model load
使用 `transformers==4.53.0`、`kernels==0.11.7` 和 `torch==2.11.0+cu130`。

## 可复现命令

```bash
/data/venvs/opencua-transformers-4.53.0/bin/python \
  -m scripts.run_open_cua_policy_smoke \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile-v03/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/OpenCUA-7B \
  --decision-step-id 4 \
  --mixed-restored-step-id 2 \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --max-new-tokens 256 \
  --output-dir /data/experiments/open_cua_policy_smoke_step4
```

完整输出、snapshot hash、latency 与显存见 [`summary.json`](summary.json)。
