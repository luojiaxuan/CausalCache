# OpenCUA-7B frozen-policy coverage

## 结论

OpenCUA-7B 没有通过预注册的 frozen-policy coverage gate，因此不进入 restoration
attribution pilot：

| Action type | Parsed | Match | Coverage |
| --- | ---: | ---: | ---: |
| tap | 6/7 | 1/7 | 14.3% |
| swipe | 1/1 | 0/1 | 0% |
| type_text | 0/1 | 0/1 | 0% |
| **overall** | **7/9** | **1/9** | **11.1%** |

Overall coverage 远低于预注册的 50% threshold，且 swipe、type_text 没有任何 match，
因此 overall gate 与 action-type gate 都失败。

## 失败分析

- decision step 4 输出 `write` 后又输出 `press('enter')`，违反每个 decision 恰好一个
  executable call 的 contract；
- decision step 3 生成非法 Python `pyautogui.click(x=109, y(90))`；
- decision step 8 选择 Threads icon，而 recorded trajectory 需要先 scroll；
- 多个 tap 的目标偏差很大，部分输出坐标还超出 current smart-resized image 并被归一化
  clamp，反映 desktop-oriented grounding 与该 mobile trajectory 不匹配；
- 唯一 match 是 decision step 7 的 share button tap。

这不是 OOM 或上下文长度失败：full-history 从 3 增长到 19 张图，最长输入 5,172
tokens，全部完成 generation；峰值 allocated GPU memory 为 18.71 GB。

## 可复现配置

- Model：`xlangai/OpenCUA-7B`
- Revision：`a2efb7d2b104d477a4a2666a357e79550a28aafc`
- Runtime：`transformers==4.53.0`、`kernels==0.11.7`
- Dataset：`gavinlaw/causalcache-guiodyssey-pilot-mobile`
- Revision：`1de9c34ff029d4c01665cdaca74436ae24bff276`
- Validation：full-history executable match
- Visual budget：每张截图 256 visual tokens

```bash
/data/venvs/opencua-transformers-4.53.0/bin/python \
  -m scripts.run_open_cua_policy_coverage \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-pilot-mobile-v03/data/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/OpenCUA-7B \
  --coverage-gate code/configs/policy_coverage_gate.json \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --max-new-tokens 256 \
  --output-dir /data/experiments/open_cua_policy_coverage_pilot
```

逐 decision 输出、parser error、image count、latency、显存和 snapshot hash 见
[`summary.json`](summary.json)。该结果只用于 frozen-policy candidate selection，不构成
对 CausalCache 方法的验证或证伪。
